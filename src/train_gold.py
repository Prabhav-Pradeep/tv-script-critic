"""Train the gold critic: lexical (TF-IDF + ridge) and semantic (embeddings + ridge), stacked.

Gold is the hidden "truth" of the experiment: it labels the proxies' training data and
scores every generated scene. It must beat the baseline (test episode-level rho = 0.51).

1. Out-of-fold (OOF) predictions on train, GroupKFold by (show, season): every train
   chunk is predicted by models that never saw its season.
2. Combine: (a) average of z-scored predictions, (b) stacking: non-negative weights
   fitted on the OOF predictions. Nothing is ever fitted on test.
3. Refit both critics on all of train, score the test seasons once, report all four.
4. Save the gold critic (models/gold.joblib) and its score for every chunk.

Gold = the equal z-score average, fixed in advance. Stacking puts ~60% of the weight on
TF-IDF and ends up correlating 0.96 with the lexical critic alone; the proxies are TF-IDF
critics too, so they would share gold's blind spots. Equal weighting is as accurate on test
(within the bootstrap CI) and makes half of gold's judgment come from meaning.

Run from the repo root:  PYTHONPATH=src .venv/Scripts/python src/train_gold.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from critics import GoldCritic, LexicalCritic, SemanticCritic
from embed_chunks import MODEL_NAME
from text_utils import find_names, mask_names

seed = yaml.safe_load(open("config/seed.yaml"))["seed"]
N_FOLDS = 5
SEM_ALPHAS = [100, 300, 1000, 3000, 10_000, 30_000, 100_000, 300_000]
N_PERM = 2000


def load_data():
    df = pd.read_parquet("data/processed/chunks.parquet")
    key = pd.read_parquet("data/processed/chunk_emb_key.parquet")
    emb = np.load("data/processed/chunk_emb.npy")
    assert len(key) == len(df) == len(emb)
    assert (key.ep_id.values == df.ep_id.values).all() and (key.chunk.values == df.chunk.values).all(), \
        "embeddings are not aligned with chunks.parquet: rerun embed_chunks.py"
    names = find_names(df[df.split == "train"].text)
    df["masked"] = [mask_names(t, names) for t in df.text]
    df["group"] = df.slug + "_" + df.season.astype(str)
    return df, emb, names


def episode_rho(frame, pred):
    ep = frame.assign(pred=pred).groupby("ep_id").agg(pred=("pred", "mean"), y=("rating_z", "first"))
    return spearmanr(ep.pred, ep.y).statistic


def permutation_p(frame, pred, rng):
    ep = (frame.assign(pred=pred).groupby(["ep_id", "slug"])
               .agg(pred=("pred", "mean"), y=("rating_z", "first")).reset_index())
    obs = spearmanr(ep.pred, ep.y).statistic
    null = [spearmanr(ep.pred, ep.groupby("slug").y.transform(lambda s: rng.permutation(s.values))).statistic
            for _ in range(N_PERM)]
    return obs, (np.sum(np.array(null) >= obs) + 1) / (N_PERM + 1)


def choose_sem_alpha(tr, tr_emb):
    folds = list(GroupKFold(N_FOLDS).split(tr, groups=tr.group))
    cv = {}
    for a in SEM_ALPHAS:
        oof = np.zeros(len(tr))
        for f, v in folds:
            oof[v] = SemanticCritic(a).fit(tr_emb[f], tr.rating_z.values[f]).predict(tr_emb[v])
        cv[a] = episode_rho(tr, oof)
    print("semantic alpha CV (episode rho):", {a: round(float(r), 3) for a, r in cv.items()})
    best = max(cv, key=cv.get)
    assert best not in (SEM_ALPHAS[0], SEM_ALPHAS[-1]), "best alpha is on the grid edge: widen SEM_ALPHAS"
    return best


def oof_predictions(tr, tr_emb, sem_alpha):
    oof_lex, oof_sem = np.zeros(len(tr)), np.zeros(len(tr))
    for f, v in GroupKFold(N_FOLDS).split(tr, groups=tr.group):
        y = tr.rating_z.values[f]
        oof_lex[v] = LexicalCritic().fit(tr.masked.values[f], y).predict(tr.masked.values[v])
        oof_sem[v] = SemanticCritic(sem_alpha).fit(tr_emb[f], y).predict(tr_emb[v])
    return oof_lex, oof_sem


def main():
    df, emb, names = load_data()
    is_tr = (df.split == "train").values
    tr, te = df[is_tr].reset_index(drop=True), df[~is_tr].reset_index(drop=True)
    tr_emb, te_emb = emb[is_tr], emb[~is_tr]
    rng = np.random.default_rng(seed)

    sem_alpha = choose_sem_alpha(tr, tr_emb)
    oof_lex, oof_sem = oof_predictions(tr, tr_emb, sem_alpha)

    # (a) z-score average: scales taken from the OOF predictions, never from test
    mu = np.array([oof_lex.mean(), oof_sem.mean()])
    sd = np.array([oof_lex.std(), oof_sem.std()])
    # (b) stacking: non-negative weights on the honest OOF predictions
    stack = LinearRegression(positive=True).fit(np.column_stack([oof_lex, oof_sem]), tr.rating_z)
    print(f"\nstack weights lexical={stack.coef_[0]:.3f} semantic={stack.coef_[1]:.3f} "
          f"intercept={stack.intercept_:.3f}")
    print("OOF corr(lexical, semantic) =", round(np.corrcoef(oof_lex, oof_sem)[0, 1], 3))
    share = stack.coef_ * sd / np.sum(stack.coef_ * sd)   # weights alone ignore each part's scale
    print(f"stacked: share of its spread from lexical {share[0]:.0%}, semantic {share[1]:.0%}")

    lex = LexicalCritic().fit(tr.masked, tr.rating_z)
    sem = SemanticCritic(sem_alpha).fit(tr_emb, tr.rating_z)
    p_lex, p_sem = lex.predict(te.masked), sem.predict(te_emb)
    variants = {
        "lexical (baseline)": (oof_lex, p_lex),
        "semantic": (oof_sem, p_sem),
        "z-average (gold)": (((np.column_stack([oof_lex, oof_sem]) - mu) / sd).mean(1),
                             ((np.column_stack([p_lex, p_sem]) - mu) / sd).mean(1)),
        "stacked": (stack.predict(np.column_stack([oof_lex, oof_sem])),
                    stack.predict(np.column_stack([p_lex, p_sem]))),
    }
    print(f"\n{'critic':20s} {'train OOF rho':>13s} {'TEST rho':>9s} {'perm p':>8s}")
    for name, (oof, test) in variants.items():
        rho, p = permutation_p(te, test, rng)
        print(f"{name:20s} {episode_rho(tr, oof):13.3f} {rho:9.3f} {p:8.4f}")

    gold_test = variants["z-average (gold)"][1]
    print("\nTEST episode rho by show (gold):")
    for s, idx in te.groupby("slug").groups.items():
        print(f"  {s:6s} {episode_rho(te.loc[idx], gold_test[idx]):.3f}")
    print("chunk-level corr(gold, lexical) on test:", round(np.corrcoef(gold_test, p_lex)[0, 1], 3),
          "| corr(stacked, lexical):", round(np.corrcoef(variants["stacked"][1], p_lex)[0, 1], 3))

    # z-average as a linear map: 0.5*(p - mu)/sd summed over both critics
    weights = 0.5 / sd
    intercept = -np.sum(0.5 * mu / sd)
    gold = GoldCritic(names, lex, sem, weights, intercept, MODEL_NAME)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(gold, "models/gold.joblib")

    # score every chunk with the saved object, through the same path generated text will take
    gold = joblib.load("models/gold.joblib")
    df["gold"] = gold.score(df.text.tolist(), emb=emb)
    assert np.allclose(df.loc[~is_tr, "gold"].values, gold_test)
    df[["ep_id", "chunk", "split", "rating_z", "gold"]].to_parquet("data/processed/gold_scores.parquet", index=False)
    print("\nsaved models/gold.joblib and data/processed/gold_scores.parquet")


if __name__ == "__main__":
    main()
