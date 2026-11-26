"""Feasibility check: can a text critic predict within-season episode quality at all?

TF-IDF + ridge regression on chunks -> rating_z. The ridge strength is chosen by
cross-validation grouped by (show, season) on the train seasons only; the test seasons
are scored once at the end. Episode-level score = mean of its chunk predictions.
"""
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from text_utils import find_names, mask_names

seed = yaml.safe_load(open("config/seed.yaml"))["seed"]
ALPHAS = [1, 3, 10, 30, 100, 300]
MASK = "--mask" in sys.argv

df = pd.read_parquet("data/processed/chunks.parquet")
if MASK:
    names = find_names(df[df.split == "train"].text)   # names learned from train text only
    print(f"masking {len(names)} names, e.g. {sorted(names)[:12]}")
    df["text"] = [mask_names(t, names) for t in df.text]
tr, te = df[df.split == "train"].copy(), df[df.split == "test"].copy()
groups = tr.slug + "_" + tr.season.astype(str)


def make_vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, sublinear_tf=True,
                           max_features=50_000, lowercase=True)


def episode_spearman(frame, pred):
    ep = frame.assign(pred=pred).groupby("ep_id").agg(pred=("pred", "mean"), y=("rating_z", "first"))
    return spearmanr(ep.pred, ep.y).statistic


# --- choose alpha with season-grouped CV on train only
cv = GroupKFold(n_splits=5)
scores = {a: [] for a in ALPHAS}
for fit_idx, val_idx in cv.split(tr, groups=groups):
    f, v = tr.iloc[fit_idx], tr.iloc[val_idx]
    vec = make_vec().fit(f.text)
    Xf, Xv = vec.transform(f.text), vec.transform(v.text)
    for a in ALPHAS:
        p = Ridge(alpha=a).fit(Xf, f.rating_z).predict(Xv)
        scores[a].append(episode_spearman(v, p))
cv_table = pd.Series({a: np.mean(s) for a, s in scores.items()}).round(3)
print("season-grouped CV, episode-level Spearman by alpha:\n", cv_table.to_string())
best = cv_table.idxmax()

# --- fit on all train, score test once
vec = make_vec().fit(tr.text)
model = Ridge(alpha=best).fit(vec.transform(tr.text), tr.rating_z)
te["pred"] = model.predict(vec.transform(te.text))

chunk_r = pearsonr(te.pred, te.rating_z).statistic
ep = te.groupby(["ep_id", "slug"]).agg(pred=("pred", "mean"), y=("rating_z", "first")).reset_index()
ep_rho = spearmanr(ep.pred, ep.y).statistic

# permutation null: shuffle episode labels within show, refit nothing, just re-score
rng = np.random.default_rng(seed)
null = []
for _ in range(2000):
    yy = ep.groupby("slug").y.transform(lambda s: rng.permutation(s.values))
    null.append(spearmanr(ep.pred, yy).statistic)
p_perm = (np.sum(np.array(null) >= ep_rho) + 1) / (len(null) + 1)

print(f"\nbest alpha = {best}")
print(f"TEST chunk-level Pearson r     = {chunk_r:.3f}  (n={len(te)} chunks)")
print(f"TEST episode-level Spearman rho = {ep_rho:.3f}  (n={len(ep)} episodes, permutation p = {p_perm:.4f})")
print("\nper show (episode-level Spearman):")
for s, g in ep.groupby("slug"):
    print(f"  {s:6s} n={len(g):3d}  rho={spearmanr(g.pred, g.y).statistic:.3f}")

terms = np.array(vec.get_feature_names_out())
order = np.argsort(model.coef_)
print("\nterms pushing the score UP:  ", ", ".join(terms[order[-25:]][::-1]))
print("terms pushing the score DOWN:", ", ".join(terms[order[:25]]))
