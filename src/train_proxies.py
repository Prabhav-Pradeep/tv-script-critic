"""Train the proxy critics: the cheap, imperfect critics the writer is optimized against.

For each subset size, K = 5 TF-IDF + ridge critics are trained, each on its own random
subset of train chunks, with the GOLD score as the label (not the IMDb rating). Gold is
the truth by construction; each proxy is an imperfect copy of it. Different subsets give
each proxy different blind spots, which is what the disagreement penalty relies on.

The K subsets are DISJOINT: no chunk is seen by two proxies. With overlapping random
subsets of 4,000 chunks (of 6,330) the proxies correlated 0.985, i.e. clones that never
disagree. Disjointness caps the size at 6,330 / 5, so the sizes are 250 / 500 / 1,200,
three proxy strengths. Weaker proxies should be hacked at lower optimization pressure.

Gold labels on train chunks are the saved gold critic's own outputs, so the target is the
exact function that will later score generated scenes.

Run from the repo root:  PYTHONPATH=src .venv/Scripts/python src/train_proxies.py
"""
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from critics import LexicalCritic, ProxyEnsemble
from text_utils import find_names, mask_names

seed = yaml.safe_load(open("config/seed.yaml"))["seed"]
K = 5
SUBSET_SIZES = [250, 500, 1200]


def main():
    df = pd.read_parquet("data/processed/chunks.parquet")
    g = pd.read_parquet("data/processed/gold_scores.parquet")
    assert (g.ep_id.values == df.ep_id.values).all() and (g.chunk.values == df.chunk.values).all()
    df["gold"] = g.gold.values

    names = find_names(df[df.split == "train"].text)
    df["masked"] = [mask_names(t, names) for t in df.text]
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)

    print("gold vs real rating_z, chunk-level r: train (in-sample) "
          f"{np.corrcoef(tr.gold, tr.rating_z)[0, 1]:.3f}, test {np.corrcoef(te.gold, te.rating_z)[0, 1]:.3f}")
    print(f"\n{'size':>5s} {'proxy~gold r':>13s} {'proxy~proxy r':>14s} {'mean~gold r':>12s} "
          f"{'mean: episode rho vs IMDb':>26s}")

    Path("models").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for size in SUBSET_SIZES:
        assert K * size <= len(tr), "disjoint subsets do not fit in the train chunks"
        order = rng.permutation(len(tr))
        members = []
        for k in range(K):
            idx = order[k * size:(k + 1) * size]          # slices of one permutation: disjoint
            members.append(LexicalCritic().fit(tr.masked.values[idx], tr.gold.values[idx]))
        ens = ProxyEnsemble(names, members, size)

        P = np.column_stack([m.predict(te.masked) for m in members])     # test chunks x K
        r_gold = [np.corrcoef(P[:, k], te.gold)[0, 1] for k in range(K)]
        r_pair = [np.corrcoef(P[:, a], P[:, b])[0, 1] for a, b in combinations(range(K), 2)]
        mean = P.mean(1)
        ep = te.assign(p=mean).groupby("ep_id").agg(p=("p", "mean"), y=("rating_z", "first"))
        print(f"{size:5d} {np.mean(r_gold):13.3f} {np.mean(r_pair):14.3f} "
              f"{np.corrcoef(mean, te.gold)[0, 1]:12.3f} {spearmanr(ep.p, ep.y).statistic:26.3f}")

        joblib.dump(ens, f"models/proxies_n{size}.joblib")

    print(f"\nsaved models/proxies_n{{{','.join(map(str, SUBSET_SIZES))}}}.joblib  (K = {K} each)")


if __name__ == "__main__":
    main()
