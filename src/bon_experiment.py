"""Best-of-N experiment: does a disagreement-penalized critic ensemble resist reward hacking?

For each prompt (pool of M = 256 candidates) and each selection rule, compute exactly the
expected GOLD score of the best-of-N pick for N = 1, 2, 4, ..., 256 (see bon.py). Rules,
applied to the 5 proxy scores of each candidate:

    single      one proxy (curve averaged over the 5 proxies, each used alone)
    mean        mean of the 5 proxies
    mean-1sd    mean - 1 x standard deviation across the 5 (disagreement penalty)
    mean-2sd    mean - 2 x standard deviation
    min         the lowest of the 5 (worst case)
    gold        select on gold itself: the ceiling no proxy can beat

Scores are expressed as improvement over the pool average, in units of that pool's standard
deviation, then averaged over the 12 prompts. Confidence intervals: bootstrap over prompts.
Hacking shows as proxy improvement that keeps rising while gold improvement stalls or falls.

Outputs: reports/bon_results.csv, reports/bon_curves.png, a summary on stdout.
Run from the repo root:  PYTHONPATH=src .venv/Scripts/python src/bon_experiment.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from bon import bon_kl, expected_best  # noqa: E402

seed = yaml.safe_load(open("config/seed.yaml"))["seed"]
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SIZES = [250, 500, 1200]
MAIN_SIZE = 500
N_BOOT = 2000
RULES = ["single", "mean", "mean-1sd", "mean-2sd", "min"]
# validated categorical slots 1-5 (dataviz reference palette, light mode)
COLORS = dict(zip(RULES, ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]))
MARKERS = dict(zip(RULES, ["o", "s", "^", "D", "v"]))
INK, INK2, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"


def selection_scores(P, rule):
    """P: (M, K) proxy scores. Returns a list of selection-score vectors (one per curve)."""
    if rule == "single":
        return [P[:, k] for k in range(P.shape[1])]
    mean, sd = P.mean(1), P.std(1, ddof=1)
    return [{"mean": mean, "mean-1sd": mean - sd, "mean-2sd": mean - 2 * sd,
             "min": P.min(1)}[rule]]


def curves_for_prompt(g, size):
    """Improvement over the pool mean, in pool SD units, for gold and for the mean proxy."""
    P = g[[c for c in g.columns if re.fullmatch(fr"p{size}_\d+", c)]].to_numpy()
    gold, pmean, words = g.gold.to_numpy(), P.mean(1), g.n_words.to_numpy()
    z = lambda v: (v - v.mean()) / v.std()       # noqa: E731
    zg, zp = z(gold), z(pmean)
    out = []
    for rule in RULES + ["gold"]:
        sels = [gold] if rule == "gold" else selection_scores(P, rule)
        for N in NS:
            out.append(dict(rule=rule, N=N,
                            d_gold=np.mean([expected_best(s, zg, N) for s in sels]),
                            d_proxy=np.mean([expected_best(s, zp, N) for s in sels]),
                            words=np.mean([expected_best(s, words, N) for s in sels])))
    return pd.DataFrame(out)


def bootstrap_ci(per_prompt, rng):
    """per_prompt: (n_prompts,) values -> (lo, hi) of the mean, resampling prompts."""
    idx = rng.integers(0, len(per_prompt), size=(N_BOOT, len(per_prompt)))
    return np.percentile(per_prompt[idx].mean(1), [2.5, 97.5])


def main():
    df = pd.read_parquet("data/generated/scores.parquet")
    rng = np.random.default_rng(seed)
    Path("reports").mkdir(exist_ok=True)

    # --- length: do the critics simply prefer longer scenes?
    print("within-prompt Spearman with scene length (mean over prompts):")
    for col in ["gold"] + [f"p{s}_0" for s in SIZES]:
        r = np.mean([spearmanr(g[col], g.n_words).statistic for _, g in df.groupby("prompt_id")])
        print(f"  {col:8s} {r:+.3f}")

    per = []
    for size in SIZES:
        for pid, g in df.groupby("prompt_id"):
            per.append(curves_for_prompt(g, size).assign(size=size, prompt_id=pid))
    per = pd.concat(per, ignore_index=True)

    rows = []
    for (size, rule, N), g in per.groupby(["size", "rule", "N"]):
        lo, hi = bootstrap_ci(g.d_gold.to_numpy(), rng)
        rows.append(dict(size=size, rule=rule, N=N, kl=bon_kl(N), d_gold=g.d_gold.mean(),
                         d_gold_lo=lo, d_gold_hi=hi, d_proxy=g.d_proxy.mean(), words=g.words.mean()))
    res = pd.DataFrame(rows)
    res.round(4).to_csv("reports/bon_results.csv", index=False)

    # --- summary: peak, value at N = 256, drop from peak
    print(f"\n{'size':>5s} {'rule':9s} {'gold@256':>9s} {'95% CI':>17s} {'peak':>6s} {'@N':>4s} "
          f"{'drop':>6s} {'proxy@256':>10s} {'words@256':>10s}")
    for (size, rule), g in res.groupby(["size", "rule"], sort=False):
        last, peak = g[g.N == 256].iloc[0], g.loc[g.d_gold.idxmax()]
        print(f"{size:5d} {rule:9s} {last.d_gold:+9.3f} [{last.d_gold_lo:+.3f}, {last.d_gold_hi:+.3f}] "
              f"{peak.d_gold:+6.3f} {int(peak.N):4d} {last.d_gold - peak.d_gold:+6.3f} "
              f"{last.d_proxy:+10.3f} {last.words:10.0f}")

    # --- paired differences at N = 256, main proxy size: each rule vs single and vs mean
    print(f"\npaired differences in gold at N = 256 (proxy size {MAIN_SIZE}, bootstrap over prompts):")
    at = per[(per["size"] == MAIN_SIZE) & (per.N == 256)].pivot(index="prompt_id", columns="rule", values="d_gold")
    for a, b in [("mean", "single"), ("mean-1sd", "mean"), ("mean-2sd", "mean"), ("min", "mean"),
                 ("mean-1sd", "single")]:
        d = (at[a] - at[b]).to_numpy()
        lo, hi = bootstrap_ci(d, rng)
        print(f"  {a:9s} - {b:7s} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  better in {np.sum(d > 0)}/{len(d)} prompts")

    plot(res)
    print("\nsaved reports/bon_results.csv and reports/bon_curves.png")


def dodge(values, min_gap):
    """Spread label y-positions so neighbours are at least `min_gap` apart, keeping order."""
    order = np.argsort(values)
    pos = np.asarray(values, float)[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    shift = (pos - np.asarray(values, float)[order]).mean()   # recentre around the data
    out = np.empty_like(pos)
    out[order] = pos - shift
    return out


def plot(res):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(2, len(SIZES), figsize=(13, 7.6), sharex=True, sharey="row")
    x = np.log2(NS)
    for j, size in enumerate(SIZES):
        for i, (col, ylabel) in enumerate([("d_gold", "Gold score gain (pool SD)"),
                                           ("d_proxy", "Proxy score gain (pool SD)")]):
            ax = axes[i, j]
            ax.set_facecolor("white")
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            g = res[(res["size"] == size) & (res.rule == "gold")]
            ax.plot(x, g[col], color=INK2, linestyle="--", linewidth=1.5, zorder=2,
                    label="select on gold (ceiling)")
            ends = {"select on gold (ceiling)": g[col].iloc[-1]}
            for rule in RULES:
                g = res[(res["size"] == size) & (res.rule == rule)]
                if col == "d_gold":
                    ax.fill_between(x, g.d_gold_lo, g.d_gold_hi, color=COLORS[rule], alpha=0.10,
                                    linewidth=0, zorder=1)
                ax.plot(x, g[col], color=COLORS[rule], linewidth=2, marker=MARKERS[rule],
                        markersize=5, markeredgecolor="white", markeredgewidth=1, zorder=3,
                        label=rule)
                ends[rule] = g[col].iloc[-1]
            if j == len(SIZES) - 1:                       # direct labels, spaced apart
                lo, hi = res[col].min(), res[col].max()
                ys = dodge(list(ends.values()), min_gap=0.055 * (hi - lo))
                for (name, y_end), y_lab in zip(ends.items(), ys):
                    ax.annotate(name, (x[-1], y_end), xytext=(x[-1] + 0.35, y_lab),
                                textcoords="data", va="center", fontsize=8.5,
                                color=INK2 if name.startswith("select") else INK,
                                arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))
            ax.axhline(0, color=INK2, linewidth=0.8)
            if i == 0:
                ax.set_title(f"Proxies trained on {size} chunks each", fontsize=10.5, color=INK)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xticks(x, [str(n) for n in NS])
                ax.set_xlabel("N (best of N candidates)")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Best-of-N against proxy critics: true (gold) gain vs what the proxies see",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "12 prompts x 256 candidates. Gain = expected score of the pick minus "
             "the pool mean, in units of the pool's SD. Shaded: 95% bootstrap CI over prompts.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 0.93, 0.96))
    fig.savefig("reports/bon_curves.png", dpi=160, facecolor="white")


if __name__ == "__main__":
    main()
