import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

raw = Path("data/raw/imdb")
shows = yaml.safe_load(open("config/shows.yaml"))["shows"]
tconsts = [v["tconst"] for v in shows.values()]
THRESH = 0.8

# transcript side: one row per transcript, from the earlier join
fd = pd.read_csv("data/interim/fd_join_v0.csv")
fd = fd[["show", "episode", "season", "ep", "fd_title", "parentTconst"]].copy()

# IMDb side: all episodes of the chosen series
read = dict(sep="\t", na_values="\\N", quoting=3, low_memory=False)
episode = pd.read_csv(raw / "title.episode.tsv.gz", **read)
episode = episode[episode.parentTconst.isin(tconsts)]
basics = pd.read_csv(raw / "title.basics.tsv.gz", usecols=["tconst", "primaryTitle"], **read)
ratings = pd.read_csv(raw / "title.ratings.tsv.gz", **read)
imdb = (episode.merge(basics, on="tconst", how="left")
               .merge(ratings, on="tconst", how="left")
               .rename(columns={"seasonNumber": "season", "episodeNumber": "ep",
                                "primaryTitle": "imdb_title"}))


def norm(s):
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)          # (1), (Part II)
    s = re.sub(r"\bparts?\b.*$", " ", s)    # ": Part 2", ": Parts 1 and 2"
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sim(a, b):
    if not isinstance(b, str):
        return 0.0
    x, y = norm(a), norm(b)
    return max(SequenceMatcher(None, x, y).ratio(),
               SequenceMatcher(None, x.replace(" ", ""), y.replace(" ", "")).ratio())


rows = []
for (show, season), F in fd.groupby(["show", "season"]):
    F = F.reset_index(drop=True)
    I = imdb[(imdb.parentTconst == F["parentTconst"].iloc[0]) & (imdb.season == season)].reset_index(drop=True)
    if I.empty:
        for _, f in F.iterrows():
            rows.append({**f.to_dict(), "matched": False, "sim": 0.0, "shared_rating": False})
        continue
    S = np.array([[sim(f.fd_title, i.imdb_title) for _, i in I.iterrows()] for _, f in F.iterrows()])
    same_ep = np.array([[f.ep == i.ep for _, i in I.iterrows()] for _, f in F.iterrows()])
    close = S >= THRESH
    r, c = linear_sum_assignment(1 - S - 0.05 * same_ep)
    assigned = {ri: ci for ri, ci in zip(r, c) if S[ri, ci] >= THRESH}
    for ri, f in F.iterrows():
        rec = f.to_dict()
        R = np.flatnonzero(close[ri])
        # resembles exactly one IMDb row that other transcripts also resemble -> merged two-parter
        shared = len(R) == 1 and close[:, R[0]].sum() >= 2
        ci = assigned.get(ri)
        if ci is None:
            rec.update(matched=False, sim=float(S[ri].max()), shared_rating=shared)
        else:
            i = I.iloc[ci]
            rec.update(matched=True, sim=float(S[ri, ci]), shared_rating=shared,
                       imdb_ep=i.ep, tconst=i.tconst, imdb_title=i.imdb_title,
                       averageRating=i.averageRating, numVotes=i.numVotes,
                       shifted=bool(f.ep != i.ep))
        rows.append(rec)

m = pd.DataFrame(rows)
for col in ["matched", "shifted", "shared_rating"]:
    m[col] = m[col].fillna(False).astype(bool)
m["critic_ok"] = m.matched & ~m.shared_rating & (m.numVotes >= 500)

summary = m.groupby("show").agg(
    n_transcripts=("matched", "size"), n_matched=("matched", "sum"),
    n_shifted=("shifted", "sum"), n_shared=("shared_rating", "sum"),
    n_critic_ok=("critic_ok", "sum"))
print(summary)
print(summary.sum())
print("\n--- unmatched ---")
print(m[~m.matched][["show", "season", "ep", "fd_title", "sim"]].to_string())
print("\n--- shifted episodes per show/season ---")
print(m[m.shifted].groupby(["show", "season"]).size())
print("\n--- matched but sim < 0.9 (spot-check) ---")
print(m[m.matched & (m.sim < 0.9)][["show", "season", "ep", "imdb_ep", "fd_title", "imdb_title", "sim"]].to_string())

m.to_csv("data/interim/fd_imdb_matched_v1.csv", index=False)