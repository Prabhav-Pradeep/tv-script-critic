from pathlib import Path

import numpy as np
import pandas as pd
import yaml

raw = Path("data/raw/imdb")
shows = yaml.safe_load(open("config/shows.yaml"))["shows"]
seed = yaml.safe_load(open("config/seed.yaml"))["seed"]
slug_by_title = {v["title"]: k for k, v in shows.items()}
tconst_by_title = {v["title"]: v["tconst"] for v in shows.values()}
CORE = {"csi", "bones", "ncis", "house"}
MIN_SEASON_EPS = 8
TEST_SHARE = 0.20

t = pd.read_csv("data/interim/transcripts_parsed_v1.csv")
t["slug"] = t["show"].map(slug_by_title)
t["parentTconst"] = t["show"].map(tconst_by_title)

# --- IMDb side
read = dict(sep="\t", na_values="\\N", quoting=3, low_memory=False)
episode = pd.read_csv(raw / "title.episode.tsv.gz", **read)
episode = episode[episode.parentTconst.isin(tconst_by_title.values())]
basics = pd.read_csv(raw / "title.basics.tsv.gz",
                     usecols=["tconst", "primaryTitle", "startYear", "runtimeMinutes"], **read)
ratings = pd.read_csv(raw / "title.ratings.tsv.gz", **read)
crew = pd.read_csv(raw / "title.crew.tsv.gz", **read)
imdb = (episode.merge(basics, on="tconst", how="left")
               .merge(ratings, on="tconst", how="left")
               .merge(crew, on="tconst", how="left")
               .rename(columns={"seasonNumber": "imdb_season", "episodeNumber": "imdb_ep",
                                "primaryTitle": "imdb_title"}))

ok = imdb[imdb.numVotes >= 500]
season_stats = (ok.groupby(["parentTconst", "imdb_season"])["averageRating"]
                  .agg(season_mean="mean", season_sd="std", season_n="size").reset_index())
season_len = (imdb.groupby(["parentTconst", "imdb_season"])["imdb_ep"]
                  .max().rename("season_len").reset_index())

# --- master table
t = t.merge(imdb[["tconst", "imdb_season", "imdb_ep", "imdb_title", "startYear",
                  "runtimeMinutes", "directors", "writers"]], on="tconst", how="left")
t = t.merge(season_stats, on=["parentTconst", "imdb_season"], how="left")
t = t.merge(season_len, on=["parentTconst", "imdb_season"], how="left")
t["rating"] = t["averageRating"]
t["rating_z"] = (t["rating"] - t["season_mean"]) / t["season_sd"]
t["position"] = t["imdb_ep"] / t["season_len"]
t["is_premiere"] = t["imdb_ep"] == 1
t["is_finale"] = t["imdb_ep"] == t["season_len"]
t["air_year"] = t["startYear"]
t["transcript_path"] = [
    f"data/processed/transcripts/{s}/s{int(a):02d}e{int(b):02d}.txt"
    for s, a, b in zip(t.slug, t.season, t.ep)]
t["double_episode"] = t["shared_rating"]
t["core"] = t["slug"].isin(CORE)
t["critic_use"] = t["core"] & t["critic_ok"] & t["parse_ok"]

# --- split by season (seeded), per show
rng = np.random.default_rng(seed)
split = {}
for slug, g in t[t.critic_use].groupby("slug"):
    counts = g.groupby("season").size()
    pool = counts[counts >= MIN_SEASON_EPS]
    for s in counts[counts < MIN_SEASON_EPS].index:
        split[(slug, s)] = "small_season"
    total = pool.sum()
    test, n = [], 0
    for s in rng.permutation(pool.index.values):
        if n >= TEST_SHARE * total:
            break
        test.append(s)
        n += pool[s]
    for s in pool.index:
        split[(slug, s)] = "test" if s in test else "train"
t["split"] = [split.get((a, b), "unused") for a, b in zip(t.slug, t.season)]

# --- checks
cu = t[t.critic_use]
assert t.duplicated(["show", "season", "ep"]).sum() == 0
assert cu.rating.between(1, 10).all() and (cu.numVotes >= 500).all()
assert all(Path(p).exists() for p in cu.transcript_path)
assert (cu.groupby(["slug", "season"]).split.nunique() == 1).all()

# --- reports
print(pd.crosstab(cu.show, cu.split, margins=True))
print("\n--- split by season ---")
print(cu.groupby(["show", "season"]).agg(n=("split", "size"), split=("split", "first")).to_string())
print("\n--- rating / within-season z ---")
print(cu[["rating", "rating_z"]].describe().round(3))

Path("data/processed").mkdir(parents=True, exist_ok=True)
t.to_parquet("data/processed/episodes.parquet", index=False)
print("\nsaved", len(t), "rows to data/processed/episodes.parquet")