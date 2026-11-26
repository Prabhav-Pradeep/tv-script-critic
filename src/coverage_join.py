import json, re
from pathlib import Path
import pandas as pd
import yaml
from datasets import load_dataset

raw = Path("data/raw/imdb")
shows = yaml.safe_load(open("config/shows.yaml"))["shows"]
tconst_by_title = {v["title"]: v["tconst"] for v in shows.values()}

# --- transcripts (fd_dialogue)
ds = load_dataset("sedthh/fd_dialogue")["train"].to_pandas()
meta = ds["METADATA"].apply(lambda m: json.loads(m) if isinstance(m, str) else m).apply(pd.Series)
fd = pd.concat([ds, meta], axis=1)
fd = fd[fd["show"].isin(tconst_by_title)].copy()
parsed = fd["episode"].str.extract(r"^\s*(\d+)x(\d+)\s*-\s*(.*)$")
fd["season"] = pd.to_numeric(parsed[0])
fd["ep"] = pd.to_numeric(parsed[1])
fd["fd_title"] = parsed[2]
print("unparsed episode strings:", fd["season"].isna().sum())
print(fd[fd["season"].isna()]["episode"].head(10).to_string())
fd["parentTconst"] = fd["show"].map(tconst_by_title)

print("\n--- sample transcript (first 1200 chars) ---")
print(fd.iloc[0]["TEXT"][:1200])

# --- IMDb episodes
read = dict(sep="\t", na_values="\\N", quoting=3, low_memory=False)
episode = pd.read_csv(raw / "title.episode.tsv.gz", **read)
episode = episode[episode.parentTconst.isin(tconst_by_title.values())]
basics = pd.read_csv(raw / "title.basics.tsv.gz", usecols=["tconst", "primaryTitle"], **read)
ratings = pd.read_csv(raw / "title.ratings.tsv.gz", **read)
imdb = (episode.merge(basics, on="tconst", how="left")
               .merge(ratings, on="tconst", how="left")
               .rename(columns={"seasonNumber": "season", "episodeNumber": "ep",
                                "primaryTitle": "imdb_title"}))

# --- join
j = fd.merge(imdb, on=["parentTconst", "season", "ep"], how="left")
print("duplicate show/season/ep rows after join:", j.duplicated(["show", "season", "ep"]).sum())
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
j["title_agree"] = [norm(a) == norm(b) for a, b in zip(j["fd_title"], j["imdb_title"])]
j["ok_votes"] = j["numVotes"] >= 500
summary = j.groupby("show").agg(
    n_transcripts=("TEXT", "size"),
    n_matched=("tconst", "count"),
    n_title_agree=("title_agree", "sum"),
    n_500plus=("ok_votes", "sum"),
)
print(summary)
print("TOTAL matched:", summary.n_matched.sum(), " with 500+ votes:", summary.n_500plus.sum())

print("\n--- title disagreements (first 15) ---")
print(j[~j.title_agree][["show", "season", "ep", "fd_title", "imdb_title"]].head(15).to_string())

print("\n--- transcripts per season ---")
print(j.pivot_table(index="show", columns="season", values="TEXT", aggfunc="size", fill_value=0).to_string())

j.drop(columns=["TEXT"]).to_csv("data/interim/fd_join_v0.csv", index=False)