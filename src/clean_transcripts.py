import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

shows = yaml.safe_load(open("config/shows.yaml"))["shows"]
slug_by_title = {v["title"]: k for k, v in shows.items()}

m = pd.read_csv("data/interim/fd_imdb_matched_v1.csv")
ds = load_dataset("sedthh/fd_dialogue")["train"].to_pandas()
meta = ds["METADATA"].apply(lambda x: json.loads(x) if isinstance(x, str) else x).apply(pd.Series)
fd = pd.concat([ds[["TEXT"]], meta], axis=1)
fd = fd[fd["show"].isin(slug_by_title)]
df = fd.merge(
    m[["show", "episode", "season", "ep", "tconst", "matched", "shared_rating",
       "critic_ok", "averageRating", "numVotes"]],
    on=["show", "episode"], how="left")
assert len(df) == len(m), (len(df), len(m))

HEADER = re.compile(
    r"^(episode\s+\d+x\d+|written\s+by|teleplay|story\s+by|directed\s+by|transcribed\s+by|"
    r"transcript\s+by|transcriber|disclaimer|original\s+air\s*date|air\s*date)\b", re.I)
TITLE = re.compile(r'^["\u201c].*["\u201d]$')
TRANSCRIBER = re.compile(r"^(transcribed|transcript)\s+by\b", re.I)

STRUCT = re.compile(
    r"^\[?\s*(teaser|cold open|prologue|epilogue|tag|act\s+(one|two|three|four|five|six|\d+)|"
    r"end of act.*|fade (in|out).*|music (in|out)|commercial.*|opening credits|main titles|"
    r"end credits|credits)\s*[:.\]]*\s*$", re.I)
ALLCAPS_SHORT = re.compile(r"^[A-Z0-9 .'\-:]{3,40}$")
SCENE = re.compile(r"^\[?\s*(INT|EXT)\b", re.I)
DLG = re.compile(r"^([A-Za-z][A-Za-z0-9 .'\-/&,]{0,40}?):\s+(\S.*)$")


def clean(text):
    out, started = [], False
    for line in str(text).replace("\r", "").split("\n"):
        s = line.strip()
        if TRANSCRIBER.match(s):
            continue
        if not started:
            if not s or HEADER.match(s) or TITLE.match(s):
                continue
            started = True
        out.append(s)
    txt = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", txt).strip()


def classify(line):
    if STRUCT.match(line) or (ALLCAPS_SHORT.match(line) and ":" not in line.rstrip(":")
                              and any(ch.isalpha() for ch in line)):
        return "struct"
    if SCENE.match(line):
        return "scene"
    if line[0] in "[(":
        return "dir"
    if DLG.match(line):
        return "dlg"
    return "other"


def stats(txt):
    counts = {"struct": 0, "scene": 0, "dir": 0, "dlg": 0, "other": 0}
    speakers = set()
    for line in txt.split("\n"):
        if not line:
            continue
        kind = classify(line)
        counts[kind] += 1
        if kind == "dlg":
            speakers.add(DLG.match(line).group(1).strip().lower())
    total = max(sum(counts.values()), 1)
    return pd.Series(dict(n_dlg=counts["dlg"], n_dir=counts["dir"], n_scene=counts["scene"],
                          n_struct=counts["struct"], n_other=counts["other"],
                          share_other=counts["other"] / total, n_speakers=len(speakers)))


df["clean"] = df["TEXT"].apply(clean)
df["n_words"] = df["clean"].str.split().str.len()
df = pd.concat([df, df["clean"].apply(stats)], axis=1)

med = df.groupby("show")["n_words"].transform("median")
df["parse_ok"] = ((df.share_other <= 0.15) & (df.n_dlg >= 50)
                  & (df.n_words >= 0.4 * med) & (df.n_words <= 2.5 * med))

out_dir = Path("data/processed/transcripts")
for r in df.itertuples():
    d = out_dir / slug_by_title[r.show]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"s{int(r.season):02d}e{int(r.ep):02d}.txt").write_text(r.clean, encoding="utf-8")

print(df.groupby("show").agg(n=("clean", "size"), parse_ok=("parse_ok", "sum"),
                             median_words=("n_words", "median"),
                             median_speakers=("n_speakers", "median"),
                             median_share_other=("share_other", "median")))
print("\n--- parse_ok == False: count per show/season ---")
print(df[~df.parse_ok].groupby(["show", "season"]).size().to_string())

others = [(r.show, line) for r in df.itertuples() for line in r.clean.split("\n")
          if line and classify(line) == "other"]
if others:
    o = pd.DataFrame(others, columns=["show", "line"])
    print("\n--- 8 random 'other' lines per show (what is still unclassified) ---")
    for show, g in o.groupby("show"):
        print(f"\n[{show}]")
        for line in g["line"].sample(min(8, len(g)), random_state=0):
            print("  ", line[:160])

df.drop(columns=["TEXT", "clean"]).to_csv("data/interim/transcripts_parsed_v1.csv", index=False)