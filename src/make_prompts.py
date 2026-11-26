"""Pick the scene openings the writer will continue, and write them for the Colab notebook.

12 prompts: 3 per show, each the first ~80 words (whole lines) of a chunk from a TEST season.
The critics never trained on test-season text, so a prompt cannot steer the writer toward
text they memorized. Each episode's first chunk is skipped (often a "Previously on" recap),
and at most one prompt is taken per episode.

Output: data/processed/prompts.jsonl  (git-ignored: it contains transcript text)
Run from the repo root:  PYTHONPATH=src .venv/Scripts/python src/make_prompts.py
"""
import json

import numpy as np
import pandas as pd
import yaml

seed = yaml.safe_load(open("config/seed.yaml"))["seed"]
shows = yaml.safe_load(open("config/shows.yaml"))["shows"]
PER_SHOW = 3
OPENING_WORDS = 80
GENRE = {"csi": "a forensic crime drama", "bones": "a forensic crime drama",
         "ncis": "a naval crime drama", "house": "a medical drama"}


def opening(text, n_words=OPENING_WORDS):
    out, n = [], 0
    for line in text.split("\n"):
        out.append(line)
        n += len(line.split())
        if n >= n_words:
            break
    return "\n".join(out)


def main():
    df = pd.read_parquet("data/processed/chunks.parquet")
    pool = df[(df.split == "test") & (df.chunk > 0)]
    rng = np.random.default_rng(seed)
    rows = []
    for slug, g in pool.groupby("slug"):
        eps = rng.choice(g.ep_id.unique(), size=PER_SHOW, replace=False)
        for ep in eps:
            c = g[g.ep_id == ep].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            rows.append(dict(prompt_id=len(rows), slug=slug, show=shows[slug]["title"],
                             genre=GENRE[slug], ep_id=ep, chunk=int(c.chunk),
                             opening=opening(c.text)))
    with open("data/processed/prompts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        print(f"[{r['prompt_id']:2d}] {r['slug']:6s} {len(r['opening'].split()):3d} words | "
              + r["opening"][:70].replace("\n", " / "))
    print(f"\nsaved {len(rows)} prompts to data/processed/prompts.jsonl")


if __name__ == "__main__":
    main()
