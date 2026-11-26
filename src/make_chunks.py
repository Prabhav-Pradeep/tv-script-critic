"""Cut each critic-usable transcript into ~450-word chunks.

Scene markers are inconsistent across shows (CSI uses [INT. ...], House uses free-form
brackets), so chunks are fixed word windows built from whole lines. Each chunk inherits
its episode's within-season rating z-score and its season-level train/test split.
"""
from pathlib import Path

import pandas as pd

TARGET_WORDS = 450   # a chunk closes once it reaches this many words
MIN_TAIL = 200       # a leftover tail shorter than this is dropped


def chunk_lines(text, target=TARGET_WORDS, min_tail=MIN_TAIL):
    chunks, cur, n = [], [], 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        cur.append(line)
        n += len(line.split())
        if n >= target:
            chunks.append("\n".join(cur))
            cur, n = [], 0
    if n >= min_tail:
        chunks.append("\n".join(cur))
    return chunks


def main():
    eps = pd.read_parquet("data/processed/episodes.parquet")
    eps = eps[eps.critic_use & eps.split.isin(["train", "test"])]
    rows = []
    for e in eps.itertuples():
        text = Path(e.transcript_path).read_text(encoding="utf-8")
        for i, c in enumerate(chunk_lines(text)):
            rows.append(dict(ep_id=e.tconst, slug=e.slug, season=int(e.season), ep=int(e.ep),
                             chunk=i, text=c, n_words=len(c.split()),
                             rating=e.rating, rating_z=e.rating_z, split=e.split))
    df = pd.DataFrame(rows)

    assert df.ep_id.nunique() == len(eps), "every usable episode must yield chunks"
    assert (df.groupby(["slug", "season"]).split.nunique() == 1).all(), "split must stay per season"

    print(df.groupby(["slug", "split"]).agg(episodes=("ep_id", "nunique"), chunks=("chunk", "size")).unstack())
    print("\nchunks per episode:", df.groupby("ep_id").size().describe().round(1).to_dict())
    print("words per chunk:", df.n_words.describe().round(0).to_dict())
    df.to_parquet("data/processed/chunks.parquet", index=False)
    print(f"\nsaved {len(df)} chunks to data/processed/chunks.parquet")


if __name__ == "__main__":
    main()
