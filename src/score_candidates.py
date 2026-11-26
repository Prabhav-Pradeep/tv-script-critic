"""Score every generated candidate with the gold critic and all proxy ensembles.

Input:  data/generated/candidates.jsonl  (from notebooks/generate_colab.ipynb)
Output: data/generated/scores.parquet     one row per candidate:
          prompt_id, cand_id, n_words, finish_reason, gold, p{size}_{k} for every proxy
        data/generated/cand_emb.npy       cached embeddings (the slow part, ~10 min on CPU)

Run from the repo root:  PYTHONPATH=src .venv/Scripts/python src/score_candidates.py
Options: --cands PATH --out DIR   (used for dry runs on fake candidates)
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from embed_chunks import embed_chunks
from text_utils import mask_names

PROXY_SIZES = [250, 500, 1200]
# chat-model boilerplate that is not part of the scene
PREAMBLE = re.compile(r"^\s*(here('s| is)\b.*|sure[,!.].*|continuation:?|scene (so far|continues?):?)\s*$", re.I)


def clean(text):
    """Strip a leading 'Here's the continuation:'-style line and surrounding whitespace."""
    lines = text.strip().split("\n")
    while lines and (not lines[0].strip() or PREAMBLE.match(lines[0])):
        lines.pop(0)
    return "\n".join(lines).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="data/generated/candidates.jsonl")
    ap.add_argument("--out", default="data/generated")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.cands, encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["raw"] = df.text
    df["text"] = df.raw.map(clean)
    df["n_words"] = df.text.str.split().str.len()
    print(f"{len(df)} candidates over {df.prompt_id.nunique()} prompts; "
          f"{(df.text != df.raw.str.strip()).sum()} had a preamble stripped; "
          f"{(df.n_words < 100).sum()} shorter than 100 words")

    gold = joblib.load("models/gold.joblib")
    emb_path = out / "cand_emb.npy"
    if emb_path.exists() and len(np.load(emb_path)) == len(df):
        emb = np.load(emb_path)
        print("reusing cached embeddings", emb_path)
    else:
        masked = [mask_names(t, gold.names) for t in df.text]
        emb, n_pieces = embed_chunks(masked, gold.embedder())
        np.save(emb_path, emb)
        print(f"embedded {n_pieces} pieces -> {emb_path}")
    df["gold"] = gold.score(df.text.tolist(), emb=emb)

    for size in PROXY_SIZES:
        ens = joblib.load(f"models/proxies_n{size}.joblib")
        P = ens.score(df.text.tolist())
        for k in range(P.shape[1]):
            df[f"p{size}_{k}"] = P[:, k]

    keep = ["prompt_id", "cand_id", "n_words", "finish_reason", "gold"] + \
           [c for c in df.columns if re.fullmatch(r"p\d+_\d+", c)]
    df[keep].to_parquet(out / "scores.parquet", index=False)
    print(f"saved {out / 'scores.parquet'}")
    print("\ngold score per prompt (mean, sd):")
    print(df.groupby("prompt_id").gold.agg(["mean", "std"]).round(3).T.to_string())


if __name__ == "__main__":
    main()
