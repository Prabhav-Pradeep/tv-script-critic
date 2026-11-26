"""Embed every (name-masked) chunk with a sentence-transformer, once, and save to disk.

MiniLM reads at most 256 tokens, but chunks are ~450 words. So each chunk is
split into pieces of whole lines, each piece is embedded, and the piece vectors are
averaged into one vector per chunk. The label stays at chunk level.

Output: data/processed/chunk_emb.npy  (row i = row i of the key file)
        data/processed/chunk_emb_key.parquet  (ep_id, chunk) in the same order
Run from the repo root:  PYTHONPATH=src .venv/Scripts/python src/embed_chunks.py
"""
import numpy as np
import pandas as pd

from text_utils import find_names, mask_names

MODEL_NAME = "all-MiniLM-L6-v2"
# Masked dialogue runs ~1.6 tokens per word. Measured on 400 chunks: at 140 words only
# 0.4% of pieces exceed 256 tokens (at 180 words it was 34%).
MAX_PIECE_WORDS = 140
BATCH_SIZE = 64


def split_into_pieces(text, max_words=MAX_PIECE_WORDS):
    """Split a chunk into pieces of whole lines, each at most `max_words` words.

    A single line longer than max_words becomes its own piece and MiniLM truncates its
    end. That is rare in dialogue, so it is accepted rather than cutting lines in half.
    """
    pieces, cur, n = [], [], 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        w = len(line.split())
        if cur and n + w > max_words:
            pieces.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += w
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def embed_chunks(texts, model, batch_size=BATCH_SIZE):
    """Embed a list of chunk texts -> (n_chunks, dim) array of mean piece embeddings."""
    pieces, owner = [], []
    for i, t in enumerate(texts):
        ps = split_into_pieces(t)
        pieces.extend(ps)
        owner.extend([i] * len(ps))
    owner = np.asarray(owner)
    vecs = model.encode(pieces, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True)
    out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
    np.add.at(out, owner, vecs)
    out /= np.bincount(owner, minlength=len(texts))[:, None]
    return out, len(pieces)


def main():
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet("data/processed/chunks.parquet")
    names = find_names(df[df.split == "train"].text)       # train only: no test leakage
    masked = [mask_names(t, names) for t in df.text]

    model = SentenceTransformer(MODEL_NAME)
    emb, n_pieces = embed_chunks(masked, model)

    assert emb.shape == (len(df), model.get_sentence_embedding_dimension())
    assert np.isfinite(emb).all()
    print(f"{len(df)} chunks -> {n_pieces} pieces ({n_pieces / len(df):.2f} per chunk), "
          f"embedding dim {emb.shape[1]}")

    np.save("data/processed/chunk_emb.npy", emb)
    df[["ep_id", "chunk"]].to_parquet("data/processed/chunk_emb_key.parquet", index=False)
    print("saved data/processed/chunk_emb.npy and chunk_emb_key.parquet")


if __name__ == "__main__":
    main()
