import re
from difflib import SequenceMatcher
import pandas as pd

j = pd.read_csv("data/interim/fd_join_v0.csv")

def norm(s):
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

j["sim"] = [SequenceMatcher(None, norm(a), norm(b)).ratio() if isinstance(b, str) else 0.0
            for a, b in zip(j["fd_title"], j["imdb_title"])]
j["title_ok"] = j["sim"] >= 0.8

print(j.groupby("show").agg(n=("sim", "size"), n_ok=("title_ok", "sum"), min_sim=("sim", "min")))
print("\n--- still disagreeing (sim < 0.8) ---")
print(j[~j.title_ok][["show", "season", "ep", "fd_title", "imdb_title", "sim"]].to_string())