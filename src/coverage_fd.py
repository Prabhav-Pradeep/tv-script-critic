import json
import pandas as pd
from datasets import load_dataset

ds = load_dataset("sedthh/fd_dialogue")
print(ds)
df = pd.concat([ds[k].to_pandas() for k in ds.keys()], ignore_index=True)
print(df.columns.tolist())
print(df.iloc[0]["METADATA"])

meta = df["METADATA"].apply(lambda m: json.loads(m) if isinstance(m, str) else m).apply(pd.Series)
df = pd.concat([df, meta], axis=1)
keys = ["csi", "bones", "ncis", "supernatural", "house", "criminal", "mentalist", "castle"]
mask = df["show"].str.lower().apply(lambda s: any(k in s for k in keys))
print(df[mask].groupby("show").size())
print(df[mask][["show", "episode"]].head(15).to_string())