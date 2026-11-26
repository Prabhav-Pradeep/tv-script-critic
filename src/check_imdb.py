import pandas as pd
from pathlib import Path

raw = Path("data/raw/imdb")
cols = {
    "title.basics": ["tconst", "titleType", "primaryTitle", "startYear", "endYear", "runtimeMinutes"],
    "title.episode": None,
    "title.ratings": None,
    "title.crew": None,
}
for name, usecols in cols.items():
    df = pd.read_csv(raw / f"{name}.tsv.gz", sep="\t", na_values="\\N",
                     quoting=3, usecols=usecols, low_memory=False)
    print(name, df.shape)
    print(df.head(3), "\n")
    