import pandas as pd
from pathlib import Path

raw = Path("data/raw/imdb")
candidates = ["CSI: Crime Scene Investigation", "Bones", "NCIS", "Supernatural", "House",
              "Criminal Minds", "The Mentalist", "Castle"]

basics = pd.read_csv(raw / "title.basics.tsv.gz", sep="\t", na_values="\\N", quoting=3,
                     usecols=["tconst", "titleType", "primaryTitle", "startYear", "endYear"],
                     low_memory=False)
series = basics[(basics.titleType == "tvSeries") & basics.primaryTitle.isin(candidates)]

episode = pd.read_csv(raw / "title.episode.tsv.gz", sep="\t", na_values="\\N", quoting=3,
                      low_memory=False)
ratings = pd.read_csv(raw / "title.ratings.tsv.gz", sep="\t", na_values="\\N", quoting=3,
                      low_memory=False)

ep = episode[episode.parentTconst.isin(series.tconst)].merge(ratings, on="tconst", how="left")
stats = ep.groupby("parentTconst").agg(
    n_episodes=("tconst", "size"),
    n_rated=("averageRating", "count"),
    n_500plus=("numVotes", lambda s: (s >= 500).sum()),
)
out = (series.merge(stats, left_on="tconst", right_index=True, how="left")
             .merge(ratings.rename(columns={"averageRating": "series_rating",
                                            "numVotes": "series_votes"}),
                    on="tconst", how="left"))
out = out.sort_values(["primaryTitle", "n_episodes"], ascending=[True, False])
pd.set_option("display.width", 200)
print(out.to_string(index=False))