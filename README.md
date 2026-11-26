# tv-script-critic

Does a critic ensemble with a disagreement penalty reduce reward hacking
compared with a single critic, when tuning a TV script writer?

## Data sources

| Source | What it provides | License / terms |
|---|---|---|
| IMDb non-commercial datasets | Ratings, vote counts, season and episode numbers, crew | Personal, non-commercial use only; see IMDb's terms |
| SummScreen (ForeverDreaming config) | Episode transcripts and recaps | Check the dataset's own terms; transcripts are fan-sourced |
| fd_dialogue (Hugging Face) | Speaker-labelled transcripts for several shows | Dataset card says MIT; the underlying transcripts may carry their own rights |

## Data policy

Raw transcripts are not redistributed. Only code and derived statistics
are published. The `data/` folder is git-ignored.