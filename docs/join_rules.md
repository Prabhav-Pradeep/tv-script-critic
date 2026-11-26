# Join rules: transcripts to IMDb episodes

Written 2026-11-26. Update if the rules change.

## Why this exists
Matching on (show, season, episode number) alone is unsafe. The transcript
source and IMDb number some episodes differently, and a wrong number attaches
the wrong rating to a transcript. Rule 1 below is the fix.

## Rules
1. **Title first, number second.** Within each (show, season), match each
   transcript to the IMDb episode with the most similar title, using a
   one-to-one assignment. A matching episode number only breaks ties.
2. **Normalization before comparing titles:**
   - lowercase
   - drop parenthetical text, e.g. "(1)", "(Part II)"
   - drop everything from "Part"/"Parts" onward
   - drop punctuation and leading articles (the, a, an)
   - compare both with and without spaces, and take the higher score
3. **Threshold.** A match needs a title similarity of at least 0.8.
   This is my chosen cutoff, not a standard. Anything below it is unmatched.
4. **Shifted episodes.** If the matched IMDb episode number differs from the
   transcript's number, the row is flagged `shifted` and kept.
5. **Shared ratings (merged two-parters).** If two transcripts resemble only
   one IMDb row, both are flagged `shared_rating`. They stay in the writer's
   training text but are excluded from critic data, because one rating
   covers both parts.
6. **Unmatched rows** (similarity below 0.8, or no IMDb row) are excluded from
   critic data. They are not forced into a match.
7. **Vote floor.** Critic data requires at least 500 IMDb votes.

`critic_ok = matched AND NOT shared_rating AND numVotes >= 500`

## Known cases from the first audit
- House season 6: transcripts from 6x02 onward are one number higher than
  IMDb, because the premiere "Broken" is two numbers in the transcripts and
  one row on IMDb. Fixed by Rule 1.
- Bones 4x01 and 4x02 ("Yanks in the U.K."): one merged IMDb row, so both
  are excluded from critic data.
- NCIS "Kill Ari" and "Hiatus", and CSI "Built to Kill": IMDb has separate
  rows per part, so these are matched normally.
- Final usable critic episodes: 668 across six shows (CSI 164, House 150,
  Supernatural 113, Bones 112, NCIS 91, The Mentalist 38). The four-show
  candidate core (CSI, Bones, NCIS, House) has 517.
## Outputs
- `data/interim/fd_imdb_matched_v1.csv`, produced by `src/match_titles.py`.
- Final usable critic episodes: <fill in n_critic_ok from the script output>.