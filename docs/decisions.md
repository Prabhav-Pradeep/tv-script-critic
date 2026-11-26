# Design decisions

Why the project is built the way it is. Each entry: the decision, the reason, and the
evidence where there is any. Read this before talking about the project.

## Setup: gold critic and proxy critics

**Decision.** Follow the synthetic setup of Gao, Schulman & Hilton (2022), *Scaling Laws for
Reward Model Overoptimization*. A strong **gold critic** is trained on real IMDb ratings and
is treated as the truth. Weaker **proxy critics** are trained on the gold critic's scores and
are what the writer optimizes against.

**Why.** Generated scenes have no IMDb rating, so "true quality" needs a stand-in. Training
the proxies on gold labels makes gold the true target by construction, the same way a
reward model in RLHF is an imperfect copy of human judgment. Reward hacking then means: the
proxy score rises while the gold score stalls or falls.

## Data

**Unit of text: ~450-word chunks of whole lines** (`make_chunks.py`). Scene markers differ by
show (CSI uses `[INT. ...]`, House uses free-form brackets), so fixed word windows are more
robust than scene detection. 508 episodes become 8,679 chunks, about 17 per episode.

**Label: within-season rating z-score**, inherited from the chunk's episode. It removes
"House is rated higher than NCIS" and "season 2 was better than season 7", leaving how good
an episode is compared with its own season. Chunk labels are noisy: a dull scene in a great
episode is labelled great.

**Train/test split by whole seasons** (seeded, per show, about 20% of episodes in test).
Chunks from one episode, or one season, share characters, plot words and transcriber habits.
A random split would put near-duplicates on both sides and inflate every score.

**Validation copies the test split.** Hyperparameters are chosen with `GroupKFold` grouped by
(show, season), on train only. A random K-fold would reward memorizing episodes and pick too
little regularization.

## Critic signal

**Baseline: TF-IDF (1–2-grams) + ridge** (`critic_baseline.py`). Test episode-level
Spearman ρ = 0.47 (within-show permutation p < 0.001, 2,000 shuffles).

**Names are masked** (`text_utils.py`). Unmasked, the top features were character names
(Ari, Zack, Taub, Kutner): the critic learned which characters and arcs appear, which a
writer could hack trivially. With names replaced by `NAME`, test ρ rose to 0.51, and the top
features became content: courtroom, autopsy, flashback, Christmas (up); drugs, money, limo
(down).

**Names come from train text only.** The name list is a fitted step, like the TF-IDF
vocabulary, so fitting it on test would leak. A guest name that appears only in test stays
unmasked, but the train-only vocabulary gives it no weight anyway.

**Why episode-level ρ beats chunk-level r (0.51 vs 0.27).** Averaging ~17 chunk predictions
cancels much of the per-chunk noise in both predictions and labels.

**Why a permutation test.** With 136 test episodes, a useless critic can reach ρ ≈ 0.2 by
luck. Shuffling ratings within each show 2,000 times gives the luck distribution; 0.51 is
beyond all but about 1 in 2,000. Shuffling within show keeps each show's rating spread intact.

## Gold critic

**Decision: hybrid of lexical and semantic critics, stacked** (`train_gold.py`,
`critics.py`).

**Why not just a bigger TF-IDF.** The proxies are TF-IDF critics. If gold sees text the
same way, the proxies inherit gold's blind spots and nothing can be hacked. Gold has to see
the text differently.

**Why sentence embeddings.** `all-MiniLM-L6-v2` maps text to a 384-number vector that
captures meaning, so paraphrases land close together and stuffing in magic words works far
less well. The same idea as hybrid BM25 + FAISS retrieval: exact words plus meaning.

**Why not fine-tune a transformer.** About 6,000 noisy training chunks, CPU only, and a
4-week timebox. Fine-tuning would likely overfit and eat the schedule. Kept as a stretch goal.

**Pieces of 140 words.** MiniLM reads at most 256 tokens. The first guess, 180 words, left
34% of pieces over the limit (masked dialogue runs ~1.6 tokens per word); measured on 400
chunks, 140 words leaves 0.4% over. Pieces are whole lines, embedded separately and averaged
into one vector per chunk; the label stays at chunk level.

**Out-of-fold predictions for any combining.** In-sample, TF-IDF with 50,000 features nearly
memorizes its training chunks, so weights fitted there would trust it too much. Each train
chunk is predicted by models that never saw its season (5 season-grouped folds), and both
combination methods use those honest predictions only.

**Semantic ridge alpha = 10,000.** The first grid stopped at 3,000 with the score still
rising, so the true optimum was off the edge. The widened grid peaks at 10,000; the script
now refuses a best alpha that sits on either edge.

**Results** (test = 136 episodes from held-out seasons, episode-level Spearman ρ, all
permutation p < 0.001):

| Critic | Train OOF ρ | Test ρ |
|---|---|---|
| Lexical (TF-IDF, the baseline) | 0.433 | 0.511 |
| Semantic (embeddings) | 0.425 | 0.443 |
| **z-score average (gold)** | **0.465** | **0.505** |
| Stacked | 0.464 | 0.511 |

Bootstrap over test episodes (2,000 resamples): lexical ρ 95% CI [0.37, 0.62], stacked
[0.37, 0.62], difference [−0.04, +0.05]. With 136 episodes, gaps of a few hundredths are
undetectable. Gold per show: Bones 0.57, CSI 0.53, House 0.56, NCIS 0.25 (only 20 test
episodes, the weakest).

**Decision: gold is the equal z-score average, not the stacked model.** Stacking gave
lexical ~60% of the weight and correlated 0.96 with the lexical critic alone at episode level.
The proxies are TF-IDF critics, so a near-TF-IDF gold would share their blind spots and
undercount hacking. Equal weights are fixed in advance (nothing tuned on test or train), are
statistically as accurate, and make half of gold's judgment come from meaning
(chunk-level correlation with lexical 0.89 rather than 0.94). Test scores for several weights
were looked at while diagnosing this; the weight was chosen on principle, not on the best test
row, which would be test leakage.

**Known limitation.** Gold is still half TF-IDF, so it is not hack-proof; a writer that games
word choice can partly fool gold too. This makes measured hacking a lower bound.

## Proxy critics

**Decision: 5 TF-IDF + ridge critics per ensemble, each trained on its own disjoint random
subset of train chunks, with gold scores as labels** (`train_proxies.py`).

**Why gold labels, not IMDb.** Gold is the truth by construction; proxies are imperfect
copies of it, like a reward model trained on a limited number of human ratings.

**Why disjoint subsets.** The first version drew overlapping random subsets; at 4,000 chunks
(of 6,330) the five proxies correlated 0.985 with each other, i.e. clones that never
disagree, which leaves a disagreement penalty nothing to work with. With disjoint slices of
one permutation, no chunk is seen by two proxies. That caps the size at 6,330 / 5.

**Three proxy strengths: 250, 500, 1,200 chunks.** Main setting: 500. Test-chunk
correlations:

| Subset size | Single proxy ~ gold | Proxy ~ proxy | Ensemble mean ~ gold |
|---|---|---|---|
| 250 | 0.62 | 0.63 | 0.73 |
| **500** | **0.70** | **0.70** | **0.80** |
| 1,200 | 0.80 | 0.89 | 0.84 |

At 500, each proxy is a decent but imperfect copy of gold, the proxies differ from each other
as much as from gold, and averaging already gains +0.10. At 1,200 they converge on the same
function and averaging adds little. All of these are measured on real transcripts; on
optimized, generated text the proxies are expected to disagree more, which is what the
penalty exploits.

**Subtlety: gold memorizes its training chunks.** Gold's correlation with the real chunk
labels is 0.73 on train chunks (in-sample) but 0.29 on test chunks. So the labels the
proxies learn from are partly gold's memorized noise. This is consistent with the setup,
because the target is defined as gold's exact function, the same one that scores generated
scenes, but it makes the proxies' job harder than copying a smooth target.

## Engineering notes

- `GoldCritic` lives in `critics.py`, not in a script: pickle saves a class by module path,
  so a class defined in a script run as `__main__` can't be loaded from another script.
- `GoldCritic.score()` takes raw text and repeats every training step (masking, both critics,
  weights). The saved object is reloaded and checked against the test predictions before
  scores are written, so generated scenes are judged exactly as validated.
- Saved critics hold a vocabulary built from transcripts, so `models/` is git-ignored, in
  line with the data policy.
