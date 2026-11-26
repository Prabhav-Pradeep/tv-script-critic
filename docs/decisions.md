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

## Generation

**Prompts: 12 scene openings, 3 per show, ~80 words each, from test seasons**
(`make_prompts.py`). The critics never trained on test-season text, so a prompt cannot steer
the writer toward memorized text. Each episode's first chunk is skipped (often a "Previously
on" recap), and each prompt comes from a different episode.

**Writer: Qwen2.5-1.5B-Instruct, prompted, no fine-tuning** (`notebooks/generate_colab.ipynb`).
Small enough for a free Colab T4, open licence (Apache-2.0). It continues each opening for
~450 words in the same transcript format, so its output looks like what the critics learned
from (they were trained on ~450-word chunks).

**256 candidates per prompt, plain sampling (temperature 1.0, top-p 0.95).** Diversity matters:
best-of-N only has something to choose from if candidates differ. The laptop has no GPU
(llama3.2 3B on CPU writes ~6–10 tokens/s, so a 3,072-scene pool would take ~20 hours), so
generation runs on a free Colab T4.

**Hugging Face `transformers`, not vLLM.** vLLM 0.34 crashed on start-up on Colab: its kernel
warm-up imports code for an unrelated model (MiniMax-M3) that needs a `cutlass` package it
does not install, and the T4 already lacked two of its fast paths. Plain `model.generate` with
batches of 128 is slower but has no fragile install. The notebook was dry-run locally on CPU
with Qwen2.5-0.5B (4 candidates, 20 tokens) before each Colab run, including resuming after a
partial run: finished prompts are kept, a half-written prompt is regenerated.

**Why one pool of 256 instead of separate runs per N.** Best-of-N for any N ≤ 256 can be
computed exactly from one pool: sort the candidates by the selection score; the chance that
the candidate ranked i of M is the best of a random N-subset is C(i−1, N−1) / C(M, N). The
expected gold score of the pick is the sum of gold scores weighted by those chances. No extra
generation, and no sampling noise from drawing subsets.

## Scoring and experiment

**What the writer produced.** 3,072 candidates (12 prompts × 256). 98% use the requested
`NAME: line` format; 12 had a "Here's the continuation"-style preamble stripped. They are
shorter than asked: median 162 words (10th–90th percentile 64–313) against ~450 requested.
Length is not a shortcut for the critics: within-prompt Spearman with length is −0.003 for
gold and −0.18 to −0.01 for the proxies.

**Selection rules** (`bon_experiment.py`), on the 5 proxy scores of each candidate: one proxy
alone (curve averaged over the five), mean, mean − 1 SD, mean − 2 SD (uncertainty-weighted),
min (worst case), and gold itself as the ceiling.

**Metric: gain over the pool mean, in units of the pool's SD**, averaged over prompts. Pools
differ in mean and spread (gold pool SD 0.74–1.01), so raw gold differences would let a few
prompts dominate. CIs bootstrap over the 12 prompts, the unit that was actually sampled;
rule-vs-rule differences are paired by prompt.

**Exact best-of-N, tested.** `bon.py` is checked in `tests/test_bon.py` against brute force over
every subset, plus the edge cases (N = 1 is the pool mean, N = M is the top-scored candidate,
weights sum to 1).

**Results.** See the README table. Ensembling beats a single proxy by +0.52 SD of gold at
N = 256 (CI [+0.19, +0.86], 11/12 prompts). Every disagreement penalty is worse than the plain
mean, at every proxy size.

**Why the penalty fails here.** Hypothesis 1, that disagreement just tracks scene length, was
tested and rejected: within-prompt Spearman of disagreement with length is −0.11 to +0.08.
What the data shows: disagreement barely predicts which candidates the proxies overrate
(Spearman with proxy-minus-gold: +0.17 at 250 chunks, +0.01 at 500, +0.02 at 1,200), and is
slightly positive with gold itself at 250 and 1,200 (+0.13, +0.16), plausibly because
distinctive scenes split opinion. A penalty on it removes good candidates and rarely removes
overrated ones.

**Scope of the claim.** Best-of-256 is mild optimization (KL ≈ 4.5 nats); gold has no peak for
the mean or single proxy, i.e. the overoptimization regime was not reached. The finding is
"the penalty costs when hacking is mild", not "the penalty never helps". Testing the strong
regime needs a stronger optimizer (larger N, or RL fine-tuning of the writer).

**Other limitations.** 12 prompts is the effective sample size. Gold is half TF-IDF, so
hacking that games word choice partly fools gold too: measured gaps are a lower bound. One
writer model (Qwen2.5-1.5B-Instruct) and one sampling temperature.

## Engineering notes

- `GoldCritic` lives in `critics.py`, not in a script: pickle saves a class by module path,
  so a class defined in a script run as `__main__` can't be loaded from another script.
- `GoldCritic.score()` takes raw text and repeats every training step (masking, both critics,
  weights). The saved object is reloaded and checked against the test predictions before
  scores are written, so generated scenes are judged exactly as validated.
- Saved critics hold a vocabulary built from transcripts, so `models/` is git-ignored, in
  line with the data policy.
