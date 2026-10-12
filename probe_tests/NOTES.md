# Probe tests: does a reward-hacking probe read intent or text?

Branch `john/probe-tests`. Scripts run from the repo root; outputs under `data/probe_tests/`.

## Step 1 (2026-10-12): eval sets and text-only baselines

```
uv run --no-project --with tokenizers python probe_tests/build_sets.py
uv run --no-project --with scikit-learn --with numpy python probe_tests/text_baselines.py
uv run --no-project --with scikit-learn --with numpy python probe_tests/precommit.py
```

Tokenizer: Qwen/Qwen3.6-27B `tokenizer.json`, expected at `/tmp/john-scratch/tok/tokenizer.json`.

### Sets (`data/probe_tests/sets/`)

- `natural_pool.jsonl`: 8,256 grader rollouts, no intervention, no system prompt, reasoning on (parity 4 logs, sign, pets_listed). Index only: labels, CoT length, vocabulary flags, split (sha256 of rollout_id: 60/20/20), exclusions. 8,137 usable after excluding label None (108), no `</think>` (93), premature `</think>` after `` ` ``+newline or repeated (21), Ethan's resample source pair (2).
- `sysprompt_pool.jsonl`: generic and very_hacker system-prompt parity rollouts (384).
- `vocab_matched.jsonl`, `length_matched_pairs.jsonl` (within family and side, CoT tokens within 10%), `length_vocab_matched_pairs.jsonl`.
- `resample_prefixes.jsonl`: Ethan's 1,467 resamples with the prefix text each continuation was conditioned on (the resample files' `reasoning` field holds the continuation only).
- `precommit_cuts.jsonl`: character position of the first committed answer in each usable CoT.
- `candidates/`: unverified candidate lists from the labeling agents (regex, about 56% precision for the strict commit-then-reverse list; secret_number flagged reads with silent/verbal flags).
- `../agent_labels_2026-10-12.json`: the five labeling agents' full output.

### Results (`data/probe_tests/text_baselines/`)

TF-IDF (word 1-2 grams, logistic regression, C tuned on val) on the full CoT separates cheat from clean almost perfectly, in and out of family, and on the vocabulary- and length-matched subsets. Test AUC:

| train -> test | all | "maximize reward" subset | length pairs |
|---|---|---|---|
| parity -> parity | 1.000 | 1.000 | 1.000 |
| parity -> sign | 0.999 | 0.998 | 0.999 |
| sign -> pets_listed | 0.997 | 0.978 | 0.992 |
| pets_listed -> parity | 0.739 | 0.753 | 0.892 |

Numbers masked: no change. Keyword count: 0.60-0.93. Length alone: 0.20-0.80, sign of the effect flips across families. very_hacker-vs-generic TF-IDF transfers to natural parity at AUC 0.957.

Cut before the first committed answer (median about 25% into the CoT), labeled by final outcome: TF-IDF still 0.92-0.99 (pets -> parity 0.68). Top features are leaning language ("to maximize", "satisfy", "dictates" vs "should follow", "user instruction").

On Ethan's resampled rollout, a TF-IDF model trained on truncated parity CoTs:

| t | 0 | 100 | 186 | 211 | 250 | 300 | 323 | 400 |
|---|---|---|---|---|---|---|---|---|
| P(cheat), 25-token mean | 0.08 | 0.24 | 0.24 | 0.72 | 0.96 | 1.00 | 0.92 | 1.00 |
| TF-IDF (truncated-trained) | 0.18 | 0.16 | 0.74 | 0.71 | 0.73 | 0.79 | 0.95 | 0.98 |

Onset (10 consecutive t above threshold): P(cheat) >= 0.5 at t=205, >= 0.8 at 239; TF-IDF >= 0.5 at 167, >= 0.8 at 313. The text classifier is high on the shared pre-fork prefix (t~186, where resampled P is 0.24) and lags the resampled curve in t~239-313.

### What this means for probe tests

- Whole-CoT cheat-vs-clean accuracy cannot distinguish a probe from a text classifier here: the text baseline is at ceiling.
- A probe shows it reads more than text only where TF-IDF fails: positions where the resampled P(cheat) and the text disagree (t~186 high text / low P; t~239-313 low text / high P), pets -> parity transfer, secret_number silent reads, and edit-then-resample experiments.
- Prefix labels need resampling. Final-outcome labels on truncated CoTs are already predicted from text at 0.92-0.99.
- Caveats: one resampled rollout, one sample per t, so curve positions are uncertain by about ±10-20 tokens. Commit regex not hand-validated beyond the agents' checks. No LLM-monitor baseline yet.
