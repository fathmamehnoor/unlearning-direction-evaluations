# Direction Extraction

How the WMDP refusal/IDK directions are extracted and selected. Implementation:
[`scripts/extract_refusal_direction.py`](scripts/extract_refusal_direction.py).

This supersedes an earlier script, `wmdp_bio_refusal_direction_eval.py` (now
removed): its `chat_template` prompt-rendering path re-tokenized an
already-templated string, silently double-inserting the BOS token (which also
undercounted its own forced-choice WMDP-Bio scoring vs. `lm_eval`); its
candidate search only ever considered the final token position rather than
the full post-instruction-token grid Arditi et al. and COSMIC both use; and
neither of its selection modes applied the KL-divergence / late-layer-exclusion
/ induce-threshold filters both papers specify.

## Prompt source

A generic, model-only harmful/harmless contrast, reused unchanged from Arditi
et al. (2024): [AdvBench](https://github.com/llm-attacks/llm-attacks) harmful
behaviors vs. [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
harmless instructions. Nothing WMDP- or bio-specific goes into the direction
itself -- the resulting direction is a general "refusal" direction, applied
afterward to a bio-knowledge benchmark. A bio-specific variant (harmful
biosecurity prompts vs. benign biology prompts) was explored and removed from
this codebase for now; it's planned as separate follow-up work rather than
bundled into this generic-only pipeline.

## Candidate extraction

Candidates are difference-in-means vectors between harmful- and
harmless-prompt activations at the residual-stream input of every decoder
layer (`resid_pre`, via forward pre-hooks), at each of the final 5
post-instruction token positions (`-5` through `-1`) -- one candidate per
(layer, position) pair. For an 8B Llama-3 model (32 layers) this is 160
candidates per run. Left-padding is used throughout so position `-1` is
always the true final token regardless of prompt length. Chat-templated
prompts are tokenized once via `apply_chat_template(tokenize=True)` and
padded with `tokenizer.pad` -- never re-tokenized from a rendered string.

## Selection

Both selection algorithms are computed from the same candidate sweep in a
single run (`--selection-method both`):

- **`mean_diff`** -- Arditi et al. (2024) Appendix C.1's causal selection.
  For each candidate: ablate it on held-out harmful prompts (`bypass_score`),
  add it on held-out harmless prompts (`induce_score`), and measure the KL
  divergence it introduces on harmless prompts (`kl_score`). Select the
  candidate minimizing `bypass_score` subject to `induce_score > 0`,
  `kl_score < 0.1`, and `layer < 0.8L`.
- **`cosmic`** -- Siu et al. (2025) COSMIC's concept-inversion,
  cosine-similarity selection, restricted to the low-similarity layers
  (bottom 10% by harmful/harmless cosine similarity). Scores each candidate
  by how much ablating it on harmful prompts makes those activations resemble
  natural harmless activations (`S_comply`), and how much adding it on
  harmless prompts makes those activations resemble natural harmful
  activations (`S_refuse`). Selects `argmax(S_refuse + S_comply)` subject to
  the same KL/layer-fraction filters plus `S_refuse > 0`.

Each run also writes a **matched-control direction**: a split-half
difference-in-means vector computed from a fresh harmless prompt pool
disjoint from every prompt used in real extraction, at the same
(position, layer) as the real direction. Same construction mechanism, same
activation statistics, no genuine refusal signal -- it's a stronger null than
an isotropic random unit vector, since it tests whether ablating *any*
difference-in-means direction built this way, at this spot, moves WMDP-Bio
accuracy, not just the specific one selected as "refusal."

## Why both methods matter: `mean_diff` fails structurally on 4 of 6 models

Across the 6 WMDP-unlearned checkpoints tested, Arditi's `induce_score > 0`
filter -- does adding the candidate direction increase the model's probability
of literally starting its response with the calibrated refusal token ("I")? --
found **zero passing candidates out of 160** for GradDiff, IDK-AP, NPO, and
NPO-ILU (induce scores uniformly and deeply negative, e.g. GradDiff:
-115.9 to -16.0; NPO: -185.4 to -87.6). Only RMU (36/160 passing) and ILU-RMU
(67/160 passing) had any candidate satisfy the filter. COSMIC's
activation-based selection, run from the same candidate sweep, succeeded for
all 6 -- a concrete instance of COSMIC's motivating claim: it doesn't assume a
specific output token, so it stays usable when a model's surface refusal
phrasing has drifted under unlearning (or, for NPO/NPO-ILU in particular,
degenerated into near-total repetition -- see the behavioral check in
[results.md](results.md)). This is why the pipeline always computes both
selection methods rather than picking one upfront.

## Selected direction per model

| model | mean_diff | cosmic |
|---|---|---|
| RMU (`ScaleAI/mhj-llama3-8b-rmu`) | pos -3, layer 13 | pos -2, layer 12 |
| ILU-RMU | pos -3, layer 12 | pos -2, layer 11 |
| GradDiff | *(no candidate passed filters)* | pos -2, layer 11 |
| IDK-AP | *(no candidate passed filters)* | pos -1, layer 16 |
| NPO | *(no candidate passed filters)* | pos -1, layer 10 |
| NPO-ILU | *(no candidate passed filters)* | pos -1, layer 11 |

## Confirming the direction does something: behavioral bypass check

A null WMDP-Bio result is only informative if the selected direction
demonstrably does something. `scripts/wmdp_refusal_behavior_check.py`
generates on held-out AdvBench prompts (disjoint from extraction prompts,
using ablation code re-imported byte-identical from
`wmdp_bio_lm_eval_ablation.py`) with and without ablation, and measures the
refusal rate. See [results.md](results.md) for the outcome -- ablation
cleanly bypasses refusal behavior for some models (ILU-RMU, GradDiff) and not
others, independent of whether WMDP-Bio accuracy moves.
