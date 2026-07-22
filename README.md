# Unlearning Direction Evaluation

This repository contains scripts used in an exploratory project investigating
whether unlearning removes knowledge from language models or suppresses
access to that knowledge via refusal direction.

Across six WMDP-unlearned Llama-3-8B-Instruct checkpoints, ablating a rank-1
refusal direction does not recover WMDP-Bio accuracy beyond
random-direction or matched-construction controls, even for the two models
(ILU-RMU, GradDiff) where the same direction demonstrably bypasses refusal-like
behavior on ordinary harmful prompts. A separate arm found that unrelated
GSM8K fine-tuning *does* raise WMDP-Bio accuracy for most of the same models,
gated by whether general utility (MMLU) moves alongside it. Low forget-set
performance alone does not distinguish genuine knowledge removal from
suppression, and a null result from one recovery probe (direction ablation)
does not generalize to another (fine-tuning).

## Results and writeups

- [results.md](results.md) -- results summary for both arms
- [direction_extraction.md](direction_extraction.md) -- prompt source, candidate
  sweep, and direction-selection procedure

---

## Installation

```bash
pip install -r requirements.txt
```

The scripts accept Hugging Face model IDs or local checkpoint paths.

---

## Model checkpoints

WMDP-unlearned checkpoints are from [OPTML-Group](https://huggingface.co/OPTML-Group)
(GradDiff, IDK-AP, ILU-RMU, NPO, NPO-ILU) and [ScaleAI](https://huggingface.co/ScaleAI/mhj-llama3-8b-rmu)
(RMU), all unlearned from `meta-llama/Meta-Llama-3-8B-Instruct`. This
repository focuses on direction extraction/evaluation and SFT-recovery
evaluation rather than reproducing the unlearning training pipelines
themselves.

---

## Direction extraction and WMDP-Bio ablation

> An earlier script, `wmdp_bio_refusal_direction_eval.py`, is removed from
> this repo. It re-tokenized already-templated prompts (silently
> double-inserting the BOS token, which also undercounted its own
> forced-choice WMDP-Bio scoring vs. `lm_eval`), only searched the final
> token position instead of the full post-instruction grid, and applied
> neither Arditi et al.'s nor COSMIC's selection filters. `extract_refusal_direction.py`
> + `wmdp_bio_lm_eval_ablation.py` (below) fix all of that -- see
> [direction_extraction.md](direction_extraction.md) for the full rationale.

**1. Extract a refusal direction.** `extract_refusal_direction.py` computes
difference-in-means candidates over the last 5 post-instruction token
positions x every layer, using Arditi et al. (2024)'s own AdvBench-harmful vs.
Alpaca-harmless setup (a general "refusal" direction, not bio-specific).
`--selection-method both` (the default) computes **both** selection
algorithms from a single candidate sweep:

- `mean_diff` -- Arditi et al. (2024) Appendix C.1's causal selection
  (minimize bypass_score subject to induce_score > 0, kl_score < 0.1,
  layer < 0.8L).
- `cosmic` -- Siu et al. (2025) COSMIC's concept-inversion cosine-similarity
  selection on the low-similarity layers, with the same KL/layer-fraction
  filters used in the official COSMIC repo (`wang-research-lab/COSMIC`).

```bash
python scripts/extract_refusal_direction.py \
  --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
  --model-label idk-ap \
  --selection-method both \
  --output-root outputs/idk-ap/direction_extraction/generic
```

Writes, per selection method: `direction_{method}.pt` (with selection
diagnostics in the metadata), `candidate_diagnostics.csv` (every candidate's
bypass/induce/KL/COSMIC scores, for auditing), and
`direction_{method}_matched_control.pt` -- a split-half difference-in-means
direction from a fresh, disjoint harmless prompt pool at the same
(position, layer), testing whether ablating *any* direction built this way at
this spot moves WMDP-Bio accuracy, not just the one selected as "refusal."
(`mean_diff` finds zero passing candidates for 4 of the 6 models tested here
-- see [direction_extraction.md](direction_extraction.md) for why the
pipeline always runs both methods rather than picking one upfront.)

**2. Evaluate WMDP-Bio accuracy under ablation.** `wmdp_bio_lm_eval_ablation.py`
scores a direction against `lm_eval`'s own `wmdp_bio` task:

```bash
python scripts/wmdp_bio_lm_eval_ablation.py \
  --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
  --model-label idk-ap \
  --direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic.pt \
  --matched-control-direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic_matched_control.pt \
  --output-root outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic \
  --apply-chat-template
```

Runs `wmdp_bio` on the same loaded model across `baseline` (no hook, unless
`--skip-baseline`), `selected_direction_ablation`, `matched_control_ablation`
(if given), and `random_direction_ablation_{0..N}` controls, writing a
comparison summary (including a random-control mean/std and a
`selected_control_z` distance from that distribution) to
`wmdp_bio_lm_eval_summary.json`.

**3. Confirm the direction does something.** A null accuracy result is only
informative if the direction demonstrably affects behavior.
`wmdp_refusal_behavior_check.py` generates on 100 held-out AdvBench prompts
(disjoint from extraction) with and without ablation and measures the
refusal rate, reusing the exact ablation code from step 2:

```bash
python scripts/wmdp_refusal_behavior_check.py \
  --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
  --model-label idk-ap \
  --direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic.pt \
  --extraction-prompts-csv outputs/idk-ap/direction_extraction/generic/direction_prompts.csv \
  --matched-control-direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic_matched_control.pt \
  --output-root outputs/idk-ap/wmdp_refusal_behavior_check/generic_cosmic
```

**Full pipeline, one model, both selection methods:**

```bash
MODEL=OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
MODEL_LABEL=idk-ap \
OUTPUT_ROOT=outputs/idk-ap \
HARDWARE_PROFILE=a100 \
bash examples/run_full_wmdp_bio_pipeline.sh
```

Repeat per model. Random-direction and matched-control conditions are only
computed once per model (the first ablation-eval call); subsequent calls for
the same model pass `--num-random-controls 0`.

**Paired significance testing.** Both `wmdp_bio_lm_eval_ablation.py` and
`wmdp_sft_recovery/eval_recovery_lm_eval.py` persist per-question correctness
to `per_doc_correctness/<task>.json` (doc_id -> bool), since every condition
in this repo is scored on the *same* question set and a paired test (exact
binomial McNemar + a paired bootstrap CI) is the correct tool -- not two
independent-sample tests. `scripts/analyze_paired_recovery.py` runs that
comparison between any two conditions' persisted files:

```bash
python scripts/analyze_paired_recovery.py \
  --correctness-a outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic/per_doc_correctness/wmdp_bio/selected_direction_ablation.json \
  --correctness-b outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic/per_doc_correctness/wmdp_bio/matched_control_ablation.json \
  --label-a selected_direction --label-b matched_control \
  --output-json outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic/mcnemar_selected_vs_matched_control.json
```

---

## SFT recovery

A different recovery mechanism: does unrelated fine-tuning undo unlearning by
disturbing the unlearned weights, rather than by ablating a direction?
[`wmdp_sft_recovery/`](wmdp_sft_recovery/) runs QLoRA SFT on `openai/gsm8k`
(near-zero mutual information with WMDP-bio, so any bio movement afterward is
the unlearning coming undone, not new knowledge going in) on each unlearned
model plus a full-knowledge control, checkpointed at 1000/3000/6000 examples,
evaluated via `lm_eval` on `wmdp_bio` (recovery signal), `mmlu` (utility
gate), and `gsm8k` (uptake check):

```bash
cd wmdp_sft_recovery
bash run_base_control.sh                          # full-knowledge control
METHODS="RMU" bash run_unlearned.sh                # one unlearned model at a time
python aggregate_table.py                          # -> results/wmdp_sft_recovery/recovery_table.md
```

See [`wmdp_sft_recovery/README.md`](wmdp_sft_recovery/README.md) for the full
design (controls, precision/chat-template gotchas, and the cross-check
against this repo's ablation-arm baselines) and
[results.md](results.md#sft-recovery-does-unrelated-fine-tuning-undo-unlearning)
for the outcome.

---

## References

- Arditi et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction.](https://arxiv.org/abs/2406.11717)
- Siu et al. (2025). [COSMIC: Generalized Refusal Direction Identification in LLM Activations.](https://arxiv.org/abs/2506.00085)
- WMDP: [The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.](https://www.wmdp.ai/)
