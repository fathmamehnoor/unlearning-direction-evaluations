# WMDP unrelated-SFT recovery

Does forgotten WMDP-bio accuracy come back when you nudge an unlearned model's
weights with data that has nothing to do with bio? If it does, low forget-set
performance was hiding **retained** knowledge -- unlearning suppressed rather
than removed it. This is the "still there" half of the project's thesis, and
the half the refusal-direction ablation arm (`../scripts/`) cannot speak to:
ablation tests whether knowledge is *readable off a direction*; this arm tests
whether it's *recoverable via fine-tuning* at all.

The whole experiment lives or dies on one principle: **make recovery
attributable.** The failure mode is fine-tuning that raises bio accuracy
simply by disrupting the unlearning in a way that also wrecks everything
else. So every recovery number is read next to a utility number, always, from
the first row.

## Why GSM8K, and the disjointness statement

The SFT data is `openai/gsm8k` -- grade-school math word problems. It has
near-zero mutual information with WMDP-bio: a model cannot learn bioweapons
facts from arithmetic, so any WMDP-bio movement after this SFT is the
unlearning coming undone, not new knowledge coming in.

## The three signals, on one row per checkpoint

| signal | task | role |
|---|---|---|
| WMDP-bio accuracy | `wmdp_bio` | recovery signal |
| WMDP-bio correct-option probability | `wmdp_bio` | knowledge surfacing *before* the answer flips |
| MMLU accuracy | `mmlu` | utility gate |
| GSM8K accuracy | `gsm8k` | confirms the fine-tune actually took (uptake check, not a score -- read the delta) |

Reading rule: bio up + MMLU flat -> real recovery (suppression, not removal).
Bio and MMLU up together -> SFT churn, not hidden knowledge. Probability
rising ahead of accuracy is the cleanest suppression-not-removal signature.

## Controls

1. **No-SFT baseline** per model (`--checkpoint-step 0`, no adapter).
2. **Full-knowledge control**: the same GSM8K SFT applied to base
   `meta-llama/Meta-Llama-3-8B-Instruct` (the checkpoint the OPTML-Group WMDP
   models were unlearned from) -- isolates what unrelated SFT does to
   WMDP-bio on its own, with no unlearning to undo.
3. **Multiple unlearning methods**, kept parallel to the ablation arm.

## Files

- `train_gsm8k_qlora.py` -- QLoRA SFT on GSM8K; saves adapters at
  1000/3000/6000 examples seen.
- `eval_recovery_lm_eval.py` -- runs `wmdp_bio,mmlu,gsm8k` through `lm_eval`,
  derives the WMDP-bio correct-option probability from the logged
  loglikelihoods (self-checked against `lm_eval`'s own reported accuracy),
  and appends one row to `recovery_results.csv`.
- `common.sh` -- single shared seed (42) and QLoRA recipe, sourced by both
  run scripts so the only difference between arms is the model.
- `run_base_control.sh` / `run_unlearned.sh` -- the two arms.
- `aggregate_table.py` -- assembles `results/wmdp_sft_recovery/recovery_table.md`
  from the CSV, with baseline deltas.
- `merge_results.py` -- dedup-merges result CSVs by
  `(arm, unlearning_method, checkpoint_step)`, latest timestamp wins; used to
  combine rows produced across separate GPU rental sessions.
- `sample_flipped_generations.py` -- for RMU specifically: samples questions
  whose forced-choice answer flipped wrong-to-right after SFT and generates
  free-form completions, to check the flip reflects the model actually
  reasoning about the biology content rather than an output-calibration
  artifact.
- `../scripts/analyze_paired_recovery.py` -- paired McNemar / bootstrap
  significance test between any two conditions' `per_doc_correctness` files;
  shared with the ablation arm since both persist that format.

## Run order

Run one model per session and clear its cache before the next (~40 GB disk,
one 24 GB GPU is enough):

```bash
cd wmdp_sft_recovery
pip install "lm_eval" transformers peft bitsandbytes accelerate datasets

bash run_base_control.sh                          # full-knowledge control
METHODS="ILU-RMU" bash run_unlearned.sh
METHODS="IDK-AP"  bash run_unlearned.sh
METHODS="RMU"     bash run_unlearned.sh
python aggregate_table.py                         # -> results/wmdp_sft_recovery/recovery_table.md
```

`recovery_results.csv` accumulates across sessions; `aggregate_table.py`
works regardless of how many sessions it took.

Smoke test first: `LIMIT=16 NUM_EXAMPLES=64 CHECKPOINTS="32 64" bash run_base_control.sh`

## Gotchas

- **Precision is uniform per model.** Reported recovery is
  `baseline - SFT`, so a precision mismatch between a model's baseline row
  and its SFT rows would land directly in the number. `common.sh` drives
  every row through one `eval_one` at a single precision (bf16 everywhere by
  default); `aggregate_table.py` refuses to trust a model whose rows
  disagree.
- **Chat template is off for every model** (`APPLY_CHAT_TEMPLATE=0`),
  including the ones that would otherwise support it, for consistency with
  `ScaleAI/mhj-llama3-8b-rmu`, whose shipped chat template renders message
  content down to ~2 tokens. The trainer uses a plain `Question:/Answer:`
  format for every model for the same reason.
- **GSM8K is capped at 200 docs, MMLU at 20 docs/subject**; `wmdp_bio`
  (the recovery signal) always runs in full.
- Cross-check the step-0 (no-SFT) row against the ablation arm's baseline for
  the same model -- both harnesses score 0-shot `wmdp_bio` on the same
  checkpoints, so they should agree within binomial noise (~1.3pp on ~1270
  items). A larger gap means something differs between the two harnesses
  before you trust the recovery delta.

## Results

See [`results/wmdp_sft_recovery/recovery_table.md`](results/wmdp_sft_recovery/recovery_table.md)
for the full per-checkpoint table, and the top-level
[results.md](../results.md) for the summarized findings.
