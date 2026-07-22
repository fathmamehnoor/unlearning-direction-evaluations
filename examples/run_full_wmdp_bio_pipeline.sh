#!/usr/bin/env bash
# WMDP-Bio refusal-direction pipeline for one model, using a generic
# harmful/harmless prompt source (Arditi et al. 2024's AdvBench-vs-Alpaca
# setup):
#
#   1. extract_refusal_direction.py, once. Produces, per selection method:
#      direction_{method}.pt (the real direction) and
#      direction_{method}_matched_control.pt (a same-construction,
#      content-free null: split-half difference-in-means over a fresh,
#      disjoint harmless prompt pool, at the same position/layer as the
#      real direction).
#
#   2. wmdp_bio_lm_eval_ablation.py, once per resulting direction (2 total),
#      scoring WMDP-Bio accuracy under ablation via lm_eval, for
#      baseline / selected_direction_ablation / matched_control_ablation /
#      random_direction_ablation_{0..N}. The no-hook baseline is always
#      skipped (assumed already evaluated separately for this model). The
#      random-direction and matched-control conditions are only computed
#      once for the model (in the first ablation-eval call) and skipped for
#      the rest.
#
# A biosecurity-flavored prompt source (harmful biosecurity-related prompts
# vs. benign biology prompts, same underlying mechanism) has been removed
# from this codebase for now -- it will be reintroduced as separate, later
# work rather than bundled into this generic-only pipeline.
#
# Usage:
#   MODEL=OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
#   MODEL_LABEL=idk-ap \
#   OUTPUT_ROOT=outputs/idk-ap \
#   HARDWARE_PROFILE=a100 \
#   bash examples/run_full_wmdp_bio_pipeline.sh
#
# Run once per model (this repo's targets: IDK-AP and ILU-RMU WMDP-unlearned
# Llama-3-8B-Instruct checkpoints from OPTML-Group).

set -euo pipefail

MODEL="${MODEL:?Set MODEL to a Hugging Face model id or local checkpoint path}"
MODEL_LABEL="${MODEL_LABEL:-$(basename "$MODEL")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${MODEL_LABEL}}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"

EXTRACT_ROOT="${OUTPUT_ROOT}/direction_extraction"
EVAL_ROOT="${OUTPUT_ROOT}/wmdp_bio_lm_eval"

SOURCE="generic"

echo "=== [1/2] extracting refusal direction for ${MODEL_LABEL} ==="
python scripts/extract_refusal_direction.py \
  --model "$MODEL" \
  --model-label "$MODEL_LABEL" \
  --selection-method both \
  --hardware-profile "$HARDWARE_PROFILE" \
  --output-root "${EXTRACT_ROOT}/${SOURCE}"

echo "=== [2/2] evaluating WMDP-Bio accuracy under ablation via lm_eval ==="
FIRST=1
for METHOD in mean_diff cosmic; do
  DIRECTION_PATH="${EXTRACT_ROOT}/${SOURCE}/direction_${METHOD}.pt"
  MATCHED_CONTROL_PATH="${EXTRACT_ROOT}/${SOURCE}/direction_${METHOD}_matched_control.pt"
  RUN_LABEL="${SOURCE}_${METHOD}"
  if [ "$FIRST" -eq 1 ]; then
    # First run: baseline already evaluated separately for this model --
    # skip it here, but still compute the random-direction and
    # matched-control conditions.
    python scripts/wmdp_bio_lm_eval_ablation.py \
      --model "$MODEL" \
      --model-label "$MODEL_LABEL" \
      --direction-path "$DIRECTION_PATH" \
      --matched-control-direction-path "$MATCHED_CONTROL_PATH" \
      --hardware-profile "$HARDWARE_PROFILE" \
      --apply-chat-template \
      --skip-baseline \
      --output-root "${EVAL_ROOT}/${RUN_LABEL}"
    FIRST=0
  else
    # Subsequent runs: reuse the random-control numbers already computed
    # above for this model -- just score this direction (and its own
    # matched control, since that's specific to this method's
    # position/layer).
    python scripts/wmdp_bio_lm_eval_ablation.py \
      --model "$MODEL" \
      --model-label "$MODEL_LABEL" \
      --direction-path "$DIRECTION_PATH" \
      --matched-control-direction-path "$MATCHED_CONTROL_PATH" \
      --hardware-profile "$HARDWARE_PROFILE" \
      --apply-chat-template \
      --skip-baseline \
      --num-random-controls 0 \
      --output-root "${EVAL_ROOT}/${RUN_LABEL}"
  fi
done

echo "=== done: results under ${OUTPUT_ROOT} ==="
