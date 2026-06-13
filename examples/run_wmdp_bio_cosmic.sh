#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:?Set MODEL to a Hugging Face model id or local checkpoint path}"
MODEL_LABEL="${MODEL_LABEL:-$(basename "$MODEL")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/wmdp_bio_cosmic_eval}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-t4x2}"

python scripts/wmdp_bio_refusal_direction_eval.py \
  --model "$MODEL" \
  --model-label "$MODEL_LABEL" \
  --selection-method cosmic \
  --output-root "$OUTPUT_ROOT" \
  --hardware-profile "$HARDWARE_PROFILE"
