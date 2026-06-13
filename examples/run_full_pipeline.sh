#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:?Set MODEL to a Hugging Face model id or local checkpoint path}"
METHOD="${METHOD:-$(basename "$MODEL")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"

python scripts/collect_refusal_prompts.py \
  --model "$MODEL" \
  --method "$METHOD" \
  --forget-config forget10 \
  --output-csv "$OUTPUT_ROOT/refusal_prompts.csv" \
  --paired-output-csv "$OUTPUT_ROOT/paired_idk_plain_prompts.csv" \
  --write-paired-output \
  --hardware-profile "$HARDWARE_PROFILE" \
  --resume

python scripts/cosmic_refusal_controller.py \
  --model "$MODEL" \
  --paired-csv "$OUTPUT_ROOT/paired_idk_plain_prompts.csv" \
  --output-root "$OUTPUT_ROOT/cosmic_refusal_controller" \
  --hardware-profile "$HARDWARE_PROFILE" \
  --prompt-format "$PROMPT_FORMAT"

python scripts/cosmic_refusal_ablation_eval.py \
  --model "$MODEL" \
  --controller "$OUTPUT_ROOT/cosmic_refusal_controller/cosmic_selected_controller.pt" \
  --forced-idk-csv "$OUTPUT_ROOT/paired_idk_plain_prompts.csv" \
  --output-root "$OUTPUT_ROOT/cosmic_refusal_ablation_eval" \
  --hardware-profile "$HARDWARE_PROFILE"
