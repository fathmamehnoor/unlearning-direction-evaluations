#!/usr/bin/env bash
# Full-knowledge control arm.
#
# Applies the SAME GSM8K QLoRA SFT to a model that never had bio knowledge
# removed (base Llama-3-8B-Instruct, the checkpoint the OPTML-Group WMDP models
# were unlearned from). This is the most important control: it tells you what
# unrelated SFT does to WMDP-bio behavior ON ITS OWN, with no unlearning to
# undo. If unrelated SFT moves the unlearned model's bio accuracy in a
# different direction, or by a different amount, than it moves this control's,
# that difference is the evidence the change is about the unlearning and not a
# generic side effect of fine-tuning.
#
# Produces: no-SFT baseline row + one row per GSM8K checkpoint (1000/3000/6000),
# all appended to $RESULTS_CSV, each carrying wmdp_bio acc + correct-option
# prob, mmlu acc, and gsm8k acc on the same line.
#
# Override any knob via env, e.g.:  LIMIT=16 bash run_base_control.sh   (smoke)

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${BASE_MODEL:=meta-llama/Meta-Llama-3-8B-Instruct}"
: "${BASE_LABEL:=base_llama3_8b_instruct}"

train_and_eval_model \
  "$BASE_MODEL" "$BASE_LABEL" "none" "full_knowledge_control" \
  "full_knowledge_control_no_sft_baseline"

echo "Base-control arm complete. Rows appended to: $RESULTS_CSV"
