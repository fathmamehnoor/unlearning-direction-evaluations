#!/usr/bin/env bash
# Unlearned arm.
#
# Runs the recovery experiment across more than one unlearning method, keeping
# it parallel to the ablation arm (IDK-AP and ILU-RMU). A recovery that shows
# on one method but not another is itself a finding. Each model gets the SAME
# GSM8K QLoRA recipe and seed as the base control (see common.sh) -- the only
# difference between arms is the model.
#
# For each method: no-SFT baseline row (the row everything is compared against)
# + one row per GSM8K checkpoint (1000/3000/6000). All appended to $RESULTS_CSV.
#
# Select methods with:  METHODS="ILU-RMU" bash run_unlearned.sh
# Override any knob via env, e.g.:  LIMIT=16 bash run_unlearned.sh   (smoke)

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# method_key -> "HF_MODEL_ID|LABEL"
declare -A UNLEARNED_MODELS=(
  ["RMU"]="ScaleAI/mhj-llama3-8b-rmu|rmu_llama3_8b"
  ["IDK-AP"]="OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct|idk_ap_wmdp_llama3_8b"
  ["ILU-RMU"]="OPTML-Group/ILU-RMU-WMDP-llama3-8b-instruct|ilu_rmu_wmdp_llama3_8b"
  ["GradDiff"]="OPTML-Group/GradDiff-WMDP-llama3-8b-instruct|graddiff_wmdp_llama3_8b"
  ["NPO"]="OPTML-Group/NPO-WMDP-llama3-8b-instruct|npo_wmdp_llama3_8b"
  ["NPO-ILU"]="OPTML-Group/NPO-ILU-WMDP-llama3-8b-instruct|npo_ilu_wmdp_llama3_8b"
)

: "${METHODS:=RMU IDK-AP ILU-RMU}"

for method in $METHODS; do
  spec="${UNLEARNED_MODELS[$method]:-}"
  if [ -z "$spec" ]; then
    echo "Unknown method '$method'. Known: ${!UNLEARNED_MODELS[*]}" >&2
    exit 2
  fi
  model_name="${spec%%|*}"
  model_label="${spec##*|}"
  train_and_eval_model \
    "$model_name" "$model_label" "$method" "unlearned" \
    "unlearned_no_sft_baseline"
done

echo "Unlearned arm complete. Rows appended to: $RESULTS_CSV"
