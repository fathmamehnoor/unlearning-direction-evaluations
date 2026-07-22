#!/usr/bin/env bash
# Shared configuration for the WMDP unrelated-SFT recovery experiment.
#
# Both run_base_control.sh and run_unlearned.sh source this file so that the
# ONE seed and ONE QLoRA recipe are provably identical across arms. The only
# thing that differs between arms is the model (and thus the checkpoint) -- not
# the fine-tuning. Do not fork these values per arm.

set -euo pipefail

# --- Shared knobs -----------------------------------------------------------
: "${SEED:=42}"
: "${NUM_EXAMPLES:=6000}"
: "${CHECKPOINTS:=1000 3000 6000}"        # sample the small end; recovery lives early
: "${TASKS:=wmdp_bio,mmlu,gsm8k}"
: "${PROB_TASKS:=wmdp_bio}"

# QLoRA recipe (single recipe across all models)
: "${LORA_R:=32}"
: "${LORA_ALPHA:=64}"
: "${LORA_DROPOUT:=0.05}"
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=16}"
: "${EPOCHS:=1}"
: "${MAX_LENGTH:=1024}"
: "${LR:=2e-4}"

# Eval knobs
: "${EVAL_BATCH_SIZE:=auto}"
# ONE precision for EVERY row of a model (baseline AND SFT checkpoints).
# Reported recovery is baseline-minus-SFT, so a precision difference between
# those two rows lands directly in the number. eval_one below applies these to
# all rows uniformly -- do not set --load-in-4bit on some rows and not others.
: "${EVAL_DTYPE:=bfloat16}"
: "${EVAL_LOAD_IN_4BIT:=0}"               # 0 = bf16 everywhere (recommended); 1 = 4-bit everywhere
: "${MMLU_LIMIT:=20}"                     # cap MMLU at N docs/subject (utility gate)
: "${GSM8K_LIMIT:=200}"                   # cap gsm8k at N docs (uptake check only); applied to every checkpoint of a model
: "${APPLY_CHAT_TEMPLATE:=1}"             # 1 = apply chat template; set 0 for models with a broken template (e.g. ScaleAI RMU)
: "${LIMIT:=}"                            # global smoke-test limit (all tasks); overrides per-task limits when set

# --- Shared paths -----------------------------------------------------------
: "${OUTPUT_ROOT:=results/wmdp_sft_recovery}"
: "${ADAPTER_ROOT:=${OUTPUT_ROOT}/adapters}"
: "${EVAL_DIR:=${OUTPUT_ROOT}/eval}"
: "${RESULTS_CSV:=${OUTPUT_ROOT}/recovery_results.csv}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${HERE}/train_gsm8k_qlora.py"
EVAL_PY="${HERE}/eval_recovery_lm_eval.py"

# --- Shared helpers ---------------------------------------------------------
# train_one MODEL_NAME MODEL_LABEL UNLEARNING_METHOD ARM
train_one() {
  local model_name="$1" model_label="$2" method="$3" arm="$4"
  echo "=== TRAIN [$arm] $model_label ($method) on GSM8K x${NUM_EXAMPLES} ==="
  python "$TRAIN_PY" \
    --model-name "$model_name" \
    --model-label "$model_label" \
    --unlearning-method "$method" \
    --arm "$arm" \
    --output-root "$ADAPTER_ROOT" \
    --dataset-name openai/gsm8k --dataset-config main \
    --num-examples "$NUM_EXAMPLES" \
    --checkpoint-examples $CHECKPOINTS \
    --seed "$SEED" \
    --max-length "$MAX_LENGTH" \
    --epochs "$EPOCHS" --lr "$LR" \
    --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
    --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT"
}

# run_dir for a given label matches train_gsm8k_qlora.py's run-name convention.
adapter_run_dir() {
  local model_label="$1"
  local slug
  slug="$(echo "$model_label" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_//; s/_$//')"
  echo "${ADAPTER_ROOT}/${slug}_gsm8k${NUM_EXAMPLES}_seed${SEED}"
}

# eval_one MODEL_NAME MODEL_LABEL UNLEARNING_METHOD ARM STEP [ADAPTER_PATH] [NOTES]
eval_one() {
  local model_name="$1" model_label="$2" method="$3" arm="$4" step="$5"
  local adapter="${6:-}" notes="${7:-}"
  echo "=== EVAL  [$arm] $model_label step=$step ${adapter:+(+adapter)} ==="
  local cmd=(python "$EVAL_PY"
    --model-name "$model_name"
    --model-label "$model_label"
    --unlearning-method "$method"
    --arm "$arm"
    --tasks "$TASKS" --prob-tasks "$PROB_TASKS"
    --checkpoint-step "$step"
    --dtype "$EVAL_DTYPE" --batch-size "$EVAL_BATCH_SIZE"
    --results-csv "$RESULTS_CSV" --output-dir "$EVAL_DIR"
    --seed "$SEED")
  [ -n "$adapter" ] && cmd+=(--adapter-path "$adapter")
  [ -n "$notes" ] && cmd+=(--notes "$notes")
  # Precision applied to EVERY row uniformly (baseline + all checkpoints).
  [ "$EVAL_LOAD_IN_4BIT" = "1" ] && cmd+=(--load-in-4bit)
  # Chat template applied uniformly to every row (baseline + checkpoints).
  [ "$APPLY_CHAT_TEMPLATE" = "0" ] && cmd+=(--no-chat-template)
  if [ -n "$LIMIT" ]; then
    # Smoke test: global limit wins for all tasks.
    cmd+=(--limit "$LIMIT")
  else
    # Real run: cap mmlu (utility gate) and gsm8k (uptake check) identically on
    # every checkpoint of this model; wmdp_bio (recovery signal) always stays full.
    local tl=""
    [ -n "$MMLU_LIMIT" ] && tl="mmlu:${MMLU_LIMIT}"
    [ -n "$GSM8K_LIMIT" ] && tl="${tl:+$tl,}gsm8k:${GSM8K_LIMIT}"
    [ -n "$tl" ] && cmd+=(--task-limits "$tl")
  fi
  "${cmd[@]}"
}

# Full per-model flow: no-SFT baseline row, then one row per SFT checkpoint.
train_and_eval_model() {
  local model_name="$1" model_label="$2" method="$3" arm="$4" baseline_note="$5"
  train_one "$model_name" "$model_label" "$method" "$arm"
  local run_dir
  run_dir="$(adapter_run_dir "$model_label")"
  eval_one "$model_name" "$model_label" "$method" "$arm" 0 "" "$baseline_note"
  for ckpt in $CHECKPOINTS; do
    eval_one "$model_name" "${model_label}_gsm8k_${ckpt}" "$method" "$arm" "$ckpt" \
      "${run_dir}/checkpoint-examples-${ckpt}" "unrelated_gsm8k_sft"
  done
}
