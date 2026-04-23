#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HUGSIM_ROOT="/root/clone/HUGSIM-ORI"
TRAIN_PYTHON="/root/miniconda3/envs/recondreamerNew-rl/bin/python"
SCENARIO_DIR="${SCENARIO_DIR:-$HUGSIM_ROOT/configs/scenarios/nuscenes}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$HUGSIM_ROOT/outputs/evaluate-auto}"
SLOTS=(${SLOTS:-0:0 1:1 2:2 3:3})
REPEAT_EVALS="${REPEAT_EVALS:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_sparsedrive_v2_no_grpo}"
RUN_ROOT="$REPO_ROOT/outputs/no_grpo_train_eval/$RUN_ID"
PROMOTE_DIR="$REPO_ROOT/egoADs/SparseDriveV2/ckpt"
DATE_TAG="$(date -u +%Y%m%d)"

PPO_CONFIG="$REPO_ROOT/script/configs/sparsedrive_v2/ppo_closed_loop_sparsedrive_v2_no_grpo.yaml"
REINFORCE_CONFIG="$REPO_ROOT/script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_no_grpo.yaml"

mkdir -p "$RUN_ROOT"

run_training() {
  local label="$1"
  local config_path="$2"
  local buffer_dir_rel="$3"
  local ckpt_name="$4"
  local log_path="$RUN_ROOT/${label}_train.log"

  echo "[train] label=$label config=$config_path log=$log_path"
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$TRAIN_PYTHON" -u "$REPO_ROOT/script/train_actor_learner_v2.py" \
      --role orchestrator \
      --config "$config_path" \
      >"$log_path" 2>&1

  local latest_ckpt="$REPO_ROOT/$buffer_dir_rel/weights/latest.ckpt"
  if [[ ! -f "$latest_ckpt" ]]; then
    echo "[error] missing training checkpoint: $latest_ckpt" >&2
    exit 1
  fi

  local promoted_ckpt="$PROMOTE_DIR/${DATE_TAG}_${ckpt_name}_latest.ckpt"
  cp -f "$latest_ckpt" "$promoted_ckpt"
  echo "[train-done] label=$label latest_ckpt=$latest_ckpt promoted_ckpt=$promoted_ckpt"
  printf '%s\n' "$promoted_ckpt" >"$RUN_ROOT/${label}_promoted_ckpt.txt"
}

run_training "ppo_no_grpo" "$PPO_CONFIG" "outputs/actor_learner_ppo_no_grpo" "ppo_noGrpo"
run_training "reinforce_no_grpo" "$REINFORCE_CONFIG" "outputs/actor_learner_reinforcepp_no_grpo" "reinforce_noGrpo"

PPO_CKPT="$(cat "$RUN_ROOT/ppo_no_grpo_promoted_ckpt.txt")"
REINFORCE_CKPT="$(cat "$RUN_ROOT/reinforce_no_grpo_promoted_ckpt.txt")"
EVAL_RUN_NAME="eval_${RUN_ID}_2x88x2"
EVAL_LOG="$RUN_ROOT/eval.log"

echo "[eval] run_name=$EVAL_RUN_NAME log=$EVAL_LOG"
"$TRAIN_PYTHON" -u "$REPO_ROOT/tools/evaluate_existing_sparsedrive_v2_ckpts.py" \
  --ckpts "$PPO_CKPT" "$REINFORCE_CKPT" \
  --scenario-dir "$SCENARIO_DIR" \
  --eval-output-root "$EVAL_OUTPUT_ROOT" \
  --run-name "$EVAL_RUN_NAME" \
  --repeat-evals "$REPEAT_EVALS" \
  --slots "${SLOTS[@]}" \
  >"$EVAL_LOG" 2>&1

echo "[done] run_root=$RUN_ROOT eval_output_root=$EVAL_OUTPUT_ROOT/$EVAL_RUN_NAME"
