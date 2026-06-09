#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE="$REPO_ROOT/script/run_train_eval_pipeline_hugsim_ori.sh"

LOG_ROOT="$REPO_ROOT/outputs/TrainEvaluationAuto"
mkdir -p "$LOG_ROOT"

active_jobs() {
  ps -eo pid=,ppid=,stat=,etime=,cmd= \
    | grep -E 'script/train_eval_pipeline.py|tools/evaluate_existing_sparsedrive_v2_ckpts.py|HUGSIM-ORI/closed_loop.py|SparseDriveV2-HF/sparsedrive_e2e.py' \
    | grep -v grep \
    | grep -v 'queue_collision_sac_train_eval_20260606.sh' \
    || true
}

wait_for_current_pipeline() {
  echo "[queue] waiting for currently running train/eval jobs to finish"
  while true; do
    current="$(active_jobs || true)"
    if [[ -z "$current" ]]; then
      echo "[queue] no active train/eval jobs detected"
      return 0
    fi
    echo "[queue] active job count: $(printf '%s\n' "$current" | wc -l)"
    printf '%s\n' "$current" | sed -n '1,20p'
    sleep 300
  done
}

run_one() {
  local label="$1"
  local config_path="$2"
  local algo_tag="$3"
  local log_path="$LOG_ROOT/queue_${label}_$(date -u +%Y%m%d_%H%M%S).log"
  echo "[queue] start ${label}"
  echo "[queue] config=${config_path}"
  echo "[queue] algo_tag=${algo_tag}"
  echo "[queue] log=${log_path}"
  if bash "$PIPELINE" \
    --reinforcepp-config "$config_path" \
    --reinforcepp-algo-tag "$algo_tag" \
    >"$log_path" 2>&1; then
    echo "[queue] finished ${label}"
  else
    local code="$?"
    echo "[queue] FAILED ${label} code=${code}; see ${log_path}"
    return 0
  fi
}

wait_for_current_pipeline

run_one \
  "collision_rpp_stable" \
  "$REPO_ROOT/script/configs/sparsedrive_v2/202606061245_HUGSM_reinforcepp_closed_loop_steppath_hd_collision_only_extreme_NoGRPOCraft_substeps1_epoch1_lr5e-7_kl0p03_diag.yaml" \
  "hugsim_ori_collision_only_rpp_stable_lr5e7_kl003"

run_one \
  "sac_collision_dense_extreme" \
  "$REPO_ROOT/script/configs/sparsedrive_v2/202606061330_HUGSM_sac_closed_loop_steppath_hd_collision_dense_extreme_NoGRPOCraft_substeps1_lr5e-7_entropy0p01.yaml" \
  "hugsim_ori_sac_collision_dense_extreme"

run_one \
  "sac_collision_terminal_only_extreme" \
  "$REPO_ROOT/script/configs/sparsedrive_v2/202606061331_HUGSM_sac_closed_loop_steppath_hd_collision_terminal_only_extreme_NoGRPOCraft_substeps1_lr3e-7_entropy0p03.yaml" \
  "hugsim_ori_sac_collision_terminal_only_extreme"

echo "[queue] all queued train/eval attempts finished"
