#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE="$REPO_ROOT/script/run_train_eval_pipeline_hugsim_ori.sh"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/reward_sweep_20260601_hugsim_ori}"

mkdir -p "$LOG_ROOT"

CONFIG_DIR="$REPO_ROOT/script/configs/sparsedrive_v2"

declare -a RUNS=(
  "close_loop|hugsim_ori_reinforcepp_close_loop_nogrpo_substeps2|$CONFIG_DIR/202606011200_HUGSM_reinforcepp_closed_loop_reward-close_loop_NoGRPOCraft_substeps2.yaml"
  "correction|hugsim_ori_reinforcepp_correction_nogrpo_substeps2|$CONFIG_DIR/202606011200_HUGSM_reinforcepp_closed_loop_reward-correction_NoGRPOCraft_substeps2.yaml"
  "step_path|hugsim_ori_reinforcepp_step_path_nogrpo_substeps2|$CONFIG_DIR/202606011200_HUGSM_reinforcepp_closed_loop_reward-step_path_NoGRPOCraft_substeps2.yaml"
)

echo "[sweep] start $(date -Is)"
echo "[sweep] log_root=$LOG_ROOT"

for run in "${RUNS[@]}"; do
  IFS="|" read -r name algo_tag config_path <<< "$run"
  run_log="$LOG_ROOT/${name}.pipeline.log"
  status_file="$LOG_ROOT/${name}.status"
  echo "[sweep] ${name} start $(date -Is) config=${config_path} algo_tag=${algo_tag}" | tee "$status_file"
  echo "[sweep] ${name} pipeline_log=${run_log}"
  set +e
  "$PIPELINE" \
    --reinforcepp-config "$config_path" \
    --reinforcepp-algo-tag "$algo_tag" \
    > "$run_log" 2>&1
  rc=$?
  set -e
  echo "[sweep] ${name} pipeline rc=${rc} $(date -Is)" | tee -a "$status_file"
  if [[ "$rc" != "0" ]]; then
    echo "[sweep] ${name} failed; see ${run_log}" >&2
    exit "$rc"
  fi
  echo "[sweep] ${name} done $(date -Is)" | tee -a "$status_file"
done

echo "[sweep] all done $(date -Is)"
