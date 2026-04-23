#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/clone/ReconDreamer-RL"
PYTHON_BIN="/root/miniconda3/envs/recondreamerNew-rl/bin/python"
LOG_DIR="${REPO_ROOT}/outputs/corner_grpo_train_eval_queue"
DATE_TAG="$(date -u +%Y%m%d)"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

nohup "${PYTHON_BIN}" -u "${REPO_ROOT}/tools/run_corner_grpo_train_eval_queue.py" \
  --date-tag "${DATE_TAG}" \
  > "${LOG_DIR}/queue_${DATE_TAG}.log" 2>&1 &

echo $!
