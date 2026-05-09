#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$REPO_ROOT/outputs/overnight_reward_train_eval_launcher"
mkdir -p "$LOG_ROOT"
LAUNCH_LOG="$LOG_ROOT/launch_$(date -u +%Y%m%d_%H%M%S).log"

nohup /root/miniconda3/envs/recondreamerNew-rl/bin/python -u \
  "$REPO_ROOT/tools/run_overnight_reward_train_eval.py" \
  "$@" \
  >"$LAUNCH_LOG" 2>&1 &

echo "$!" >"$LOG_ROOT/latest_pid.txt"
echo "$LAUNCH_LOG" >"$LOG_ROOT/latest_log.txt"
echo "[launched] pid=$(cat "$LOG_ROOT/latest_pid.txt") log=$LAUNCH_LOG"
