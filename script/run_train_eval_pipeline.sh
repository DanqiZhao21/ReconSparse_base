#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_EVAL_SEED=1
for arg in "$@"; do
  if [[ "$arg" == "--no-default-eval-seed" ]]; then
    DEFAULT_EVAL_SEED=0
    break
  fi
done

if [[ "$DEFAULT_EVAL_SEED" == "1" ]]; then
  export HUGSIM_RANDOM_SEED="${HUGSIM_RANDOM_SEED:-0}"
  export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
  unset HUGSIM_DISABLE_DEFAULT_EVAL_SEED
else
  export HUGSIM_DISABLE_DEFAULT_EVAL_SEED=1
fi

exec /root/miniconda3/envs/recondreamerNew-rl/bin/python -u \
  "$REPO_ROOT/script/train_eval_pipeline.py" "$@"
