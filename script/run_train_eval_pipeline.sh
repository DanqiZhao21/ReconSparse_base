#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_EVAL_SEED=1
HAS_EXPLICIT_SLOTS=0
for arg in "$@"; do
  if [[ "$arg" == "--no-default-eval-seed" ]]; then
    DEFAULT_EVAL_SEED=0
  elif [[ "$arg" == "--slots" ]]; then
    HAS_EXPLICIT_SLOTS=1
  fi
done

if [[ "$DEFAULT_EVAL_SEED" == "1" ]]; then
  export HUGSIM_RANDOM_SEED="${HUGSIM_RANDOM_SEED:-0}"
  export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
  unset HUGSIM_DISABLE_DEFAULT_EVAL_SEED
else
  export HUGSIM_DISABLE_DEFAULT_EVAL_SEED=1
fi

EXTRA_ARGS=()
if [[ "$HAS_EXPLICIT_SLOTS" == "0" ]]; then
  EXTRA_ARGS=(--slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7)
fi

exec /root/miniconda3/envs/recondreamerNew-rl/bin/python -u \
  "$REPO_ROOT/script/train_eval_pipeline.py" "$@" "${EXTRA_ARGS[@]}"
