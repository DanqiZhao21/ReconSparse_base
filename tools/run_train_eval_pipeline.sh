#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec /root/miniconda3/envs/recondreamerNew-rl/bin/python -u \
  "$REPO_ROOT/tools/train_eval_pipeline.py" "$@"
