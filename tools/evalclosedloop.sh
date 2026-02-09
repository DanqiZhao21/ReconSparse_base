#!/bin/bash
set -euo pipefail

# CUDA headers and libs (fix nvdiffrast JIT missing cuda_runtime.h)
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python paths (ensure repo + DiffusionDriveV2 are importable)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDV2_ROOT="$REPO_ROOT/DiffusionDriveV2"
NAVSIM_ROOT="$DDV2_ROOT/navsim"
export PYTHONPATH="$REPO_ROOT:$DDV2_ROOT:$NAVSIM_ROOT:${PYTHONPATH:-}"

# Run eval closed loop (no-grad scorer)
python -u "$REPO_ROOT/script/eval_closed_loop.py" 2>&1 | tee eval_closed_loop.log
