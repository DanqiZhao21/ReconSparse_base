#!/usr/bin/env bash
set -euo pipefail

# Repo-root relative runner for generate_video.py
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# CUDA headers/libs are needed for some JIT-compiled render extensions (e.g., nvdiffrast).
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Keep torch JIT build artifacts inside the repo (avoid clutter/permission issues).
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/.cache/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/DiffusionDriveV2:$REPO_ROOT/DiffusionDriveV2/navsim:${PYTHONPATH:-}"
if [[ -n "${NUPLAN_DEVKIT_ROOT:-}" ]]; then
  export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$PYTHONPATH"
fi

python "$REPO_ROOT/tools/smalltool/visualize/generate_video.py" "$@"


# '''
# bash tools/smalltool/visualize/generate_video.sh --scene-list 413 --duration-s 18.0 --step-frames 5 --interp-method blend
# '''
