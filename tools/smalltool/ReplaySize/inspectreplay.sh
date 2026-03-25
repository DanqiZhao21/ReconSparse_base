#!/bin/bash
set -euo pipefail

# CUDA 头文件与库路径（修复 nvdiffrast JIT 编译缺少 cuda_runtime.h）
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/DiffusionDriveV2:$REPO_ROOT/DiffusionDriveV2/navsim:${PYTHONPATH:-}"

python "$REPO_ROOT/tools/smalltool/ReplaySize/inspect_replay_sizes.py" "$@"