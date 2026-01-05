#!/bin/bash
set -euo pipefail

# CUDA headers and libs (fix nvdiffrast JIT missing cuda_runtime.h)
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python paths (ensure repo + DiffusionDriveV2 are importable)
export PYTHONPATH=/root/clone/ReconDreamer-RL:/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:${PYTHONPATH:-}

# Run eval closed loop (no-grad scorer)
python -u /root/clone/ReconDreamer-RL/script/eval_closed_loop.py 2>&1 | tee eval_closed_loop.log
