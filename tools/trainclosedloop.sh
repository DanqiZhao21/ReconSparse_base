#!/bin/bash
set -euo pipefail

# CUDA 头文件与库路径（修复 nvdiffrast JIT 编译缺少 cuda_runtime.h）
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python 路径（确保本仓库与 DiffusionDriveV2 可见）
export PYTHONPATH=/root/clone/ReconDreamer-RL:/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:${PYTHONPATH:-}

# python /root/clone/ReconDreamer-RL/script/train_closed_loop.py
python -u /root/clone/ReconDreamer-RL/script/train_closed_loop.py 2>&1 | tee train_closed_loop.log
