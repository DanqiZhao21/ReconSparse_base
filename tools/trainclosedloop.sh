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

# Multi-GPU launcher: run one training per GPU (default 0 1 2 3)
GPUS=${GPUS:-"0"}
LOG_DIR=${LOG_DIR:-"."}
echo "Launching training on GPUs: ${GPUS}" 
for gid in ${GPUS}; do
	suffix="gpu${gid}"
	CUDA_VISIBLE_DEVICES=${gid} RUN_SUFFIX="${suffix}" \
		python -u /root/clone/ReconDreamer-RL/script/train_closed_loop.py \
		2>&1 | tee "${LOG_DIR}/train_closed_loop_${suffix}.log" &
done
wait
echo "All trainings finished. Logs in ${LOG_DIR}" 
