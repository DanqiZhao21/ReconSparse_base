#!/bin/bash
set -euo pipefail

# Enable job control so each background job runs in its own process group.
# This allows Ctrl+C to terminate the whole pipeline (python + tee) per GPU.
set -m

pids=()

cleanup() {
	local sig="${1:-INT}"
	echo "[trainclosedloop.sh] Caught ${sig}, stopping background trainings..." >&2

	# Try graceful termination first.
	for pid in "${pids[@]:-}"; do
		kill -TERM -- "-${pid}" 2>/dev/null || true
	done

	sleep 1

	# Force kill if still alive.
	for pid in "${pids[@]:-}"; do
		kill -KILL -- "-${pid}" 2>/dev/null || true
	done

	# Reap children to avoid zombies.
	wait || true

	# 130 is the conventional exit code for SIGINT.
	if [[ "${sig}" == "INT" ]]; then
		exit 130
	fi
	exit 1
}

trap 'cleanup INT' INT
trap 'cleanup TERM' TERM

# CUDA 头文件与库路径（修复 nvdiffrast JIT 编译缺少 cuda_runtime.h）
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python 路径（确保本仓库与 DiffusionDriveV2 可见）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDV2_ROOT="$REPO_ROOT/DiffusionDriveV2"
NAVSIM_ROOT="$DDV2_ROOT/navsim"
export PYTHONPATH="$REPO_ROOT:$DDV2_ROOT:$NAVSIM_ROOT:${PYTHONPATH:-}"

# Multi-GPU launcher: run one training per GPU (default 0 1 2 3)
GPUS=${GPUS:-"0 1 2 3"}
LOG_DIR=${LOG_DIR:-"."}
echo "Launching training on GPUs: ${GPUS}" 
for gid in ${GPUS}; do
	suffix="gpu${gid}"
	(
		export CUDA_VISIBLE_DEVICES=${gid}
		export RUN_SUFFIX="${suffix}"
		python -u "$REPO_ROOT/script/train_closed_loop.py" \
			2>&1 | tee "${LOG_DIR}/train_closed_loop_${suffix}.log"
	) &
	pids+=("$!")
done
wait
echo "All trainings finished. Logs in ${LOG_DIR}" 
