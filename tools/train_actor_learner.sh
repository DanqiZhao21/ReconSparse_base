#!/bin/bash
set -euo pipefail

# Actor-Learner launcher (process-level parallelism):
# - 1 learner process
# - N actor processes
# - actual GPU mapping is defined in script/configs/*.yaml
#   via train.actor_learner.{learner_gpu_id, actor_gpu_ids, actor_per_gpu}

# CUDA 环境变量
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 定位 repo 路径
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDV2_ROOT="$REPO_ROOT/DiffusionDriveV2"
NAVSIM_ROOT="$DDV2_ROOT/navsim"
export PYTHONPATH="$REPO_ROOT:$DDV2_ROOT:$NAVSIM_ROOT:${PYTHONPATH:-}"

# Keep torch JIT artifacts under the repo to avoid stale global locks.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_ROOT/.cache/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# Choose algorithm/config.
# - Recommended: set CONFIG explicitly.
#   e.g. CONFIG=.../reinforcepp_closed_loop.yaml bash tools/train_actor_learner.sh
# - Convenience: set ALGO=reinforcepp to switch the default CONFIG.
# ALGO=${ALGO:-"reinforcepp"}
ALGO=${ALGO:-"ppo_sparsedrive_v2"}

if [[ -z "${CONFIG+x}" ]]; then
	if [[ "${ALGO}" == "reinforcepp" || "${ALGO}" == "reinforce++" || "${ALGO}" == "reinforce_pp" ]]; then
		CONFIG="$REPO_ROOT/script/configs/reinforcepp_closed_loop.yaml"
		printf "💗[train_actor_learner.sh] Using default CONFIG for ALGO=%s: %s\n" "${ALGO}" "${CONFIG}" >&2
	elif [[ "${ALGO}" == "ppo_sparsedrive" || "${ALGO}" == "sparsedrive" ]]; then
		CONFIG="$REPO_ROOT/script/configs/ppo_closed_loop_sparsedrive.yaml"
		printf "💗[train_actor_learner.sh] Using default CONFIG for ALGO=%s: %s\n" "${ALGO}" "${CONFIG}" >&2
	elif [[ "${ALGO}" == "ppo_sparsedrive_v2" || "${ALGO}" == "sparsedrive_v2" || "${ALGO}" == "sparsedrive-v2" || "${ALGO}" == "sdv2" ]]; then
		CONFIG="$REPO_ROOT/script/configs/ppo_closed_loop_sparsedrive_v2.yaml"
		printf "💗[train_actor_learner.sh] Using default CONFIG for ALGO=%s: %s\n" "${ALGO}" "${CONFIG}" >&2
	elif [[ "${ALGO}" == "ppo" ]]; then
		CONFIG="$REPO_ROOT/script/configs/ppo_closed_loop.yaml"
		printf "💗[train_actor_learner.sh] Using default CONFIG for ALGO=%s: %s\n" "${ALGO}" "${CONFIG}" >&2
	fi
fi

LOG_DIR=${LOG_DIR:-"."}
mkdir -p "${LOG_DIR}"

printf "[train_actor_learner.sh] CONFIG=%s\n" "${CONFIG}" >&2
printf "[train_actor_learner.sh] LOG_DIR=%s\n" "${LOG_DIR}" >&2
printf "[train_actor_learner.sh] TORCH_EXTENSIONS_DIR=%s\n" "${TORCH_EXTENSIONS_DIR}" >&2
printf "[train_actor_learner.sh] Launching orchestrator-based training\n" >&2

python -u "$REPO_ROOT/script/train_actor_learner_v2.py" \
	--role orchestrator \
	--config "${CONFIG}" \
	2>&1 | tee "${LOG_DIR}/actor_learner_orchestrator.log"
