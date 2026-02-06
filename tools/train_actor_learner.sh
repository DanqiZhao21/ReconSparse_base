#!/bin/bash
set -euo pipefail

# Actor-Learner launcher:
# - Learners on GPUs 0-1 (torchrun DDP)
# - Actors on GPUs 2-3 (standalone, no DDP)

set -m
pids=()

cleanup() {
	local sig="${1:-INT}"
	echo "[train_actor_learner.sh] Caught ${sig}, stopping..." >&2
	for pid in "${pids[@]:-}"; do
		kill -TERM -- "-${pid}" 2>/dev/null || true
	done
	sleep 1
	for pid in "${pids[@]:-}"; do
		kill -KILL -- "-${pid}" 2>/dev/null || true
	done
	wait || true
	if [[ "${sig}" == "INT" ]]; then
		exit 130
	fi
	exit 1
}

trap 'cleanup INT' INT
trap 'cleanup TERM' TERM

export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=/root/clone/ReconDreamer-RL:/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:${PYTHONPATH:-}

# Choose algorithm/config.
# - Recommended: set CONFIG explicitly.
#   e.g. CONFIG=.../reinforcepp_closed_loop.yaml bash tools/train_actor_learner.sh
# - Convenience: set ALGO=reinforcepp to switch the default CONFIG.
ALGO=${ALGO:-"reinforcepp"}

if [[ -z "${CONFIG+x}" ]]; then
	if [[ "${ALGO}" == "reinforcepp" || "${ALGO}" == "reinforce++" || "${ALGO}" == "reinforce_pp" ]]; then
		CONFIG="/root/clone/ReconDreamer-RL/script/configs/reinforcepp_closed_loop.yaml"
		printf "💗[train_actor_learner.sh] Using default CONFIG for ALGO=%s: %s\n" "${ALGO}" "${CONFIG}" >&2
	else
		CONFIG="/root/clone/ReconDreamer-RL/script/configs/ppo_closed_loop.yaml"
		printf "💗[train_actor_learner.sh] Using default CONFIG for ALGO=%s: %s\n" "${ALGO}" "${CONFIG}" >&2
	fi
fi

LOG_DIR=${LOG_DIR:-"."}

# Start learners (GPU0-1)
(
	export CUDA_VISIBLE_DEVICES=0,1
	torchrun --nproc_per_node=2 /root/clone/ReconDreamer-RL/script/train_actor_learner.py --role learner --config "${CONFIG}" \
		2>&1 | tee "${LOG_DIR}/learner.log"
) &
pids+=("$!")

# Start actors (GPU2-3)
for aid in 0 1; do
	gid=$((aid+2))
	(
		export CUDA_VISIBLE_DEVICES=${gid}
		python -u /root/clone/ReconDreamer-RL/script/train_actor_learner.py --role actor --actor-id ${aid} --config "${CONFIG}" \
			2>&1 | tee "${LOG_DIR}/actor${aid}_gpu${gid}.log"
	) &
	pids+=("$!")
done

wait
