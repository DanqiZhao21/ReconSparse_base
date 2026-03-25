#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDV2_ROOT="$REPO_ROOT/DiffusionDriveV2"
NAVSIM_ROOT="$DDV2_ROOT/navsim"

export PYTHONPATH="$NAVSIM_ROOT:$DDV2_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
if [[ -n "${NUPLAN_DEVKIT_ROOT:-}" ]]; then
	export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$PYTHONPATH"
fi

export TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest_mini}
# export CHECKPOINT="$REPO_ROOT/diffusion_drive/ckpt/diffusiondrive_navsim_88p1_PDMS.pth"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

python "$NAVSIM_ROOT/planning/script/run_pdm_score.py" \
        agent=diffusiondrivev2_rl_agent \
        experiment_name=diffusiondrivev2_agent_eval \
        worker=ray_distributed \
        train_test_split="${TRAIN_TEST_SPLIT}" \
        agent.checkpoint_path="$REPO_ROOT/outputs/weight/20260129_ppo_ver27_latest.ckpt" \
        metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/" 

                # Reference:
                # "$NAVSIM_ROOT/planning/script/config/common/train_test_split"
                # "$REPO_ROOT/outputs/weight/20260129_ppo_ver27_latest.ckpt"
                # NOTE "$REPO_ROOT/DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt"