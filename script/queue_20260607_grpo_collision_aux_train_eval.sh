#!/usr/bin/env bash
set -euo pipefail

exec /root/miniconda3/envs/recondreamerNew-rl/bin/python -u \
  /root/clone/ReconDreamer-RL/script/queue_20260607_grpo_collision_aux_train_eval.py "$@"
