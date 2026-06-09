#!/usr/bin/env bash
set -euo pipefail

exec /root/miniconda3/envs/recondreamerNew-rl/bin/python -u \
  /root/clone/ReconDreamer-RL/script/overnight_reward_diagnostic_queue_20260608.py "$@"
