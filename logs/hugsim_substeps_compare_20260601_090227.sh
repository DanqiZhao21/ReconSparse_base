#!/usr/bin/env bash
set -euo pipefail

cd /root/clone/ReconDreamer-RL
RUN_LOG=/root/clone/ReconDreamer-RL/logs/hugsim_substeps_compare_20260601_090227.log
mkdir -p /root/clone/ReconDreamer-RL/logs
exec > >(tee -a "$RUN_LOG") 2>&1

printf '[%s] tmux session: %s\n' "$(date '+%F %T')" "hugsim_substeps_compare_20260601_090227"
printf '[%s] log file: %s\n' "$(date '+%F %T')" "$RUN_LOG"

CONFIGS=(
  "/root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202606011200_HUGSM_reinforcepp_closed_loop_reward-close_loop_NoGRPOCraft_substeps2.yaml"
  "/root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202606011200_HUGSM_reinforcepp_closed_loop_reward-close_loop_oldParams_NoGRPOCraft_substeps2.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  printf '================================================================\n'
  printf '[%s] START %s\n' "$(date '+%F %T')" "$cfg"
  bash script/run_train_eval_pipeline_hugsim_ori.sh \
    --reinforcepp-config "$cfg" \
    --reinforcepp-algo-tag hugsim_ori_reinforcepp_substeps2_compare
  status=$?
  printf '[%s] END status=%s %s\n' "$(date '+%F %T')" "$status" "$cfg"
  if [ "$status" -ne 0 ]; then
    printf '[%s] STOP sequence because previous config failed.\n' "$(date '+%F %T')"
    exit "$status"
  fi
done

printf '[%s] ALL DONE\n' "$(date '+%F %T')"
