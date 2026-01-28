#!/bin/bash
set -euo pipefail

# CUDA 头文件与库路径（修复 nvdiffrast JIT 编译缺少 cuda_runtime.h）
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=/root/clone/ReconDreamer-RL:/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:${PYTHONPATH:-}
#ADD
# python /root/clone/ReconDreamer-RL/tools/debug_scene_yaw_video.py "$@"



# # '''
# #  cd /root/clone/ReconDreamer-RL && bash tools/debugyaw.sh --scene 0 --start-frame 0 --max-steps 5 --draw-traj-overlay --out outputs/yaw_debug/scene000_check_yaw_err_fields.mp4 --disable-threshold-termination
# # '''

# # '''
# # bash tools/debugyaw.sh --scene 512 --start-frame 0 --max-steps 185 --draw-traj-overlay --out outputs/yaw_debug/scene000_no_yaw_reward_smoke.mp4 --disable-threshold-termination
# # '''
#ADD
python /root/clone/ReconDreamer-RL/tools/debug_ddv2_scene1_interp_video.py