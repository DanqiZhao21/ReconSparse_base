
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:$PYTHONPATH
export PYTHONPATH=/root/clone/nuplan-devkit:$PYTHONPATH
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/nuplan-devkit:$PYTHONPATH

export TRAIN_TEST_SPLIT=navtest_mini
export PYTHONPATH=/root/clone/ReconDreamer-RL:$PYTHONPATH
# export CHECKPOINT=/root/clone/ReconDreamer-RL/diffusion_drive/ckpt/diffusiondrive_navsim_88p1_PDMS.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

python /root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim/planning/script/run_pdm_score.py \
        agent=diffusiondrivev2_rl_agent \
        experiment_name=diffusiondrivev2_agent_eval \
        worker=ray_distributed \
        train_test_split=navtest_mini \
        agent.checkpoint_path=/root/clone/ReconDreamer-RL/outputs/weight/20260129_ppo_ver27_latest.ckpt \
        metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/" 

        # /root/clone/DiffusionDriveV2/navsim/planning/script/config/common/train_test_split
        # /root/clone/ReconDreamer-RL/outputs/weight/20260129_ppo_ver27_latest.ckpt
        #NOTE /root/clone/ReconDreamer-RL/DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt