export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:$PYTHONPATH
export PYTHONPATH=/root/clone/nuplan-devkit:$PYTHONPATH
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/nuplan-devkit:$PYTHONPATH
export TRAIN_TEST_SPLIT=navtest
# export CHECKPOINT=/root/clone/ReconDreamer-RL/diffusion_drive/ckpt/diffusiondrive_navsim_88p1_PDMS.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python /root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim/planning/script/run_pdm_score_fast.py \
        agent=diffusiondrivev2_sel_agent \
        experiment_name=diffusiondrivev2_agent_eval \
        train_test_split=navtest \
        agent.checkpoint_path=/root/clone/ReconDreamer-RL/DiffusionDriveV2/ckpt/diffusiondrivev2_sel.ckpt \
        +metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/" \
        +test_cache_path="${NAVSIM_EXP_ROOT}/metric_feature_cache/"  