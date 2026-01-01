
===================1️⃣Caching=========================
# cache dataset for training
PYTHONPATH=~/clone/DiffusionDriveV2 \
export PYTHONPATH=/root/clone/DiffusionDriveV2:$PYTHONPATH
export NAVSIM_SKIP_MISSING=1
python navsim/planning/script/run_dataset_caching.py agent=diffusiondrivev2_rl_agent experiment_name=diffusiondrivev2_cache train_test_split=navtrain
#默认在 $NAVSIM_EXP_ROOT/experiments/diffusiondrivev2_cache/navtrain（通常脚本会按 experiment_name 和 split 建文件夹）

#====================================================
# cache dataset for evaluation ✔👌🎈
#========================================================
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:$PYTHONPATH
export PYTHONPATH=/root/clone/nuplan-devkit:$PYTHONPATH
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/nuplan-devkit:$PYTHONPATH

PYTHONPATH=~/clone/DiffusionDriveV2 \
python /root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim/planning/script/run_metric_caching.py \
train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
# train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache

# cache dataset for calculating PDMS during training.
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain cache.cache_path=$NAVSIM_EXP_ROOT/train_pdm_cache

#====================================================
# cache dataset for fast evaluation  (optional) ✔👌🎈
#========================================================
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:$PYTHONPATH
export PYTHONPATH=/root/clone/nuplan-devkit:$PYTHONPATH
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/nuplan-devkit:$PYTHONPATH
python /root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim/planning/script/run_dataset_caching.py agent=diffusiondrivev2_rl_agent experiment_name=diffusiondrivev2_cache train_test_split=navtest cache_path=$NAVSIM_EXP_ROOT/metric_feature_cache


====================2️⃣Evaluating=========================
#====================================================
#  fast evaluation  (optional) ✔👌🎈
#========================================================
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

#====================================================
#  No fast evaluation ✔👌🎈
#========================================================
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:$PYTHONPATH
export PYTHONPATH=/root/clone/nuplan-devkit:$PYTHONPATH
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/nuplan-devkit:$PYTHONPATH
export TRAIN_TEST_SPLIT=navtest
export PYTHONPATH=/root/clone/ReconDreamer-RL:$PYTHONPATH


export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2:$PYTHONPATH
# export CHECKPOINT=/root/clone/ReconDreamer-RL/diffusion_drive/ckpt/diffusiondrive_navsim_88p1_PDMS.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

python /root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim/planning/script/run_pdm_score.py \
        agent=diffusiondrivev2_sel_agent \
        experiment_name=diffusiondrivev2_agent_eval \
        worker=sequential \
        train_test_split=navtest \
        agent.checkpoint_path=/root/clone/ReconDreamer-RL/DiffusionDriveV2/ckpt/diffusiondrivev2_sel.ckpt \
        metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/" 
