
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