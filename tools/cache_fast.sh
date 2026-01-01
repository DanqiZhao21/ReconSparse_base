
#====================================================
# cache dataset for fast evaluation  (optional) ✔👌🎈
#========================================================
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim:$PYTHONPATH
export PYTHONPATH=/root/clone/nuplan-devkit:$PYTHONPATH
export PYTHONPATH=/root/clone/ReconDreamer-RL/DiffusionDriveV2:/root/clone/nuplan-devkit:$PYTHONPATH
python /root/clone/ReconDreamer-RL/DiffusionDriveV2/navsim/planning/script/run_dataset_caching.py agent=diffusiondrivev2_rl_agent experiment_name=diffusiondrivev2_cache train_test_split=navtest cache_path=$NAVSIM_EXP_ROOT/metric_feature_cache

