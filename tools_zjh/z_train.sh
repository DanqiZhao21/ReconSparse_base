
# conda activate recondreamerNew-rl
cd /root/clone/ReconDreamer-RL
pp=/root/miniconda3/envs/recondreamerNew-rl/bin/python

config=/root/clone/ReconDreamer-RL/tools_zjh/202605280011_HUGSM_reinforcepp_closed_loop_closeCloseloop_NoGRPOCraft.yaml
config=/root/clone/ReconDreamer-RL/tools_zjh/202606041345_HUGSM_reinforcepp_closed_loop_closeCloseloop_NoGRPOCraft_yieldExtreme.yaml

echo "$pp -u  /root/clone/ReconDreamer-RL/script/train_actor_learner_v2.py   \
        --role orchestrator   \
        --config $config"

$pp -u  /root/clone/ReconDreamer-RL/script/train_actor_learner_v2.py   \
        --role orchestrator   \
        --config $config

