

cd /root/clone/ReconDreamer-RL
pp=/root/miniconda3/envs/recondreamerNew-rl/bin/python

scene_num=013
hugsim_scene=$1
# hugsim_scene=scene-0013-medium-front-car-slowdonw
# hugsim_scene=scene-0013-medium-front-car-cutin
# hugsim_scene=scene-0013-medium-person-cross
# hugsim_scene=scene-0013-easy-00

# for org sparsedrivev2 eval vis 
# ckpt=/root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt
config=/root/clone/ReconDreamer-RL/tools_zjh/202605280011_HUGSM_reinforcepp_closed_loop_closeCloseloop_NoGRPOCraft.yaml
config=/root/clone/ReconDreamer-RL/tools_zjh/202606041345_HUGSM_reinforcepp_closed_loop_closeCloseloop_NoGRPOCraft_yieldExtreme.yaml


# for RL eval vis
# ckpt=/root/clone/ReconDreamer-RL/outputs/recondiff_outputs/actor_learner/20260526_074819_Noclose_OpenCraftGrpo/weights/latest.ckpt
# ckpt=/root/clone/ReconDreamer-RL/outputs/actor_learner/20260528_071819_HUGSIM_CloseCloseloop_OpenGRPOCraft_FullPara/weights/latest.ckpt
# ckpt=/root/clone/ReconDreamer-RL/outputs/actor_learner/20260528_081109_HUGSIM_CloseCloseloop_OpenGRPOCraft_FullPara/weights/latest.ckpt
# ckpt=/root/clone/ReconDreamer-RL/outputs/actor_learner/20260529_014619_HUGSIM_CloseCloseloop_OpenGRPOCraft_FullPara/weights/latest.ckpt
# ckpt=/root/clone/ReconDreamer-RL/checkpoints/actor_learner/20260601_045330_HUGSIM_StepPath_NoGRPOCraft/weights/latest.ckpt
# ckpt=/root/clone/ReconDreamer-RL/checkpoints/actor_learner/20260601_045453_HUGSIM_StepPath_NoGRPOCraft/weights/latest.ckpt
# ckpt=/root/clone/ReconDreamer-RL/checkpoints/actor_learner/20260604_095902_HUGSIM_StepPathYieldSafety_NoGRPOCraft/weights/latest.ckpt
ckpt=/root/clone/ReconDreamer-RL/checkpoints/actor_learner/20260605_015643_HUGSIM_StepPathYieldSafety_NoGRPOCraft_yieldExtreme/weights/latest.ckpt

echo " $pp tools/smalltool/visualize/generate_video_sparsedrive_v2-HUGSIMori.py \
        --scene $scene_num --config $config \
        --ckpt $ckpt \
        --hugsim-scene $hugsim_scene"

CUDA_VISIBLE_DEVICES=3 $pp tools/smalltool/visualize/generate_video_sparsedrive_v2-HUGSIMori.py \
        --scene $scene_num --config $config \
        --ckpt $ckpt \
        --hugsim-scene $hugsim_scene \
        # --out $output_dir/$scene_num.mp4 \
        # --traj-csv $output_dir/$scene_num.csv \
        # --traj-plot $output_dir/$scene_num.svg \
        # --reward-detail-format ipynb --save-keyframes --debug
