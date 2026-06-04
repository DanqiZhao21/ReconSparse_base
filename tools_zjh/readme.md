
如果在训练中测试的话，要在config里面更改 output_root: checkpoints/hugsim_rl_eval 不要和训练同目录


以下是调用 仿真器 命令，acttor中调用
```bash
/root/clone/HUGSIM-ORI/.pixi/envs/default/bin/python \
    /root/clone/ReconDreamer-RL/framework/env_wrapper/hugsim_fifo_runner.py \
    --scenario_path /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes_denso_benchmark_v1/scene-0013-medium-person-cross.yaml \
    --base_path /root/clone/HUGSIM-ORI/configs/sim/nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml \
    --camera_path /root/clone/HUGSIM-ORI/configs/sim/nuscenes_camera.yaml \
    --kinematic_path /root/clone/HUGSIM-ORI/configs/sim/kinematic.yaml \
    --output_dir /OpenDataset/zjh/recondiff_outputs_checkpoints/hugsim_rl/scene-0013-medium-person-cross \
    --ad sparsedrive_v2 \
    --fifo_timeout_s 1800.0 \
    --fifo_poll_interval_s 0.2 \
    --actor_id 16 \
    --worker_id 16
```