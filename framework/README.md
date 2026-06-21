# 🚗 ReconDreamer-RL Framework

`framework/` 是 ReconDreamer-RL 的强化学习训练核心。它把自动驾驶策略模型、闭环仿真环境、actor-learner 采样训练、trajectory batch 构建、PPO / ReinforcePP / SAC-style objective，以及 Lightning learner 生命周期组织在同一条训练链路中。

当前主路径是文件缓冲区形式的 actor-learner：

```text
actor -> rollout -> shard buffer -> learner -> checkpoint/version publish -> actor reload
```

如果你是第一次阅读这个仓库，可以先看根目录的 [`README.md`](../README.md)，了解 ReconDreamer-RL 与 HUGSIM-ORI、assets、outputs 的整体关系；本文件则聚焦 `framework/` 内部如何支撑训练。

## 🧭 Overview & Architecture

ReconDreamer-RL 的训练框架围绕四个核心对象展开：

- **Agent**：把 DiffusionDriveV2、SparseDrive、SparseDriveV2 等策略模型封装成统一 RL 接口。
- **Environment**：把 Recon / HUGSIM 闭环仿真包装成 actor 可调用的 `reset` / `step` 接口。
- **Shard Buffer**：actor 将采样轨迹写成 shard，learner 从文件缓冲区读取并消费。
- **Learner**：通过 PyTorch Lightning 执行策略更新，并发布新 checkpoint 与 version。

整体数据流如下：

```text
../script/train_actor_learner_v2.py
  -> runner/orchestrator.py
      -> runner/actor_runtime.py
          -> env_wrapper/ + agent/
          -> rollout/collector.py
          -> io/buffer.py
      -> runner/learner_runtime.py
          -> lightning/actor_learner_datamodule.py
          -> batch/actor_learner.py
          -> lightning/actor_learner_module.py
          -> algorithms/trajectory_policy_core.py
          -> io/buffer.py
```

这条链路的目标是让策略在闭环仿真中持续采样、持续更新，并通过 `weights/latest.ckpt` 与 `weights/version.txt` 将 learner 的最新权重广播给 actor。

## 🧩 Environment & Assets

训练环境通常依赖 HUGSIM-ORI、3DGS 场景资产、nuScenes 数据和若干本地索引文件。公开仓库中需要特别注意：**这些大体积数据不应该提交到 Git**，而应通过环境变量或软链接接入。

推荐仓库布局：

```text
/root/clone/ReconDreamer-RL
/root/clone/HUGSIM-ORI
```

默认情况下，ReconDreamer-RL 会从下面的位置寻找 HUGSIM-ORI：

```bash
export HUGSIM_ROOT=/root/clone/HUGSIM-ORI
```

如果没有显式设置 `HUGSIM_ROOT`，部分配置会使用 `/root/clone/HUGSIM-ORI` 作为默认路径。公开复现时，请根据自己的机器路径调整 YAML 中的 `env.hugsim` 字段。

常见资产入口包括：

- `assets/`：ReconDreamer-RL 侧的数据入口，通常是指向共享盘或数据盘的软链接。
- `outputs/`：训练、评估和可视化输出入口，也通常是软链接。
- `env.hugsim.scenario_dir`：HUGSIM 场景 YAML 目录。
- `env.hugsim.model_base`：HUGSIM / 3DGS 场景资产根目录。
- `env.hugsim.nuscenes_root`：nuScenes 数据路径。
- `env.hugsim.frame2token_dir`：frame 到 nuScenes token 的索引路径。
- `env.hugsim.recon_data_root`：ReconDreamer 侧重建数据路径。

环境包装层的详细说明见 [`env_wrapper/README.md`](env_wrapper/README.md)。如果遇到 HUGSIM 路径、FIFO 启动或 3DGS 渲染相关问题，优先检查 `env.backend`、`env.hugsim.repo`、`env.hugsim.launch_mode` 和 `env.hugsim.pixi_cmd`。

## 🚀 Training Usage

### 仅训练

主训练入口是 [`../script/train_actor_learner_v2.py`](../script/train_actor_learner_v2.py)。通常使用 `orchestrator` 角色启动，它会负责拉起 learner 和多个 actor：

```bash
cd /root/clone/ReconDreamer-RL

CONFIG=script/configs/sparsedrive_v2/202606151204_HUGSM_reinforcepp_closed_loop_steppath_hd_EncourageProgress_CollisionPenalty_advnorm_NoGRPOCraft_substeps1.yaml

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config "$CONFIG"
```

`--role` 支持三种运行模式：

- `orchestrator`：主入口，负责启动并管理 actor / learner 子进程。
- `actor`：单独启动 actor，用于采样轨迹并写入 shard。
- `learner`：单独启动 learner，用于读取 shard、训练并发布新权重。

训练输出默认位于：

```text
outputs/actor_learner/<timestamp>_<run_name>/
```

关键产物包括：

```text
buffer/shards/       # actor 写入的待训练 shard
buffer/consumed/     # learner 消费后的 shard
weights/latest.ckpt  # learner 发布的最新权重
weights/version.txt  # 权重版本号
*.yaml               # 本次运行实际生效的配置
```

### 训练 + 自动评估

如果希望训练结束后自动评估，可以使用 [`../script/train_eval_pipeline.py`](../script/train_eval_pipeline.py)：

```bash
cd /root/clone/ReconDreamer-RL

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_eval_pipeline.py \
  --reinforcepp-config script/configs/sparsedrive_v2/202606151204_HUGSM_reinforcepp_closed_loop_steppath_hd_EncourageProgress_CollisionPenalty_advnorm_NoGRPOCraft_substeps1.yaml
```

也可以使用 shell wrapper：

```bash
bash script/run_train_eval_pipeline.sh
```

自动 train + eval 的运行结果通常写到：

```text
outputs/TrainEvaluationAuto/<run_id>/
```

评估侧结果会写入 HUGSIM-ORI 的 `outputs/evaluate-auto/`。训练脚本和 pipeline 的更高层说明见根目录 [`README.md`](../README.md)。

## 🗂️ Module Guide

`framework/` 的每个一级目录都有相对明确的职责。主 README 只保留导航式说明，细节请进入对应子页面。

### [`runner/`](runner/README.md)

训练运行时的总调度层。负责配置规范化、actor / learner / orchestrator 三类角色、进程启动、GPU 分配、运行环境变量和对象装配。

关键文件：`orchestrator.py`、`actor_runtime.py`、`learner_runtime.py`、`config_normalization.py`、`agent_factory.py`、`env_factory.py`、`learner_factory.py`。

### [`agent/`](agent/README.md)

策略模型适配层。把 DiffusionDriveV2、SparseDrive、SparseDriveV2 等模型封装成统一 Agent 接口，负责动作采样、replay 保存、log-prob 重算、checkpoint 保存和加载。

关键文件：`base.py`、`policy_diffusiondrivev2.py`、`policy_sparsedrive.py`、`policy_sparsedrive_v2.py`。

### [`env_wrapper/`](env_wrapper/README.md)

闭环仿真环境包装层。把底层 Recon / HUGSIM 环境转成 RL 训练可用的 reset / step 接口，并处理场景采样、终止条件、碰撞信息和 reward 接入。

关键文件：`rl_wrapper.py`、`subproc_vec_env.py`、`hugsim_adapter.py`、`hugsim_fifo_runner.py`、`tool.py`。

### [`rollout/`](rollout/README.md)

Actor 侧采样层。调用 agent 和 environment 完成闭环 rollout，并把 `obs`、`reward`、`done`、`old_logp`、`replay` 等字段打包成 shard。

关键文件：`collector.py`。

### [`io/`](io/README.md)

Actor 和 learner 之间的文件通信层。定义 buffer 路径、shard 生命周期、STOP 标记、权重版本文件、原子保存和 shard 选择策略。

关键文件：`buffer.py`、`shard_policy.py`。

### [`batch/`](batch/README.md)

Learner 侧 batch 构建层。把 shard 转换成训练 batch，负责 return、advantage、GAE 和 advantage normalization 等准备工作。

关键文件：`actor_learner.py`。

### [`algorithms/`](algorithms/README.md)

算法目标函数与规格描述层。提供 PPO、ReinforcePP、SAC-style 的配置对象和 trajectory policy objective，不直接负责启动 Trainer。

关键文件：`ppo.py`、`reinforcepp.py`、`sac.py`、`trajectory_policy_core.py`。

### [`lightning/`](lightning/README.md)

Learner 训练生命周期层。通过 PyTorch Lightning 管理 datamodule、training step、optimizer、checkpoint 发布、version 推进和 WandB 日志。

关键文件：`actor_learner_datamodule.py`、`actor_learner_module.py`、`trajectory_module.py`、`config.py`。

### [`rewards/`](rewards/README.md)

Step reward 计算层。当前主流程使用 path-based tracking reward，并支持 progress、collision、comfort、terminal penalty 等分项。

关键文件：`tracking.py`。

### [`rewardmodel/`](rewardmodel/README.md)

Reward model 相关类型和配置层，用于承载更结构化的 reward / scorer 配置。

关键文件：`config.py`、`constants.py`、`types.py`。

### [`utils/`](utils/README.md)

共享工具层。包含 repo path、observation 处理、gsplat 后端选择与预热、HUGSIM / tracker 执行辅助、NuScenes token 工具等。

关键文件：`repo_paths.py`、`obs.py`、`gsplat_backend.py`、`gsplat_warmup.py`、`hugsim_execution.py`。

### `debug/`

调试辅助层。用于 log-prob、batch replay、策略输出一致性等问题定位，不属于主训练闭环的必要依赖。

## 🛠️ Runtime Files, Extension & Debugging

### Runtime files

这里的 runtime files 指 actor 和 learner 在训练过程中共享的文件约定。它不是 HTTP/TCP 这类网络协议，而是文件缓冲 actor-learner 的运行时协作规则。

核心文件和目录包括：

```text
buffer/shards/       # actor 产出的待训练 shard
buffer/consumed/     # learner 消费后的 shard
weights/latest.ckpt  # 最新策略权重
weights/version.txt  # 权重版本号，actor 根据它判断是否 reload
STOP                 # 停止信号
TRAINING_LOCK        # learner 更新 / 权重发布阶段的互斥标记
```

常见 shard 字段包括：

- `obs` / `next_obs`
- `reward`
- `done` / `terminated` / `truncated`
- `old_logp`
- `replay`
- `meta`

如果修改 shard schema、checkpoint 命名或 version 行为，需要同时检查 actor、learner、IO、batch 和 Lightning 生命周期。相关实现主要在 [`io/`](io/README.md)、[`rollout/`](rollout/README.md)、[`batch/`](batch/README.md) 和 [`lightning/`](lightning/README.md)。

### Extension points

常见扩展位置：

- **接入新策略模型**：实现 `agent/base.py` 中的 Agent 协议，并在 `runner/agent_factory.py` 中注册。
- **添加新 reward**：优先在 `rewards/` 或 reward 配置中扩展，再由 `env_wrapper/rl_wrapper.py` 接入。
- **添加新算法目标**：在 `algorithms/` 中定义 objective / spec，并确认 `lightning/trajectory_module.py` 的 training step 能调用。
- **接入新环境后端**：在 `env_wrapper/` 中封装 reset / step 语义，并通过 `runner/env_factory.py` 构建。
- **修改配置字段**：同步检查训练入口、`runner/config_normalization.py`、相关 factory 和 YAML 示例。

### Troubleshooting

常见问题可以优先按下面方向排查：

- **HUGSIM 找不到**：检查 `HUGSIM_ROOT`、`env.hugsim.repo` 和 HUGSIM-ORI 是否与 ReconDreamer-RL 并列放置。
- **场景或 3DGS asset 找不到**：检查 `env.hugsim.scenario_dir`、`model_base`、`nuscenes_root`、`frame2token_dir` 和本地软链接。
- **gsplat / CUDA extension 编译卡住**：可以清理本仓库下的 torch extension cache 后重试，例如 `.cache/torch_extensions/`。
- **actor 一直等待权重**：检查 `weights/latest.ckpt`、`weights/version.txt` 是否由 learner 正常写出。
- **learner 等不到 shard**：检查 `buffer/shards/`、actor 日志、scene 配置和环境启动是否成功。
- **log-prob 或 replay 不一致**：优先查看 `agent/` 的 replay 保存与 `algorithms/trajectory_policy_core.py` 的 log-prob 重算路径。

## 📚 Documentation Index

建议按任务选择入口：

- 想快速跑训练：看本文件的 **Training Usage** 和根目录 [`README.md`](../README.md)。
- 想理解进程如何启动：看 [`runner/README.md`](runner/README.md)。
- 想接入或调试 policy：看 [`agent/README.md`](agent/README.md)。
- 想理解 HUGSIM / 环境交互：看 [`env_wrapper/README.md`](env_wrapper/README.md)。
- 想理解 shard、version、STOP：看 [`io/README.md`](io/README.md)。
- 想理解 batch 和 advantage：看 [`batch/README.md`](batch/README.md)。
- 想理解 PPO / ReinforcePP objective：看 [`algorithms/README.md`](algorithms/README.md)。
- 想理解 learner 生命周期：看 [`lightning/README.md`](lightning/README.md)。
- 想调 reward：看 [`rewards/README.md`](rewards/README.md)。

主训练入口始终是 [`../script/train_actor_learner_v2.py`](../script/train_actor_learner_v2.py)。在修改主训练链路时，请优先保持 actor-learner 数据流、runtime files 和 checkpoint/version 行为的一致性。
