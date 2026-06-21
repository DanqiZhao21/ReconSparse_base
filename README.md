# 🚗 ReconDreamer-RL

ReconDreamer-RL 是一个面向自动驾驶闭环仿真的强化学习训练仓库。它把策略模型、重建仿真环境、actor-learner 采样训练、PPO / ReinforcePP / SAC-style objective、reward shaping、checkpoint 发布和训练后评估组织在同一个训练工程中。

当前主训练路径采用文件缓冲区形式的 actor-learner：

```text
actor -> rollout -> shard buffer -> learner -> checkpoint/version publish -> actor reload
```

主入口：

- [`script/train_actor_learner_v2.py`](script/train_actor_learner_v2.py)：启动 actor-learner 训练。
- [`script/train_eval_pipeline.py`](script/train_eval_pipeline.py)：训练完成后自动评估最新 checkpoint。
- [`framework/`](framework/README.md)：RL 框架核心实现。

## ✨ Highlights

- **Closed-loop RL for autonomous driving**：在 Recon / HUGSIM 闭环环境中采样并更新策略。
- **Actor-learner training**：actor 负责 rollout 收集形成 buffer，learner 负责从 shard buffer 读取数据并更新权重。
- **Multiple Ego-car Policy backends**：支持 SparseDriveV2 以及大部分 E2E 自动驾驶系统。
- **Multiple RL Algorithms**：支持 PPO、ReinforcePP 、SAC 以及自行扩展新的强化学习算法。
- **Multiple 3DGS assets **：支持 HUGSIM-ORI、Recondreamer 提供的 3DGS 场景资产进行闭环训练与评估。
- **Lightning learner**：使用 PyTorch Lightning 管理 learner 训练步。

## 🧭 Architecture

ReconDreamer-RL 的训练框架由四个核心对象组成：

- **Agent**：把具体自动驾驶策略模型封装成统一 RL 接口，负责动作采样、shard replay 保存、log-prob 重算。
- **Environment**：把 3DGS 仿真环境包装成 actor 可调用的 `reset` / `step` 接口，并接入自定义 reward 环境奖励与惩罚。
- **Shard Buffer**：actor 将采样轨迹写入文件缓冲区，learner 从中选择 shard 并构建 training batch 展开训练。
- **Learner**：执行策略更新，保存 `weights/latest.ckpt`,同步更新 `weights/version.txt`。

主链路如下：

```text
script/train_actor_learner_v2.py
  -> framework/runner/orchestrator.py
      -> framework/runner/actor_runtime.py
          -> framework/env_wrapper/ + framework/agent/
          -> framework/rollout/collector.py
          -> framework/io/buffer.py
      -> framework/runner/learner_runtime.py
          -> framework/lightning/actor_learner_datamodule.py
          -> framework/batch/actor_learner.py
          -> framework/lightning/actor_learner_module.py
          -> framework/algorithms/trajectory_policy_core.py
          -> framework/io/buffer.py
```

更细的框架说明见 [`framework/README.md`](framework/README.md)。

## 🧩 Environment & Assets

`HUGSIM-ORI` 是独立仓库，不是本仓库的 Git submodule，也不应该作为 `ReconDreamer-RL` 的子目录提交。

推荐保持两个仓库并列：

```text
/root/clone/ReconDreamer-RL
/root/clone/HUGSIM-ORI
```

ReconDreamer-RL 默认从 `/root/clone/HUGSIM-ORI` 读取 HUGSIM 代码和配置。需要改路径时设置：

```bash
export HUGSIM_ROOT=/path/to/HUGSIM-ORI
```

HUGSIM 自己的运行环境仍由 HUGSIM-ORI 仓库管理。当前 FIFO 后端会在 HUGSIM-ORI 目录下执行 `pixi run python ...`，本仓库只负责调度、采样、训练和评估。

评估视频等output请自行创建软连接。常见本地入口包括：

- `assets/`：ReconDreamer-RL 侧数据入口，指向共享盘或数据盘的软链接。
- `outputs/`：训练、评估和可视化输出入口，通常也是软链接。
- `env.hugsim.scenario_dir`：HUGSIM 场景 YAML 目录。
- `env.hugsim.model_base`：HUGSIM / 3DGS 场景资产根目录。
- `env.hugsim.nuscenes_root`：nuScenes 数据路径。
- `env.hugsim.frame2token_dir`：frame 到 nuScenes token 的索引路径。
- `env.hugsim.recon_data_root`：ReconDreamer 侧重建数据路径。

常见 HUGSIM 本地数据软链接示例：

```bash
ln -s /OpenDataset/HUGSIM_data/scenarios /root/clone/HUGSIM-ORI/configs/scenarios
ln -s /OpenDataset/zhaodanqi/HUGSIM_data/outputs /root/clone/HUGSIM-ORI/outputs
```

## ⚡ Quick Start

### 1. 检查仓库和资产路径

```bash
cd /root/clone/ReconDreamer-RL

echo "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}"
ls "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}/configs"

ls -l assets
ls -l outputs
```

如果使用 HUGSIM-ORI 后端，还需要确认场景配置可访问：

```bash
ls "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}/configs/scenarios/nuscenes" | head
```

### 2. 启动 actor-learner 训练

使用 `orchestrator` 角色启动训练，它会负责拉起 learner 和多个 actor：

```bash
cd /root/clone/ReconDreamer-RL

CONFIG=script/configs/sparsedrive_v2/202606151204_HUGSM_reinforcepp_closed_loop_steppath_hd_EncourageProgress_CollisionPenalty_advnorm_NoGRPOCraft_substeps1.yaml

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config "$CONFIG"
```

`--role` 支持：

- `orchestrator`：主入口，启动并管理 actor / learner 子进程。
- `actor`：单独启动 actor，负责采样并写 shard。
- `learner`：单独启动 learner，负责训练并发布新权重。

训练输出默认位于：

```text
outputs/actor_learner/<timestamp>_<run_name>/
```

关键产物：

```text
buffer/shards/       # actor 写入的待训练 shard
buffer/consumed/     # learner 消费后的 shard
weights/latest.ckpt  # learner 发布的最新权重
weights/version.txt  # 权重版本号
*.yaml               # 本次运行实际生效的配置
```

### 3. 训练后自动评估

直接调用 pipeline，默认训练完后自动进行2 repeat 88 nuscenes yaml scenario 的评估：

```bash
cd /root/clone/ReconDreamer-RL

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_eval_pipeline.py \
  --reinforcepp-config script/configs/sparsedrive_v2/202606151204_HUGSM_reinforcepp_closed_loop_steppath_hd_EncourageProgress_CollisionPenalty_advnorm_NoGRPOCraft_substeps1.yaml
```

也可以使用 wrapper：

```bash
bash script/run_train_eval_pipeline.sh
```

或使用 HUGSIM-ORI 相关 wrapper：

```bash
bash script/run_train_eval_pipeline_hugsim_ori.sh
```

默认只跑 ReinforcePP；如果要额外跑 PPO，给 `train_eval_pipeline.py` 加 `--ppo` 并传入 `--ppo-config`。

自动 train + eval 的运行结果会放在：

```text
outputs/TrainEvaluationAuto/<run_id>/
```

评估侧结果会写到 HUGSIM-ORI 的：

```text
outputs/evaluate-auto/
```

## ⚙️ Configuration & Runtime Files

训练配置位于 [`script/configs/`](script/configs/)。常见顶层字段包括：

- `env`：环境后端、最大步数、渲染尺寸、scene sampling、reward 和 HUGSIM 参数。
- `env.hugsim`：HUGSIM-ORI 路径、scenario、base config、3DGS asset、nuScenes 路径和 FIFO 启动参数。
- `env.reward`：step reward、terminal penalty、collision、comfort、CRAFT / PDM scorer 等配置。
- `train`：算法类型、WandB、学习率、batch/update、actor-learner 运行参数等。

HUGSIM-ORI 后端通常使用：

```yaml
env:
  backend: hugsim_ori
  hugsim:
    repo: /root/clone/HUGSIM-ORI
    scenario_dir: /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes
    model_base: /OpenDataset/HUGSIM_data/scenes/nuscenes
    pixi_cmd: pixi
```

运行时 actor 和 learner 通过文件约定协作：

```text
buffer/shards/       # actor 产出的待训练 shard
buffer/consumed/     # learner 消费后的 shard
weights/latest.ckpt  # 最新策略权重
weights/version.txt  # 权重版本号，actor 根据它判断是否 reload
STOP                 # 停止信号
TRAINING_LOCK        # learner 更新 / 权重发布阶段的互斥标记
```

常见 shard 字段包括 `obs`、`next_obs`、`reward`、`done`、`terminated`、`truncated`、`old_logp`、`replay` 和 `meta`。如果修改 shard schema、checkpoint 命名或 version 行为，需要同时检查 actor、learner、IO、batch 和 Lightning 生命周期。

## 🗂️ Repository Map

- [`framework/`](framework/README.md)：当前主训练框架，包含 runner、rollout、io、batch、algorithms、lightning、agent、env wrapper 等模块。
- [`script/`](script/)：训练、评估、debug 和 pipeline 入口，以及 YAML 配置。
- [`tools/`](tools/)：可视化、视频生成、诊断和辅助脚本。
- [`reconsimulator/`](reconsimulator/)：ReconSimulator 相关环境实现。
- [`policy/`](policy/)：策略侧代码和模型相关内容。
- [`egoADs/`](egoADs/)：DiffusionDriveV2、SparseDrive、SparseDriveV2 等外部策略代码适配位置。
- `assets/`：本地资产入口，通常是指向共享盘的软链接。
- `outputs/`：训练、评估和可视化输出入口，通常是指向共享盘的软链接。

## 🧱 Framework Modules

`framework/` 是训练系统的核心。每个子目录都有更细的 README，主职责如下：

- [`framework/runner/`](framework/runner/README.md)：配置规范化、actor / learner / orchestrator 进程、GPU 分配和运行时对象装配。
- [`framework/agent/`](framework/agent/README.md)：策略模型适配，负责动作采样、replay、log-prob 重算和 checkpoint 读写。
- [`framework/env_wrapper/`](framework/env_wrapper/README.md)：Recon / HUGSIM 环境包装、场景采样、终止条件、碰撞信息和 reward 接入。
- [`framework/rollout/`](framework/rollout/README.md)：actor 侧 rollout 采样，并把轨迹打包成 shard。
- [`framework/io/`](framework/io/README.md)：buffer、shard、STOP、TRAINING_LOCK、权重版本和原子保存。
- [`framework/batch/`](framework/batch/README.md)：shard 到 training batch 的转换，包含 return、GAE 和 advantage normalization。
- [`framework/algorithms/`](framework/algorithms/README.md)：PPO、ReinforcePP、SAC-style objective 与算法规格对象。
- [`framework/lightning/`](framework/lightning/README.md)：Lightning datamodule/module、training step、optimizer、checkpoint/version 发布和 WandB 日志。
- [`framework/rewards/`](framework/rewards/README.md)：path-based tracking reward、collision、comfort 和 terminal penalty。
- [`framework/rewardmodel/`](framework/rewardmodel/README.md)：结构化 reward / scorer 类型和配置。
- [`framework/utils/`](framework/utils/README.md)：repo path、observation、gsplat、HUGSIM 执行和 NuScenes token 工具。

## 🛠️ Extending & Debugging

常见扩展入口：

- **接入新策略模型**：实现 [`framework/agent/base.py`](framework/agent/base.py) 中的 Agent 协议，并在 [`framework/runner/agent_factory.py`](framework/runner/agent_factory.py) 中注册。
- **添加新 reward**：优先在 [`framework/rewards/`](framework/rewards/README.md) 或 reward 配置中扩展，再由 [`framework/env_wrapper/rl_wrapper.py`](framework/env_wrapper/rl_wrapper.py) 接入。
- **添加新算法目标**：在 [`framework/algorithms/`](framework/algorithms/README.md) 中定义 objective / spec，并确认 [`framework/lightning/trajectory_module.py`](framework/lightning/trajectory_module.py) 能调用。
- **接入新环境后端**：在 [`framework/env_wrapper/`](framework/env_wrapper/README.md) 中封装 reset / step 语义，并通过 [`framework/runner/env_factory.py`](framework/runner/env_factory.py) 构建。
- **修改配置字段**：同步检查训练入口、[`framework/runner/config_normalization.py`](framework/runner/config_normalization.py)、相关 factory 和 YAML 示例。

常见问题排查：

- **HUGSIM 找不到**：检查 `HUGSIM_ROOT`、`env.hugsim.repo` 和 HUGSIM-ORI 是否与 ReconDreamer-RL 并列放置。
- **场景或 3DGS asset 找不到**：检查 `env.hugsim.scenario_dir`、`model_base`、`nuscenes_root`、`frame2token_dir` 和本地软链接。
- **gsplat / CUDA extension 编译卡住**：可以清理本仓库下的 torch extension cache 后重试，例如 `.cache/torch_extensions/`。
- **actor 一直等待权重**：检查 `weights/latest.ckpt`、`weights/version.txt` 是否由 learner 正常写出。
- **learner 等不到 shard**：检查 `buffer/shards/`、actor 日志、scene 配置和环境启动是否成功。
- **log-prob 或 replay 不一致**：优先查看 agent 的 replay 保存与 [`framework/algorithms/trajectory_policy_core.py`](framework/algorithms/trajectory_policy_core.py) 的 log-prob 重算路径。

## 📚 Documentation Index

- [`framework/README.md`](framework/README.md)：actor-learner 框架总览。
- [`framework/runner/README.md`](framework/runner/README.md)：训练运行时和进程编排。
- [`framework/agent/README.md`](framework/agent/README.md)：策略适配层。
- [`framework/env_wrapper/README.md`](framework/env_wrapper/README.md)：Recon / HUGSIM 环境包装。
- [`framework/io/README.md`](framework/io/README.md)：buffer、shard、权重版本和 STOP 文件协议。
- [`framework/batch/README.md`](framework/batch/README.md)：shard 到 training batch 的转换。
- [`framework/algorithms/README.md`](framework/algorithms/README.md)：PPO / ReinforcePP / trajectory objective。
- [`framework/lightning/README.md`](framework/lightning/README.md)：Lightning learner 生命周期。
- [`framework/rewards/README.md`](framework/rewards/README.md)：reward 计算与调试。

主训练入口始终是 [`script/train_actor_learner_v2.py`](script/train_actor_learner_v2.py)。修改 actor-learner 主链路时，请保持 shard、runtime files、checkpoint/version 发布和 actor reload 行为的一致性。
