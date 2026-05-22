# framework

这个目录是 ReconDreamer-RL 当前使用的强化学习训练主框架，核心目标是把自动驾驶策略模型接入 actor-learner 训练流程，并支持 PPO、ReinforcePP 等算法在闭环仿真环境中持续采样和更新。

## 整体职责

- 把具体策略模型封装成统一 Agent 接口。
- 把仿真环境包装成 RL 可采样的 reset 和 step 接口。
- 在 Actor 侧采集轨迹，生成 shard。
- 在 Learner 侧把 shard 组装成 batch，并执行策略更新。
- 通过文件缓冲区完成 actor 和 learner 之间的异步协作。
- 用 Lightning 管理训练循环、日志、checkpoint 和权重版本切换。

## 训练入口

当前有两种主入口：

### 1. 仅训练

入口是 `script/train_actor_learner_v2.py`。它只负责拉起 actor-learner 训练，不做自动评估。

```bash
cd /root/clone/ReconDreamer-RL

PYTHONPATH=/root/clone/ReconDreamer-RL python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config script/configs/sparsedrive_v2/202605211155_reinforcepp_closed_loop_sparsedrive_v2_craft_closeNo_openGRPO.yaml
```

默认运行目录会放在：

```text
outputs/actor_learner/<timestamp>_<run_name>/
```

其中会包含：

- `buffer/shards/`
- `buffer/consumed/`
- `weights/latest.ckpt`
- `weights/version.txt`
- 本轮实际生效的 YAML 配置

### 2. 训练 + 自动评估

入口是 `script/run_train_eval_pipeline.sh`，或者直接调用 `script/train_eval_pipeline.py`。它会先训练，再把最新 checkpoint 自动送去评估。

默认只跑 `reinforcepp`：

```bash
cd /root/clone/ReconDreamer-RL

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_eval_pipeline.py \
  --reinforcepp-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202605211155_reinforcepp_closed_loop_sparsedrive_v2_craft_closeNo_openGRPO.yaml
```

如果要额外跑 PPO，再加 `--ppo`：

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_eval_pipeline.py \
  --ppo \
  --ppo-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/xxx.yaml \
  --reinforcepp-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202605211155_reinforcepp_closed_loop_sparsedrive_v2_craft_closeNo_openGRPO.yaml
```

自动 train + eval 的运行结果会放在：

```text
outputs/TrainEvaluationAuto/<run_id>/
```

其中训练阶段会在该 run 目录下生成 `actor_learner/`，评估结果则写到 HUGSIM-ORI 的 `outputs/evaluate-auto/` 下。

## 主训练链路

仅训练这条链路的核心流程可以概括为：

1. `runner/config_normalization.py` 先规范化 actor-learner 相关配置。
2. `runner/orchestrator.py`、`runner/actor_runtime.py`、`runner/learner_runtime.py` 分别负责 orchestrator、actor、learner 三类角色。
3. Actor 通过 `env_wrapper` 和 `rollout` 不断与环境交互，调用 agent 产出动作、logp 和 replay。
4. `rollout/collector.py` 把观测、奖励、done、replay 等信息打包成 shard。
5. `io/buffer.py` 把 shard 写入共享缓冲区，并维护权重版本、消费状态和停止标记；`io/shard_policy.py` 负责 learner 侧 shard 选择策略。
6. Learner 通过 `lightning/actor_learner_datamodule.py` 读取 shard，交给 `batch/actor_learner.py` 生成训练 batch。
7. `algorithms/trajectory_policy_core.py` 和 `lightning/trajectory_module.py` 共同完成 PPO 或 ReinforcePP 更新，`configure_optimizers()` 负责主动优化器构建。
8. `lightning/actor_learner_module.py` 在 update 结束后保存新权重、推进 version，并让 Actor 检测新版本后继续采样。

简化后的数据流如下：

`script/train_actor_learner_v2.py -> runner -> rollout + io -> batch -> algorithms + lightning -> 新权重 -> Actor`

## 目录分工

### agent/

策略适配层。

- 把 DiffusionDriveV2、SparseDrive、SparseDriveV2 这类具体模型封装成统一 Agent 接口。
- 负责动作采样、replay 保存、logp 重算、checkpoint 保存和加载。

### algorithms/

算法层。

- 实现 PPO、ReinforcePP 及其底层目标函数。
- 负责从 replay 重算 logp、计算 advantage 相关损失和训练指标。

### batch/

batch 构建入口层。

- 向 Learner 提供稳定的 build_training_batch 接口。
- 实际把 shard 变成训练 batch 的逻辑集中在 batch/actor_learner.py。

### env_wrapper/

环境包装层。

- 把 ReconSimulator 包装成 RL 环境接口。
- 处理 scene 采样、起始帧选择、碰撞标记、终止逻辑和 3DGS 相关工具。

### io/

actor-learner 通信层。

- 定义 buffer 目录结构、版本文件、停止标记和 shard 生命周期。
- 支撑 actor 和 learner 之间的异步协作。

### lightning/

训练调度层。

- 把算法更新接入 PyTorch Lightning。
- 管理 datamodule、training_step、训练锁、WandB 日志和更新后权重回写。

### rewards/

奖励计算层。

- 定义轨迹跟踪、碰撞、jerk、terminal penalty 等 reward 逻辑。

### rollout/

采样层。

- 负责 Actor 侧与环境交互并把结果打包成 shard。

### runner/

总调度层。

- 负责真正拉起训练进程，是整个 framework 的运行中枢。

### utils/

支撑工具层。

- 提供观测预处理、路径解析、gsplat 后端选择与预热、环境缓存构建等公共能力。

## 推荐阅读顺序

如果是第一次阅读这个框架，建议按下面顺序看：

1. script/train_actor_learner_v2.py
2. runner/orchestrator.py / runner/actor_runtime.py / runner/learner_runtime.py
3. runner/config_normalization.py / runner/agent_factory.py / runner/env_factory.py / runner/learner_factory.py
4. rollout/collector.py
5. lightning/actor_learner_datamodule.py
6. batch/actor_learner.py
7. lightning/trajectory_module.py
8. algorithms/ppo.py 或 algorithms/reinforcepp.py
9. agent/ 对应的具体策略实现

这样可以先抓住主链路，再回头看各个子模块的细节。

## 子目录 README

各一级子目录都已经补充了更细的说明，适合按模块深入阅读：

- agent/README.md
- algorithms/README.md
- batch/README.md
- env_wrapper/README.md
- io/README.md
- lightning/README.md
- rewards/README.md
- rollout/README.md
- runner/README.md
- utils/README.md

## 维护约定

- 新训练逻辑优先补到现有模块中，不要在 framework 下堆叠历史备份文件或兼容残留文件。
- 如果某个辅助脚本不属于当前 actor-learner 主链路，优先放到 framework 之外，或者在对应 README 中写清楚它的保留原因。
- 如果修改配置字段，最好同步检查 runner/config_normalization.py、对应 runtime 模块和相关 factory，确认这些字段在当前代码路径里真的会被读取。
