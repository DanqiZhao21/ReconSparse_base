# framework

这个目录是 ReconDreamer-RL 当前使用的强化学习训练主框架，核心目标是把自动驾驶策略模型接入 actor-learner 训练流程，并支持 PPO、ReinforcePP 等算法在闭环仿真环境中持续采样和更新。

## 整体职责

- 把具体策略模型封装成统一 Agent 接口。
- 把仿真环境包装成 RL 可采样的 reset 和 step 接口。
- 在 Actor 侧采集轨迹，生成 shard。
- 在 Learner 侧把 shard 组装成 batch，并执行策略更新。
- 通过文件缓冲区完成 actor 和 learner 之间的异步协作。
- 用 Lightning 管理训练循环、日志、checkpoint 和权重版本切换。

## 主训练链路

当前主入口是 script/train_actor_learner_v2.py。整体运行流程可以概括为：

1. runner/factories.py 根据 YAML 配置构建环境、Agent、算法和值函数组件。
2. runner/actor_learner.py 启动 orchestrator、actor、learner 三类角色。
3. Actor 通过 env_wrapper 和 rollout 不断与环境交互，调用 agent 产出动作、logp 和 replay。
4. rollout/collector.py 把观测、奖励、done、replay 等信息打包成 shard。
5. io/buffer.py 把 shard 写入共享缓冲区，并维护权重版本和消费状态。
6. Learner 通过 lightning/actor_learner_datamodule.py 读取 shard，交给 batch 和 algorithms 生成训练 batch。
7. algorithms 和 lightning 共同完成 PPO 或 ReinforcePP 更新。
8. 更新后的权重重新写回 buffer，Actor 检测到新版本后继续采样。

简化后的数据流如下：

Actor -> env_wrapper -> agent -> rollout -> io/buffer -> batch -> algorithms -> lightning -> 新权重 -> Actor

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
- 实际把 shard 变成训练 batch 的逻辑主要下沉在 algorithms/trajectory_batch.py。

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
2. runner/actor_learner.py
3. runner/factories.py
4. rollout/collector.py
5. lightning/actor_learner_datamodule.py
6. algorithms/trajectory_batch.py
7. algorithms/ppo.py 或 algorithms/reinforcepp.py
8. agent/ 对应的具体策略实现

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
- 如果修改配置字段，最好同步检查 runner/factories.py 和 runner/actor_learner.py，确认这些字段在当前代码路径里真的会被读取。