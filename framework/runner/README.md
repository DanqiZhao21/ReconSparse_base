# framework/runner

这个目录负责把整个 actor-learner 训练系统真正跑起来。它是训练流程的调度中心，连接配置、环境、Agent、算法、IO 和 Lightning。

## 目录职责

- 根据配置构建环境、策略和算法（由按职责拆分的 factory 模块负责）。
- 启动 orchestrator、actor、learner 三种运行角色。
- 管理权重版本、采样节奏、环境变量和多 GPU 启动细节。

不属于这个目录的职责：

- 不在这里实现 PPO / Reinforce 的目标函数。
- 不在这里定义 shard 文件协议细节。
- 不在这里直接实现 minibatch 级训练步。

## 文件说明

### __init__.py

包导出层。

- 通过 lazy export 暴露 `actor_main`、`learner_main`、`orchestrator_main` 等主要运行入口。
- 外部脚本通常通过这里导入 runner 的核心能力。

### actor_runtime.py

Actor 侧主循环。

- 负责加载最新权重、采样 shard、处理 backpressure、等待 Learner 版本推进。

### learner_runtime.py

Learner 侧主循环。

- 负责组装 LightningModule/DataModule、驱动训练、保存更新后的权重版本。
- 现在通过 `framework/lightning/config.py` 里的显式 handoff 对象，把优化器、trainer 和 datamodule 相关配置统一交给 Lightning。

### orchestrator.py

进程拉起与回收逻辑。

- 负责启动 learner 和所有 actor 子进程，并在退出时广播 STOP。

### logging.py

运行时日志与 WandB 辅助。

- 提供统一 stage 日志输出。
- 管理 learner 侧 WandB 初始化。

### dist.py

分布式初始化辅助。

- 负责 learner DDP 环境变量读取和 process group 初始化。

### config_normalization.py

配置规范化与 actor GPU 规划。

- `normalize_actor_learner_cfg` 负责 actor-learner 相关配置补全与推导。
- `resolve_actor_gpu_ids` 负责 actor 到 GPU 的分配策略。

### env_factory.py

Actor 环境构建。

- `discover_scene_ids` 负责场景发现。
- `build_actor_env` 负责根据配置和 actor 上下文创建环境实例。

### agent_factory.py

Agent 构建。

- `build_agent` 负责按配置实例化 DiffusionDriveV2 / SparseDrive / SparseDriveV2 等策略适配器。

### learner_factory.py

Learner 算法组件构建。

- `ValueNet` 价值网络定义。
- `build_algorithm_bundle` 负责组装 PPO / ReinforcePP 规格对象、value net 与元信息。
- 主优化器的主动构建已经下沉到 Lightning `configure_optimizers()`。

### launch_env.py

运行环境准备器。

- 构建训练子进程需要的环境变量，如 PYTHONPATH、CUDA_HOME、扩展编译目录等。
- 保证 egoADs 下的不同 agent 代码和 CUDA 依赖在启动时可以被正确找到。
- 它不参与 loss 计算，但直接影响训练能否成功启动。

## 训练时如何经过这里

`script/train_actor_learner_v2.py` 和 `script/train_eval_pipeline.py` 的训练阶段都会进入这个目录：

1. `config_normalization.py` 先补齐 actor-learner 相关配置。
2. `orchestrator.py` 负责拉起 learner / actor 进程。
3. `actor_runtime.py` 和 `learner_runtime.py` 分别进入采样与训练主循环。
4. `agent_factory.py`、`env_factory.py`、`learner_factory.py` 提供运行时所需对象。
5. `launch_env.py` 为子进程准备可运行的 Python 和 CUDA 环境。
6. `train_eval_pipeline.py` 会先准备一份按 run 目录落盘的训练配置，再把它交给这里的 orchestrator 执行。

因此这个目录就是 framework 的“总调度台”。
