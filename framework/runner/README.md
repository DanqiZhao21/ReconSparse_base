# framework/runner

这个目录负责把整个 actor-learner 训练系统真正跑起来。它是训练流程的调度中心，连接配置、环境、Agent、算法、IO 和 Lightning。

## 目录职责

- 根据配置构建环境、策略和算法。
- 启动 orchestrator、actor、learner 三种运行角色。
- 管理权重版本、采样节奏、环境变量和多 GPU 启动细节。

## 文件说明

### __init__.py

包导出层。

- 暴露 actor_main、learner_main、orchestrator_main 等主要运行入口。
- 外部脚本通常通过这里导入 runner 的核心能力。

### actor_learner.py

整个框架最核心的运行文件之一。

- 实现 actor_main、learner_main、orchestrator_main。
- Actor 侧负责加载最新权重、构建环境与 agent、收集 shard、等待 Learner 版本更新。
- Learner 侧负责读取 shard、调用 Lightning 训练、保存 checkpoint、递增权重版本。
- Orchestrator 负责把 actor 和 learner 这些进程按配置真正拉起来。

如果只看一个文件来理解训练是怎么跑通的，优先看这里。

### factories.py

训练组件工厂。

- 根据配置选择具体的 Agent 类型，例如 DiffusionDriveV2、SparseDrive、SparseDriveV2。
- 构建 actor 侧环境、算法 bundle、value net 和优化器。
- 还负责 actor GPU 分配、scene sharding 和配置规范化。

它决定了训练配置最终会被翻译成哪些真实对象。

### launch_env.py

运行环境准备器。

- 构建训练子进程需要的环境变量，如 PYTHONPATH、CUDA_HOME、扩展编译目录等。
- 保证 egoADs 下的不同 agent 代码和 CUDA 依赖在启动时可以被正确找到。
- 它不参与 loss 计算，但直接影响训练能否成功启动。

## 训练时如何经过这里

script/train_actor_learner_v2.py 会先进入这个目录：

1. factories.py 规范化配置并构建组件。
2. actor_learner.py 按角色执行 orchestrator、actor 或 learner 主逻辑。
3. launch_env.py 为子进程准备可运行的 Python 和 CUDA 环境。

因此这个目录就是 framework 的“总调度台”。