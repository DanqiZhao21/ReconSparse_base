# framework/io

这个目录负责 actor 和 learner 之间的数据传输与权重同步。它属于训练系统的基础设施层，不直接定义 RL 目标函数，但决定了 actor-learner 架构能否稳定运行。

## 文件说明

### __init__.py

包导出层。

- 统一导出 BufferPaths、list_shards、wait_for_version、atomic_torch_save 等 IO 辅助函数。
- 让 runner 和 lightning 模块可以直接从 framework.io 取用这些工具。

### buffer.py

actor-learner 文件式缓冲区的核心实现。

- 定义 BufferPaths，统一描述 shard、consumed、weights、version、STOP 等路径。
- 提供 shard 枚举、原子保存、版本号读写、已消费 shard 回收、停止信号检测等能力。
- 提供 actor heartbeat 与 stale heartbeat 标记 failed 的辅助函数，用于区分慢 actor 和已经长期无进展的 actor。
- actor_main 和 learner_main 都直接依赖这里，它决定了采样数据如何落盘、权重如何广播以及旧 shard 如何清理。

### shard_policy.py

Learner 侧 shard 选择策略。

- 负责 stale shard 丢弃、replay 兼容性过滤，以及 sync/async 模式下的 shard 选择规则。
- 让 DataModule 不再直接拥有 buffer 协议策略。

## 训练时如何经过这里

Actor 每次采样完一个 shard 会通过这里定义的路径规则写入 buffer。Learner 则不断轮询这些 shard，先经过 `shard_policy.py` 做筛选，再把训练完成的 shard 移到 consumed，并把最新权重和 version 写回 weights 目录。整个 actor-learner 协作是围绕这个目录提供的文件协议展开的。
