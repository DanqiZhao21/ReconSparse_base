# framework/batch

这个目录负责提供 Learner 侧“构建训练 batch”的稳定入口。现在这里已经是 canonical 实现层，而不只是薄兼容层。

## 文件说明

### __init__.py

包导出层。

- 对外暴露 LoadedShardBatch 和 build_training_batch。
- 让上层模块只依赖 `framework.batch`，而不用直接耦合具体实现文件。

### actor_learner.py

当前的 canonical 实现文件。

- 负责 shard 到训练 batch 的主要实现。
- 包含 GAE、return、advantage normalization 和 batch 组装逻辑。
- 当前主路径直接依赖这里，不再通过 algorithms 层的兼容别名转发。

## 训练时如何经过这里

Learner 在 `lightning/actor_learner_datamodule.py` 里选好 shard 之后，会调用这里的 `build_training_batch`。这里会直接完成 advantage、return 和 batch 结构整理，然后交给 Lightning 的 dataloader / training_step 使用。
