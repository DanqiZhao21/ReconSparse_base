# framework/batch

这个目录负责提供 Learner 侧“构建训练 batch”的稳定入口。它本身很轻，但在训练主流程中承担了兼容层的作用：上层 datamodule 不需要知道底层 batch 构建逻辑具体放在哪个文件里。

## 文件说明

### __init__.py

包导出层。

- 对外暴露 LoadedShardBatch 和 build_training_batch。
- 让上层模块只依赖 framework.batch，而不用直接耦合 algorithms/trajectory_batch.py。

### actor_learner.py

薄封装文件。

- 实际工作几乎都委托给 algorithms/trajectory_batch.py。
- 作用是给 actor-learner 训练链路提供稳定的导入路径。
- 这样即使底层 batch 实现继续演进，Learner datamodule 的调用方式也可以尽量保持不变。

## 训练时如何经过这里

Learner 在 lightning/actor_learner_datamodule.py 里选好 shard 之后，会调用这里的 build_training_batch。然后底层再转到 algorithms/trajectory_batch.py，计算 advantage、return 并整理成最终 minibatch 可读的数据结构。