# framework/lightning

这个目录负责把 RL 更新流程接入 PyTorch Lightning。它位于训练链路的 Learner 中央控制层，作用是把“算法逻辑”和“训练循环调度”分开。

## 目录职责

- 把 trajectory batch 封装成 DataModule。
- 把 PPO 或 ReinforcePP 的损失计算封装成 LightningModule。
- 在 `configure_optimizers()` 中主动构建当前 learner 使用的优化器。
- 在 actor-learner 模式下，把 shard 选择、训练结束后的 checkpoint 回写、WandB 日志等流程接进 Lightning 生命周期。
- 这里的 actor-learner 专用模块已经是主路径，不是辅助兼容层。

## 文件说明

### __init__.py

导出通用的 `TrajectoryUpdateDataModule` 和 `TrajectoryLightningModule`。

- 当前主路径里，这些类型主要由 `runner/learner_runtime.py` 和 actor-learner 专用模块间接使用。
- `algorithms/` 已经不再把自己实现成 Lightning runner。

### config.py

Lightning 侧 learner handoff 配置。

- 提供 `LearnerOptimizerConfig` 和 `ActorLearnerLightningConfig`。
- 负责把 algorithm spec、trainer 参数和 actor-learner datamodule 参数收口成一个显式 handoff 对象。
- `runner/learner_runtime.py` 会通过这里生成 Trainer kwargs，而不是自己零散读取各个 algo 字段。
- 这里也明确区分了 `max_updates` 和 `inner_epochs`：前者是 actor-learner update 次数，后者是同一批 shard 在单次 update 内重复训练的轮数。

### trajectory_datamodule.py

通用训练数据模块。

- 把已经准备好的 batch 切成 dataset 和 dataloader。
- 负责 minibatch 粒度的数据读取、collate 和分布式 sampler 配置。
- 是算法层和 Lightning Trainer 之间的标准数据桥梁。

### trajectory_module.py

通用训练模块。

- 在 training_step 中根据 algo_kind 分流到 PPO 或 Reinforce 目标函数。
- 调用 algorithms/trajectory_policy_core.py 计算 loss 和 metrics。
- 在 `configure_optimizers()` 中主动创建当前训练用 optimizer。
- 这是 Learner 真正执行反向传播时最直接的训练步入口。

### actor_learner_datamodule.py

专门给 actor-learner 训练使用的数据模块。

- 不只是读 batch，还负责在训练开始前等待 shard 到齐。
- 会通过 `framework.io.shard_policy` 调用版本过滤、兼容性过滤和 sync/async shard 选择策略，再调用 batch/build_training_batch 生成训练数据。
- 它连接了 IO 层和 Lightning 层，是 Learner 每次 update 前的第一站。
- 当 `inner_epochs > 1` 时，它会在同一 update 内复用同一批 shard/batch，只重新走 dataloader 迭代与 minibatch 打乱。
- 文件名保留 `actor_learner_*`，是为了明确它服务的是文件缓冲 actor-learner 协议，而不是所有 Lightning 训练场景的通用 datamodule。

### actor_learner_module.py

专门给 actor-learner 训练使用的训练模块。

- 继承 TrajectoryLightningModule，在 epoch 开始和结束时额外处理训练锁、消费 shard、保存权重、递增 version。
- 还负责把一次 update 的统计信息发到 stage 日志和 WandB。
- 可以看作“Learner 更新生命周期控制器”。
- 文件名保留 `actor_learner_*`，因为这一层仍然绑定 actor-learner buffer 协议和版本发布语义。

## 训练时如何经过这里

Learner 从 `runner/learner_runtime.py` 进入后，会先构建 `actor_learner_datamodule.py` 和 `actor_learner_module.py`。真正进入每个 minibatch 的 loss 计算时，又会下沉到 `trajectory_module.py` 和 `trajectory_datamodule.py`。这个目录因此同时管理“整轮 update 的生命周期”和“单个 minibatch 的训练步”。
