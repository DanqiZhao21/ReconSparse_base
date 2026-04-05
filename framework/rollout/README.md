# framework/rollout

这个目录负责 actor 侧的数据采样，也就是 rollout。它处在训练链路的前半段，负责把环境交互结果整理成 learner 能消费的 shard。

## 文件说明

### __init__.py

导出 collect_single_env_shard 和 collect_vector_env_shards，供 `runner/actor_runtime.py` 直接调用。

### collector.py

actor 采样核心文件。

- 调用 agent.act 或 agent.act_batch 生成动作、旧 logp 和 replay。
- 调用环境 step 拿到 reward、done 和 next obs。
- 把 obs、old_logp、reward、done、terminated、truncated、next_obs、replay 和 meta 打包成 shard。

这个文件决定了 learner 之后能看到哪些训练字段，所以它是连接环境、策略和 batch 构建的关键桥梁。

## 训练时如何经过这里

`runner/actor_runtime.py` 中的 Actor 主循环会持续调用这里的函数收集固定 horizon 的轨迹片段。收集完成的 shard 会交给 `io/buffer.py` 写入磁盘，然后等待 Learner 消费。
