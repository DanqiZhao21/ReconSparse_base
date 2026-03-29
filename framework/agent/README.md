# framework/agent

这个目录负责把具体的自动驾驶策略模型封装成统一的 RL Agent 接口。它位于训练链路中“策略执行”和“重算 log-prob”这一层：

- Actor 采样时，runner 会先通过 factories 构建这里的 Agent。
- rollout/collector 会调用 Agent 的 act 或 act_batch 产出动作、旧 logp 和 replay 数据。
- Learner 更新时，algorithms 和 lightning 会再次调用 Agent 的 logp_from_replay 系列接口，在当前参数下重算策略概率。

## 文件说明

### __init__.py

包导出文件。把 Agent 抽象基类和三种具体策略实现统一暴露给外部模块使用，同时保留 DiffusionDriveV2Agent 这个兼容别名。

### base.py

定义整个训练框架依赖的 Agent 协议，是这个目录最基础的接口文件。

- 规定了 act、act_batch、logp_from_replay、save_checkpoint、load_checkpoint 等统一入口。
- PPO 和 ReinforcePP 都依赖这里定义的 replay 重算接口，而不是直接依赖某个具体策略的内部实现。
- optimizer 和 trainable_module 这两个属性也在这里统一暴露，方便算法层和 DDP 包装层直接使用。

### policy_diffusiondrivev2.py

DiffusionDriveV2 的 RL 适配器。

- 负责加载 DiffusionDriveV2 RL 版本模型，并把它变成框架能直接调用的 Agent。
- 在 act 阶段把模型输出的轨迹转换成环境可执行的动作，当前主要走 first-step 执行模式。
- 维护 replay 信息，使 Learner 能在 PPO 或 Reinforce 更新时重新计算当前策略下的 logp。
- 处理设备迁移、DDP 包装、checkpoint 保存与加载，是 DiffusionDriveV2 进入训练主链路的关键适配层。

### policy_sparsedrive.py

SparseDrive 的 RL 适配器。

- 把多相机观测、相机内外参和 ego pose 组装成 SparseDrive 规划头需要的输入。
- 通过规划 mode 的分类分数定义策略分布，并从中取出动作与 logp。
- PPO 更新时会依赖这里的 replay 重放逻辑，在当前参数下重算选中 mode 的概率。
- 同时负责 SparseDrive 模型的 checkpoint、DDP 和优化器管理。

### policy_sparsedrive_v2.py

SparseDriveV2 的 RL 适配器，是当前仓库较新的策略接入层。

- 负责加载 SparseDriveV2 模型和锚点配置，把候选轨迹与候选分数接入 RL 训练框架。
- 支持通过 trainable_prefixes 控制哪些参数可训练，所以它直接决定 actor-learner 训练时到底在微调哪些层。
- 在 act 阶段从候选动作中选择执行轨迹，同时保存 replay，供 Learner 端重算 logp 使用。
- 负责设备迁移、DDP 包装、checkpoint 读写，因此既参与 rollout，也直接参与 Learner 的反向传播更新。

## 训练时如何经过这里

主入口 script/train_actor_learner_v2.py 启动后，runner/factories.py 会优先进入这个目录构建 Agent。随后：

- Actor 进程在 rollout 期间不断调用这里的策略实现来采样动作。
- Learner 进程在训练时通过 replay 再次进入这里，计算新旧策略概率比值。
- 权重保存、广播和重新加载也都通过这里的具体 Agent 实现完成。