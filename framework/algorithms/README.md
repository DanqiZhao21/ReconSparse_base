# framework/algorithms

这个目录负责实现 RL 更新逻辑，是训练链路里真正计算损失函数和做参数更新的地方。它位于“采样数据已经落盘之后、模型权重回写之前”的核心 Learner 阶段。

## 目录职责

- 从 batch 模块拿到已经整理好的 obs、adv、ret、old_logp、replay。
- 调用 agent 的 replay 接口，重算当前策略下的 logp。
- 计算 PPO 或 ReinforcePP 的目标函数、裁剪项、价值损失和统计指标。
- 通过 lightning 模块把这些目标函数放进 Trainer 的训练循环中执行。

## 文件说明

### __init__.py

懒加载导出层。

- 暴露 Algorithm、PPO、ReinforcePP 三个对外入口。
- 避免在包初始化时过早导入重模块，减少入口脚本被旧导入路径拖垮的风险。

### base.py

算法抽象基类。

- 定义 update 接口，约束所有算法都以同一种方式接收 agent、batch 和 device。
- get_value_components 用于让 PPO 这类带 value net 的算法把附属组件回传给上层。

### ppo.py

PPO 的高层封装。

- 负责把 PPO 相关超参数组织起来。
- 创建 TrajectoryLightningModule 和 TrajectoryUpdateDataModule。
- 用 Lightning Trainer 跑多 epoch、多 minibatch 的更新流程。
- 适合看作“PPO 的训练调度入口”。

### ppo_core.py

PPO 的底层数值实现。

- 逐 minibatch 计算新 logp、value 预测、裁剪目标和梯度。
- 处理 grad accumulation、梯度裁剪、DDP sampler 等训练细节。
- 这里更接近纯算法内核，适合单独分析 PPO 公式如何落到代码里。

### reinforcepp.py

ReinforcePP 的高层封装。

- 组织 ReinforcePP 训练所需参数。
- 与 ppo.py 类似，负责把算法挂到 Lightning 的训练循环里。
- 适合看作“无 value head 版本的策略梯度训练入口”。

### reinforcepp_core.py

ReinforcePP 的底层更新逻辑。

- 根据 replay 重算 logp，用 advantage 直接构建策略梯度损失。
- 支持可选的 KL 正则和参考策略比较。
- 负责 minibatch 更新、梯度裁剪和分布式采样细节。

### trajectory_batch.py

把 actor 侧 shard 变成 learner 可训练 batch 的核心文件。

- 从 shard 中读取 obs、reward、done、replay 等字段。
- 对 PPO 计算 value、GAE 和 return。
- 对 Reinforce 家族计算 return 和 advantage。
- 负责 advantage 归一化，并做各种长度一致性检查。

这是连接 rollout 数据和算法更新的关键中间层。

### trajectory_policy_core.py

策略目标函数公共库。

- 提供 agent_logp_from_replay_batch，屏蔽不同 Agent 的 replay 接口差异。
- 统一实现 PPO 目标函数、Reinforce 目标函数和对应 metrics 统计。
- lightning/trajectory_module.py 的 training_step 会直接依赖这里。

### readme

旧的简短说明文件，只概括了 Trainer -> Algorithm -> algorithm_core 的关系。现在的 README.md 更完整，覆盖了每个 Python 文件的具体职责。

## 训练时如何经过这里

Learner 端主流程大致是：

1. lightning/actor_learner_datamodule.py 选出一批 shard。
2. trajectory_batch.py 把 shard 组装成训练 batch。
3. ppo.py 或 reinforcepp.py 建立 Lightning 训练循环。
4. trajectory_policy_core.py 和 ppo_core.py 或 reinforcepp_core.py 计算目标并更新参数。