# framework/algorithms

这个目录负责 **算法目标函数与算法配置描述**。它是 learner 链路里的 objective/config 层，不负责直接运行 Trainer。

## 目录职责

- 从 batch 模块拿到已经整理好的 obs、adv、ret、old_logp、replay。
- 调用 agent 的 replay 接口，重算当前策略下的 logp。
- 计算 PPO 或 ReinforcePP 的目标函数、裁剪项、价值损失和统计指标。
- 提供 PPO/ReinforcePP/SAC 规格对象（裁剪系数、优化超参数、梯度与采样相关参数）给 learner runtime 使用。

不属于这个目录的职责：

- 不在这里构建 Lightning `Trainer`。
- 不在这里驱动训练循环。
- 不在这里做 actor-learner 协调与 checkpoint/version 发布。

## 文件说明

### __init__.py

懒加载导出层。

- 暴露 Algorithm、PPO、ReinforcePP、SAC 四个对外入口。
- 避免在包初始化时过早导入重模块，减少入口脚本被旧导入路径拖垮的风险。

### base.py

算法抽象基类。

- 定义算法规格对象的公共接口。
- `update` 仅保留兼容入口，不再由 algorithms 层执行训练循环。

### ppo.py

PPO 的配置/规格容器。

- 组织 PPO 所需超参数与 value net 引用。
- 暴露 learner runtime 需要读取的字段（如 `clip_eps`、`vf_coef`、`minibatch_size`、`grad_accum_steps` 等）。
- 由 Lightning 侧根据这些字段创建优化器；算法规格对象本身不再持有 optimizer。

### reinforcepp.py

ReinforcePP 的配置/规格容器。

- 组织 ReinforcePP 训练所需参数。
- 暴露 learner runtime 读取的训练相关字段。
- 不直接创建 Trainer 或执行训练循环。

### sac.py

SAC-style 的配置/规格容器。

- 当前 actor-learner shard 只保存 replay、闭环 reward、done 和 log-prob，不是标准 off-policy SAC replay buffer。
- 因此这里的 `SAC` 是兼容现有闭环 reward 路径的 entropy-regularized policy-gradient 规格对象。
- 它通过 `trajectory_policy_core.py` 里的 SAC-style objective 使用闭环 return/advantage 和 `entropy_coef`，不改变 shard schema。

### trajectory_policy_core.py

策略目标函数公共库。

- 提供 agent_logp_from_replay_batch，屏蔽不同 Agent 的 replay 接口差异。
- 统一实现 PPO、Reinforce、SAC-style 目标函数和对应 metrics 统计。
- `framework/lightning/trajectory_module.py` 的 `training_step` 会直接依赖这里。

### nuscenes_pdm_scorer.py

NuScenes PDM 风格业务 scorer。

- 作为默认的 NuScenes counterfactual scorer 后端。
- 负责 sample context、occupancy、drivable area、collision TTC 等 PDM 评分逻辑。

### nuscenes_craft_scorer.py

NuScenes CRAFT/CARL 风格业务 scorer。

- 负责 CRAFT CaRL forward-sim reward 评分。
- 内部复用 PDM scorer 的 route / collision / map 工具。

### nuscenes_scorer_utils.py

NuScenes 共享工具层，不是独立业务 scorer。

- 提供 `_wrap_angle`、`_polyline_arclength`、`_project_progress` 等基础函数。
- 承载通用 map/context/visualization 辅助逻辑。
- 供 PDM 和 CRAFT scorer 复用。

## 训练时如何经过这里

Learner 端主流程大致是：

1. `lightning/actor_learner_datamodule.py` 选出一批 shard。
2. `framework/batch/actor_learner.py` 把 shard 组装成训练 batch。
3. `runner/learner_factory.py` 构建 PPO/ReinforcePP 规格对象（仅配置承载，PPO 同时携带 value net）。
4. `runner/learner_runtime.py` 组装 Lightning 训练入口，`trajectory_module.py` 负责实际 `training_step`，`actor_learner_module.py` 负责 actor-learner 生命周期钩子。
5. `trajectory_policy_core.py` 为 Lightning 模块提供 PPO/Reinforce 目标函数与 metrics 计算。

当前 canonical 目标函数入口就是 `trajectory_policy_core.py`，batch 组装入口就是 `framework/batch/actor_learner.py`。
