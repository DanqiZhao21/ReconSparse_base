# framework/rewards

这个目录负责定义奖励函数。它直接决定策略在训练中被鼓励去做什么，因此虽然文件很少，但对整体训练目标非常关键。

## 文件说明

### __init__.py

包导出层。

- 对外暴露 TrackingRewardComputer 和 TrackingRewardResult。
- env_wrapper/rl_wrapper.py 通过这里拿到奖励计算器。

### tracking.py

当前训练主流程使用的奖励计算核心。

- 统一使用 path-based smooth reward 计算，不再保留旧的 legacy reward 分支。
- 把 expert clip 视作 path，而不是严格的逐时刻监督点。
- 计算 ego 在 densify 后 path 上的投影进度 `progress_s` 和增量 `progress_delta_s`。
- 把 dense reward 显式拆成 `positive_reward` 和 `cost_reward`：
  - `positive_reward` 主要来自 progress / completion / anchor progress
  - `cost_reward` 主要来自 lateral / yaw / comfort / dense collision penalty
- `collision.mode=constraint_gate` 时，碰撞不再直接扣 dense reward，而是只门控正向收益。
- 还支持对严重横向偏差或严重朝向偏差触发 `severe_gate_scale`，同样只门控正向收益。
- 在 episode 结束时，可根据 failure、timeout 或 env_done 再施加 terminal penalty。
- 奖励输出不仅有 reward，还会把各个分项、gate 来源和路径指标写回 info，方便调试和记录。

## 推荐分层职责

- `step reward`:
  - 提供 dense learning signal
  - 鼓励沿路径推进、靠近终点、保持平顺
- `constraint gate`:
  - 用于 collision / severe tracking 这类约束事件
  - 只削弱正向收益，不洗白已有代价
- `terminal penalty`:
  - 用于 episode-level failure / timeout / success 边界
  - 表达“这局结束得好不好”
- `PDM / GRPO scorer`:
  - 用于轨迹级偏好和候选排序
  - 不要和 step reward 混成一套手调加减法

## 训练时如何经过这里

Actor 调用环境 step 时，env_wrapper/rl_wrapper.py 会把仿真状态传给 TrackingRewardComputer。它返回的 reward 会被 rollout/collector.py 收进 shard，最后进入 Learner 用于计算 return 和 advantage。
