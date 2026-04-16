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
- 对横向偏移 `lateral_error_m` 和路径朝向误差 `yaw_path_err_deg` 使用带死区的平滑惩罚。
- 对纵向 jerk 和 yaw jerk 使用带死区的舒适度惩罚。
- 在 episode 结束时，可根据 failure、timeout 或 env_done 再施加 terminal penalty。
- 奖励输出不仅有 reward，还会把各个分项、路径指标和碰撞指标写回 info，方便调试和记录。

## 训练时如何经过这里

Actor 调用环境 step 时，env_wrapper/rl_wrapper.py 会把仿真状态传给 TrackingRewardComputer。它返回的 reward 会被 rollout/collector.py 收进 shard，最后进入 Learner 用于计算 return 和 advantage。
