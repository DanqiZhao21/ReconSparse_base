# framework/rewards

这个目录负责定义奖励函数。它直接决定策略在训练中被鼓励去做什么，因此虽然文件很少，但对整体训练目标非常关键。

## 文件说明

### __init__.py

包导出层。

- 对外暴露 TrackingRewardComputer 和 TrackingRewardResult。
- env_wrapper/rl_wrapper.py 通过这里拿到奖励计算器。

### tracking.py

当前训练主流程使用的奖励计算核心。

- 根据位置偏差、航向误差、静态和动态碰撞、纵向 jerk、航向 jerk 计算 step reward。
- 在 episode 结束时，可根据 failure、timeout 或 env_done 再施加 terminal penalty。
- 奖励输出不仅有 reward，还会把各个惩罚项写回 info，方便调试和记录。

## 训练时如何经过这里

Actor 调用环境 step 时，env_wrapper/rl_wrapper.py 会把仿真状态传给 TrackingRewardComputer。它返回的 reward 会被 rollout/collector.py 收进 shard，最后进入 Learner 用于计算 return 和 advantage。