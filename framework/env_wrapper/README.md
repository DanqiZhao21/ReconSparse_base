# framework/env_wrapper

这个目录负责把底层重建仿真环境包装成 RL 训练可直接使用的环境接口。它位于训练链路的最前端，是 Actor 采样时接触最多的模块之一。

## 目录职责

- 把 ReconSimulator 包装成类似 Gymnasium 的 reset 和 step 接口。
- 在环境层补充奖励、碰撞、终止条件和场景采样逻辑。
- 提供多进程向量环境和 3DGS 渲染辅助工具。

## 文件说明

### __init__.py

包导出层。

- 对外导出 RLReconEnv、SceneSamplingEnv、SubprocVecEnv、SerialVecEnv 以及 3DGS 相关工具函数。
- `runner/env_factory.py` 主要通过这里统一拿环境构造能力。

### rl_wrapper.py

训练环境的核心包装器。

- 把 ReconSimulator 包装成 RLReconEnv，输出 obs、reward、terminated、truncated、info。
- 在 step 中补充 yaw 误差修正、静态和动态碰撞判定、阈值终止和 terminal kind 标记。
- 奖励本身不是在这里直接硬编码，而是委托 rewards/tracking.py 计算。

这是环境和奖励逻辑真正汇合的地方。

### subproc_vec_env.py

并行环境与场景采样控制器。

- SceneSamplingEnv 负责在 reset 时自动选择 scene 和 start frame。
- make_scene_sampling_env 是 `runner/env_factory.py` 构建采样环境时的主要入口。
- SubprocVecEnv 提供多进程并行环境执行能力。
- 这个文件还承担“每个 worker 自己管理场景轮换”的职责，让主进程只关心 reset 和 step。

### tool.py

3DGS 和渲染相关的辅助工具。

- 负责加载重建 trainer、缓存 splat 模型、提取渲染状态。
- 提供相机射线、SLERP、设备搬运等几何和渲染辅助函数。
- 它不是 PPO 更新核心，但直接影响环境观测是如何从重建资产里生成出来的。

## 训练时如何经过这里

`runner/env_factory.py` 会调用 `make_scene_sampling_env` 或直接构建 `RLReconEnv`。随后：

- Actor 在 `rollout/collector.py` 中不断对这里的环境调用 step。
- `rl_wrapper.py` 会在每一步把仿真结果转成 RL 能理解的 reward 和终止信号。
- `rewards/tracking.py` 负责具体 reward 细节，环境包装器只负责把它接到 step 流程里。
- 若开启多进程环境，则 `subproc_vec_env.py` 负责把这些 step 和 reset 分发到各 worker。
