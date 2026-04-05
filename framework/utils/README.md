# framework/utils

这个目录放的是跨模块共享的工具代码。它多数不直接定义 RL 算法，但会影响观测预处理、路径解析、gsplat 后端可用性以及离线缓存准备，因此属于训练系统的支撑层。

## 文件说明

### __init__.py

导出 gsplat 相关工具和 torch extension patch，供 runner、环境模块和测试代码复用。

### obs.py

观测预处理工具。

- 把 6 路相机图像从环境字典转换成 PPO value net 使用的张量格式。
- 输出形状是 1 x 18 x H x W，是 rollout -> batch -> value net 这条链路的标准图像输入格式。

### repo_paths.py

仓库路径解析工具。

- 统一定位 REPO_ROOT、egoADs 根目录以及具体 agent 子目录。
- `runner/agent_factory.py` 和 `runner/launch_env.py` 会直接依赖这里来解析 ckpt、config 和子仓库路径。

### gsplat_backend.py

gsplat 后端兼容层。

- 根据当前 Torch 版本决定使用现代后端还是 legacy 后端。
- 在需要时会动态编译并加载 gsplat 的 CUDA 扩展。
- 主要服务于环境渲染和相关 warmup，而不是直接服务 PPO 本身。

### gsplat_warmup.py

gsplat CUDA 预热工具。

- 在 actor 大规模启动前，先串行触发一次 gsplat 后端加载。
- 避免多个 actor 同时触发 JIT 编译，降低启动期冲突和等待时间。

### torch_extension.py

Torch C++ 扩展加载补丁。

- 修补 torch.utils.cpp_extension.load 在构建目录处理上的细节问题。
- 主要被 gsplat_backend.py 调用，用于保证扩展 JIT 编译稳定进行。

### build_metrics_cache.py

离线环境缓存构建脚本。

- 预先为 NuScenes 场景生成 drivable area、中心线、静态和动态物体等快照缓存。
- env_wrapper/rl_wrapper.py 在训练中会使用这些缓存加速碰撞和地图相关判断。
- 它通常不在每次训练时执行，但会影响 reward 和终止逻辑的数据来源。

## 训练时如何经过这里

训练过程中最常见的路径是：

- Agent 和 runner 通过 repo_paths.py 找模型和配置。
- rollout 或 value net 通过 obs.py 处理图像观测。
- 环境渲染和相关测试通过 gsplat_backend.py、gsplat_warmup.py、torch_extension.py 保障 CUDA 扩展可用。
- reward 相关地图缓存则依赖 build_metrics_cache.py 预处理产物。
