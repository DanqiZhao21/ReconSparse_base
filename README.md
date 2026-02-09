# ReconDreamer-RL / ReconDiff（闭环强化学习）

本仓库用于训练与评测“ReconDiff”闭环策略：将 DiffusionDriveV2（轨迹扩散策略）与 ReconDreamer 3D Gaussian Splatting（3DGS）仿真环境结合，通过 Actor-Learner 框架进行 PPO / Reinforce++ 强化学习。

## 近期改动

- **统一到 framework/**：训练/评测/可视化脚本统一从 `framework.*` 导入；历史 `rl/*` 命名空间已收敛。
- **Actor-Learner v2 入口**：统一使用 [script/train_actor_learner_v2.py](script/train_actor_learner_v2.py)。
- **rollout 语义修正（commit-to-plan）**：引入 `train.actor_learner.commit_steps=K`，每次从 DDV2 采样一次 plan，然后连续执行前 $K$ 个点，并聚合为 1 条 macro transition 写入 buffer；对应折扣使用 $\gamma_{eff}=\gamma^K$。
- **时长对齐**：默认 `step_frames=5` 且 `frame_dt=0.1s`，因此 1 个 `env.step` 对应 $0.5s$；18s 一局最多 36 步（见 [script/configs/ppo_closed_loop.yaml](script/configs/ppo_closed_loop.yaml)）。
- **buffer 磁盘卫生**：learner 消费过的 shard 会自动移动/清理，避免长期堆积占满磁盘。
- **视频生成工具**：新增 [tools/smalltool/visualize/generate_video.py](tools/smalltool/visualize/generate_video.py) + [tools/smalltool/visualize/generate_video.sh](tools/smalltool/visualize/generate_video.sh)，可按“秒”控制 rollout 时长（必须是 0.5s 的整数倍，或更一般地是 `step_frames*frame_dt` 的整数倍）。

## 代码结构

- **Env 封装**：`framework/env_wrapper/`（ReconSimulator + gymnasium 封装，vec-env 适配等）
- **Agent/Policy**：`framework/agent/`（DDV2 RL 适配器/采样 + logp 接口）
- **算法**：
  - PPO：`framework/algorithms/ppo.py` + `framework/algorithms/ppo_ddv2_core.py`
  - Reinforce++：`framework/algorithms/reinforcepp.py` + `framework/algorithms/reinforcepp_core.py`
- **Buffer IO**：`framework/io/buffer.py`（shard/weights/version 管理、消费与清理）

## 环境配置

推荐直接使用 [environment.yml](environment.yml)（CUDA 11.8 + Python 3.10）：

```bash
conda env create -f environment.yml
conda activate recondreamerNew-rl
```

## 数据与模型准备（Denso/OpenDataset 场景）

如果使用Denso 服务器环境，数据/模型通常已经在 `OpenDataset` 盘中准备好；需要确认环境变量 + 软连接。

### 1) navsim 数据集

参考官方安装说明：https://github.com/autonomousvision/navsim/blob/main/docs/install.md

确认以下环境变量（路径以你的服务器实际为准）：

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/OpenDataset/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/OpenDataset/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/OpenDataset/navsim/navsim"
export OPENSCENE_DATA_ROOT="/OpenDataset/navsim/dataset"
```

### 2) DiffusionDriveV2 预训练资源（GTRS/ckpt）

需要保证 [DiffusionDriveV2/gtrs_traj](DiffusionDriveV2/gtrs_traj) 与 [DiffusionDriveV2/ckpt](DiffusionDriveV2/ckpt) 指向共享盘或已下载的真实目录，例如：

```bash
ls -l DiffusionDriveV2/gtrs_traj
ls -l DiffusionDriveV2/ckpt
```

### 3) ReconSimulator 资产（3DGS anchors 等）

根目录 [assets](assets) 通常应为指向共享盘的软连接：

```bash
ls -l assets
```

## 快速评测（DiffusionDriveV2 / navsim）

### 生成 cache

```bash
bash tools/cache_fast.sh
# 或完整版本
bash tools/cache.sh
```

### 运行评测

```bash
bash tools/evaluate_fast.sh
bash tools/evaluate.sh
```

如果要评测 RL checkpoint（PDM score），可参考 [tools/evaluate_rl.sh](tools/evaluate_rl.sh)。

## 闭环 RL 训练（推荐：Actor-Learner）

入口脚本：

- Launcher：[tools/train_actor_learner.sh](tools/train_actor_learner.sh)
- Python：[script/train_actor_learner_v2.py](script/train_actor_learner_v2.py)

示例：

```bash
# PPO（默认配置为 ppo_closed_loop.yaml）
LOG_DIR=./logs ALGO=ppo bash tools/train_actor_learner.sh

# Reinforce++（默认配置为 reinforcepp_closed_loop.yaml）
LOG_DIR=./logs ALGO=reinforcepp bash tools/train_actor_learner.sh

# 显式指定配置（推荐做法）
LOG_DIR=./logs CONFIG=script/configs/ppo_closed_loop.yaml bash tools/train_actor_learner.sh
```

重要配置项（建议先读一遍）：

- 环境时长：`env.max_steps=36`，`env.step_frames=5`（对应 18s/局，0.5s/step）
- 宏动作：`train.actor_learner.commit_steps=K`（一次采样 plan 后连续执行 K 步，再写 1 条宏 transition）
- `actor_horizon` 语义：引入 `commit_steps` 后，`actor_horizon` 是“宏决策次数”，而不是原始 env.step 数。

<!-- ## 闭环训练（单进程/多 GPU，偏调试用途）

如果你想用“每 GPU 一个进程”的方式跑闭环训练（更像旧版脚本行为），可用 [tools/trainclosedloop.sh](tools/trainclosedloop.sh)：

```bash
LOG_DIR=./logs GPUS="0 1 2 3" bash tools/trainclosedloop.sh
``` -->

## 生成 rollout 视频

新增工具：

- 脚本：[tools/smalltool/visualize/generate_video.py](tools/smalltool/visualize/generate_video.py)
- Wrapper：[tools/smalltool/visualize/generate_video.sh](tools/smalltool/visualize/generate_video.sh)

典型用法：

```bash
# 生成 1 个 scene 的完整 18s（36步）rollout：duration 必须是 0.5s 的整数倍（默认 step_frames=5, frame_dt=0.1）
bash tools/smalltool/visualize/generate_video.sh --scene-list 413 --duration-s 18.0 --step-frames 5 --fps 10 --outdir outputs/visualize

# 指定多个 scene + 只跑 5 秒
bash tools/smalltool/visualize/generate_video.sh --num-scenes 3 --duration-s 5.0 --seed 0 --outdir outputs/visualize

# 插帧（让视频更顺滑，非改变仿真步长）
bash tools/smalltool/visualize/generate_video.sh --scene-list 413 --duration-s 18.0 --interp-method blend --interp-frames-per-step 5
```

## 常见问题（Actor-Learner 框架）

- **权重版本**：learner 更新后由 rank0 保存 `latest.ckpt` 并将 `version.txt` 加 1；actor 在写 shard 前会尝试加载最新版本。
- **shard 消费机制**：learner 选定本次更新使用的 shard 文件名，通过 DDP 广播；更新完成后 rank0 会把已消费 shard 移动到 `consumed/` 并按策略清理。
- **DDP 超时（NCCL/Gloo）**：常见原因是 actor/learner 在 `broadcast/barrier` 前后状态不一致或 shard 堆积。建议降低 `shards_per_update`、适当调小 `num_envs_per_actor`、或将 `actor_learner.mode` 设为 `sync`。
- **ninja / cpp_extension 报错**：通常是运行时触发 JIT 编译（如 nvdiffrast）但找不到 CUDA headers/libs。优先使用各类 `.sh` wrapper（训练/评测/视频）启动，它们会补齐 `CUDA_HOME/CPATH/LD_LIBRARY_PATH` 并设置 torch extension cache 目录。

## 参考

- 环境文件：[environment.yml](environment.yml)
- DiffusionDriveV2 官方仓库：https://github.com/hustvl/DiffusionDriveV2
- navsim 安装说明：https://github.com/autonomousvision/navsim/blob/main/docs/install.md
- Anchor/资产说明（示例数据集）：https://huggingface.co/datasets/Ni1111/ReconDreamer-RL/tree/main/assets
   
