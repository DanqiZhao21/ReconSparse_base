# ReconDreamer-RL

ReconDreamer-RL 是当前用于自动驾驶闭环强化学习训练的项目仓库。它把策略模型、闭环仿真环境、actor-learner 采样训练流程、PPO / ReinforcePP / GRPO 相关 objective，以及训练后评估脚本放在同一个训练侧工程中。

当前主链路是文件缓冲区形式的 actor-learner：

```text
actor -> rollout -> shard buffer -> learner -> checkpoint/version publish -> actor reload
```

训练主入口是：

- `script/train_actor_learner_v2.py`

`framework/README.md` 是框架内部说明；本文件只作为整个仓库的入口文档。

## 仓库关系

`HUGSIM-ORI` 是独立仓库，不是本仓库的 Git submodule，也不应该作为 `ReconDreamer-RL` 的子目录提交。

推荐保持两个仓库并列：

```text
/root/clone/ReconDreamer-RL
/root/clone/HUGSIM-ORI
```

ReconDreamer-RL 默认从 `/root/clone/HUGSIM-ORI` 读取 HUGSIM 代码和配置。需要改路径时设置：

```bash
export HUGSIM_ROOT=/path/to/HUGSIM-ORI
```

HUGSIM 自己的运行环境仍由 HUGSIM-ORI 仓库管理。当前 FIFO 后端会在 HUGSIM-ORI 目录下执行 `pixi run python ...`，训练环境只负责调度和与策略训练交互。

HUGSIM 的本地数据目录不要提交到 Git。通常在 HUGSIM-ORI 仓库内用软链接指向共享盘：

```bash
ln -s /OpenDataset/HUGSIM_data/scenarios /root/clone/HUGSIM-ORI/configs/scenarios
ln -s /OpenDataset/zhaodanqi/HUGSIM_data/outputs /root/clone/HUGSIM-ORI/outputs
```

## 快速检查

进入训练仓库：

```bash
cd /root/clone/ReconDreamer-RL
```

确认 HUGSIM-ORI 可被找到：

```bash
echo "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}"
ls "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}/configs"
```

确认 ReconDreamer-RL 的共享目录软链接存在：

```bash
ls -l assets
ls -l outputs
```

如果使用 HUGSIM-ORI 后端，还需要确认 HUGSIM 场景配置存在：

```bash
ls "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}/configs/scenarios/nuscenes" | head
```

## 仅训练

使用 orchestrator 拉起 actor 和 learner：

```bash
cd /root/clone/ReconDreamer-RL

CONFIG=script/configs/sparsedrive_v2/202605301357_HUGSM_grpo_only_closed_loop_reward-step_path_openGRPOCraft.yaml

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config "$CONFIG"
```

训练输出默认在：

```text
outputs/actor_learner/<timestamp>_<run_name>/
```

关键产物包括：

- `buffer/shards/`
- `buffer/consumed/`
- `weights/latest.ckpt`
- `weights/version.txt`
- 本次实际生效的 YAML 配置

## 训练后自动评估

直接调用 Python pipeline：

```bash
cd /root/clone/ReconDreamer-RL

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_eval_pipeline.py \
  --reinforcepp-config script/configs/sparsedrive_v2/202605301356_HUGSM_reinforcepp_closed_loop_reward-step_path_openGRPOCraft.yaml
```

也可以使用 HUGSIM-ORI wrapper：

```bash
bash script/run_train_eval_pipeline_hugsim_ori.sh
```

默认只跑 ReinforcePP；如果要额外跑 PPO，给 `train_eval_pipeline.py` 加 `--ppo` 并传入 `--ppo-config`。

自动 train + eval 的运行结果会放在：

```text
outputs/TrainEvaluationAuto/<run_id>/
```

评估侧结果会写到 HUGSIM-ORI 的：

```text
outputs/evaluate-auto/
```

## 目录说明

- `framework/`：当前主训练框架，包含 runner、rollout、io、batch、algorithms、lightning、agent、env wrapper 等模块。
- `script/`：训练、评估和 pipeline 入口，以及 YAML 配置。
- `tools/`：可视化、视频生成和辅助脚本。
- `reconsimulator/`：ReconSimulator 相关环境实现。
- `policy/`：策略侧代码和模型相关内容。
- `egoADs/`：SparseDrive / SparseDriveV2 等外部策略代码适配位置。
- `assets`：本地资产入口，通常是指向共享盘的软链接。
- `outputs`：训练、评估和可视化输出入口，通常是指向共享盘的软链接。

## 主训练链路

仅训练流程大致如下：

1. `script/train_actor_learner_v2.py` 读取配置并启动 orchestrator。
2. `framework/runner/` 负责配置规范化、actor / learner 进程启动和运行时装配。
3. Actor 通过 `framework/env_wrapper/` 与 Recon 或 HUGSIM-ORI 环境交互。
4. `framework/rollout/collector.py` 把轨迹写成 shard。
5. `framework/io/` 维护 shard、STOP、TRAINING_LOCK、权重版本文件和消费状态。
6. Learner 通过 `framework/lightning/actor_learner_datamodule.py` 读取 shard。
7. `framework/batch/` 把 shard 组装成训练 batch。
8. `framework/algorithms/` 和 `framework/lightning/` 执行策略更新。
9. 更新后的权重写入 `weights/latest.ckpt`，并推进 `weights/version.txt`，actor 检测新版本后继续采样。

更细的模块职责见 `framework/README.md`。

## 配置注意事项

HUGSIM-ORI 后端通常出现在 YAML 的 `env.backend: hugsim_ori` 和 `env.hugsim` 字段下。常见字段包括：

- `repo`：HUGSIM-ORI 仓库路径；未设置时走 `HUGSIM_ROOT` 或默认 `/root/clone/HUGSIM-ORI`。
- `scenario_dir`：HUGSIM 场景 YAML 目录。
- `base_path`、`camera_path`、`kinematic_path`：HUGSIM 侧仿真模板和相机/运动学配置。
- `model_base`：HUGSIM 模型或场景资产根目录。
- `pixi_cmd`：启动 HUGSIM 环境时使用的 pixi 命令。
- `output_root`：HUGSIM 运行输出目录。

如果修改配置字段，优先同步检查：

- `script/train_actor_learner_v2.py`
- `framework/runner/config_normalization.py`
- `framework/runner/env_factory.py`
- `framework/runner/actor_runtime.py`
- `framework/runner/learner_runtime.py`

## 更多文档

- `framework/README.md`：actor-learner 框架总览。
- `framework/agent/README.md`：策略适配层。
- `framework/algorithms/README.md`：PPO / ReinforcePP / trajectory objective。
- `framework/batch/README.md`：shard 到 training batch 的转换。
- `framework/env_wrapper/README.md`：Recon 和 HUGSIM 环境包装。
- `framework/io/README.md`：buffer、shard、权重版本和 STOP 文件协议。
- `framework/lightning/README.md`：Lightning module / datamodule 和训练生命周期。
- `framework/rollout/README.md`：Actor 侧采样。
- `framework/runner/README.md`：训练运行时和进程编排。

## 维护约定

- 根 README 只描述项目入口、仓库关系和常用运行方式。
- 框架内部职责和实现细节放在 `framework/README.md` 及子目录 README。
- 不要把 HUGSIM-ORI 作为本仓库子目录、备份目录或 submodule 引入。
- 不要提交本地数据、HUGSIM 输出、训练输出或共享盘软链接目标内容。
- 修改 actor-learner 协议时，需要同时检查 actor、learner、buffer 和 checkpoint/version 逻辑。
