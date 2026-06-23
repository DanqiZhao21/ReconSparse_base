# ReconSparse

ReconSparse is a reinforcement learning framework for closed-loop autonomous driving policy optimization. It integrates policy backends, reconstructed simulation environments, actor-learner data collection, PPO / ReinforcePP / SAC-style objectives, reward shaping, checkpoint publication, and post-training evaluation within a unified training system.

The primary training path uses a file-buffer-based actor-learner protocol:

```text
actor -> rollout -> shard buffer -> learner -> checkpoint/version publish -> actor reload
```

Primary entry points:

- [`script/train_actor_learner_v2.py`](script/train_actor_learner_v2.py): actor-learner training launcher.
- [`script/train_eval_pipeline.py`](script/train_eval_pipeline.py): training followed by automatic evaluation of the latest checkpoint.
- [`framework/`](framework/README.md): core reinforcement learning framework.

## Highlights

- **Closed-loop reinforcement learning for autonomous driving**: policy rollouts and updates are performed in Recon / HUGSIM closed-loop environments.
- **Actor-learner training**: actors collect rollouts and write shard files; learners consume shard buffers and update policy weights.
- **Multiple ego-policy backends**: SparseDriveV2 is supported, with adapter structure for additional end-to-end autonomous driving policies.
- **Multiple reinforcement learning objectives**: PPO, ReinforcePP, SAC-style objectives, and extensible algorithm specifications are supported.
- **Multiple 3DGS assets**: HUGSIM-ORI and ReconDreamer 3D Gaussian Splatting assets can be used for closed-loop training and evaluation.
- **Lightning-based learner stack**: PyTorch Lightning is used to organize learner-side training steps and lifecycle hooks.

## Architecture

The actor-learner system can be understood at two levels: runtime roles and functional components.

Runtime roles:

- **Orchestrator**: the supervisory process that starts and manages learner and actor subprocesses.
- **Actor**: a rollout process that owns a policy replica, interacts with the environment, collects trajectories, and writes shard files.
- **Learner**: a training process that owns the trainable policy, reads data from the shard buffer, performs policy updates, and publishes new weights.

Functional components:

- **Agent**: a model-specific policy adapter that exposes a common reinforcement learning interface for action sampling, replay serialization, log-probability recomputation, and checkpoint I/O.
- **Environment**: a wrapper around 3DGS-based simulation environments, exposing `reset` / `step` semantics and integrating reward and termination logic.
- **Shard Buffer**: the file-based protocol through which actors write trajectory shards and learners select shards for training.
- **Learner Training Stack**: the learner-side optimization path that updates the policy, writes `weights/latest.ckpt`, and increments `weights/version.txt`.

Actors and learners both use Agents, but they are not Agents themselves. An Agent is the policy-model interface; an actor is a runtime role that uses an Agent for sampling, and a learner is a runtime role that uses an Agent for optimization and checkpoint publication.

Main execution path:

```text
script/train_actor_learner_v2.py
  -> framework/runner/orchestrator.py
      -> framework/runner/actor_runtime.py
          -> framework/env_wrapper/ + framework/agent/
          -> framework/rollout/collector.py
          -> framework/io/buffer.py
      -> framework/runner/learner_runtime.py
          -> framework/lightning/actor_learner_datamodule.py
          -> framework/batch/actor_learner.py
          -> framework/lightning/actor_learner_module.py
          -> framework/algorithms/trajectory_policy_core.py
          -> framework/io/buffer.py
```

For a more detailed description of the framework, see [`framework/README.md`](framework/README.md).

## Environment and Assets

### Python and CUDA Environment

The main training environment is defined by [`environment.yml`](environment.yml). The current environment is built around:

- Python 3.10
- CUDA toolkit / nvcc 11.8
- PyTorch 2.1.0 + cu118
- PyTorch Lightning 2.2.1 / Lightning 2.5.5
- SparseDrive / SparseDriveV2 / nuScenes / nuPlan / gsplat / nvdiffrast dependencies

Create the environment with Conda or Mamba:

```bash
cd /root/clone/ReconDreamer-RL

conda env create -f environment.yml
conda activate recondreamerNew-rl
```

If the environment already exists, update it from the environment file:

```bash
conda env update -n recondreamerNew-rl -f environment.yml --prune
conda activate recondreamerNew-rl
```

After installation, run a basic dependency check:

```bash
python - <<'PY'
import torch
import lightning
import pytorch_lightning
import gsplat
import nvdiffrast.torch as dr

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("lightning", lightning.__version__)
print("pytorch_lightning", pytorch_lightning.__version__)
print("gsplat", getattr(gsplat, "__version__", "unknown"))
print("nvdiffrast ok", dr is not None)
PY
```

The repository root should be added explicitly to `PYTHONPATH` during training:

```bash
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}
```

### CUDA Extensions

Rendering and policy execution may trigger PyTorch CUDA extension compilation through `gsplat`, `nvdiffrast`, or policy-specific CUDA operators. Ensure that CUDA 11.8 is discoverable by the build system:

```bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/root/clone/ReconDreamer-RL/.cache/torch_extensions

nvcc --version
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch cuda:", torch.version.cuda)
print("cpp_extension CUDA_HOME:", CUDA_HOME)
PY
```

If actor startup stalls during `gsplat` compilation, clear the local extension cache and retry:

```bash
rm -rf /root/clone/ReconDreamer-RL/.cache/torch_extensions/gsplat_cuda_legacy
```

### Policy Backend Resources

Ego-policy implementations are resolved from [`egoADs/`](egoADs/) by default. The root environment has been consolidated for the actor-learner training path with SparseDriveV2. If DiffusionDriveV2 or SparseDrive is used instead, update the Conda environment and policy resources accordingly.

The typical SparseDriveV2 configuration uses:

```yaml
agent:
  type: sparsedrive_v2
  ckpt: SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt
```

Check the corresponding files or symbolic links:

```bash
ls -l egoADs/SparseDriveV2/ckpt
ls -l egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt
ls -l egoADs/SparseDriveV2/ckpt/resnet34.bin
ls -l egoADs/SparseDriveV2/ckpt/kmeans
```

The original SparseDriveV2 environment file is [`egoADs/SparseDriveV2/environment.yml`](egoADs/SparseDriveV2/environment.yml), but it describes the upstream policy project's Python 3.9 / torch 2.0.1 environment. For this repository's actor-learner training path, use the root-level [`environment.yml`](environment.yml) instead.

If another policy backend is selected, verify that `agent.type`, `agent.ckpt`, and `agent.config` in the training YAML match the actual backend and checkpoint layout.

### HUGSIM-ORI Runtime

`HUGSIM-ORI` is an independent repository with its own runtime environment. It should not be nested under `ReconDreamer-RL`. Communication between HUGSIM-ORI and the ego-policy stack is performed through a FIFO-based interface.

The recommended directory layout is:

```text
/root/clone/ReconDreamer-RL
/root/clone/HUGSIM-ORI
```

By default, ReconSparse reads HUGSIM code and configuration from `/root/clone/HUGSIM-ORI`. To use another location, set:

```bash
export HUGSIM_ROOT=/path/to/HUGSIM-ORI
```

HUGSIM-ORI manages its own runtime environment. The FIFO backend executes `pixi run python ...` inside the HUGSIM-ORI repository. ReconSparse is responsible for orchestration, sampling, training, and evaluation.

In practical terms:

- `ReconDreamer-RL` runs actors, learners, policy code, and reinforcement learning logic in the `recondreamerNew-rl` Conda environment.
- `HUGSIM-ORI` runs the simulator-side FIFO backend in its own pixi environment.

Verify the HUGSIM-side runtime:

```bash
cd "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}"
pixi run python --version
ls configs
```

### Local Data Entrypoints and Symbolic Links

Datasets, 3DGS assets, evaluation videos, and training outputs should be exposed through local directories or symbolic links. Common entrypoints include:

- `assets/`: data entrypoint on the ReconSparse side, usually a symbolic link to shared storage.
- `outputs/`: training, evaluation, and visualization outputs, usually a symbolic link to shared storage.
- `HUGSIM-ORI/configs/scenarios`: HUGSIM scenario YAML directory.
- `HUGSIM-ORI/outputs`: HUGSIM evaluation, rendering, and video output directory.

Example symbolic links:

```bash
ln -s /OpenDataset/HUGSIM_data/scenarios /root/clone/HUGSIM-ORI/configs/scenarios
ln -s /OpenDataset/zhaodanqi/HUGSIM_data/outputs /root/clone/HUGSIM-ORI/outputs
ln -s /OpenDataset/ReconDreamer-RL/outputs /root/clone/ReconDreamer-RL/outputs
ln -s /OpenDataset/ReconDreamer-RL/assets /root/clone/ReconDreamer-RL/assets
```

If `assets/` or `outputs/` already exists as a real directory, do not overwrite it blindly. Inspect the contents first, then either migrate the data or update the YAML paths.

### HUGSIM Path Configuration

Symbolic links only define local data entrypoints. The effective runtime paths are taken from the YAML file passed through `--config`:

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config script/configs/sparsedrive_v2/xxx.yaml
```

HUGSIM-related paths are mainly configured under `env.hugsim`:

```yaml
env:
  backend: hugsim_ori
  hugsim:
    repo: /root/clone/HUGSIM-ORI
    scenario_dir: /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes
    base_path: /root/clone/HUGSIM-ORI/configs/sim/nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml
    camera_path: /root/clone/HUGSIM-ORI/configs/sim/nuscenes_camera.yaml
    kinematic_path: /root/clone/HUGSIM-ORI/configs/sim/kinematic.yaml
    model_base: /OpenDataset/HUGSIM_data/scenes/nuscenes
    output_root: outputs/hugsim_rl
    recon_data_root: /root/clone/ReconDreamer-RL/assets/nus/data
    nuscenes_root: /root/clone/ReconDreamer-RL/assets/nuscenes/v1.0-trainval
    frame2token_dir: /root/clone/ReconDreamer-RL/assets/nus/information/frame2token
```

Key fields:

- `env.hugsim.repo`: HUGSIM-ORI repository path.
- `env.hugsim.scenario_dir`: HUGSIM scenario YAML directory.
- `env.hugsim.model_base`: HUGSIM / 3DGS scene asset root.
- `env.hugsim.nuscenes_root`: nuScenes data root.
- `env.hugsim.frame2token_dir`: frame-to-nuScenes-token index directory.
- `env.hugsim.recon_data_root`: ReconDreamer-side reconstructed data root.
- `env.hugsim.output_root`: output directory used by the HUGSIM FIFO process, usually under `outputs/`.

If a YAML file omits some of these fields, defaults are resolved in `framework/runner/env_factory.py` and `framework/utils/repo_paths.py`. `HUGSIM_ROOT` only affects HUGSIM-relative path resolution and the default HUGSIM root.

### nuScenes and NAVSIM Variables

SparseDriveV2 and some reward / scorer components may indirectly use nuScenes, nuPlan, or NAVSIM resources. The training YAML remains the source of truth for runtime paths. If an external policy or utility script expects environment variables, configure them according to the local storage layout:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/OpenDataset/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/OpenDataset/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/OpenDataset/navsim/navsim"
export OPENSCENE_DATA_ROOT="/OpenDataset/navsim/dataset"
```

Common resource checks:

```bash
ls -l /root/clone/ReconDreamer-RL/assets/nus/data
ls -l /root/clone/ReconDreamer-RL/assets/nuscenes/v1.0-trainval
ls -l /root/clone/ReconDreamer-RL/assets/nus/information/frame2token
ls -l /OpenDataset/HUGSIM_data/scenes/nuscenes
```

### Environment Sanity Check

After configuring Conda, CUDA, policy weights, HUGSIM, and data links, run:

```bash
cd /root/clone/ReconDreamer-RL
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}

python - <<'PY'
from framework.runner.agent_factory import build_agent
from framework.utils.repo_paths import resolve_hugsim_root, resolve_repo_path

print("HUGSIM_ROOT:", resolve_hugsim_root())
print("SparseDriveV2 ckpt:", resolve_repo_path("SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt"))
print("agent factory import ok:", build_agent is not None)
PY
```

To verify that the training entrypoint can be imported and parsed:

```bash
python -u script/train_actor_learner_v2.py --help
```

## Quick Start

### 1. Check Repository and Asset Paths

```bash
cd /root/clone/ReconDreamer-RL
conda activate recondreamerNew-rl
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}

echo "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}"
ls "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}/configs"

ls -l assets
ls -l outputs
```

If the HUGSIM-ORI backend is used, also check that scenario files are accessible:

```bash
ls "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}/configs/scenarios/nuscenes" | head
```

### 2. Start Actor-Learner Training

Use the `orchestrator` role to launch the learner and actors:

```bash
cd /root/clone/ReconDreamer-RL

CONFIG=script/configs/sparsedrive_v2/202606151204_HUGSM_reinforcepp_closed_loop_steppath_hd_EncourageProgress_CollisionPenalty_advnorm_NoGRPOCraft_substeps1.yaml

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config "$CONFIG"
```

Supported roles:

- `orchestrator`: main launcher that starts and supervises actor and learner subprocesses.
- `actor`: standalone actor process for rollout collection and shard writing.
- `learner`: standalone learner process for optimization and checkpoint publication.

Training outputs are written by default to:

```text
outputs/actor_learner/<timestamp>_<run_name>/
```

Important runtime artifacts:

```text
buffer/shards/       # pending training shards written by actors
buffer/consumed/     # shards consumed by the learner
weights/latest.ckpt  # latest checkpoint published by the learner
weights/version.txt  # checkpoint version number
*.yaml               # materialized configuration for the run
```

### 3. Run Automatic Post-Training Evaluation

The pipeline can train a policy and then evaluate the latest checkpoint. By default, it runs repeated evaluation over nuScenes YAML scenarios:

```bash
cd /root/clone/ReconDreamer-RL

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_eval_pipeline.py \
  --reinforcepp-config script/configs/sparsedrive_v2/202606151204_HUGSM_reinforcepp_closed_loop_steppath_hd_EncourageProgress_CollisionPenalty_advnorm_NoGRPOCraft_substeps1.yaml
```

Wrapper scripts are also available:

```bash
bash script/run_train_eval_pipeline.sh
```

For HUGSIM-ORI-oriented evaluation:

```bash
bash script/run_train_eval_pipeline_hugsim_ori.sh
```

The default pipeline runs ReinforcePP only. To evaluate PPO as well, add `--ppo` and provide `--ppo-config`.

Automatic train-and-evaluate outputs are written to:

```text
outputs/TrainEvaluationAuto/<run_id>/
```

Evaluation-side outputs are written under HUGSIM-ORI:

```text
outputs/evaluate-auto/
```

## Configuration and Runtime Files

Training configurations are stored under [`script/configs/`](script/configs/). The YAML passed through `--config` is authoritative at runtime. To change data roots, the HUGSIM repository, or nuScenes paths, update the active YAML, especially its `env.hugsim.*` fields.

Key configuration groups:

- `env`: backend selection, maximum episode length, rendering size, scene sampling, rewards, and HUGSIM parameters.
- `env.hugsim`: HUGSIM-ORI paths, scenarios, base configuration, 3DGS assets, nuScenes data, and FIFO launch parameters.
- `env.reward`: step rewards, terminal penalties, collision and comfort terms, and CRAFT / PDM scorer options.
- `train`: algorithm selection, WandB logging, learning rate, batch/update settings, and actor-learner runtime parameters.

A typical HUGSIM-ORI backend configuration is:

```yaml
env:
  backend: hugsim_ori
  hugsim:
    repo: /root/clone/HUGSIM-ORI
    scenario_dir: /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes
    model_base: /OpenDataset/HUGSIM_data/scenes/nuscenes
    recon_data_root: /root/clone/ReconDreamer-RL/assets/nus/data
    nuscenes_root: /root/clone/ReconDreamer-RL/assets/nuscenes/v1.0-trainval
    frame2token_dir: /root/clone/ReconDreamer-RL/assets/nus/information/frame2token
    output_root: outputs/hugsim_rl
    pixi_cmd: pixi
```

Actors and learners coordinate through file-based runtime artifacts:

```text
buffer/shards/       # pending training shards produced by actors
buffer/consumed/     # shards moved after learner consumption
weights/latest.ckpt  # latest policy weights
weights/version.txt  # version number used by actors for reload decisions
STOP                 # stop signal
TRAINING_LOCK        # mutual-exclusion marker during learner update / publication
```

Common shard fields include `obs`, `next_obs`, `reward`, `done`, `terminated`, `truncated`, `old_logp`, `replay`, and `meta`. Any change to shard schema, checkpoint naming, or version behavior should be reviewed across actors, learners, I/O, batch construction, and Lightning lifecycle code.

## Repository Map

- [`framework/`](framework/README.md): core training framework, including runner, rollout, I/O, batch, algorithms, Lightning, agent, and environment-wrapper modules.
- [`script/`](script/): training, evaluation, debugging, pipeline entrypoints, and YAML configurations.
- [`tools/`](tools/): visualization, video generation, diagnostics, and auxiliary scripts.
- [`reconsimulator/`](reconsimulator/): ReconSimulator-related environment implementation.
- [`policy/`](policy/): policy-side code and model-related components.
- [`egoADs/`](egoADs/): external policy adapters and code for DiffusionDriveV2, SparseDrive, and SparseDriveV2.
- `assets/`: local asset entrypoint, usually a symbolic link to shared storage.
- `outputs/`: local output entrypoint, usually a symbolic link to shared storage.

## Framework Modules

The [`framework/`](framework/README.md) package is the core of the training system. Each subpackage contains a more detailed README.

- [`framework/runner/`](framework/runner/README.md): configuration normalization, actor / learner / orchestrator processes, GPU assignment, and runtime object construction.
- [`framework/agent/`](framework/agent/README.md): policy-model adapters for action sampling, replay serialization, log-probability recomputation, and checkpoint I/O.
- [`framework/env_wrapper/`](framework/env_wrapper/README.md): Recon / HUGSIM environment wrappers, scene sampling, termination conditions, collision metadata, and reward integration.
- [`framework/rollout/`](framework/rollout/README.md): actor-side rollout collection and trajectory-to-shard packaging.
- [`framework/io/`](framework/io/README.md): buffer, shard, STOP marker, TRAINING_LOCK marker, weight versioning, and atomic checkpoint saves.
- [`framework/batch/`](framework/batch/README.md): shard-to-training-batch conversion, returns, GAE, and advantage normalization.
- [`framework/algorithms/`](framework/algorithms/README.md): PPO, ReinforcePP, SAC-style, GRPO objectives, algorithm specifications, and nuScenes PDM / CRAFT counterfactual scorers.
- [`framework/lightning/`](framework/lightning/README.md): Lightning data modules, Lightning modules, training steps, optimizers, checkpoint publication, and WandB logging.
- [`framework/rewards/`](framework/rewards/README.md): path-based tracking rewards, collision penalties, comfort terms, and terminal penalties.
- [`framework/rewardmodel/`](framework/rewardmodel/README.md): structured reward / scorer types and configuration.
- [`framework/utils/`](framework/utils/README.md): repository path utilities, observation helpers, gsplat utilities, HUGSIM execution helpers, and nuScenes token utilities.

## Extension and Debugging Notes

Common extension points:

- **Add a new policy model**: implement the Agent protocol in [`framework/agent/base.py`](framework/agent/base.py) and register the adapter in [`framework/runner/agent_factory.py`](framework/runner/agent_factory.py).
- **Add a new reward**: extend [`framework/rewards/`](framework/rewards/README.md) or the reward configuration, then integrate it through [`framework/env_wrapper/rl_wrapper.py`](framework/env_wrapper/rl_wrapper.py).
- **Add a new algorithm**: define the objective or specification under [`framework/algorithms/`](framework/algorithms/README.md), and ensure that [`framework/lightning/trajectory_module.py`](framework/lightning/trajectory_module.py) can call it.
- **Add a new environment backend**: implement `reset` / `step` semantics under [`framework/env_wrapper/`](framework/env_wrapper/README.md), then construct it through [`framework/runner/env_factory.py`](framework/runner/env_factory.py).
- **Modify configuration fields**: update the training entrypoint, [`framework/runner/config_normalization.py`](framework/runner/config_normalization.py), related factories, and representative YAML files together.

Common diagnostics:

- **HUGSIM cannot be found**: check `HUGSIM_ROOT`, `env.hugsim.repo`, and whether HUGSIM-ORI is placed alongside ReconSparse.
- **Scenario or 3DGS assets cannot be found**: check `env.hugsim.scenario_dir`, `model_base`, `nuscenes_root`, `frame2token_dir`, and local symbolic links.
- **gsplat / CUDA extension compilation stalls**: clear the local torch extension cache, for example `.cache/torch_extensions/`.
- **Actors wait indefinitely for weights**: check that the learner writes `weights/latest.ckpt` and `weights/version.txt`.
- **The learner waits indefinitely for shards**: inspect `buffer/shards/`, actor logs, scene configuration, and environment startup.
- **Log-probability or replay mismatch**: inspect replay serialization in the agent and log-probability recomputation in [`framework/algorithms/trajectory_policy_core.py`](framework/algorithms/trajectory_policy_core.py).

## Documentation Index

- [`framework/README.md`](framework/README.md): actor-learner framework overview.
- [`framework/runner/README.md`](framework/runner/README.md): training runtime and process orchestration.
- [`framework/agent/README.md`](framework/agent/README.md): policy adapter layer.
- [`framework/env_wrapper/README.md`](framework/env_wrapper/README.md): Recon / HUGSIM environment wrappers.
- [`framework/io/README.md`](framework/io/README.md): buffer, shard, weight versioning, and STOP-file protocol.
- [`framework/batch/README.md`](framework/batch/README.md): shard-to-training-batch conversion.
- [`framework/algorithms/README.md`](framework/algorithms/README.md): PPO / ReinforcePP / SAC-style / GRPO objectives and nuScenes scorers.
- [`framework/lightning/README.md`](framework/lightning/README.md): Lightning learner lifecycle.
- [`framework/rewards/README.md`](framework/rewards/README.md): reward computation and debugging.

For broader environment-configuration context, consult [GigaAI-research/ReconDreamer-RL](https://github.com/GigaAI-research/ReconDreamer-RL) and [swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2).

The canonical training entrypoint is [`script/train_actor_learner_v2.py`](script/train_actor_learner_v2.py). When modifying the actor-learner path, preserve the shard protocol, runtime files, checkpoint/version publication, and actor reload behavior.

## Acknowledgement

ReconSparse builds on ideas, environments, and policy infrastructures from the broader autonomous driving, closed-loop simulation, and reconstructed-scene learning communities. Please refer to the upstream projects listed above for additional environment setup details and policy-specific implementation context.
