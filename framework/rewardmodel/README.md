# framework/rewardmodel

This package is a standalone reward-model subsystem for scoring candidate
driving trajectories. It is inspired by DreamerAD's autoregressive dense reward
model, but intentionally differs in one important way: it does not require a
latent world model. The model directly consumes image observations from the
dataset or environment, ego-state features, and candidate trajectories.

The current model is image-only and uses query-compressed image tokens rather
than a single global image vector. Candidate trajectory prefixes, horizon
embeddings, ego features, and metric-specific reward queries attend over those
image tokens to predict dense step-wise reward metrics.

## Inputs And Outputs

The main model is `ObservationTrajectoryRewardModel`.

Inputs:

- `observations`: tensor `[B, C, H, W]`, where `C` can be current-frame cameras
  or history-stacked cameras.
- `ego_states`: tensor `[B, E]` with current/history ego features chosen by the
  dataset builder.
- `candidate_trajectories`: tensor `[B, G, T, 3]` containing candidate future
  `(x, y, yaw)` trajectories.

Outputs:

- `metric_logits`: `[B, G, H, 8]`
- `metric_scores`: `[B, G, H, 8]`
- `horizon_score`: `[B, G, H]`
- `final_score`: `[B, G]`

The eight metrics follow the paper naming:

`rnc, rdac, rddc, rtlc, rep, rttc, rlk, rhc`

The first four are treated as safety metrics and the last four as task metrics.
Aggregation uses a safety-first log formulation so safety failures strongly
lower the final score.

## Training Data

The included trainer expects cached `.pt` samples containing:

- `image_paths`: list of camera image paths, ordered by frame then camera
- `ego_states`: `[E]`
- `candidate_trajectories`: `[G, T, 3]`
- `targets`: `[G, H, 8]`
- `valid_mask`: optional `[G, H, 8]`

Images are loaded by `CachedRewardModelDataset` at training time and converted
to `[C, H, W]` tensors. The cache intentionally stores paths rather than image
tensors so full-split caches stay small and resize/augmentation choices can be
changed without rebuilding PDM labels.

Teacher labels can come from NavSim/PDM. If the teacher can emit the full eight
metrics, use those directly. If only a scalar PDM score is available, the helper
`normalize_teacher_scores` can broadcast it across horizons and metrics for
bootstrap smoke training, but that fallback is less informative than true
metric-level supervision.

## Train

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL \
python -m framework.rewardmodel.training.train_reward_model \
  --data-root /path/to/cached_reward_samples \
  --output /path/to/reward_model.pt \
  --observation-channels 18 \
  --ego-state-dim 8 \
  --image-height 256 \
  --image-width 448 \
  --num-observation-queries 32 \
  --num-attention-heads 4
```

## Build NavSim Reward Cache

If you already have NavSim metric cache and a large candidate trajectory
vocabulary file, you can
build rewardmodel `.pt` samples directly:

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL/egoADs/SparseDriveV2:/root/clone/ReconDreamer-RL \
python -m framework.rewardmodel.data.build_navsim_reward_cache \
  --navsim-root /OpenDataset/navsim/dataset \
  --split trainval \
  --metric-cache-path /path/to/metric_cache_navtrainv2 \
  --candidate-path /path/to/trajectory_vocabulary.npy \
  --output-root /path/to/reward_cache
```

Or build the large vocabulary directly from NavSim GT futures:

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL/egoADs/SparseDriveV2:/root/clone/ReconDreamer-RL \
python -m framework.rewardmodel.data.build_navsim_reward_cache \
  --navsim-root /OpenDataset/navsim/dataset \
  --split trainval \
  --metric-cache-path /path/to/metric_cache_navtrainv2 \
  --build-vocabulary-from-gt \
  --max-vocabulary-size 8192 \
  --output-root /path/to/reward_cache
```

This script reads image paths and ego state from NavSim, reads the matching
metric cache, filters the large trajectory vocabulary per token using that
scene's GT future endpoint, evaluates the filtered candidates with the NavSim
PDM teacher at each trajectory prefix horizon, maps the teacher metrics into the
internal 8-dim schema, and saves one `.pt` sample per token. The dense target
shape is `[G, H, 8]`, where `H` is usually eight 0.5s horizons over four
seconds.

## Frozen Inference

```python
from framework.rewardmodel.inference import FrozenRewardModelScorer

scorer = FrozenRewardModelScorer.from_checkpoint("/path/to/reward_model.pt", device="cuda:0")
out = scorer.score(
    observations=observations,
    ego_states=ego_states,
    candidate_trajectories=candidate_trajectories,
)
reward = out.final_score
```

This package does not modify actor-learner shard schemas, checkpoint publishing,
or runtime launch behavior. It is designed to be trained offline and frozen
before being integrated into closed-loop reward code.
