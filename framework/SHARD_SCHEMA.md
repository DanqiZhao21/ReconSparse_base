# Shard Schema 说明

本文档说明 actor-learner 主路径中 shard 的数据结构，以及每条 replay 的当前结构化格式。目标读者是需要更换闭环算法、接入新 egoAD/self-driving agent，或者排查 buffer shard 兼容性问题的开发者。

当前主路径是：

```text
actor -> rollout collector -> shard buffer -> learner -> training batch -> policy update
```

shard 是 actor 写入文件缓冲区的最小训练数据包；replay 是 shard 中每个 step 对应的策略重放信息。

## 总体关系

一个 shard 保存一段 rollout，长度为 `T`：

```python
shard = {
    "old_logp": Tensor[T],
    "reward": Tensor[T],
    "done": Tensor[T],
    "terminated": Tensor[T],
    "truncated": Tensor[T],
    "done_last": Tensor[()],
    "terminated_last": Tensor[()],
    "replay": list[dict],  # len == T
    "meta": dict,

    # optional
    "obs": Tensor[T, ...],
    "next_obs": Tensor[...],
    "next_value_feature": Tensor[F],
}
```

索引必须对齐：

```text
old_logp[i]  <-> reward[i] <-> done[i] <-> replay[i]
```

learner 会把多个 shard 拼成一个 batch，并把 `batch["replay"]` 原样传回 agent：

```python
agent.logp_from_replay_batch(batch["replay"])
```

因此，框架层不应该理解具体 egoAD 的 policy replay 细节；具体字段由 agent adapter 负责。

## Shard 外层字段

### 必需字段

`old_logp: Tensor[T]`

Actor 采样动作时旧策略下的 log probability。PPO 和 ReinforcePP 用它计算 ratio/clip/KL；SAC 当前实现可用它做 ratio/KL 统计；普通 Reinforce 可以不依赖它，但 collector 当前仍统一保存。

`reward: Tensor[T]`

环境每一步返回的 reward，float32。

`done: Tensor[T]`

每一步是否结束 episode。通常是 `terminated or truncated`。

`terminated: Tensor[T]`

环境真实终止，例如失败、碰撞、任务结束等。PPO bootstrap 时用它判断是否允许从最后状态估值。

`truncated: Tensor[T]`

时间截断或 rollout 截断。它会使 `done=True`，但语义上不同于真实 terminated。

`done_last: Tensor[()]`

shard 最后一步之后的 done 状态。

`terminated_last: Tensor[()]`

shard 最后一步之后是否真实 terminated。PPO GAE 用它判断最后状态是否可以 bootstrap。

`replay: list[dict]`

长度必须等于 `T`。每个元素是一个 step 的结构化 replay，详见下文。

`meta: dict`

运行和统计信息。典型字段：

```python
meta = {
    "actor_id": int,
    "env_id": int,
    "horizon": int,
    "num_steps": int,
    "weights_version": int,
    "time": float,
    "shard_idx": int,
    "timing": dict,
    "reward_summary": dict,
}
```

单环境且 `end_shard_on_done=True` 时，还可能包含：

```python
"needs_reset_after": bool
```

### 可选字段

`obs: Tensor[T, ...]`

只在 PPO fallback critic 需要原始 observation feature 时保存。若配置使用 agent replay feature critic，则不需要保存。

`next_obs: Tensor[...]`

PPO fallback critic 的最后 bootstrap 状态。

`next_value_feature: Tensor[F]`

PPO replay-feature critic 的最后 bootstrap feature。由 actor 侧调用：

```python
agent.value_features_from_observation_batch([next_observation])
```

得到。

## Replay 内部结构

当前 replay 使用 schema version 2：

```python
replay = {
    "schema_version": 2,
    "policy": {...},
    "env": {...},
    "grpo": {...},
    "aux": {...},
    "debug": {...},
}
```

只有 `policy` 是普通闭环训练必需。其他分区按功能启用。

### policy

`policy` 是 agent 重算 logp 的唯一主入口。框架不解析其中模型细节。

```python
"policy": {
    "backend": "sparsedrive_v2",
    "schema_version": 1,
    "model_inputs": dict,
    "action_id": dict,
    "extra": dict,  # optional
}
```

`backend`

标识 replay 属于哪个 egoAD adapter，例如 `"sparsedrive_v2"`。新 agent 应设置自己的 backend 名称。

`model_inputs`

重算 `new_logp` 所需的最小模型输入。不同 egoAD 可以完全不同。例如 SparseDriveV2 当前使用：

```python
"model_inputs": {
    "camera_feature": dict[str, Tensor],
    "status_feature": Tensor[1, D],
}
```

`action_id`

标识 actor 当时选中的动作。它应该是稳定的动作身份，而不是仅用于 debug 的路径编号。例如 SparseDriveV2 当前使用：

```python
"action_id": {
    "global_mode_idx": int,
}
```

新 agent 接入时必须保证：

```python
agent.logp_from_replay_batch(replays)
```

只依赖 `replay["policy"]` 就能在当前参数下重算同一个动作的 logp。

### env

`env` 只服务环境执行，不服务 policy loss。

```python
"env": {
    "plan_xyyaw": Tensor[H, 3],
    "plan_frame": "ego",
    "dt_s": float,       # optional
    "action_flag": int,  # optional
}
```

`plan_xyyaw`

完整局部轨迹计划。collector 会读取它并调用环境的：

```python
set_external_plan_local_xyyaw(plan_xyyaw)
```

如果某个环境只执行 `action` 的 first step，可以不依赖此字段。但 HUGSIM/Recon closed-loop 路径通常需要保留它。

### grpo

`grpo` 只在启用 GRPO 或 GRPO debug/scorer 时需要。当前 GRPO 语义是 actor-stored candidate clipped-ratio GRPO，因此候选和 scorer context 放在同一个分区。

```python
"grpo": {
    "candidates": {
        "mode_indices": Tensor[K],
        "old_log_probs": Tensor[K],
        "traj_xyyaw": Tensor[K, H, 3],
        "score_logits": Tensor[K],  # optional
    },
    "scorer": {
        "sample_token": str,
        "scene_id": int,   # optional
        "frame_idx": int,  # optional
        "gt_sample_token_override": str,  # optional
        "gt_frame_idx_override": int,     # optional
        "future_dt_s": float,             # optional
        "object_context": dict,           # optional
    },
}
```

`grpo.candidates.mode_indices`

Actor 侧采样出的候选动作身份。learner 用当前策略重新计算这些候选的 new logp。

`grpo.candidates.old_log_probs`

Actor 采样时旧策略对这些候选的 logp。GRPO ratio 必需。

`grpo.candidates.traj_xyyaw`

候选轨迹，供 PDM/Craft scorer 打分。

`grpo.candidates.score_logits`

可选。当前主要给 auxiliary loss 或 debug 使用；GRPO ratio 本身不依赖它。

`grpo.scorer.sample_token`

NuScenes/PDM/Craft scorer 的核心上下文。启用 scorer 时缺失会报错。

`grpo.scorer.object_context`

可选动态物体上下文。若需要 replay-scoped object context，可包含：

```python
"object_context": {
    "scene_objects": list,
    "ea_agent_states": list,
    "ttc_agent_states": list,
}
```

### aux

`aux` 只服务辅助 loss，不应该被普通 PPO/Reinforce/SAC 主 loss 依赖。

当前 risk-decel auxiliary 使用：

```python
"aux": {
    "front_obstacle": {
        "available": bool,
        "gap_m": float,
        "lateral_m": float,
        "closing_speed_mps": float,
        "ttc_s": float,
        "category": str,
    },
    "ego_speed_mps": float,  # optional
}
```

若 `ego_speed_mps` 缺失，当前实现会尝试从 `policy.model_inputs.status_feature` 中估计速度。

### debug

`debug` 只放分析和排查字段。训练主逻辑不应依赖 debug 字段。

SparseDriveV2 当前可能写入：

```python
"debug": {
    "selected_path_idx": int,
    "selected_vel_idx": int,
    "candidate_scores": Tensor,
    "execute_mode": str,
    "feature_missing_fields": list[str],
    "timestamp_s": float,
}
```

新 agent 不应把 logp 重算必需字段放在 `debug`。必需字段必须放在 `policy`。

## 不同算法需要哪些内容

### Reinforce

必需：

```text
reward, done, replay[i].policy
```

说明：`reward/done` 用于计算 return；`policy` 用于重算 `new_logp`。

### ReinforcePP / clipped policy gradient

必需：

```text
old_logp, reward, done, replay[i].policy
```

说明：`old_logp` 用于 ratio/clip/KL；`policy` 用于重算 `new_logp`。

### SAC-style policy objective

必需：

```text
reward, done, replay[i].policy
```

可选：

```text
old_logp
```

说明：当前 SAC-style 实现用 `new_logp` 和 advantage；若有 `old_logp`，会计算 ratio/KL/clip 统计。

### PPO

必需：

```text
old_logp, reward, done, terminated, replay[i].policy
```

并且还需要 critic 输入，二选一：

```text
obs + next_obs
```

或：

```text
agent.value_features_from_replay_batch(replay)
next_value_feature
```

若使用 fallback critic 且 shard 缺 `obs/next_obs`，batch 构建会报错。

### GRPO / grpo_only

必需：

```text
replay[i].policy
replay[i].grpo.candidates.mode_indices
replay[i].grpo.candidates.old_log_probs
replay[i].grpo.candidates.traj_xyyaw
replay[i].grpo.scorer.sample_token
```

说明：当前 GRPO 只保留 actor-stored candidate clipped-ratio 语义。候选缺字段会直接报错。

### Auxiliary risk-decel

必需或建议：

```text
replay[i].aux.front_obstacle
replay[i].aux.ego_speed_mps 或 policy.model_inputs.status_feature
```

若没有前车上下文，该样本不会被判为 high-risk。

## 接入新 egoAD / agent 的要求

新 agent adapter 至少需要实现：

```python
act(obs) -> action, old_logp, replay
act_batch(obs_list) -> actions, old_logps, replays
logp_from_replay_batch(replays) -> Tensor[B]
replay_is_compatible(replay) -> bool
```

普通闭环训练最小 replay：

```python
replay = {
    "schema_version": 2,
    "policy": {
        "backend": "<your_agent>",
        "schema_version": 1,
        "model_inputs": {...},
        "action_id": {...},
    },
}
```

如果环境需要完整轨迹执行，再加：

```python
"env": {
    "plan_xyyaw": Tensor[H, 3],
    "plan_frame": "ego",
}
```

如果启用 GRPO，再加：

```python
"grpo": {
    "candidates": {
        "mode_indices": Tensor[K],
        "old_log_probs": Tensor[K],
        "traj_xyyaw": Tensor[K, H, 3],
    },
    "scorer": {
        "sample_token": str,
    },
}
```

如果启用 PPO replay-feature critic，需要：

```python
supports_value_features() -> True
value_feature_dim -> int
value_features_from_replay_batch(replays) -> Tensor[B, F]
value_features_from_observation_batch(obs_list) -> Tensor[B, F]
```

## 兼容性与报错

当前设计倾向 fail-fast：

- replay 缺 `policy`，agent 应判定不兼容或直接报错。
- GRPO 缺 `grpo.candidates` 或 `old_log_probs`，learner 会报错。
- scorer 缺 `grpo.scorer.sample_token`，NuScenes/PDM/Craft scorer 会报错。
- PPO fallback critic 缺 `obs/next_obs`，batch 构建会报错。

learner 选择 shard 时会调用：

```python
agent.replay_is_compatible(replay)
```

不兼容 shard 会被 buffer policy 过滤到 consumed。若希望训练严格中断，应在调用路径上改成 fail-fast 策略。

## 检查 shard 的最小脚本

```python
import torch

shard = torch.load("path/to/shard.pt", map_location="cpu")

T = int(shard["reward"].numel())
assert shard["old_logp"].shape[0] == T
assert shard["done"].shape[0] == T
assert len(shard["replay"]) == T

for replay in shard["replay"]:
    assert replay["schema_version"] == 2
    assert "policy" in replay
    assert "model_inputs" in replay["policy"]
    assert "action_id" in replay["policy"]
```

如果启用 GRPO：

```python
for replay in shard["replay"]:
    candidates = replay["grpo"]["candidates"]
    scorer = replay["grpo"]["scorer"]
    assert "mode_indices" in candidates
    assert "old_log_probs" in candidates
    assert "traj_xyyaw" in candidates
    assert "sample_token" in scorer
```

## 设计原则

1. Shard 外层只存 RL transition 和 batch 构建所需字段。
2. `replay["policy"]` 是 agent 私有重放协议，框架不解析具体模型输入。
3. `replay["env"]` 只服务环境执行。
4. `replay["grpo"]` 只服务 GRPO 候选和 scorer。
5. `replay["aux"]` 只服务辅助 loss。
6. `replay["debug"]` 不参与训练主逻辑。

遵守这些边界后，更换 egoAD 时只需要重新定义 `policy.model_inputs` 和 `policy.action_id`，而不需要改动 shard 外层协议。
