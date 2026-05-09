from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from framework.algorithms.ppo import PPO
from framework.algorithms.reinforcepp import ReinforcePP
from framework.utils.repo_paths import resolve_repo_path


class ValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 18, 64, 64)
            n_flat = int(self.conv(dummy).view(1, -1).shape[1])
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, 512), nn.ReLU(inplace=True))
        self.v = nn.Linear(512, 1)

    def forward(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.v(self.fc(self.conv(obs_t))).squeeze(-1)


class ValueHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.expects_value_features = True
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _resolve_policy_lr(train_cfg: Dict[str, Any], agent: Any) -> float:
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is not None and len(getattr(optimizer, "param_groups", [])) > 0:
        try:
            return float(optimizer.param_groups[0].get("lr", 1e-5))
        except Exception:
            pass
    return float(train_cfg.get("policy_lr", train_cfg.get("ddv2_lr", 1e-5)))


def _trainable_parameters(module: Any) -> List[torch.nn.Parameter]:
    if module is None or not hasattr(module, "parameters"):
        return []
    return [param for param in module.parameters() if getattr(param, "requires_grad", False)]


def _resolve_optional_repo_path(value: Any) -> str | None:
    if value is None:
        return None
    return str(resolve_repo_path(str(value)))


def build_algorithm_bundle(
    cfg: Dict[str, Any],
    *,
    agent: Any,
    device: torch.device,
    ddp_enabled: bool,
    world_size: int,
    rank: int,
    process_group: Any | None = None,
) -> tuple[Any, Any | None, Dict[str, Any]]:
    train_cfg = cfg.get("train", {}) or {}
    algo_key = str(train_cfg.get("algo", "ppo")).strip().lower()
    if algo_key in {"reinforce++", "reinforce_pp", "reinforce_clip"}:
        algo_key = "reinforcepp"
    if algo_key in {"reinforce_vanilla", "vanilla_reinforce"}:
        algo_key = "reinforce"

    minibatch_size = int(train_cfg.get("minibatch_size", 16))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))
    clip_eps = float(train_cfg.get("clip_eps", 0.2))
    ppo_epochs = int(train_cfg.get("epochs", 2))
    vf_coef = float(train_cfg.get("vf_coef", 0.5))
    value_clip_eps = float((train_cfg.get("ppo", {}) or {}).get("value_clip_eps", 0.0))
    eta = float(train_cfg.get("eta", 1.0))
    replay_mode_idx = int(train_cfg.get("mode_idx", -1))
    ddp_seed = int(((train_cfg.get("ddp", {}) or {}).get("seed", 0)))
    grad_accum_steps = int(((train_cfg.get("ddp", {}) or {}).get("grad_accum_steps", 1)))
    rpp_cfg = (train_cfg.get("reinforcepp", {}) or {})
    ppo_cfg = (train_cfg.get("ppo", {}) or {})
    reinforce_cfg = (train_cfg.get("reinforce", {}) or {})
    policy_lr = _resolve_policy_lr(train_cfg, agent)
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    if algo_key in {"ppo", "ppo_kl", "ppo_dual_clip", "ppo_value_clip"}:
        policy_params = _trainable_parameters(getattr(agent, "trainable_module", None) or agent)
        if len(policy_params) == 0:
            raise RuntimeError("No trainable policy parameters found for PPO")
        use_agent_value_features = bool(
            train_cfg.get("critic_use_agent_features", True)
            and getattr(agent, "supports_value_features", lambda: False)()
            and getattr(agent, "value_feature_dim", None) is not None
        )
        if use_agent_value_features:
            value_hidden_dim = int(train_cfg.get("critic_hidden_dim", 256))
            value_net = ValueHead(
                input_dim=int(agent.value_feature_dim),
                hidden_dim=int(value_hidden_dim),
            ).to(device)
        else:
            value_net = ValueNet().to(device)
        if ddp_enabled and torch.cuda.is_available():
            value_net = DDP(
                value_net,
                device_ids=[int(device.index)] if device.index is not None else None,
                output_device=int(device.index) if device.index is not None else None,
                process_group=process_group,
                find_unused_parameters=False,
            )
        value_lr = float(train_cfg.get("lr_value", 1e-4))

        algo = PPO(
            value_net=value_net,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
            grad_accum_steps=grad_accum_steps,
            ddp_seed=ddp_seed,
            eta=eta,
            variant=algo_key,
            kl_coef=float(ppo_cfg.get("kl_coef", 0.0)) if algo_key == "ppo_kl" else 0.0,
            dual_clip=(float(ppo_cfg.get("dual_clip", 3.0)) if algo_key == "ppo_dual_clip" else None),
            value_clip_eps=(float(value_clip_eps) if algo_key in {"ppo", "ppo_value_clip"} else 0.0),
            policy_lr=float(policy_lr),
            value_lr=float(value_lr),
            weight_decay=float(weight_decay),
            forward_kl_coef=float(ppo_cfg.get("forward_kl_coef", 0.0)),
            reverse_kl_coef=float(ppo_cfg.get("reverse_kl_coef", 0.0)),
            distill_temperature=float(ppo_cfg.get("distill_temperature", 1.0)),
            teacher_ckpt=_resolve_optional_repo_path(ppo_cfg.get("teacher_ckpt", None)),
        )
    elif algo_key in {"reinforce++", "reinforce", "reinforcepp", "reinforce_vanilla", "grpo_only"}:
        policy_params = _trainable_parameters(getattr(agent, "trainable_module", None) or agent)
        if len(policy_params) == 0:
            raise RuntimeError("No trainable policy parameters found for policy-gradient learner")
        algo = ReinforcePP(
            clip_eps=clip_eps,
            kl_coef=float(reinforce_cfg.get("kl_coef", rpp_cfg.get("kl_coef", 0.0))) if algo_key in {"reinforce_kl", "reinforcepp"} else 0.0,
            epochs=int(reinforce_cfg.get("epochs", rpp_cfg.get("epochs", 1))),
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
            grad_accum_steps=grad_accum_steps,
            ddp_seed=ddp_seed,
            eta=eta,
            variant=algo_key,
            policy_lr=float(policy_lr),
            weight_decay=float(weight_decay),
            forward_kl_coef=float(reinforce_cfg.get("forward_kl_coef", rpp_cfg.get("forward_kl_coef", 0.0))),
            reverse_kl_coef=float(reinforce_cfg.get("reverse_kl_coef", rpp_cfg.get("reverse_kl_coef", 0.0))),
            distill_temperature=float(reinforce_cfg.get("distill_temperature", rpp_cfg.get("distill_temperature", 1.0))),
            teacher_ckpt=_resolve_optional_repo_path(
                reinforce_cfg.get("teacher_ckpt", rpp_cfg.get("teacher_ckpt", None))
            ),
        )
        value_net = None

    meta = {
        "algo_key": algo_key,
        "eta": eta,
        "mode_idx": replay_mode_idx,
        "clip_eps": clip_eps,
        "minibatch_size": minibatch_size,
        "max_grad_norm": max_grad_norm,
        "rpp_norm_eps": float(rpp_cfg.get("norm_eps", 1e-8)),
        "value_clip_eps": float(value_clip_eps),
        "critic_use_agent_features": bool(
            train_cfg.get("critic_use_agent_features", True)
            and getattr(agent, "supports_value_features", lambda: False)()
            and getattr(agent, "value_feature_dim", None) is not None
        ),
    }
    return algo, value_net, meta
