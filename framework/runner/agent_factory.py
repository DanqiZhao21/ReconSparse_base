from __future__ import annotations

import os
from typing import Any, Dict

import torch

from framework.utils.repo_paths import resolve_ego_ads_subdir, resolve_repo_path


def _normalize_policy_execute_mode(mode: Any) -> str:
    text = str(mode if mode is not None else "first_step").strip().lower().replace("-", "_")
    if text in {"", "continuous", "first_step", "step1", "traj_first_step"}:
        return "first_step"
    return "first_step"


def build_agent(cfg: Dict[str, Any], *, device: torch.device) -> Any:
    train_cfg = cfg.get("train", {}) or {}
    agent_cfg = cfg.get("agent", {}) or {}
    agent_type = str(agent_cfg.get("type", "ddv2")).strip().lower().replace("-", "_")
    ckpt_path = agent_cfg.get("ckpt", None)
    policy_execute_mode = _normalize_policy_execute_mode(
        train_cfg.get(
            "policy_execute_mode",
            train_cfg.get("ddv2_execute_mode", "continuous"),
        )
    )

    if agent_type in {"dummy", "test_dummy", "framework_dummy"}:
        from framework.agent.policy_dummy import DummyPolicy

        resolved_ckpt = resolve_repo_path(str(ckpt_path)) if ckpt_path is not None else None
        return DummyPolicy(
            ckpt_path=(str(resolved_ckpt) if resolved_ckpt is not None else None),
            device=str(device),
            rl_lr=float(train_cfg.get("policy_lr", train_cfg.get("ddv2_lr", 1e-3))),
        )

    if ckpt_path is None:
        raise RuntimeError("agent.ckpt is required")
    ckpt_path = resolve_repo_path(str(ckpt_path))

    if agent_type == "sparsedrive":
        from framework.agent.policy_sparsedrive import SparseDrivePolicy

        sparse_root = resolve_ego_ads_subdir("SparseDrive")
        config_path = agent_cfg.get("config", os.path.join(sparse_root, "projects", "configs", "sparsedrive_small_stage2.py"))
        config_path = resolve_repo_path(str(config_path))
        return SparseDrivePolicy(
            config_path=str(config_path),
            ckpt_path=str(ckpt_path),
            device=str(device),
            rl_lr=float(train_cfg.get("policy_lr", train_cfg.get("ddv2_lr", 1e-5))),
            execute_mode=policy_execute_mode,
        )
    if agent_type in {"sparsedrive_v2", "sparsedrivev2", "sdv2"}:
        from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy

        trainable_prefixes = agent_cfg.get("trainable_prefixes", [])
        frozen_prefixes = agent_cfg.get("frozen_prefixes", [])
        nuscenes_scorer_config = dict(agent_cfg.get("nuscenes_scorer", {}) or {})
        for key in ("scene_cache_root", "agent_state_cache_root", "ea_project_src", "nuscenes_dataroot"):
            if key in nuscenes_scorer_config and nuscenes_scorer_config[key] is not None:
                nuscenes_scorer_config[key] = str(resolve_repo_path(str(nuscenes_scorer_config[key])))
        return SparseDriveV2Policy(
            ckpt_path=str(ckpt_path),
            device=str(device),
            rl_lr=float(train_cfg.get("policy_lr", train_cfg.get("ddv2_lr", 1e-5))),
            execute_mode=policy_execute_mode,
            trainable_prefixes=trainable_prefixes,
            frozen_prefixes=frozen_prefixes,
            nuscenes_scorer_config=nuscenes_scorer_config,
        )
    from framework.agent.policy_diffusiondrivev2 import DiffusionDriveV2Policy

    return DiffusionDriveV2Policy(
        ckpt_path=str(ckpt_path),
        device=str(device),
        rl_lr=float(train_cfg.get("ddv2_lr", 1e-5)),
        execute_mode=policy_execute_mode,
    )
