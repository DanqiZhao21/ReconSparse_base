from __future__ import annotations

import pytest
import torch
import yaml

from framework.lightning.config import resolve_auxiliary_objectives_config, resolve_grpo_config
from framework.runner.agent_factory import build_agent
from framework.runner.config_normalization import normalize_actor_learner_cfg


def test_new_sparsedrive_v2_template_uses_train_owned_grpo_scorer() -> None:
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml"
    )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "kind" not in cfg["train"]["closed_loop"]

    normalize_actor_learner_cfg(cfg)

    assert cfg["train"]["algo"] == "reinforcepp"
    assert cfg["train"]["algo_spec"] == "reinforcepp-nogrpo-noaux"
    assert cfg["train"]["grpo"]["enable"] is False
    assert cfg["train"]["grpo"]["coef"] == 0.0
    assert cfg["train"]["auxiliary"]["enable"] is False
    assert cfg["train"]["grpo"]["scorer"]["backend"] == "craft_carl"
    assert "reinforcepp" in cfg["train"]["closed_loop"]
    assert "ppo" in cfg["train"]["closed_loop"]
    assert "sac" in cfg["train"]["closed_loop"]
    assert "reinforcepp" not in cfg["train"]
    assert "ppo" not in cfg["train"]
    assert "sac" not in cfg["train"]
    assert "nuscenes_scorer" not in cfg["agent"]


def test_train_algo_dsl_derives_algorithm_switches() -> None:
    cfg = {
        "train": {
            "algo": "reinforcepp-grpo-aux",
            "grpo": {"coef": 0.4, "num_candidates": 8},
            "auxiliary": {"risk_decel": {"coef": 0.7}},
            "actor_learner": {"num_envs_per_actor": 1},
        }
    }

    normalize_actor_learner_cfg(cfg)

    train_cfg = cfg["train"]
    assert train_cfg["algo"] == "reinforcepp"
    assert train_cfg["algo_spec"] == "reinforcepp-grpo-aux"
    assert train_cfg["closed_loop"]["kind"] == "reinforcepp"
    assert train_cfg["grpo"]["enable"] is True
    assert train_cfg["grpo"]["coef"] == 0.4
    assert train_cfg["auxiliary"]["enable"] is True
    assert train_cfg["auxiliary"]["risk_decel"]["enable"] is True
    assert resolve_grpo_config(train_cfg)["enabled"] is True
    assert resolve_auxiliary_objectives_config(train_cfg)["risk_decel_enabled"] is True


def test_train_algo_dsl_disables_grpo_and_auxiliary_with_zeroed_loss_weights() -> None:
    cfg = {
        "train": {
            "algo": "sac-nogrpo-noaux",
            "grpo": {"enable": True, "coef": 99.0, "num_candidates": 8},
            "auxiliary": {
                "enable": True,
                "risk_decel": {"enable": True, "coef": 99.0},
            },
            "actor_learner": {"num_envs_per_actor": 1},
        }
    }

    normalize_actor_learner_cfg(cfg)

    train_cfg = cfg["train"]
    assert train_cfg["algo"] == "sac"
    assert train_cfg["closed_loop"]["kind"] == "sac"
    assert train_cfg["grpo"]["enable"] is False
    assert train_cfg["grpo"]["coef"] == 0.0
    assert train_cfg["grpo"]["num_candidates"] == 0
    assert train_cfg["auxiliary"]["enable"] is False
    assert train_cfg["auxiliary"]["risk_decel"]["enable"] is False
    assert resolve_grpo_config(train_cfg)["coef"] == 0.0
    assert resolve_auxiliary_objectives_config(train_cfg)["risk_decel_coef"] == 0.0


def test_train_algo_dsl_rejects_ambiguous_specs() -> None:
    cfg = {
        "train": {
            "algo": "reinforcepp-ppo-grpo-noaux",
            "actor_learner": {"num_envs_per_actor": 1},
        }
    }

    with pytest.raises(ValueError, match="closed-loop algorithm"):
        normalize_actor_learner_cfg(cfg)


def test_train_algo_dsl_accepts_reinforce_plus_plus_alias() -> None:
    cfg = {
        "train": {
            "algo": "reinforce++-nogrpo-noaux",
            "actor_learner": {"num_envs_per_actor": 1},
        }
    }

    normalize_actor_learner_cfg(cfg)

    assert cfg["train"]["algo"] == "reinforcepp"


def test_train_algo_dsl_keeps_zero_coef_auxiliary_disabled() -> None:
    cfg = {
        "train": {
            "algo": "reinforcepp-nogrpo-aux",
            "auxiliary": {"risk_decel": {"coef": 0.0}},
            "actor_learner": {"num_envs_per_actor": 1},
        }
    }

    normalize_actor_learner_cfg(cfg)

    assert cfg["train"]["auxiliary"]["enable"] is True
    assert cfg["train"]["auxiliary"]["risk_decel"]["enable"] is False


def test_build_sparsedrive_v2_agent_uses_train_grpo_scorer(monkeypatch) -> None:
    captured = {}

    class FakeSparseDriveV2Policy:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "framework.agent.policy_sparsedrive_v2.SparseDriveV2Policy",
        FakeSparseDriveV2Policy,
    )

    cfg = {
        "train": {
            "policy_lr": 1.0e-5,
            "algo": "reinforcepp-grpo-noaux",
            "grpo": {
                "enable": True,
                "num_candidates": 6,
                "scorer": {
                    "backend": "craft_carl",
                    "scene_cache_root": "assets/nus/cache",
                    "carl": {"w_prog": 10.0},
                },
            },
        },
        "agent": {
            "type": "sparsedrive_v2",
            "ckpt": "missing-but-not-loaded.ckpt",
            "trainable_prefixes": [],
            "frozen_prefixes": ["_backbone"],
        },
    }

    normalize_actor_learner_cfg(cfg)
    build_agent(cfg, device=torch.device("cpu"))

    assert captured["grpo_num_candidates"] == 6
    assert captured["nuscenes_scorer_config"]["backend"] == "craft_carl"
    assert "nuscenes_scorer" not in cfg["agent"]


def test_build_algorithm_bundle_accepts_unnormalized_algo_dsl() -> None:
    from framework.runner.learner_factory import build_algorithm_bundle

    class Agent:
        def __init__(self) -> None:
            self.trainable_module = torch.nn.Linear(1, 1)

    cfg = {
        "train": {
            "algo": "reinforcepp-grpo-noaux",
            "policy_lr": 1.0e-5,
            "clip_eps": 0.2,
            "minibatch_size": 4,
            "closed_loop": {"reinforcepp": {"epochs": 3, "norm_eps": 1.0e-7}},
            "grpo": {"coef": 0.2, "num_candidates": 8},
        }
    }

    _algo, _value_net, meta = build_algorithm_bundle(
        cfg,
        agent=Agent(),
        device=torch.device("cpu"),
        ddp_enabled=False,
        world_size=1,
        rank=0,
    )

    assert cfg["train"]["algo"] == "reinforcepp"
    assert cfg["train"]["grpo"]["enable"] is True
    assert meta["rpp_norm_eps"] == 1.0e-7
    assert getattr(_algo, "epochs") == 3
    assert meta["algo_key"] == "reinforcepp"


def test_train_eval_pipeline_tags_use_normalized_algo_dsl() -> None:
    from script.train_eval_pipeline import build_run_tags

    cfg = {
        "env": {"reward": {"mode": "step_path"}},
        "train": {
            "algo": "reinforcepp-grpo-aux",
            "grpo": {
                "coef": 0.2,
                "num_candidates": 8,
                "scorer": {"ea_gate_enabled": True},
            },
            "auxiliary": {"risk_decel": {"coef": 0.3}},
        },
    }

    tags = build_run_tags(config=cfg, algo_tag="demo")

    assert "ReinforcePP" in tags
    assert "GRPO" in tags
    assert "EA" in tags
    assert cfg["train"]["algo"] == "reinforcepp-grpo-aux"
