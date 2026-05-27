from __future__ import annotations

from pathlib import Path

import yaml

from framework.runner.config_normalization import normalize_actor_learner_cfg


TEMPLATE_CONFIGS = [
    "20260519_reinforcepp_closed_loop_sparsedrive_v2_craft_onlyGRPO.yaml",
    "202605211154_reinforcepp_closed_loop_sparsedrive_v2_craft_closeCorrection_openGrpo.yaml",
    "202605211155_reinforcepp_closed_loop_sparsedrive_v2_craft_closeNo_openGRPO.yaml",
]

FOUR_WAY_20260524_CONFIGS = {
    "202605241004_reinforcepp_closed_loop_sparsedrive_v2_craft_closeCorrection_openGRPOCraft-FullPara.yaml": {
        "algo": "reinforcepp",
        "craft_enabled": True,
        "corrective_progress_enabled": True,
        "trainable_prefixes": [],
        "frozen_prefixes": [],
        "run_key": "CloseCorrection_OpenGRPOCraft_FullPara",
    },
    "202605241004_reinforcepp_closed_loop_sparsedrive_v2_craft_closeCorrection_openGRPOCraft-MetricsPara.yaml": {
        "algo": "reinforcepp",
        "craft_enabled": True,
        "corrective_progress_enabled": True,
        "trainable_prefixes": ["_trajectory_head.decoder.layers.1.metric_heads"],
        "frozen_prefixes": ["_backbone"],
        "run_key": "CloseCorrection_OpenGRPOCraft_MetricsPara",
    },
    "202605241004_reinforcepp_closed_loop_sparsedrive_v2_craft_closeNo_openGRPOCraft-FullPara.yaml": {
        "algo": "grpo_only",
        "craft_enabled": False,
        "corrective_progress_enabled": False,
        "trainable_prefixes": [],
        "frozen_prefixes": [],
        "run_key": "CloseNo_OpenGRPOCraft_FullPara",
    },
    "202605241004_reinforcepp_closed_loop_sparsedrive_v2_craft_closeNo_openGRPOCraft-MetricsPara.yaml": {
        "algo": "grpo_only",
        "craft_enabled": False,
        "corrective_progress_enabled": False,
        "trainable_prefixes": ["_trajectory_head.decoder.layers.1.metric_heads"],
        "frozen_prefixes": ["_backbone"],
        "run_key": "CloseNo_OpenGRPOCraft_MetricsPara",
    },
}

TWO_WAY_20260525_CLOSELOOP_CONFIGS = {
    "202605250049_reinforcepp_closed_loop_sparsedrive_v2_craft_closeCloseloop_openGRPOCraft-FullPara.yaml": {
        "run_key": "CloseCloseloop_OpenGRPOCraft_FullPara",
        "trainable_prefixes": [],
    },
    "202605250049_reinforcepp_closed_loop_sparsedrive_v2_craft_closeCloseloop_openGRPOCraft-MetricsPara.yaml": {
        "run_key": "CloseCloseloop_OpenGRPOCraft_MetricsPara",
        "trainable_prefixes": ["_trajectory_head.decoder.layers.1.metric_heads"],
    },
}

HUGSIM_ORI_20260525_CONFIG = (
    "202605251610_HUGSM_reinforcepp_closed_loop_closeCloselopop_openGRPOCraft-try.yaml"
)


def _key_paths(obj: object, prefix: str = "") -> set[str]:
    if not isinstance(obj, dict):
        return set()
    out: set[str] = set()
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        out.add(path)
        out.update(_key_paths(value, path))
    return out


def test_craft_training_yaml_uses_clean_corrective_and_carl_sections() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_craft.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    craft_reward = cfg["env"]["reward"]["CRAFT"]
    assert craft_reward["real_reward_model"] == "corrective"
    assert craft_reward["corrective"] == {
        "cost_off_road": 0.5,
        "cost_emergency_lane": 0.2,
        "cost_off_global_route": 0.5,
        "cost_red_light": 2.0,
        "cost_stop_sign": 2.0,
        "cost_collision": 5.0,
    }

    scorer = cfg["agent"]["nuscenes_scorer"]
    assert scorer["backend"] == "craft_carl"
    assert "carl" in scorer

    old_scorer_keys = {
        "dac_weight",
        "ttc_weight",
        "history_comfort_weight",
        "lane_keeping_weight",
        "progress_weight",
    }
    assert old_scorer_keys.isdisjoint(scorer)

    old_closed_loop_keys = {
        "progress_weight",
        "w_lateral_efficiency",
        "w_heading_efficiency",
        "correction_lateral_weight",
        "correction_heading_weight",
    }
    assert old_closed_loop_keys.isdisjoint(craft_reward)


def test_sparsedrive_v2_training_templates_share_identical_schema() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "script/configs/sparsedrive_v2"
    loaded = [
        (name, yaml.safe_load((config_dir / name).read_text(encoding="utf-8")))
        for name in TEMPLATE_CONFIGS
    ]
    reference_name, reference_cfg = loaded[0]
    reference_keys = _key_paths(reference_cfg)

    for name, cfg in loaded[1:]:
        assert _key_paths(cfg) == reference_keys, (
            f"{name} schema differs from {reference_name}; "
            f"missing={sorted(reference_keys - _key_paths(cfg))}; "
            f"extra={sorted(_key_paths(cfg) - reference_keys)}"
        )


def test_sparsedrive_v2_training_templates_use_canonical_grpo_and_craft_keys() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "script/configs/sparsedrive_v2"
    for name in TEMPLATE_CONFIGS:
        cfg = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        train_cfg = cfg["train"]
        reward_cfg = cfg["env"]["reward"]

        assert "grpo" in train_cfg
        assert "reinforce" not in train_cfg
        assert not any(str(key).startswith("grpo_") for key in train_cfg["reinforcepp"])

        assert "CRAFT" in reward_cfg
        assert "craft" not in reward_cfg
        assert "corrective_progress" in reward_cfg["CRAFT"]


def test_20260524_sparsedrive_v2_four_way_craft_configs_match_intended_ablation() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "script/configs/sparsedrive_v2"
    for name, expected in FOUR_WAY_20260524_CONFIGS.items():
        cfg = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        train_cfg = cfg["train"]
        actor_learner = train_cfg["actor_learner"]
        grpo = train_cfg["grpo"]
        craft = cfg["env"]["reward"]["CRAFT"]
        agent = cfg["agent"]

        assert train_cfg["algo"] == expected["algo"]
        assert train_cfg["wandb"]["group"] == expected["run_key"]
        assert actor_learner["buffer_dir"].endswith(expected["run_key"])
        assert actor_learner["actor_gpu_pool"] == [2, 3, 4, 5, 6, 7]
        assert actor_learner["actors_per_gpu"] == 4
        assert actor_learner["shards_per_update"] == 24
        assert actor_learner["learner_gpu_id"] == 0

        normalized_cfg = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        normalize_actor_learner_cfg(normalized_cfg)
        normalized_al = normalized_cfg["train"]["actor_learner"]
        assert normalized_al["num_actors"] == 24
        assert normalized_al["actor_gpu_ids"] == [
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
        ]

        assert grpo["enable"] is True
        assert grpo["coef"] == 1.0
        assert grpo["num_candidates"] == 8
        assert grpo["debug_dir"].endswith(expected["run_key"])

        assert craft["enable"] is expected["craft_enabled"]
        assert craft["real_reward_model"] == "corrective"
        assert craft["corrective_progress"]["enable"] is expected["corrective_progress_enabled"]

        assert agent["trainable_prefixes"] == expected["trainable_prefixes"]
        assert agent["frozen_prefixes"] == expected["frozen_prefixes"]
        assert agent["nuscenes_scorer"]["backend"] == "craft_carl"


def test_20260525_sparsedrive_v2_closeloop_configs_match_training_identity() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "script/configs/sparsedrive_v2"
    for name, expected in TWO_WAY_20260525_CLOSELOOP_CONFIGS.items():
        cfg = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        train_cfg = cfg["train"]
        actor_learner = train_cfg["actor_learner"]
        grpo = train_cfg["grpo"]
        craft = cfg["env"]["reward"]["CRAFT"]
        agent = cfg["agent"]

        assert train_cfg["algo"] == "reinforcepp"
        assert train_cfg["wandb"]["group"] == expected["run_key"]
        assert actor_learner["buffer_dir"].endswith(expected["run_key"])
        assert grpo["debug_dir"].endswith(expected["run_key"])
        assert craft["enable"] is True
        assert craft["real_reward_model"] == "close loop"
        assert craft["corrective_progress"]["enable"] is True
        assert agent["trainable_prefixes"] == expected["trainable_prefixes"]
        assert agent["frozen_prefixes"] == ["_backbone"]


def test_20260525_hugsim_ori_config_collects_shards_with_large_collision_penalty() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "script/configs/sparsedrive_v2"
    cfg = yaml.safe_load((config_dir / HUGSIM_ORI_20260525_CONFIG).read_text(encoding="utf-8"))

    assert cfg["env"]["backend"] == "hugsim_ori"
    assert cfg["env"]["use_all_scenes"] is True
    assert "scenes" not in cfg["env"]["hugsim"]

    craft = cfg["env"]["reward"]["CRAFT"]
    terminal = cfg["env"]["reward"]["terminal"]
    assert craft["enable"] is True
    assert craft["real_reward_model"] == "close loop"
    assert craft["term_collision"] == 200.0
    assert terminal["penalty"] == -200.0
    assert terminal["apply_on_failure"] is True
    assert terminal["terminate_on_dynamic_collision"] is True

    train_cfg = cfg["train"]
    actor_learner = train_cfg["actor_learner"]
    assert train_cfg["algo"] == "reinforcepp"
    assert actor_learner["learner_gpu_id"] == 0
    assert actor_learner["actor_gpu_pool"] == [1, 2, 3, 4, 5, 6, 7]
    assert actor_learner["actors_per_gpu"] == 3
    assert actor_learner["shards_per_update"] == 21

    normalized_cfg = yaml.safe_load((config_dir / HUGSIM_ORI_20260525_CONFIG).read_text(encoding="utf-8"))
    normalize_actor_learner_cfg(normalized_cfg)
    normalized_al = normalized_cfg["train"]["actor_learner"]
    assert normalized_al["num_actors"] == 21
    assert normalized_al["actor_gpu_ids"] == [
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
        5,
        5,
        6,
        6,
        6,
        7,
        7,
        7,
    ]
