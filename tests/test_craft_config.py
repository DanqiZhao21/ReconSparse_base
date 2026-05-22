from __future__ import annotations

from pathlib import Path

import yaml


TEMPLATE_CONFIGS = [
    "20260519_reinforcepp_closed_loop_sparsedrive_v2_craft_onlyGRPO.yaml",
    "202605211154_reinforcepp_closed_loop_sparsedrive_v2_craft_closeCorrection_openGrpo.yaml",
    "202605211155_reinforcepp_closed_loop_sparsedrive_v2_craft_closeNo_openGRPO.yaml",
]


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
