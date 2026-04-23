from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_corner_train_config_variants_exist_and_match_requested_matrix() -> None:
    config_specs = [
        (
            REPO_ROOT / "script/configs/sparsedrive_v2/ppo_closed_loop_sparsedrive_v2_corner_baseline.yaml",
            "outputs/actor_learner_ppo_corner_baseline",
            "ppo_closed_loop_sparsedrive_v2_corner_baseline",
            0.1,
            8.0,
        ),
        (
            REPO_ROOT / "script/configs/sparsedrive_v2/ppo_closed_loop_sparsedrive_v2_corner_grpo003.yaml",
            "outputs/actor_learner_ppo_corner_grpo003",
            "ppo_closed_loop_sparsedrive_v2_corner_grpo003",
            0.03,
            8.0,
        ),
        (
            REPO_ROOT / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_corner_baseline.yaml",
            "outputs/actor_learner_reinforcepp_corner_baseline",
            "reinforcepp_closed_loop_sparsedrive_v2_corner_baseline",
            0.1,
            8.0,
        ),
        (
            REPO_ROOT / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_corner_grpo003.yaml",
            "outputs/actor_learner_reinforcepp_corner_grpo003",
            "reinforcepp_closed_loop_sparsedrive_v2_corner_grpo003",
            0.03,
            8.0,
        ),
    ]

    for path, buffer_dir, wandb_group, grpo_coef, progress_weight in config_specs:
        cfg = _load_yaml(path)
        train_cfg = cfg["train"]
        actor_learner_cfg = train_cfg["actor_learner"]
        agent_cfg = cfg["agent"]

        assert actor_learner_cfg["buffer_dir"] == buffer_dir
        assert train_cfg["wandb"]["group"] == wandb_group
        assert float(train_cfg["grpo"]["coef"]) == grpo_coef
        assert float(actor_learner_cfg["shard_collect_timeout_s"]) == 60.0
        assert float(agent_cfg["nuscenes_scorer"]["progress_weight"]) == progress_weight


def test_reinforce_grpo_no_lane_no_dirdir_config_matches_requested_setup() -> None:
    path = REPO_ROOT / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_grpo_no_lane_no_dirdir.yaml"
    cfg = _load_yaml(path)

    train_cfg = cfg["train"]
    actor_learner_cfg = train_cfg["actor_learner"]
    agent_cfg = cfg["agent"]
    scorer_cfg = agent_cfg["nuscenes_scorer"]

    assert train_cfg["algo"] == "reinforcepp"
    assert bool(train_cfg["grpo"]["enable"]) is True
    assert float(train_cfg["grpo"]["coef"]) == 0.1
    assert actor_learner_cfg["buffer_dir"] == "outputs/actor_learner_reinforcepp_grpo_no_lane_no_dirdir"
    assert train_cfg["wandb"]["group"] == "reinforcepp_closed_loop_sparsedrive_v2_grpo_no_lane_no_dirdir"
    assert scorer_cfg["backend"] == "nuscenes_pdm"
    assert bool(scorer_cfg["ea_gate_enabled"]) is False
    assert bool(scorer_cfg["driving_direction_gate_enabled"]) is False
    assert float(scorer_cfg["lane_keeping_weight"]) == 0.0
    assert float(scorer_cfg["progress_weight"]) == 8.0


def test_reinforce_grpo_lane_dir_on_config_matches_requested_setup() -> None:
    path = REPO_ROOT / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_grpo_lane_dir_on.yaml"
    cfg = _load_yaml(path)

    train_cfg = cfg["train"]
    actor_learner_cfg = train_cfg["actor_learner"]
    agent_cfg = cfg["agent"]
    scorer_cfg = agent_cfg["nuscenes_scorer"]

    assert train_cfg["algo"] == "reinforcepp"
    assert bool(train_cfg["grpo"]["enable"]) is True
    assert float(train_cfg["grpo"]["coef"]) == 0.1
    assert actor_learner_cfg["buffer_dir"] == "outputs/actor_learner_reinforcepp_grpo_lane_dir_on"
    assert train_cfg["wandb"]["group"] == "reinforcepp_closed_loop_sparsedrive_v2_grpo_lane_dir_on"
    assert scorer_cfg["backend"] == "nuscenes_pdm"
    assert bool(scorer_cfg["ea_gate_enabled"]) is False
    assert bool(scorer_cfg["driving_direction_gate_enabled"]) is True
    assert float(scorer_cfg["lane_keeping_weight"]) == 2.0
    assert float(scorer_cfg["progress_weight"]) == 8.0
