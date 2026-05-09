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


def test_reinforcepp_dac9_grpo_coef_sweep_configs_match_requested_matrix() -> None:
    config_specs = [
        (
            "reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef1.yaml",
            "outputs/actor_learner_reinforcepp_dac9_grpo_coef1",
            "reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef1",
            "outputs/visualize/grpo_nuscenes_reinforcepp_dac9_coef1",
            1.0,
        ),
        (
            "reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef01.yaml",
            "outputs/actor_learner_reinforcepp_dac9_grpo_coef01",
            "reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef01",
            "outputs/visualize/grpo_nuscenes_reinforcepp_dac9_coef01",
            0.1,
        ),
        (
            "reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef05.yaml",
            "outputs/actor_learner_reinforcepp_dac9_grpo_coef05",
            "reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef05",
            "outputs/visualize/grpo_nuscenes_reinforcepp_dac9_coef05",
            0.5,
        ),
    ]

    for filename, buffer_dir, wandb_group, debug_dir, grpo_coef in config_specs:
        cfg = _load_yaml(REPO_ROOT / "script/configs/sparsedrive_v2" / filename)
        train_cfg = cfg["train"]
        actor_learner_cfg = train_cfg["actor_learner"]
        grpo_cfg = train_cfg["grpo"]
        scorer_cfg = cfg["agent"]["nuscenes_scorer"]

        assert train_cfg["algo"] == "reinforcepp"
        assert bool(grpo_cfg["enable"]) is True
        assert float(grpo_cfg["coef"]) == grpo_coef
        assert actor_learner_cfg["buffer_dir"] == buffer_dir
        assert train_cfg["wandb"]["group"] == wandb_group
        assert grpo_cfg["debug_dir"] == debug_dir
        assert scorer_cfg["backend"] == "nuscenes_pdm"
        assert scorer_cfg["score_mode"] == "drivable_area_only"
        assert bool(scorer_cfg["ea_gate_enabled"]) is False
        assert bool(scorer_cfg["driving_direction_gate_enabled"]) is False
        assert float(scorer_cfg["progress_weight"]) == 0.0
        assert float(scorer_cfg["ttc_weight"]) == 0.0
        assert float(scorer_cfg["lane_keeping_weight"]) == 0.0
        assert float(scorer_cfg["history_comfort_weight"]) == 0.0


def test_reinforcepp_dac9_grpo_coef08_config_matches_requested_setup() -> None:
    cfg = _load_yaml(
        REPO_ROOT
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef08.yaml"
    )
    train_cfg = cfg["train"]
    actor_learner_cfg = train_cfg["actor_learner"]
    grpo_cfg = train_cfg["grpo"]
    scorer_cfg = cfg["agent"]["nuscenes_scorer"]

    assert train_cfg["algo"] == "reinforcepp"
    assert bool(grpo_cfg["enable"]) is True
    assert float(grpo_cfg["coef"]) == 0.8
    assert actor_learner_cfg["buffer_dir"] == "outputs/actor_learner_reinforcepp_dac9_grpo_coef08"
    assert train_cfg["wandb"]["group"] == "reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef08"
    assert grpo_cfg["debug_dir"] == "outputs/visualize/grpo_nuscenes_reinforcepp_dac9_coef08"
    assert scorer_cfg["backend"] == "nuscenes_pdm"
    assert scorer_cfg["score_mode"] == "drivable_area_only"
    assert bool(scorer_cfg["ea_gate_enabled"]) is False
    assert bool(scorer_cfg["driving_direction_gate_enabled"]) is False
    assert float(scorer_cfg["progress_weight"]) == 0.0
    assert float(scorer_cfg["ttc_weight"]) == 0.0
    assert float(scorer_cfg["lane_keeping_weight"]) == 0.0
    assert float(scorer_cfg["history_comfort_weight"]) == 0.0


def test_reinforcepp_progress_comfort_grpo_dac_ttc_coef08_config_matches_requested_setup() -> None:
    cfg = _load_yaml(
        REPO_ROOT
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_progress_comfort_grpo_dac_ttc_coef08.yaml"
    )
    train_cfg = cfg["train"]
    actor_learner_cfg = train_cfg["actor_learner"]
    grpo_cfg = train_cfg["grpo"]
    reward_cfg = cfg["env"]["reward"]
    path_reward_cfg = reward_cfg["path"]
    comfort_reward_cfg = reward_cfg["comfort"]
    scorer_cfg = cfg["agent"]["nuscenes_scorer"]

    assert train_cfg["algo"] == "reinforcepp"
    assert bool(grpo_cfg["enable"]) is True
    assert float(grpo_cfg["coef"]) == 0.8
    assert actor_learner_cfg["buffer_dir"] == "outputs/actor_learner_reinforcepp_progress_comfort_grpo_dac_ttc_coef08"
    assert train_cfg["wandb"]["group"] == "reinforcepp_closed_loop_sparsedrive_v2_progress_comfort_grpo_dac_ttc_coef08"
    assert grpo_cfg["debug_dir"] == "outputs/visualize/grpo_nuscenes_progress_comfort_grpo_dac_ttc_coef08"

    assert reward_cfg["mode"] == "step_path"
    assert float(path_reward_cfg["w_progress"]) == 2.0
    assert float(path_reward_cfg["w_completion_ratio"]) == 2.0
    assert float(comfort_reward_cfg["w_longitudinal_jerk"]) == 0.05
    assert float(comfort_reward_cfg["w_yaw_jerk"]) == 0.05

    assert scorer_cfg["backend"] == "nuscenes_pdm"
    assert scorer_cfg["score_mode"] == "full"
    assert bool(scorer_cfg["ea_gate_enabled"]) is False
    assert bool(scorer_cfg["driving_direction_gate_enabled"]) is False
    assert float(scorer_cfg["progress_weight"]) == 0.0
    assert float(scorer_cfg["dac_weight"]) == 5.0
    assert bool(scorer_cfg["dac_gate_enabled"]) is True
    assert float(scorer_cfg["ttc_weight"]) == 5.0
    assert float(scorer_cfg["lane_keeping_weight"]) == 0.0
    assert float(scorer_cfg["history_comfort_weight"]) == 0.0


def test_reinforcepp_progress_comfort_dac_weight_coef_configs_match_requested_matrix() -> None:
    config_specs = [
        (
            "reinforcepp_closed_loop_sparsedrive_v2_progress_comfort_dac_weight_grpo_coef01.yaml",
            "outputs/actor_learner_reinforcepp_progress_comfort_dac_weight_grpo_coef01",
            "reinforcepp_closed_loop_sparsedrive_v2_progress_comfort_dac_weight_grpo_coef01",
            "outputs/visualize/grpo_nuscenes_progress_comfort_dac_weight_coef01",
            0.1,
        ),
        (
            "reinforcepp_closed_loop_sparsedrive_v2_progress_comfort_dac_weight_grpo_coef08.yaml",
            "outputs/actor_learner_reinforcepp_progress_comfort_dac_weight_grpo_coef08",
            "reinforcepp_closed_loop_sparsedrive_v2_progress_comfort_dac_weight_grpo_coef08",
            "outputs/visualize/grpo_nuscenes_progress_comfort_dac_weight_coef08",
            0.8,
        ),
    ]

    for filename, buffer_dir, wandb_group, debug_dir, grpo_coef in config_specs:
        cfg = _load_yaml(REPO_ROOT / "script/configs/sparsedrive_v2" / filename)
        train_cfg = cfg["train"]
        actor_learner_cfg = train_cfg["actor_learner"]
        grpo_cfg = train_cfg["grpo"]
        reward_cfg = cfg["env"]["reward"]
        scorer_cfg = cfg["agent"]["nuscenes_scorer"]

        assert train_cfg["algo"] == "reinforcepp"
        assert bool(grpo_cfg["enable"]) is True
        assert float(grpo_cfg["coef"]) == grpo_coef
        assert actor_learner_cfg["buffer_dir"] == buffer_dir
        assert train_cfg["wandb"]["group"] == wandb_group
        assert grpo_cfg["debug_dir"] == debug_dir
        assert float(reward_cfg["path"]["w_progress"]) == 2.0
        assert float(reward_cfg["path"]["w_completion_ratio"]) == 2.0
        assert float(reward_cfg["comfort"]["w_longitudinal_jerk"]) == 0.05
        assert float(reward_cfg["comfort"]["w_yaw_jerk"]) == 0.05
        assert scorer_cfg["backend"] == "nuscenes_pdm"
        assert scorer_cfg["score_mode"] == "full"
        assert bool(scorer_cfg["ea_gate_enabled"]) is False
        assert bool(scorer_cfg["driving_direction_gate_enabled"]) is False
        assert float(scorer_cfg["progress_weight"]) == 0.0
        assert float(scorer_cfg["dac_weight"]) == 5.0
        assert bool(scorer_cfg["dac_gate_enabled"]) is True
        assert float(scorer_cfg["ttc_weight"]) == 5.0
        assert float(scorer_cfg["lane_keeping_weight"]) == 0.0
        assert float(scorer_cfg["history_comfort_weight"]) == 0.0


def test_reinforcepp_baseline_reward_dac_weight_coef_configs_match_requested_matrix() -> None:
    config_specs = [
        (
            "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef003.yaml",
            "outputs/actor_learner_reinforcepp_baseline_reward_dac_weight_grpo_coef003",
            "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef003",
            "outputs/visualize/grpo_nuscenes_baseline_reward_dac_weight_coef003",
            0.03,
        ),
        (
            "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef01.yaml",
            "outputs/actor_learner_reinforcepp_baseline_reward_dac_weight_grpo_coef01",
            "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef01",
            "outputs/visualize/grpo_nuscenes_baseline_reward_dac_weight_coef01",
            0.1,
        ),
    ]

    for filename, buffer_dir, wandb_group, debug_dir, grpo_coef in config_specs:
        cfg = _load_yaml(REPO_ROOT / "script/configs/sparsedrive_v2" / filename)
        train_cfg = cfg["train"]
        actor_learner_cfg = train_cfg["actor_learner"]
        grpo_cfg = train_cfg["grpo"]
        reward_cfg = cfg["env"]["reward"]
        scorer_cfg = cfg["agent"]["nuscenes_scorer"]

        assert train_cfg["algo"] == "reinforcepp"
        assert bool(grpo_cfg["enable"]) is True
        assert float(grpo_cfg["coef"]) == grpo_coef
        assert actor_learner_cfg["buffer_dir"] == buffer_dir
        assert train_cfg["wandb"]["group"] == wandb_group
        assert grpo_cfg["debug_dir"] == debug_dir
        assert float(reward_cfg["path"]["w_progress"]) == 1.0
        assert float(reward_cfg["path"]["w_completion_ratio"]) == 1.0
        assert float(reward_cfg["comfort"]["w_longitudinal_jerk"]) == 0.01
        assert float(reward_cfg["comfort"]["w_yaw_jerk"]) == 0.01
        assert scorer_cfg["backend"] == "nuscenes_pdm"
        assert scorer_cfg["score_mode"] == "full"
        assert bool(scorer_cfg["ea_gate_enabled"]) is False
        assert bool(scorer_cfg["driving_direction_gate_enabled"]) is False
        assert float(scorer_cfg["progress_weight"]) == 0.0
        assert float(scorer_cfg["dac_weight"]) == 5.0
        assert bool(scorer_cfg["dac_gate_enabled"]) is True
        assert float(scorer_cfg["ttc_weight"]) == 5.0
        assert float(scorer_cfg["lane_keeping_weight"]) == 0.0
        assert float(scorer_cfg["history_comfort_weight"]) == 0.0


def test_reinforcepp_minimal_grpo_ablation_configs_match_requested_matrix() -> None:
    config_specs = [
        (
            "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef001.yaml",
            "outputs/actor_learner_reinforcepp_baseline_reward_dac_weight_grpo_coef001",
            "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef001",
            "outputs/visualize/grpo_nuscenes_baseline_reward_dac_weight_coef001",
            0.01,
            True,
        ),
        (
            "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef003_gateoff.yaml",
            "outputs/actor_learner_reinforcepp_baseline_reward_dac_weight_grpo_coef003_gateoff",
            "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef003_gateoff",
            "outputs/visualize/grpo_nuscenes_baseline_reward_dac_weight_coef003_gateoff",
            0.03,
            False,
        ),
    ]

    for filename, buffer_dir, wandb_group, debug_dir, grpo_coef, dac_gate_enabled in config_specs:
        cfg = _load_yaml(REPO_ROOT / "script/configs/sparsedrive_v2" / filename)
        train_cfg = cfg["train"]
        actor_learner_cfg = train_cfg["actor_learner"]
        grpo_cfg = train_cfg["grpo"]
        reward_cfg = cfg["env"]["reward"]
        scorer_cfg = cfg["agent"]["nuscenes_scorer"]

        assert train_cfg["algo"] == "reinforcepp"
        assert bool(grpo_cfg["enable"]) is True
        assert float(grpo_cfg["coef"]) == grpo_coef
        assert actor_learner_cfg["buffer_dir"] == buffer_dir
        assert train_cfg["wandb"]["group"] == wandb_group
        assert grpo_cfg["debug_dir"] == debug_dir
        assert float(reward_cfg["path"]["w_progress"]) == 1.0
        assert float(reward_cfg["path"]["w_completion_ratio"]) == 1.0
        assert float(reward_cfg["comfort"]["w_longitudinal_jerk"]) == 0.01
        assert float(reward_cfg["comfort"]["w_yaw_jerk"]) == 0.01
        assert scorer_cfg["backend"] == "nuscenes_pdm"
        assert scorer_cfg["score_mode"] == "full"
        assert bool(scorer_cfg["ea_gate_enabled"]) is False
        assert bool(scorer_cfg["driving_direction_gate_enabled"]) is False
        assert float(scorer_cfg["progress_weight"]) == 0.0
        assert float(scorer_cfg["dac_weight"]) == 5.0
        assert bool(scorer_cfg["dac_gate_enabled"]) is dac_gate_enabled
        assert float(scorer_cfg["ttc_weight"]) == 5.0
        assert float(scorer_cfg["lane_keeping_weight"]) == 0.0
        assert float(scorer_cfg["history_comfort_weight"]) == 0.0
