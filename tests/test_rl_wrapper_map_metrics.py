from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _pose(x: float, y: float, yaw_rad: float = 0.0) -> np.ndarray:
    c = np.cos(float(yaw_rad))
    s = np.sin(float(yaw_rad))
    out = np.eye(4, dtype=np.float64)
    out[0, 0] = c
    out[0, 2] = -s
    out[2, 0] = s
    out[2, 2] = c
    out[0, 3] = float(x)
    out[2, 3] = float(y)
    return out


def _world_pose(x: float, y: float, yaw_rad: float = 0.0) -> np.ndarray:
    c = np.cos(float(yaw_rad))
    s = np.sin(float(yaw_rad))
    out = np.eye(4, dtype=np.float64)
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c
    out[0, 3] = float(x)
    out[1, 3] = float(y)
    return out


class _DummySim:
    scene = 12
    start_ego = _pose(2.0, 1.0, np.pi)
    camera_front_start = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _load_rl_recon_env(monkeypatch: pytest.MonkeyPatch):
    nus_mod = types.ModuleType("reconsimulator.envs.nus")
    nus_mod.ReconSimulator = object
    metrics_mod = types.ModuleType("reconsimulator.envs.metrics")
    metrics_mod.EGO_LENGTH = 4.2
    metrics_mod.EGO_WIDTH = 1.9
    metrics_mod.oriented_box = lambda *args, **kwargs: None
    metrics_cache_mod = types.ModuleType("reconsimulator.envs.metrics_cache")
    metrics_cache_mod.load_scene_env_cache = lambda scene_id: None
    monkeypatch.setitem(sys.modules, "reconsimulator.envs.nus", nus_mod)
    monkeypatch.setitem(sys.modules, "reconsimulator.envs.metrics", metrics_mod)
    monkeypatch.setitem(sys.modules, "reconsimulator.envs.metrics_cache", metrics_cache_mod)

    from framework.env_wrapper.rl_wrapper import RLReconEnv

    return RLReconEnv


def test_rl_wrapper_map_metrics_falls_back_to_pdm_context_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RLReconEnv = _load_rl_recon_env(monkeypatch)
    sample_token = "tok-a"
    cache_root = tmp_path / "_sample_pdm_context"
    cache_root.mkdir()
    with (cache_root / f"{sample_token}_ctx.pkl").open("wb") as handle:
        pickle.dump(
            {
                "sample_token": sample_token,
                "lane_centerlines": [[[0.0, 0.0], [10.0, 0.0]]],
                "drivable_polygons": [[[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]],
            },
            handle,
        )

    monkeypatch.setattr(
        "framework.env_wrapper.rl_wrapper.load_scene_env_cache",
        lambda scene_id: {0: {"sample_token": sample_token}},
    )

    wrapper = RLReconEnv.__new__(RLReconEnv)
    wrapper.env = _DummySim()
    wrapper.reward_cfg = {
        "CRAFT": {
            "enable": True,
            "map": {
                "pdm_context_cache_root": str(cache_root),
                "center_dev_max_m": 2.0,
                "heading_dev_max_deg": 90.0,
                "reverse_dot_threshold": -0.5,
            },
        }
    }
    wrapper._step_idx = 0
    wrapper._env_cache_scene_id = None
    wrapper._env_cache = {}
    wrapper._env_cache_keys = []

    metrics = wrapper._compute_map_metrics()

    assert metrics["map_has_lane_centerline"] is True
    assert metrics["centerline_lateral_error_m"] == pytest.approx(1.0)
    assert metrics["center_dev_ratio"] == pytest.approx(0.5)
    assert metrics["map_heading_dev_ratio"] == pytest.approx(1.0)
    assert metrics["off_road"] is False
    assert metrics["opposite_lane"] is True


def test_rl_wrapper_adds_closed_loop_ea_metrics_to_step_info(monkeypatch: pytest.MonkeyPatch) -> None:
    RLReconEnv = _load_rl_recon_env(monkeypatch)

    wrapper = RLReconEnv.__new__(RLReconEnv)
    wrapper.env = types.SimpleNamespace(
        scene=4,
        now_frame=10,
        step=lambda _action: ({"obs": np.asarray([1.0])}, False, False, {}),
        start_ego=_pose(1.0, 2.0),
        camera_front_start=np.eye(4, dtype=np.float64),
    )
    wrapper.reward_cfg = {"ea": {"enable": True}}
    wrapper._step_idx = 0
    wrapper._compute_collision_flags = lambda: (False, False)
    wrapper._compute_reward = lambda info, done=False: (1.0 - float(info.get("ea_cost", 0.0)), dict(info))
    wrapper._closed_loop_ea_scorer = types.SimpleNamespace(
        score_current_step=lambda **kwargs: {
            "ea_enabled": True,
            "ea_available": True,
            "ea_max": 3.0,
            "ea_min": 3.0,
            "ea_mean": 3.0,
            "ea_risk": 0.25,
            "ea_evaluated_pairs": 1.0,
        }
    )

    obs, reward, terminated, truncated, info = wrapper.step((0.0, 0.0, 0.0, 2))

    assert obs["obs"].tolist() == pytest.approx([1.0])
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(1.0)
    assert info["ea_available"] is True
    assert info["ea_max"] == pytest.approx(3.0)
    assert info["ea_risk"] == pytest.approx(0.25)


def test_rl_wrapper_computes_front_obstacle_metrics_from_env_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    RLReconEnv = _load_rl_recon_env(monkeypatch)

    wrapper = RLReconEnv.__new__(RLReconEnv)
    wrapper.env = types.SimpleNamespace(
        scene=7,
        start_ego=_world_pose(0.0, 0.0, np.pi / 2.0),
        camera_front_start=np.eye(4, dtype=np.float64),
        _status_vel_xy=np.asarray([0.0, 4.0], dtype=np.float64),
    )
    wrapper.reward_cfg = {"safety": {"lookahead_m": 20.0, "corridor_half_width_m": 2.5}}
    wrapper._step_idx = 0
    wrapper._env_cache_scene_id = None
    wrapper._env_cache = {}
    wrapper._env_cache_keys = []
    monkeypatch.setattr(
        "framework.env_wrapper.rl_wrapper.load_scene_env_cache",
        lambda scene_id: {
            0: {
                "dynamic_objects": [
                    {
                        "category": "vehicle.car",
                        "poly": [[-1.0, 5.0], [1.0, 5.0], [1.0, 9.0], [-1.0, 9.0]],
                        "velocity_xy": [0.0, 0.0],
                    },
                    {
                        "category": "vehicle.car",
                        "poly": [[5.0, 4.0], [7.0, 4.0], [7.0, 8.0], [5.0, 8.0]],
                        "velocity_xy": [0.0, 0.0],
                    },
                ],
            }
        },
    )

    metrics = wrapper._compute_front_obstacle_metrics()

    assert metrics["front_obstacle_available"] is True
    assert metrics["front_obstacle_gap_m"] == pytest.approx(2.9)
    assert metrics["front_obstacle_lateral_m"] == pytest.approx(0.0)
    assert metrics["front_obstacle_closing_speed_mps"] == pytest.approx(4.0)
    assert metrics["front_obstacle_ttc_s"] == pytest.approx(2.9 / 4.0)
    assert metrics["front_obstacle_category"] == "vehicle.car"
