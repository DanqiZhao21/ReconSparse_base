from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from framework.rewards.closed_loop_ea import ClosedLoopEAScorer


def _pose(x: float, y: float, yaw_rad: float = 0.0) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    out = np.eye(4, dtype=np.float64)
    out[0, 0] = c
    out[0, 2] = -s
    out[2, 0] = s
    out[2, 2] = c
    out[0, 3] = float(x)
    out[2, 3] = float(y)
    return out


def _write_agent_state_cache(root: Path, *, scene_id: int, frame_idx: int, agents: list[dict[str, object]]) -> None:
    scene_dir = root / f"{int(scene_id):03d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "agent_state_cache.json").write_text(
        json.dumps(
            {
                "meta": {"coordinate_frame": "local"},
                str(int(frame_idx)): {"agents": agents},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_closed_loop_ea_scores_current_step_vehicle_pairs_and_reports_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "agent_cache"
    _write_agent_state_cache(
        cache_root,
        scene_id=3,
        frame_idx=10,
        agents=[
            {
                "category": "vehicle.car",
                "center_xy": [3.0, 0.0],
                "velocity_xy": [0.0, 0.0],
                "yaw_rad": 0.0,
                "yaw_rate_rps": 0.0,
                "length_m": 4.0,
                "width_m": 2.0,
            },
            {
                "category": "human.pedestrian",
                "center_xy": [1.0, 0.0],
                "velocity_xy": [0.0, 0.0],
                "yaw_rad": 0.0,
                "yaw_rate_rps": 0.0,
                "length_m": 1.0,
                "width_m": 1.0,
            },
        ],
    )
    scorer = ClosedLoopEAScorer(
        {
            "enable": True,
            "agent_state_cache_root": str(cache_root),
            "good_threshold": 0.0,
            "bad_threshold": 8.0,
        }
    )

    monkeypatch.setattr(scorer, "_ensure_compute_fn", lambda: (lambda **kwargs: float(kwargs["xB"])))

    result = scorer.score_current_step(
        scene_id=3,
        frame_idx=10,
        ego_pose=_pose(0.0, 0.0),
        ego_velocity_xy=np.asarray([2.0, 0.0], dtype=np.float32),
        previous_ego_pose=_pose(-1.0, 0.0),
        dt_s=0.5,
    )

    assert result["ea_enabled"] is True
    assert result["ea_available"] is True
    assert result["ea_evaluated_pairs"] == pytest.approx(1.0)
    assert result["ea_max"] == pytest.approx(3.0)
    assert result["ea_min"] == pytest.approx(3.0)
    assert result["ea_mean"] == pytest.approx(3.0)
    assert result["ea_risk"] == pytest.approx(3.0 / 8.0)
