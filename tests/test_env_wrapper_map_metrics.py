from __future__ import annotations

import math

import pytest

from framework.env_wrapper.map_metrics import compute_craft_map_metrics


def test_craft_map_metrics_marks_ego_inside_drivable_area() -> None:
    snapshot = {
        "drivable_polygons": [
            [
                [-5.0, -5.0],
                [5.0, -5.0],
                [5.0, 5.0],
                [-5.0, 5.0],
            ],
        ],
    }

    metrics = compute_craft_map_metrics(snapshot, ego_x=0.0, ego_y=0.0, ego_yaw=0.0)

    assert metrics["map_has_drivable"] is True
    assert metrics["off_road"] is False


def test_craft_map_metrics_marks_ego_outside_drivable_area() -> None:
    snapshot = {
        "drivable_polygons": [
            [
                [-5.0, -5.0],
                [5.0, -5.0],
                [5.0, 5.0],
                [-5.0, 5.0],
            ],
        ],
    }

    metrics = compute_craft_map_metrics(snapshot, ego_x=20.0, ego_y=0.0, ego_yaw=0.0)

    assert metrics["map_has_drivable"] is True
    assert metrics["off_road"] is True


def test_craft_map_metrics_computes_center_and_heading_ratios_from_lanes() -> None:
    snapshot = {
        "lanes_centerlines": [
            [
                [0.0, 0.0],
                [10.0, 0.0],
            ],
        ],
    }

    metrics = compute_craft_map_metrics(
        snapshot,
        ego_x=2.0,
        ego_y=1.0,
        ego_yaw=math.pi,
        center_dev_max_m=2.0,
        heading_dev_max_deg=90.0,
        reverse_dot_threshold=-0.5,
    )

    assert metrics["map_has_lane_centerline"] is True
    assert metrics["centerline_lateral_error_m"] == pytest.approx(1.0)
    assert metrics["center_dev_ratio"] == pytest.approx(0.5)
    assert metrics["map_heading_dev_ratio"] == pytest.approx(1.0)
    assert metrics["opposite_lane"] is True
    assert metrics["driving_direction_violation"] is True


def test_craft_map_metrics_prefers_plausible_same_direction_lane_over_nearest_reverse_lane() -> None:
    snapshot = {
        "lanes_centerlines": [
            [
                [0.0, 2.24],
                [-10.0, 2.24],
            ],
            [
                [0.0, -2.50],
                [10.0, -2.50],
            ],
        ],
    }

    metrics = compute_craft_map_metrics(
        snapshot,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        center_dev_max_m=3.0,
        heading_dev_max_deg=90.0,
        reverse_dot_threshold=-0.5,
    )

    assert metrics["map_has_lane_centerline"] is True
    assert metrics["opposite_lane"] is False
    assert metrics["driving_direction_violation"] is False
    assert metrics["lane_tangent_dot"] == pytest.approx(1.0)
    assert metrics["centerline_lateral_error_m"] == pytest.approx(2.50)
