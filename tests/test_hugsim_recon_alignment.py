import math
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from framework.env_wrapper.hugsim_recon_alignment import (
    Sim2Transform,
    build_hugsim_recon_alignment,
    fit_sim2,
    transform_hugsim_box_to_recon_poly,
    transform_hugsim_ego_box_to_reward_pose,
)


def _pose(x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_euler("Z", yaw).as_matrix()
    out[0, 3] = x
    out[1, 3] = y
    return out


def _hugsim_pose_for_box_xy(x: float, y: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[2, 3] = x
    out[0, 3] = -y
    return out


def test_fit_sim2_recovers_scale_rotation_translation():
    src = np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [20.0, 5.0]], dtype=np.float64)
    theta = math.radians(30.0)
    scale = 1.25
    rot = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    dst = scale * (src @ rot.T) + np.asarray([[100.0, -20.0]])

    transform = fit_sim2(src, dst)

    np.testing.assert_allclose(transform.transform_points(src), dst, atol=1.0e-6)
    assert transform.scale == pytest.approx(scale)
    assert transform.yaw_rad == pytest.approx(theta)
    np.testing.assert_allclose(transform.translation_xy, [100.0, -20.0], atol=1.0e-6)


def test_build_hugsim_recon_alignment_uses_ground_path_and_recon_ego_path(tmp_path: Path):
    hugsim_scene_root = tmp_path / "hugsim" / "scene-0001"
    recon_scene_root = tmp_path / "recon" / "012" / "ego_pose"
    hugsim_scene_root.mkdir(parents=True)
    recon_scene_root.mkdir(parents=True)

    hugsim_poses = np.stack(
        [_hugsim_pose_for_box_xy(0.0, 0.0), _hugsim_pose_for_box_xy(0.0, 0.0), _hugsim_pose_for_box_xy(10.0, 0.0)]
    )
    commands = [2, 2, 2]
    with (hugsim_scene_root / "ground_param.pkl").open("wb") as handle:
        pickle.dump((hugsim_poses, [0.0, 0.0, 0.0], commands), handle)
    for idx, xy in enumerate([[100.0, 50.0], [100.0, 50.0], [120.0, 50.0]]):
        np.savetxt(recon_scene_root / f"{idx:03d}.txt", _pose(xy[0], xy[1]))

    alignment = build_hugsim_recon_alignment(
        official_scene_name="scene-0001",
        recon_scene_id=12,
        hugsim_model_base=tmp_path / "hugsim",
        recon_data_root=tmp_path / "recon",
    )

    assert alignment.valid
    assert alignment.transform.scale == pytest.approx(2.0)
    np.testing.assert_allclose(
        alignment.transform.transform_points([[0.0, 0.0], [10.0, 0.0]]),
        [[100.0, 50.0], [120.0, 50.0]],
    )


def test_alignment_transforms_hugsim_box_to_recon_polygon():
    transform = Sim2Transform(scale=2.0, rotation=np.eye(2), translation_xy=np.asarray([100.0, 50.0]))

    poly = transform_hugsim_box_to_recon_poly(
        [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
        transform,
    )

    np.testing.assert_allclose(
        poly,
        np.asarray([[124.0, 52.0], [124.0, 48.0], [116.0, 48.0], [116.0, 52.0]], dtype=np.float64),
    )


def test_alignment_builds_reward_pose_in_recon_reward_coordinates():
    transform = Sim2Transform(scale=1.0, rotation=np.eye(2), translation_xy=np.asarray([100.0, 50.0]))

    pose = transform_hugsim_ego_box_to_reward_pose(
        [10.0, 2.0, 0.0, 2.0, 4.0, 1.5, math.pi / 2.0],
        transform,
    )

    assert pose.shape == (4, 4)
    np.testing.assert_allclose(pose[:3, 3], [110.0, 0.0, 52.0], atol=1.0e-6)
