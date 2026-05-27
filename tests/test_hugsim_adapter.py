import numpy as np
import pytest
import pickle

from framework.env_wrapper.hugsim_adapter import build_recondreamer_obs_from_hugsim
from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping
from framework.rewards import TrackingRewardResult


def _camera_cfg():
    return {
        "intrinsic": {"H": 450, "W": 800, "fovx": 1.0, "fovy": 0.8, "cx": 400.0, "cy": 225.0},
        "v2c": np.eye(4, dtype=np.float32),
        "l2c": np.eye(4, dtype=np.float32),
    }


def _fake_hugsim_obs(image):
    return {
        "rgb": {
            "CAM_FRONT_LEFT": image,
            "CAM_FRONT": image + 1,
            "CAM_FRONT_RIGHT": image + 2,
            "CAM_BACK_LEFT": image + 3,
            "CAM_BACK": image + 4,
            "CAM_BACK_RIGHT": image + 5,
        }
    }


def _fake_hugsim_info(timestamp):
    return {
        "ego_velo": 2.0,
        "ego_steer": 0.0,
        "accelerate": 0.5,
        "command": 2,
        "timestamp": float(timestamp),
        "ego_pos": [1.0, 2.0, 3.0],
        "ego_rot": [0.0, 0.0, 0.0],
        "cam_params": {
            "CAM_FRONT_LEFT": _camera_cfg(),
            "CAM_FRONT": _camera_cfg(),
            "CAM_FRONT_RIGHT": _camera_cfg(),
            "CAM_BACK_LEFT": _camera_cfg(),
            "CAM_BACK": _camera_cfg(),
            "CAM_BACK_RIGHT": _camera_cfg(),
        },
    }


def test_build_recondreamer_obs_uses_all_six_cameras():
    image = np.full((450, 800, 3), 7, dtype=np.uint8)
    hugsim_obs = _fake_hugsim_obs(image)
    hugsim_info = _fake_hugsim_info(0.5)
    mapping = HUGSIMFrameMapping(
        official_scene_name="scene-0013",
        recon_scene_id=12,
        sample_token="tok1",
        frame_idx=5,
        sample_index=1,
        sample_relative_time_s=0.5,
        hugsim_relative_time_s=0.5,
    )

    obs = build_recondreamer_obs_from_hugsim(
        hugsim_obs=hugsim_obs,
        hugsim_info=hugsim_info,
        mapping=mapping,
        hugsim_step_idx=6,
    )

    assert set(["front_left", "front", "front_right", "back_left", "back", "back_right"]).issubset(obs)
    assert obs["front"].shape == (450, 800, 3)
    assert obs["back_left"][0, 0].tolist() == [10, 10, 10]
    assert obs["back"][0, 0].tolist() == [11, 11, 11]
    assert obs["back_right"][0, 0].tolist() == [12, 12, 12]
    assert obs["scene_id"] == np.int32(12)
    assert obs["frame_idx"] == np.int32(5)
    assert obs["sample_token"] == "tok1"
    assert obs["ego_status"].shape == (8,)
    assert obs["cam2ego"].shape == (6, 4, 4)
    assert obs["cam_intrinsics"].shape == (6, 3, 3)


class FakeHUGSIMEnv:
    def __init__(self):
        self.actions = []
        self.info = {
            "ego_velo": 1.0,
            "ego_steer": 0.0,
            "accelerate": 0.0,
            "command": 2,
            "timestamp": 0.0,
            "ego_pos": [0.0, 0.0, 0.0],
            "ego_rot": [0.0, 0.0, 0.0],
            "cam_params": {},
        }

    def step(self, action):
        self.actions.append(action)
        self.info = dict(self.info)
        self.info["timestamp"] = 0.25 * len(self.actions)
        return {"rgb": {}}, 0.0, False, False, self.info


def test_execute_hugsim_substeps_reuses_same_control(monkeypatch):
    from framework.env_wrapper import hugsim_adapter

    calls = []

    def fake_traj2control(plan, info):
        calls.append((plan.copy(), dict(info)))
        return 0.2, -0.1

    monkeypatch.setattr(hugsim_adapter, "traj2control", fake_traj2control)

    env = FakeHUGSIMEnv()
    obs, reward, terminated, truncated, info = hugsim_adapter.execute_hugsim_control_horizon(
        env=env,
        plan_traj=np.zeros((8, 2), dtype=np.float32),
        initial_info=env.info,
        substeps_per_rl_step=2,
    )

    assert obs == {"rgb": {}}
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert len(calls) == 1
    assert len(env.actions) == 2
    assert env.actions[0] == env.actions[1]
    assert info["timestamp"] == 0.5


def test_hugsim_recon_env_reset_and_step(monkeypatch, tmp_path):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping

    image = np.zeros((450, 800, 3), dtype=np.uint8)

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok0" if relative_time_s < 0.25 else "tok1",
                frame_idx=0 if relative_time_s < 0.25 else 5,
                sample_index=0 if relative_time_s < 0.25 else 1,
                sample_relative_time_s=0.0 if relative_time_s < 0.25 else 0.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def __init__(self):
            self.timestamp = 0.0

        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(self.timestamp)

        def step(self, action):
            self.timestamp += 0.25
            return _fake_hugsim_obs(image), 0.0, False, False, _fake_hugsim_info(self.timestamp)

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            env.step({"acc": 0.0, "steer_rate": 0.0})[0],
            0.0,
            False,
            False,
            _fake_hugsim_info(0.5),
        ),
    )

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        reward_cfg={},
        output_root=tmp_path,
    )

    obs, info = env.reset()
    assert obs["frame_idx"] == np.int32(0)
    assert info["sample_token"] == "tok0"

    next_obs, reward, terminated, truncated, info = env.step((0.0, 0.0, 0.0, 2))
    assert next_obs["frame_idx"] == np.int32(5)
    assert info["scene_id"] == 12
    assert info["frame_idx"] == 5
    assert info["sample_token"] == "tok1"


def test_hugsim_reward_proxy_loads_recon_expert_trajectory(tmp_path):
    from framework.env_wrapper.hugsim_adapter import HUGSIMRewardProxy

    scene_dir = tmp_path / "012" / "ego_pose"
    scene_dir.mkdir(parents=True)
    for frame, x in [(0, 0.0), (5, 1.0), (10, 2.0)]:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = x
        np.savetxt(scene_dir / f"{frame:03d}.txt", pose)

    proxy = HUGSIMRewardProxy(recon_data_root=tmp_path)
    proxy.update_from_hugsim_info(
        recon_scene_id=12,
        frame_idx=5,
        hugsim_info={
            "ego_pos": [1.0, 0.0, 0.0],
            "ego_rot": [0.0, 0.0, 0.0],
            "ego_velo": 2.0,
            "accelerate": 0.5,
            "command": 2,
        },
    )

    assert proxy.scene == 12
    assert proxy.now_frame == 5
    assert len(proxy.all_expert_ego) == 3
    assert len(proxy.expert_pair) > 3
    assert np.asarray(proxy.expert_pair)[0].tolist() == [0.0, 0.0]
    assert np.asarray(proxy.expert_pair)[-1].tolist() == [2.0, 0.0]
    assert proxy._status_vel_xy.tolist() == [2.0, 0.0]
    assert proxy._status_acc_xy.tolist() == [0.5, 0.0]


def test_hugsim_reward_proxy_converts_recon_ego_pose_xy_to_reward_xz(tmp_path):
    from framework.env_wrapper.hugsim_adapter import HUGSIMRewardProxy

    scene_dir = tmp_path / "048" / "ego_pose"
    scene_dir.mkdir(parents=True)
    for frame, xy in [(0, (775.0, 1570.0)), (5, (775.0, 1565.0)), (10, (775.0, 1560.0))]:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = xy[0]
        pose[1, 3] = xy[1]
        np.savetxt(scene_dir / f"{frame:03d}.txt", pose)

    proxy = HUGSIMRewardProxy(recon_data_root=tmp_path)
    reward_pose = np.eye(4, dtype=np.float64)
    reward_pose[:3, 3] = [775.0, 0.0, 1565.0]
    proxy.update_from_hugsim_info(
        recon_scene_id=48,
        frame_idx=5,
        hugsim_info={"ego_velo": 0.0, "accelerate": 0.0, "command": 2},
        reward_pose=reward_pose,
    )

    assert proxy.start_ego[:3, 3].tolist() == pytest.approx([775.0, 0.0, 1565.0])
    assert np.asarray(proxy.expert_pair)[0].tolist() == pytest.approx([775.0, 1570.0])
    assert np.asarray(proxy.expert_pair)[-1].tolist() == pytest.approx([775.0, 1560.0])


def test_hugsim_recon_env_uses_recondreamer_reward_not_hugsim_reward(monkeypatch, tmp_path):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping

    image = np.zeros((450, 800, 3), dtype=np.uint8)

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok1",
                frame_idx=5,
                sample_index=1,
                sample_relative_time_s=0.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

    calls = []

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            self.reward_cfg = reward_cfg

        def reset(self):
            calls.append(("reset",))

        def compute(self, *, env, info, step_idx, done):
            calls.append(("compute", env.scene, env.now_frame, info["hugsim_base_reward"], step_idx, done))
            out = dict(info)
            out["reward_mode"] = "fake_recondreamer"
            out["reward"] = 7.5
            return TrackingRewardResult(reward=7.5, info=out)

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            _fake_hugsim_obs(image),
            123.0,
            False,
            False,
            _fake_hugsim_info(0.5),
        ),
    )
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        reward_cfg={"mode": "step_path"},
        output_root=tmp_path,
        recon_data_root=tmp_path,
    )

    env.reset()
    _obs, reward, terminated, truncated, info = env.step((0.0, 0.0, 0.0, 2))

    assert reward == 7.5
    assert info["reward_mode"] == "fake_recondreamer"
    assert info["hugsim_base_reward"] == 123.0
    assert not terminated
    assert not truncated
    assert ("compute", 12, 5, 123.0, 5, False) in calls


def test_hugsim_recon_env_exposes_collision_terminal_metadata(monkeypatch, tmp_path):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping

    image = np.zeros((450, 800, 3), dtype=np.uint8)

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok1",
                frame_idx=5,
                sample_index=1,
                sample_relative_time_s=0.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            pass

        def reset(self):
            pass

        def compute(self, *, env, info, step_idx, done):
            out = dict(info)
            out["reward"] = -3.0
            return TrackingRewardResult(reward=-3.0, info=out)

    terminal_info = _fake_hugsim_info(0.5)
    terminal_info["collision"] = True
    terminal_info["rc"] = 0.25
    terminal_info["ego_box"] = [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]
    terminal_info["obj_boxes"] = [[0.25, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]]

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            _fake_hugsim_obs(image),
            -100.0,
            True,
            False,
            terminal_info,
        ),
    )
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        reward_cfg={"mode": "step_path"},
        output_root=tmp_path,
        recon_data_root=tmp_path,
    )

    env.reset()
    _obs, reward, terminated, truncated, info = env.step((0.0, 0.0, 0.0, 2))

    assert reward == -3.0
    assert terminated
    assert not truncated
    assert info["dynamic_collision"] is True
    assert info["static_collision"] is False
    assert info["collision"] is True
    assert info["terminal_kind"] == "failure"
    assert info["done_reason"] == "hugsim_collision"
    assert info["ego_box"] == terminal_info["ego_box"]
    assert info["obj_boxes"] == terminal_info["obj_boxes"]


def test_hugsim_recon_env_forces_done_when_hugsim_reports_collision_without_termination(monkeypatch, tmp_path):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping

    image = np.zeros((450, 800, 3), dtype=np.uint8)
    calls = []

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok1",
                frame_idx=5,
                sample_index=1,
                sample_relative_time_s=0.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            pass

        def reset(self):
            pass

        def compute(self, *, env, info, step_idx, done):
            calls.append((done, dict(info)))
            return TrackingRewardResult(reward=-9.0, info=dict(info))

    collision_info = _fake_hugsim_info(0.5)
    collision_info["collision"] = True
    collision_info["rc"] = 0.2

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            _fake_hugsim_obs(image),
            0.0,
            False,
            False,
            collision_info,
        ),
    )
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        reward_cfg={"mode": "step_path"},
        output_root=tmp_path,
        recon_data_root=tmp_path,
    )

    env.reset()
    _obs, reward, terminated, truncated, info = env.step((0.0, 0.0, 0.0, 2))

    assert reward == -9.0
    assert terminated
    assert not truncated
    assert calls[-1][0] is True
    assert info["dynamic_collision"] is True
    assert info["terminal_kind"] == "failure"
    assert info["done_reason"] == "hugsim_collision"


def test_hugsim_recon_env_forces_done_when_aligned_bev_objects_collide(monkeypatch, tmp_path):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_recon_alignment import HUGSIMReconAlignment, Sim2Transform

    image = np.zeros((450, 800, 3), dtype=np.uint8)

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok1",
                frame_idx=5,
                sample_index=1,
                sample_relative_time_s=0.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            pass

        def reset(self):
            pass

        def compute(self, *, env, info, step_idx, done):
            assert done is True
            return TrackingRewardResult(reward=-30.0, info=dict(info))

    step_info = _fake_hugsim_info(0.5)
    step_info["collision"] = False
    step_info["ego_box"] = [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]
    step_info["obj_boxes"] = [[11.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]]
    alignment = HUGSIMReconAlignment(
        official_scene_name="scene-0013",
        recon_scene_id=12,
        transform=Sim2Transform(scale=1.0, rotation=np.eye(2), translation_xy=np.asarray([100.0, 50.0])),
        valid=True,
    )

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            _fake_hugsim_obs(image),
            0.0,
            False,
            False,
            step_info,
        ),
    )
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)
    monkeypatch.setattr(hugsim_adapter, "build_hugsim_recon_alignment", lambda **kwargs: alignment)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        reward_cfg={"mode": "step_path"},
        output_root=tmp_path,
        recon_data_root=tmp_path,
        hugsim_model_base=tmp_path / "hugsim",
    )

    env.reset()
    _obs, reward, terminated, truncated, info = env.step((0.0, 0.0, 0.0, 2))

    assert reward == -30.0
    assert terminated
    assert not truncated
    assert info["collision"] is True
    assert info["dynamic_collision"] is True
    assert info["terminal_kind"] == "failure"
    assert info["done_reason"] == "hugsim_collision"
    assert info["hugsim_aligned_collision_tokens"] == ["hugsim_obj_0"]


def test_hugsim_recon_env_collides_with_recon_cache_objects_using_local_alignment_when_global_invalid(
    monkeypatch, tmp_path
):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_recon_alignment import HUGSIMReconAlignment, Sim2Transform
    import json

    image = np.zeros((450, 800, 3), dtype=np.uint8)
    hugsim_scene_root = tmp_path / "hugsim" / "scene-0013"
    recon_scene_root = tmp_path / "recon" / "012" / "ego_pose"
    recon_cache_root = tmp_path / "recon" / "012"
    hugsim_scene_root.mkdir(parents=True)
    recon_scene_root.mkdir(parents=True)

    hugsim_poses = []
    for xy in [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]:
        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = xy[0]
        pose[0, 3] = -xy[1]
        hugsim_poses.append(pose)
    with (hugsim_scene_root / "ground_param.pkl").open("wb") as handle:
        pickle.dump((np.stack(hugsim_poses), [0.0, 0.5, 1.0], [2, 2, 2]), handle)
    for idx, xy in enumerate([(100.0, 50.0), (110.0, 50.0), (120.0, 50.0)]):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = xy[0]
        pose[1, 3] = xy[1]
        np.savetxt(recon_scene_root / f"{idx:03d}.txt", pose)
    with (recon_cache_root / "env_cache.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "1": {
                    "dynamic_objects": [
                        {
                            "token": "veh_cache",
                            "category": "vehicle.car",
                            "poly": [[112.0, 51.0], [112.0, 49.0], [108.0, 49.0], [108.0, 51.0]],
                        }
                    ]
                }
            },
            handle,
        )

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok1",
                frame_idx=1,
                sample_index=1,
                sample_relative_time_s=0.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

    step_info = _fake_hugsim_info(0.5)
    step_info["collision"] = False
    step_info["ego_box"] = [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]
    global_alignment = HUGSIMReconAlignment(
        official_scene_name="scene-0013",
        recon_scene_id=12,
        transform=Sim2Transform(
            scale=1.0,
            rotation=np.eye(2),
            translation_xy=np.asarray([1000.0, 1000.0]),
            rmse_m=9.0,
        ),
        valid=False,
        reason="rmse_m>2.000",
    )

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            pass

        def reset(self):
            pass

        def compute(self, *, env, info, step_idx, done):
            assert done is True
            return TrackingRewardResult(reward=-12.0, info=dict(info))

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            _fake_hugsim_obs(image),
            0.0,
            False,
            False,
            step_info,
        ),
    )
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)
    monkeypatch.setattr(hugsim_adapter, "build_hugsim_recon_alignment", lambda **kwargs: global_alignment)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        output_root=tmp_path,
        recon_data_root=tmp_path / "recon",
        hugsim_model_base=tmp_path / "hugsim",
    )

    env.reset()
    _obs, reward, terminated, truncated, info = env.step((0.0, 0.0, 0.0, 2))

    assert reward == -12.0
    assert terminated
    assert not truncated
    assert info["collision"] is True
    assert info["dynamic_collision"] is True
    assert info["hugsim_recon_alignment_valid"] is True
    assert info["hugsim_recon_alignment_mode"] == "local_frame"
    assert info["hugsim_aligned_collision_tokens"] == ["veh_cache"]


def test_hugsim_recon_env_selects_recon_cache_frame_by_aligned_ego_position(
    monkeypatch, tmp_path
):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_recon_alignment import HUGSIMReconAlignment, Sim2Transform
    import json

    image = np.zeros((450, 800, 3), dtype=np.uint8)
    recon_scene_root = tmp_path / "recon" / "012" / "ego_pose"
    recon_cache_root = tmp_path / "recon" / "012"
    recon_scene_root.mkdir(parents=True)
    for frame, xy in [(125, (125.0, 0.0)), (169, (169.0, 0.0))]:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = xy[0]
        pose[1, 3] = xy[1]
        np.savetxt(recon_scene_root / f"{frame:03d}.txt", pose)
    with (recon_cache_root / "env_cache.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "125": {
                    "dynamic_objects": [
                        {
                            "token": "wrong_time_vehicle",
                            "category": "vehicle.car",
                            "poly": [[171.0, 1.0], [171.0, -1.0], [167.0, -1.0], [167.0, 1.0]],
                        }
                    ]
                },
                "169": {
                    "dynamic_objects": [
                        {
                            "token": "nearest_pose_vehicle",
                            "category": "vehicle.car",
                            "poly": [[205.0, 1.0], [205.0, -1.0], [201.0, -1.0], [201.0, 1.0]],
                        }
                    ]
                },
            },
            handle,
        )

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok125",
                frame_idx=125,
                sample_index=25,
                sample_relative_time_s=12.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

    step_info = _fake_hugsim_info(12.5)
    step_info["collision"] = False
    step_info["ego_box"] = [169.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]
    alignment = HUGSIMReconAlignment(
        official_scene_name="scene-0013",
        recon_scene_id=12,
        transform=Sim2Transform(scale=1.0, rotation=np.eye(2), translation_xy=np.zeros((2,))),
        valid=True,
        mode="global",
    )

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            pass

        def reset(self):
            pass

        def compute(self, *, env, info, step_idx, done):
            assert done is False
            return TrackingRewardResult(reward=1.0, info=dict(info))

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            _fake_hugsim_obs(image),
            0.0,
            False,
            False,
            step_info,
        ),
    )
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)
    monkeypatch.setattr(hugsim_adapter, "build_hugsim_recon_alignment", lambda **kwargs: alignment)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        output_root=tmp_path,
        recon_data_root=tmp_path / "recon",
        hugsim_model_base=tmp_path / "hugsim",
    )

    env.reset()
    _obs, _reward, terminated, truncated, info = env.step((0.0, 0.0, 0.0, 2))

    assert not terminated
    assert not truncated
    assert info["collision"] is False
    assert info["recon_cache_frame_idx"] == 169
    assert info["recon_cache_frame_source"] == "nearest_pose"
    assert info["recon_cache_time_frame_idx"] == 125
    assert info["recon_cache_frame_pose_dist_m"] == pytest.approx(0.0)
    assert info["recon_cache_dynamic_objects"][0]["token"] == "nearest_pose_vehicle"


def test_hugsim_recon_env_uses_aligned_recon_global_pose_for_reward(monkeypatch, tmp_path):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_recon_alignment import HUGSIMReconAlignment, Sim2Transform

    image = np.zeros((450, 800, 3), dtype=np.uint8)

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok1",
                frame_idx=5,
                sample_index=1,
                sample_relative_time_s=0.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

    step_info = _fake_hugsim_info(0.5)
    step_info["ego_box"] = [10.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.0]

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            pass

        def reset(self):
            pass

        def compute(self, *, env, info, step_idx, done):
            out = dict(info)
            out["reward"] = 1.0
            return TrackingRewardResult(reward=1.0, info=out)

    alignment = HUGSIMReconAlignment(
        official_scene_name="scene-0013",
        recon_scene_id=12,
        transform=Sim2Transform(scale=1.0, rotation=np.eye(2), translation_xy=np.asarray([100.0, 50.0])),
        valid=True,
    )

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            _fake_hugsim_obs(image),
            0.0,
            False,
            False,
            step_info,
        ),
    )
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)
    monkeypatch.setattr(hugsim_adapter, "build_hugsim_recon_alignment", lambda **kwargs: alignment)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        reward_cfg={"mode": "step_path"},
        output_root=tmp_path,
        recon_data_root=tmp_path,
        hugsim_model_base=tmp_path / "hugsim",
    )

    env.reset()
    _obs, _reward, _terminated, _truncated, info = env.step((0.0, 0.0, 0.0, 2))

    assert env._reward_proxy.start_ego[:3, 3].tolist() == pytest.approx([110.0, 0.0, 52.0])
    assert info["hugsim_recon_alignment_valid"] is True
    assert info["recon_global_ego_xy"] == pytest.approx([110.0, 52.0])
    assert info["reward_pose_source"] == "hugsim_recon_alignment"


def test_hugsim_recon_env_adds_recon_cache_dynamic_objects_and_front_obstacle_metrics(monkeypatch, tmp_path):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_recon_alignment import HUGSIMReconAlignment, Sim2Transform
    import json

    image = np.zeros((450, 800, 3), dtype=np.uint8)
    cache_dir = tmp_path / "012"
    cache_dir.mkdir(parents=True)
    with (cache_dir / "env_cache.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "5": {
                    "dynamic_objects": [
                        {
                            "token": "veh_a",
                            "category": "vehicle.car",
                            "poly": [[113.0, 51.0], [113.0, 49.0], [111.0, 49.0], [111.0, 51.0]],
                            "velocity": [0.0, 0.0],
                        }
                    ]
                }
            },
            handle,
        )

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok1",
                frame_idx=5,
                sample_index=1,
                sample_relative_time_s=0.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

    step_info = _fake_hugsim_info(0.5)
    step_info["ego_box"] = [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]
    alignment = HUGSIMReconAlignment(
        official_scene_name="scene-0013",
        recon_scene_id=12,
        transform=Sim2Transform(scale=1.0, rotation=np.eye(2), translation_xy=np.asarray([100.0, 50.0])),
        valid=True,
    )

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            pass

        def reset(self):
            pass

        def compute(self, *, env, info, step_idx, done):
            return TrackingRewardResult(reward=1.0, info=dict(info))

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            _fake_hugsim_obs(image),
            0.0,
            False,
            False,
            step_info,
        ),
    )
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)
    monkeypatch.setattr(hugsim_adapter, "build_hugsim_recon_alignment", lambda **kwargs: alignment)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        output_root=tmp_path,
        recon_data_root=tmp_path,
        hugsim_model_base=tmp_path / "hugsim",
    )

    env.reset()
    _obs, _reward, _terminated, _truncated, info = env.step((0.0, 0.0, 0.0, 2))

    assert info["recon_cache_dynamic_objects"][0]["token"] == "veh_a"
    assert info["front_obstacle_available"] is True
    assert info["front_obstacle_gap_m"] == pytest.approx(2.0)
    assert info["front_obstacle_lateral_m"] == pytest.approx(0.0)


def test_hugsim_recon_env_transforms_inserted_hugsim_objects_to_recon_global(monkeypatch, tmp_path):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_recon_alignment import HUGSIMReconAlignment, Sim2Transform

    image = np.zeros((450, 800, 3), dtype=np.uint8)

    class FakeSceneIndex:
        def map_time(self, official_scene_name, relative_time_s):
            return HUGSIMFrameMapping(
                official_scene_name=official_scene_name,
                recon_scene_id=12,
                sample_token="tok1",
                frame_idx=5,
                sample_index=1,
                sample_relative_time_s=0.5,
                hugsim_relative_time_s=relative_time_s,
            )

    class FakeEnv:
        def reset(self):
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

    step_info = _fake_hugsim_info(0.5)
    step_info["ego_box"] = [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]
    step_info["obj_boxes"] = [[12.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]]
    alignment = HUGSIMReconAlignment(
        official_scene_name="scene-0013",
        recon_scene_id=12,
        transform=Sim2Transform(scale=1.0, rotation=np.eye(2), translation_xy=np.asarray([100.0, 50.0])),
        valid=True,
    )

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            pass

        def reset(self):
            pass

        def compute(self, *, env, info, step_idx, done):
            return TrackingRewardResult(reward=1.0, info=dict(info))

    monkeypatch.setattr(hugsim_adapter, "create_hugsim_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        hugsim_adapter,
        "execute_hugsim_control_horizon",
        lambda env, plan_traj, initial_info, substeps_per_rl_step, hugsim_repo: (
            _fake_hugsim_obs(image),
            0.0,
            False,
            False,
            step_info,
        ),
    )
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)
    monkeypatch.setattr(hugsim_adapter, "build_hugsim_recon_alignment", lambda **kwargs: alignment)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        output_root=tmp_path,
        recon_data_root=tmp_path,
        hugsim_model_base=tmp_path / "hugsim",
    )

    env.reset()
    _obs, _reward, _terminated, _truncated, info = env.step((0.0, 0.0, 0.0, 2))

    assert len(info["hugsim_ego_box_recon_global_poly"]) >= 4
    assert info["hugsim_obj_boxes_recon_global"][0]["source"] == "hugsim_inserted"
    assert info["hugsim_obj_boxes_recon_global"][0]["token"] == "hugsim_obj_0"
    assert len(info["hugsim_obj_boxes_recon_global"][0]["poly"]) >= 4
