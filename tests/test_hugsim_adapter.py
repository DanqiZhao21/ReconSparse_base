import numpy as np

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
        },
    }


def test_build_recondreamer_obs_uses_three_front_cameras():
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

    assert set(["front_left", "front", "front_right"]).issubset(obs)
    assert obs["front"].shape == (450, 800, 3)
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
