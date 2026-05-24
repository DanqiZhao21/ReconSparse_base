from pathlib import Path
import sys

import numpy as np

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


def test_hugsim_fifo_runner_parser_has_required_args():
    from framework.env_wrapper.hugsim_fifo_runner import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--scenario_path",
            "/tmp/scenario.yaml",
            "--base_path",
            "/tmp/base.yaml",
            "--camera_path",
            "/tmp/camera.yaml",
            "--kinematic_path",
            "/tmp/kinematic.yaml",
            "--output_dir",
            "/tmp/out",
        ]
    )

    assert args.ad == "sparsedrive_v2"


def test_hugsim_fifo_runner_adds_hugsim_cwd_to_sys_path(monkeypatch, tmp_path: Path):
    from framework.env_wrapper.hugsim_fifo_runner import ensure_hugsim_import_paths

    monkeypatch.chdir(tmp_path)
    repo = str(tmp_path)
    sim_dir = str(tmp_path / "sim")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in {repo, sim_dir}])

    ensure_hugsim_import_paths()

    assert repo in sys.path
    assert sim_dir in sys.path


def test_hugsim_fifo_client_launches_pixi_runner(monkeypatch, tmp_path: Path):
    from framework.env_wrapper.hugsim_adapter import HUGSIMFifoClient

    launched = {}

    class FakeProcess:
        def __init__(self, cmd, cwd, env):
            launched["cmd"] = cmd
            launched["cwd"] = cwd
            launched["env"] = env

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("subprocess.Popen", lambda cmd, cwd=None, env=None, **kwargs: FakeProcess(cmd, cwd, env))

    client = HUGSIMFifoClient(
        hugsim_repo="/root/clone/HUGSIM-ORI",
        scenario_path="/tmp/scenario.yaml",
        base_path="/tmp/base.yaml",
        camera_path="/tmp/camera.yaml",
        kinematic_path="/tmp/kinematic.yaml",
        output_dir=tmp_path,
        pixi_cmd="pixi",
        cuda=1,
    )
    client.start()

    assert launched["cwd"] == "/root/clone/HUGSIM-ORI"
    assert launched["cmd"][:3] == ["pixi", "run", "python"]
    assert "hugsim_fifo_runner.py" in launched["cmd"][3]
    assert launched["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    output_idx = launched["cmd"].index("--output_dir") + 1
    assert Path(launched["cmd"][output_idx]).is_absolute()


def test_hugsim_recon_env_fifo_mode_uses_client(monkeypatch, tmp_path: Path):
    from framework.env_wrapper import hugsim_adapter
    from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping

    calls = []
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

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def reset(self):
            calls.append(("reset",))
            return _fake_hugsim_obs(image), _fake_hugsim_info(0.0)

        def step(self, plan_traj):
            calls.append(("step", plan_traj.copy()))
            info = _fake_hugsim_info(0.5)
            info["collision"] = False
            info["rc"] = 0.1
            return _fake_hugsim_obs(image), 0.0, False, False, info

        def close(self):
            calls.append(("close",))

    class FakeRewardComputer:
        def __init__(self, reward_cfg):
            pass

        def reset(self):
            calls.append(("reward_reset",))

        def compute(self, *, env, info, step_idx, done):
            calls.append(("reward_compute", env.scene, env.now_frame, step_idx, done))
            out = dict(info)
            out["reward"] = 2.5
            return TrackingRewardResult(reward=2.5, info=out)

    monkeypatch.setattr(hugsim_adapter, "HUGSIMFifoClient", FakeClient)
    monkeypatch.setattr(hugsim_adapter, "TrackingRewardComputer", FakeRewardComputer)

    env = hugsim_adapter.HUGSIMReconEnv(
        scenario_name="scene-0013",
        scenario_path="/tmp/scene-0013-easy-00.yaml",
        scene_index=FakeSceneIndex(),
        reward_cfg={},
        output_root="outputs/test_hugsim_fifo_relative",
        launch_mode="fifo",
        pixi_cmd="pixi",
        fifo_timeout_s=120.0,
        cuda=1,
        recon_data_root=tmp_path,
    )

    obs, info = env.reset()
    assert obs["frame_idx"] == np.int32(0)
    assert info["sample_token"] == "tok0"

    env.set_external_plan_local_xyyaw([[1.0, 0.0, 0.0]] * 8)
    _obs, reward, terminated, truncated, info = env.step((0.0, 0.0, 0.0, 2))
    env.close()

    assert reward == 2.5
    assert not terminated
    assert not truncated
    assert info["sample_token"] == "tok1"
    assert any(call[0] == "step" for call in calls)
    assert any(call[0] == "close" for call in calls)
    init_kwargs = next(call[1] for call in calls if call[0] == "init")
    assert init_kwargs["pixi_cmd"] == "pixi"
    assert init_kwargs["fifo_timeout_s"] == 120.0
    assert init_kwargs["cuda"] == 1
    assert Path(init_kwargs["output_dir"]).is_absolute()
