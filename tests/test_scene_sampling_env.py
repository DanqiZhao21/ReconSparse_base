from __future__ import annotations

import sys
import types

from framework.env_wrapper.subproc_vec_env import SceneSamplingEnv, SceneSamplingSpec


def test_hugsim_scene_switch_closes_previous_env(monkeypatch):
    closed: list[int] = []
    created: list[int] = []

    class FakeHUGSIMReconEnv:
        def __init__(self, *, scenario_name, scenario_path, reward_cfg, cuda, **kwargs):
            self.scene_id = int(scenario_path)
            created.append(self.scene_id)
            self.env = types.SimpleNamespace(now_frame=0)

        def reset(self, scene=None, start_frame=None, step_frames=None):
            return {"obs": self.scene_id}, {"scene": self.scene_id}

        def close(self):
            closed.append(self.scene_id)

    fake_module = types.ModuleType("framework.env_wrapper.hugsim_adapter")
    fake_module.HUGSIMReconEnv = FakeHUGSIMReconEnv
    monkeypatch.setitem(sys.modules, "framework.env_wrapper.hugsim_adapter", fake_module)

    env = SceneSamplingEnv(
        cuda=0,
        reward_cfg={},
        debug=False,
        spec=SceneSamplingSpec(
            scene_ids=[0, 1],
            scene_sampling="sequential",
            ddp_seed=0,
            rank=0,
            worker_id=0,
            start_mode="zero",
            allow_short_tail=False,
            start_min=0,
            start_max=None,
            start_stride=None,
            max_steps=1,
        ),
        env_backend="hugsim_ori",
        hugsim_scenarios=[
            {"official_scene_name": "scene-0000", "scenario_path": "0"},
            {"official_scene_name": "scene-0001", "scenario_path": "1"},
        ],
        hugsim_kwargs={"scene_index": object()},
    )

    env.reset()
    env.reset()

    assert created == [0, 1]
    assert closed == [0]


def test_scene_sampling_env_close_closes_current_env(monkeypatch):
    closed: list[int] = []

    class FakeHUGSIMReconEnv:
        def __init__(self, *, scenario_name, scenario_path, reward_cfg, cuda, **kwargs):
            self.scene_id = int(scenario_path)
            self.env = types.SimpleNamespace(now_frame=0)

        def close(self):
            closed.append(self.scene_id)

    fake_module = types.ModuleType("framework.env_wrapper.hugsim_adapter")
    fake_module.HUGSIMReconEnv = FakeHUGSIMReconEnv
    monkeypatch.setitem(sys.modules, "framework.env_wrapper.hugsim_adapter", fake_module)

    env = SceneSamplingEnv(
        cuda=0,
        reward_cfg={},
        debug=False,
        spec=SceneSamplingSpec(
            scene_ids=[0],
            scene_sampling="sequential",
            ddp_seed=0,
            rank=0,
            worker_id=0,
            start_mode="zero",
            allow_short_tail=False,
            start_min=0,
            start_max=None,
            start_stride=None,
            max_steps=1,
        ),
        env_backend="hugsim_ori",
        hugsim_scenarios=[{"official_scene_name": "scene-0000", "scenario_path": "0"}],
        hugsim_kwargs={"scene_index": object()},
    )

    env.close()

    assert closed == [0]
