import json

from framework.runner.env_factory import discover_hugsim_scenarios


def test_discover_hugsim_scenarios_reads_yaml_names(tmp_path):
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "scene-0013-easy-00.yaml").write_text("scene_name: scene-0013\nmode: easy_00\n", encoding="utf-8")
    (scenario_dir / "scene-0038-hard-00.yaml").write_text("scene_name: scene-0038\nmode: hard_00\n", encoding="utf-8")

    scenarios = discover_hugsim_scenarios(str(scenario_dir))

    assert [s.official_scene_name for s in scenarios] == ["scene-0013", "scene-0038"]


def test_build_actor_env_filters_hugsim_by_scenario_file_stem(monkeypatch, tmp_path):
    from framework.runner import env_factory

    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "scene-0051-easy-00.yaml").write_text("scene_name: scene-0051\nmode: easy_00\n", encoding="utf-8")
    (scenario_dir / "scene-0051-medium-00.yaml").write_text(
        "scene_name: scene-0051\nmode: medium_00\n", encoding="utf-8"
    )
    nusc_root = tmp_path / "nuscenes" / "v1.0-trainval"
    frame2token = tmp_path / "assets" / "nus" / "information" / "frame2token"
    nusc_root.mkdir(parents=True)
    frame2token.mkdir(parents=True)
    (nusc_root / "scene.json").write_text(json.dumps([{"name": "scene-0051", "token": "scene-token"}]), encoding="utf-8")
    (nusc_root / "sample.json").write_text(
        json.dumps([{"token": "tok0", "scene_token": "scene-token", "timestamp": 1000000}]),
        encoding="utf-8",
    )
    (frame2token / "048.json").write_text(json.dumps({"tok0": 0}), encoding="utf-8")

    captured = {}

    def fake_make_scene_sampling_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("framework.env_wrapper.make_scene_sampling_env", fake_make_scene_sampling_env)

    cfg = {
        "env": {
            "backend": "hugsim_ori",
            "use_all_scenes": True,
            "scene_sampling": "sequential",
            "hugsim": {
                "scenario_dir": str(scenario_dir),
                "scenes": ["scene-0051-medium-00"],
                "nuscenes_root": str(nusc_root),
                "frame2token_dir": str(frame2token),
            },
        },
        "train": {"actor_learner": {"scene_shard_by_actor": True}},
    }

    env_factory.build_actor_env(cfg, cuda=0, actor_id=0, total_actors=1)

    assert captured["hugsim_scenarios"] == [
        {"official_scene_name": "scene-0051", "scenario_path": str(scenario_dir / "scene-0051-medium-00.yaml")}
    ]


def test_build_actor_env_passes_hugsim_backend_without_recon_ckpt(monkeypatch, tmp_path):
    from framework.runner import env_factory

    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "scene-0013-easy-00.yaml").write_text("scene_name: scene-0013\nmode: easy_00\n", encoding="utf-8")
    nusc_root = tmp_path / "nuscenes" / "v1.0-trainval"
    frame2token = tmp_path / "assets" / "nus" / "information" / "frame2token"
    nusc_root.mkdir(parents=True)
    frame2token.mkdir(parents=True)
    (nusc_root / "scene.json").write_text(json.dumps([{"name": "scene-0013", "token": "scene-token"}]), encoding="utf-8")
    (nusc_root / "sample.json").write_text(
        json.dumps([{"token": "tok0", "scene_token": "scene-token", "timestamp": 1000000}]),
        encoding="utf-8",
    )
    (frame2token / "012.json").write_text(json.dumps({"tok0": 0}), encoding="utf-8")

    captured = {}

    def fake_make_scene_sampling_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("framework.env_wrapper.make_scene_sampling_env", fake_make_scene_sampling_env)

    cfg = {
        "env": {
            "backend": "hugsim_ori",
            "use_all_scenes": True,
            "scene_sampling": "sequential",
            "hugsim": {
                "scenario_dir": str(scenario_dir),
                "scenes": ["scene-0013"],
                "nuscenes_root": str(nusc_root),
                "frame2token_dir": str(frame2token),
                "launch_mode": "fifo",
                "pixi_cmd": "pixi",
                "fifo_timeout_s": 120.0,
                "min_gt_route_points": 3,
            },
        },
        "train": {"actor_learner": {"scene_shard_by_actor": True}},
    }

    env_factory.build_actor_env(cfg, cuda=0, actor_id=0, total_actors=1)

    assert captured["env_backend"] == "hugsim_ori"
    assert captured["scene_ids"] == [0]
    assert captured["hugsim_scenarios"] == [
        {"official_scene_name": "scene-0013", "scenario_path": str(scenario_dir / "scene-0013-easy-00.yaml")}
    ]
    assert "substeps_per_rl_step" not in captured["hugsim_kwargs"]
    assert captured["hugsim_kwargs"]["launch_mode"] == "fifo"
    assert captured["hugsim_kwargs"]["pixi_cmd"] == "pixi"
    assert captured["hugsim_kwargs"]["fifo_timeout_s"] == 120.0
    assert captured["hugsim_kwargs"]["min_gt_route_points"] == 3
    assert captured["hugsim_kwargs"]["output_namespace"] == "actor0_worker0"


def test_build_actor_env_can_random_sample_all_hugsim_scenes_with_worker_namespace(monkeypatch, tmp_path):
    from framework.runner import env_factory

    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "scene-0013-easy-00.yaml").write_text("scene_name: scene-0013\nmode: easy_00\n", encoding="utf-8")
    (scenario_dir / "scene-0013-medium-00.yaml").write_text(
        "scene_name: scene-0013\nmode: medium_00\n", encoding="utf-8"
    )
    nusc_root = tmp_path / "nuscenes" / "v1.0-trainval"
    frame2token = tmp_path / "assets" / "nus" / "information" / "frame2token"
    nusc_root.mkdir(parents=True)
    frame2token.mkdir(parents=True)
    (nusc_root / "scene.json").write_text(json.dumps([{"name": "scene-0013", "token": "scene-token"}]), encoding="utf-8")
    (nusc_root / "sample.json").write_text(
        json.dumps([{"token": "tok0", "scene_token": "scene-token", "timestamp": 1000000}]),
        encoding="utf-8",
    )
    (frame2token / "012.json").write_text(json.dumps({"tok0": 0}), encoding="utf-8")

    captured = {}

    def fake_make_scene_sampling_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("framework.env_wrapper.make_scene_sampling_env", fake_make_scene_sampling_env)

    cfg = {
        "env": {
            "backend": "hugsim_ori",
            "use_all_scenes": True,
            "scene_sampling": "random",
            "hugsim": {
                "scenario_dir": str(scenario_dir),
                "scenes": ["scene-0013-easy-00", "scene-0013-medium-00"],
                "nuscenes_root": str(nusc_root),
                "frame2token_dir": str(frame2token),
            },
        },
        "train": {"actor_learner": {"scene_shard_by_actor": False}},
    }

    env_factory.build_actor_env(cfg, cuda=0, actor_id=1, total_actors=2, worker_id=1001)

    assert captured["scene_ids"] == [0, 1]
    assert captured["scene_sampling"] == "random"
    assert captured["hugsim_kwargs"]["output_namespace"] == "actor1_worker1001"


def test_build_actor_env_defaults_hugsim_launch_mode_to_fifo(monkeypatch, tmp_path):
    from framework.runner import env_factory

    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "scene-0013-easy-00.yaml").write_text("scene_name: scene-0013\nmode: easy_00\n", encoding="utf-8")
    nusc_root = tmp_path / "nuscenes" / "v1.0-trainval"
    frame2token = tmp_path / "assets" / "nus" / "information" / "frame2token"
    nusc_root.mkdir(parents=True)
    frame2token.mkdir(parents=True)
    (nusc_root / "scene.json").write_text(json.dumps([{"name": "scene-0013", "token": "scene-token"}]), encoding="utf-8")
    (nusc_root / "sample.json").write_text(
        json.dumps([{"token": "tok0", "scene_token": "scene-token", "timestamp": 1000000}]),
        encoding="utf-8",
    )
    (frame2token / "012.json").write_text(json.dumps({"tok0": 0}), encoding="utf-8")

    captured = {}

    def fake_make_scene_sampling_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("framework.env_wrapper.make_scene_sampling_env", fake_make_scene_sampling_env)

    cfg = {
        "env": {
            "backend": "hugsim_ori",
            "use_all_scenes": True,
            "hugsim": {
                "scenario_dir": str(scenario_dir),
                "nuscenes_root": str(nusc_root),
                "frame2token_dir": str(frame2token),
            },
        }
    }

    env_factory.build_actor_env(cfg, cuda=0, actor_id=0, total_actors=1)

    assert captured["hugsim_kwargs"]["launch_mode"] == "fifo"
