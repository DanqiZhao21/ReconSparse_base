from __future__ import annotations

from framework.runner.actor_runtime import _actor_should_pause_for_learner
from framework.runner.config_normalization import normalize_actor_learner_cfg, resolve_actor_gpu_ids, resolve_learner_gpu_ids
from framework.runner.orchestrator import build_learner_launch_specs


def test_resolve_learner_gpu_ids_defaults_to_existing_scalar() -> None:
    assert resolve_learner_gpu_ids({"learner_gpu_id": 2}) == [2]


def test_resolve_learner_gpu_ids_accepts_comma_string() -> None:
    assert resolve_learner_gpu_ids({"learner_gpu_ids": "0,2,3"}) == [0, 2, 3]


def test_normalize_actor_learner_cfg_preserves_first_learner_gpu_for_compatibility() -> None:
    cfg = {"train": {"actor_learner": {"learner_gpu_ids": [2, 3]}}}

    normalize_actor_learner_cfg(cfg)

    al_cfg = cfg["train"]["actor_learner"]
    assert al_cfg["learner_gpu_ids"] == [2, 3]
    assert al_cfg["learner_gpu_id"] == 2


def test_build_learner_launch_specs_sets_rank_world_and_local_rank() -> None:
    specs = build_learner_launch_specs(
        learner_gpu_ids=[0, 2],
        base_env={"PYTHONPATH": "/repo"},
        entry="entry.py",
        config_path="cfg.yaml",
    )

    assert [spec.rank for spec in specs] == [0, 1]
    assert [spec.local_rank for spec in specs] == [0, 2]
    assert [spec.env["RANK"] for spec in specs] == ["0", "1"]
    assert [spec.env["WORLD_SIZE"] for spec in specs] == ["2", "2"]
    assert [spec.env["LOCAL_RANK"] for spec in specs] == ["0", "2"]
    assert all(spec.env["MASTER_ADDR"] == "127.0.0.1" for spec in specs)
    assert all(spec.env["MASTER_PORT"] == "29500" for spec in specs)
    assert all(spec.cmd == ["python", "entry.py", "--config", "cfg.yaml", "--role", "learner"] for spec in specs)


def test_resolve_actor_gpu_ids_excludes_all_learner_gpus_when_possible(monkeypatch) -> None:
    monkeypatch.setattr("framework.runner.config_normalization.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("framework.runner.config_normalization.torch.cuda.device_count", lambda: 4)

    plan = resolve_actor_gpu_ids(
        {"learner_gpu_ids": [0, 1], "actor_per_gpu": 1},
        num_actors=2,
    )

    assert plan == [2, 3]


def test_resolve_actor_gpu_ids_preserves_single_learner_gpu_order(monkeypatch) -> None:
    monkeypatch.setattr("framework.runner.config_normalization.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("framework.runner.config_normalization.torch.cuda.device_count", lambda: 4)

    plan = resolve_actor_gpu_ids(
        {"learner_gpu_id": 0, "actor_per_gpu": 1},
        num_actors=2,
    )

    assert plan == [0, 1]


def test_actor_pause_detects_any_learner_gpu() -> None:
    al_cfg = {"pause_actor_on_learner_gpu": True, "learner_gpu_ids": [0, 2]}

    assert _actor_should_pause_for_learner(al_cfg, cuda=2)
    assert not _actor_should_pause_for_learner(al_cfg, cuda=3)
