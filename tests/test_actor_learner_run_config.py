from __future__ import annotations

from pathlib import Path

import yaml

from framework.runner.config_normalization import timestamp_actor_learner_buffer_dir
from script.train_actor_learner_v2 import materialize_orchestrator_config


def test_timestamp_actor_learner_buffer_dir_prefixes_new_run_name() -> None:
    cfg = {
        "train": {
            "actor_learner": {
                "buffer_dir": "outputs/actor_learner_reinforcepp_overnight_safe",
            }
        }
    }

    resolved = timestamp_actor_learner_buffer_dir(cfg, timestamp="20260514_153000")

    assert resolved == "outputs/20260514_153000_actor_learner_reinforcepp_overnight_safe"
    assert cfg["train"]["actor_learner"]["buffer_dir"] == resolved


def test_timestamp_actor_learner_buffer_dir_does_not_double_prefix() -> None:
    cfg = {
        "train": {
            "actor_learner": {
                "buffer_dir": "outputs/20260514_153000_actor_learner_reinforcepp_overnight_safe",
            }
        }
    }

    resolved = timestamp_actor_learner_buffer_dir(cfg, timestamp="20260514_154500")

    assert resolved is None
    assert (
        cfg["train"]["actor_learner"]["buffer_dir"]
        == "outputs/20260514_153000_actor_learner_reinforcepp_overnight_safe"
    )


def test_timestamp_actor_learner_buffer_dir_can_be_disabled() -> None:
    cfg = {
        "train": {
            "actor_learner": {
                "buffer_dir": "outputs/actor_learner_fixed_resume",
                "timestamp_buffer_dir": False,
            }
        }
    }

    resolved = timestamp_actor_learner_buffer_dir(cfg, timestamp="20260514_153000")

    assert resolved is None
    assert cfg["train"]["actor_learner"]["buffer_dir"] == "outputs/actor_learner_fixed_resume"


def test_materialize_orchestrator_config_writes_resolved_child_config(tmp_path: Path) -> None:
    config_path = tmp_path / "reinforcepp_closed_loop_sparsedrive_v2_overnight_safe.yaml"
    cfg = {
        "train": {
            "actor_learner": {
                "buffer_dir": "outputs/actor_learner_reinforcepp_overnight_safe",
            }
        },
        "agent": {"type": "sparsedrive_v2"},
    }

    resolved_path = materialize_orchestrator_config(
        cfg,
        config_path=str(config_path),
        timestamp="20260514_153000",
        generated_config_dir=tmp_path / "generated_configs",
    )

    assert resolved_path == str(
        tmp_path
        / "generated_configs"
        / "20260514_153000_reinforcepp_closed_loop_sparsedrive_v2_overnight_safe.yaml"
    )
    resolved = yaml.safe_load(Path(resolved_path).read_text(encoding="utf-8"))
    assert (
        resolved["train"]["actor_learner"]["buffer_dir"]
        == "outputs/20260514_153000_actor_learner_reinforcepp_overnight_safe"
    )
    assert cfg["train"]["actor_learner"]["buffer_dir"] == resolved["train"]["actor_learner"]["buffer_dir"]


def test_materialize_orchestrator_config_archives_yaml_in_timestamped_buffer_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "closed_loop.yaml"
    cfg = {
        "train": {
            "actor_learner": {
                "buffer_dir": str(tmp_path / "outputs" / "actor_learner" / "Noclose_OpenCraftGrpo"),
            }
        },
        "agent": {"type": "sparsedrive_v2"},
    }

    resolved_path = materialize_orchestrator_config(
        cfg,
        config_path=str(config_path),
        timestamp="20260522_010203",
    )

    run_dir = tmp_path / "outputs" / "actor_learner" / "20260522_010203_Noclose_OpenCraftGrpo"
    assert resolved_path == str(run_dir / "20260522_010203_closed_loop.yaml")
    assert Path(resolved_path).is_file()

    resolved = yaml.safe_load(Path(resolved_path).read_text(encoding="utf-8"))
    assert resolved["train"]["actor_learner"]["buffer_dir"] == str(run_dir)
    assert resolved["train"]["actor_learner"]["resolved_from_config"] == str(config_path)
    assert resolved["train"]["actor_learner"]["run_timestamp"] == "20260522_010203"
    assert resolved["train"]["actor_learner"]["resolved_config_path"] == str(run_dir / "20260522_010203_closed_loop.yaml")
