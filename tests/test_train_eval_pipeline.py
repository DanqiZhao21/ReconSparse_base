from __future__ import annotations

from pathlib import Path

import yaml

from tools.train_eval_pipeline import (
    build_eval_run_name,
    build_promoted_ckpt_name,
    build_training_summary,
    detect_next_version,
    rewrite_hugsim_eval_config,
    write_run_manifest,
)


def test_detect_next_version_uses_existing_latest_versions(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "20260421_ppo_ver03_latest.ckpt").write_text("a", encoding="utf-8")
    (ckpt_dir / "20260422_ppo_ver11_latest.ckpt").write_text("b", encoding="utf-8")
    (ckpt_dir / "20260420_reinforcepp_ver02_latest.ckpt").write_text("c", encoding="utf-8")

    assert detect_next_version(ckpt_dir=ckpt_dir, algo_tag="ppo") == 12
    assert detect_next_version(ckpt_dir=ckpt_dir, algo_tag="reinforcepp") == 3
    assert detect_next_version(ckpt_dir=ckpt_dir, algo_tag="newalgo") == 1


def test_build_promoted_ckpt_name_matches_expected_pattern() -> None:
    out = build_promoted_ckpt_name(date_tag="20260421", algo_tag="ppo", version=32)
    assert out == "20260421_ppo_ver32_latest.ckpt"


def test_rewrite_hugsim_eval_config_updates_ckpt_and_output_dir(tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "sparsedrive_v2_ckpt": "/old/model.ckpt",
                "sparsedrive_v2_pretrain_ckpt": "/old/pretrain.ckpt",
                "output_dir": "/old/output_prefix_",
                "untouched": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_config = tmp_path / "generated.yaml"

    rewrite_hugsim_eval_config(
        template_config=template,
        output_config=output_config,
        ckpt_path=Path("/new/20260421_ppo_ver32_latest.ckpt"),
        output_prefix="/root/clone/HUGSIM-ORI/outputs/evaluate-auto/ppo/20260421_ppo_ver32_latest/repeat_01/nusc_",
    )

    payload = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    assert payload["sparsedrive_v2_ckpt"] == "/new/20260421_ppo_ver32_latest.ckpt"
    assert payload["sparsedrive_v2_pretrain_ckpt"] == "/new/20260421_ppo_ver32_latest.ckpt"
    assert payload["output_dir"] == "/root/clone/HUGSIM-ORI/outputs/evaluate-auto/ppo/20260421_ppo_ver32_latest/repeat_01/nusc_"
    assert payload["untouched"] == 1


def test_build_eval_run_name_is_partitioned_by_algo_ckpt_and_repeat() -> None:
    run_name = build_eval_run_name(algo_tag="reinforcepp", ckpt_stem="20260421_reinforcepp_ver07_latest", repeat_idx=2)
    assert run_name == "reinforcepp/20260421_reinforcepp_ver07_latest/repeat_02"


def test_write_run_manifest_persists_key_training_metadata(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = write_run_manifest(
        manifest_path=manifest_path,
        data={
            "algo_tag": "ppo",
            "version": 32,
            "promoted_ckpt": "/tmp/20260421_ppo_ver32_latest.ckpt",
            "config_path": "/tmp/ppo.yaml",
            "buffer_dir": "/tmp/buffer",
        },
    )

    assert manifest_path.exists()
    assert payload["algo_tag"] == "ppo"
    assert payload["version"] == 32
    assert payload["promoted_ckpt"] == "/tmp/20260421_ppo_ver32_latest.ckpt"


def test_build_training_summary_contains_core_run_identifiers() -> None:
    summary = build_training_summary(
        algo_tag="reinforcepp",
        config_path=Path("/tmp/reinforcepp.yaml"),
        buffer_dir=Path("/tmp/actor_learner_reinforce"),
        latest_ckpt=Path("/tmp/actor_learner_reinforce/weights/latest.ckpt"),
        promoted_ckpt=Path("/tmp/20260421_reinforcepp_ver07_latest.ckpt"),
        version=7,
    )

    assert "algo=reinforcepp" in summary
    assert "version=ver07" in summary
    assert "/tmp/reinforcepp.yaml" in summary
    assert "/tmp/actor_learner_reinforce" in summary
    assert "/tmp/20260421_reinforcepp_ver07_latest.ckpt" in summary
