from __future__ import annotations

from pathlib import Path

import yaml

from tools.train_eval_pipeline import (
    build_eval_environment,
    build_eval_run_name,
    build_promoted_ckpt_name,
    build_run_tags,
    build_eval_existing_ckpt_command,
    build_train_specs,
    build_training_summary,
    cleanup_training_artifacts,
    detect_next_version,
    parse_train_eval_args,
    prepare_training_config,
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


def test_promoted_ckpt_name_accepts_detailed_run_id(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    existing = "20260520_123456_Craft_ReinforcePP_GRPO_safety_ver03_latest.ckpt"
    (ckpt_dir / existing).write_text("old", encoding="utf-8")

    run_id = "20260520_123456_Craft_ReinforcePP_GRPO_safety"

    assert detect_next_version(ckpt_dir=ckpt_dir, algo_tag=run_id) == 4
    assert (
        build_promoted_ckpt_name(date_tag="20260520", algo_tag=run_id, version=4)
        == "20260520_123456_Craft_ReinforcePP_GRPO_safety_ver04_latest.ckpt"
    )


def test_build_run_tags_describes_training_config() -> None:
    payload = {
        "train": {
            "algo": "reinforcepp",
            "CRAFT": {"enable": True},
            "grpo": {"enable": True},
        },
        "agent": {
            "nuscenes_scorer": {"backend": "craft_carl", "ea_gate_enabled": True},
        },
    }

    assert build_run_tags(config=payload, algo_tag="reinforcepp_craft_safety") == [
        "Craft",
        "ReinforcePP",
        "GRPO",
        "EA",
        "safety",
    ]


def test_prepare_training_config_copies_yaml_and_points_buffer_inside_run_dir(tmp_path: Path) -> None:
    src = tmp_path / "source.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "train": {
                    "algo": "grpo_only",
                    "actor_learner": {
                        "buffer_dir": "outputs/old_buffer",
                        "timestamp_buffer_dir": True,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "TrainEvaluationAuto" / "20260520_123456_NoCraft_GRPOOnly_GRPO"

    prepared = prepare_training_config(
        source_config=src,
        run_dir=run_dir,
        run_id="20260520_123456_NoCraft_GRPOOnly_GRPO",
    )

    copied = yaml.safe_load(prepared.config_path.read_text(encoding="utf-8"))
    expected_buffer = run_dir / "actor_learner"
    assert prepared.config_path == run_dir / "configs" / "20260520_123456_NoCraft_GRPOOnly_GRPO_source.yaml"
    assert prepared.buffer_dir == expected_buffer
    assert copied["train"]["actor_learner"]["buffer_dir"] == str(expected_buffer)
    assert copied["train"]["actor_learner"]["timestamp_buffer_dir"] is False
    assert copied["train"]["actor_learner"]["resolved_from_config"] == str(src)


def test_cleanup_training_artifacts_removes_large_actor_learner_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    buffer_dir = run_dir / "actor_learner"
    for rel in [
        "buffer/shards/a.pt",
        "buffer/consumed/b.pt",
        "weights/latest.ckpt",
        "actors/actor0.log",
    ]:
        path = buffer_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    for name in ["STOP", "TRAINING_LOCK"]:
        (buffer_dir / name).write_text("marker", encoding="utf-8")
    keep = run_dir / "configs" / "train.yaml"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("train: {}", encoding="utf-8")
    manifest = run_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    (run_dir / "train.log").write_text("log", encoding="utf-8")
    (run_dir / "promoted_ckpt.txt").write_text("/ckpt/latest.ckpt", encoding="utf-8")

    cleanup_training_artifacts(run_dir=run_dir, buffer_dir=buffer_dir)

    assert keep.exists()
    assert manifest.exists()
    assert not (run_dir / "train.log").exists()
    assert not (run_dir / "promoted_ckpt.txt").exists()
    assert not (buffer_dir / "buffer").exists()
    assert not (buffer_dir / "weights").exists()
    assert not (buffer_dir / "actors").exists()
    assert not (buffer_dir / "STOP").exists()
    assert not (buffer_dir / "TRAINING_LOCK").exists()
    assert not buffer_dir.exists()


def test_build_train_specs_allows_distinct_reinforcepp_tag_for_craft() -> None:
    specs = build_train_specs(
        ppo_config=Path("/configs/ppo.yaml"),
        reinforcepp_config=Path("/configs/reinforcepp_craft.yaml"),
        ppo_algo_tag="ppo",
        reinforcepp_algo_tag="reinforcepp_craft",
    )

    assert len(specs) == 1
    assert specs[0].algo_tag == "reinforcepp_craft"
    assert specs[0].config_path == Path("/configs/reinforcepp_craft.yaml")


def test_build_train_specs_defaults_to_reinforcepp_only() -> None:
    specs = build_train_specs(
        ppo_config=Path("/configs/ppo.yaml"),
        reinforcepp_config=Path("/configs/reinforcepp.yaml"),
        ppo_algo_tag="ppo",
        reinforcepp_algo_tag="reinforcepp",
    )

    assert len(specs) == 1
    assert specs[0].algo_tag == "reinforcepp"


def test_build_train_specs_can_add_ppo_explicitly() -> None:
    specs = build_train_specs(
        ppo_config=Path("/configs/ppo.yaml"),
        reinforcepp_config=Path("/configs/reinforcepp.yaml"),
        ppo_algo_tag="ppo",
        reinforcepp_algo_tag="reinforcepp",
        run_ppo=True,
    )

    assert [spec.algo_tag for spec in specs] == ["ppo", "reinforcepp"]


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


def test_build_eval_environment_defaults_hugsim_seed_for_reproducible_pipeline_eval() -> None:
    env = build_eval_environment(base_env={"PATH": "/bin"}, hugsim_random_seed=None, default_eval_seed=True)

    assert env["HUGSIM_RANDOM_SEED"] == "0"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["PATH"] == "/bin"


def test_build_eval_environment_allows_manual_seed_and_explicit_no_default_seed() -> None:
    manual = build_eval_environment(base_env={}, hugsim_random_seed="123", default_eval_seed=True)
    no_default = build_eval_environment(base_env={}, hugsim_random_seed=None, default_eval_seed=False)

    assert manual["HUGSIM_RANDOM_SEED"] == "123"
    assert manual["PYTHONHASHSEED"] == "123"
    assert "HUGSIM_RANDOM_SEED" not in no_default
    assert "PYTHONHASHSEED" not in no_default


def test_build_eval_environment_honors_disable_default_seed_sentinel() -> None:
    env = build_eval_environment(
        base_env={"HUGSIM_DISABLE_DEFAULT_EVAL_SEED": "1"},
        hugsim_random_seed=None,
        default_eval_seed=True,
    )

    assert "HUGSIM_RANDOM_SEED" not in env
    assert "PYTHONHASHSEED" not in env


def test_build_eval_existing_ckpt_command_keeps_plain_evaluation_args() -> None:
    cmd = build_eval_existing_ckpt_command(
        python_bin=Path("/env/bin/python"),
        ckpt_path=Path("/ckpts/model.ckpt"),
        hugsim_template=Path("/hugsim/template.yaml"),
        eval_output_root=Path("/hugsim/outputs"),
        run_name="algo/ckpt",
        repeat_evals=2,
        slots=["0:0"],
        scenario_dir=Path("/hugsim/scenarios"),
        max_scenes=88,
        retry_count=3,
    )

    assert "--no-default-seed" not in cmd
    assert "--hugsim-random-seed" not in cmd
    assert cmd[-2:] == ["--slots", "0:0"]


def test_build_eval_existing_ckpt_command_targets_promoted_ckpt_and_all_scenes() -> None:
    cmd = build_eval_existing_ckpt_command(
        python_bin=Path("/env/bin/python"),
        ckpt_path=Path("/ckpts/20260421_reinforcepp_craft_ver01_latest.ckpt"),
        hugsim_template=Path("/hugsim/configs/sim/template.yaml"),
        eval_output_root=Path("/hugsim/outputs/evaluate-auto"),
        run_name="reinforcepp_craft/20260421_reinforcepp_craft_ver01_latest",
        repeat_evals=2,
        slots=["0:0", "1:1"],
        scenario_dir=Path("/hugsim/configs/scenarios/nuscenes"),
        max_scenes=None,
        retry_count=3,
    )

    assert cmd[:4] == [
        "/env/bin/python",
        "-u",
        str(Path(__file__).resolve().parents[1] / "tools" / "evaluate_existing_sparsedrive_v2_ckpts.py"),
        "--ckpts",
    ]
    assert "/ckpts/20260421_reinforcepp_craft_ver01_latest.ckpt" in cmd
    assert "--repeat-evals" in cmd
    assert cmd[cmd.index("--repeat-evals") + 1] == "2"
    assert "--scenario-dir" in cmd
    assert cmd[cmd.index("--scenario-dir") + 1] == "/hugsim/configs/scenarios/nuscenes"
    assert "--max-scenes" not in cmd
    assert cmd[-3:] == ["--slots", "0:0", "1:1"]


def test_parse_train_eval_args_defaults_to_hugsim_ori_88_scenes_two_repeats_and_8_slots() -> None:
    args = parse_train_eval_args([])

    assert args.repeat_evals == 2
    assert args.max_scenes == 88
    assert args.scenario_dir == Path("/root/clone/HUGSIM-ORI/configs/scenarios/nuscenes")
    assert args.slots == ["0:0", "1:1", "2:2", "3:3", "4:4", "5:5", "6:6", "7:7"]
    assert args.no_default_eval_seed is False


def test_parse_train_eval_args_allows_overriding_default_88_scene_limit() -> None:
    args = parse_train_eval_args(["--max-scenes", "12", "--slots", "0:0"])

    assert args.max_scenes == 12
    assert args.slots == ["0:0"]


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
