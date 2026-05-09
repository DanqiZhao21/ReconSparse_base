from __future__ import annotations

from pathlib import Path

from tools.run_reinforcepp_dac9_grpo_coef_sweep_train_eval import (
    build_promoted_ckpt_name,
    default_run_specs,
    detect_next_version,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_run_specs_cover_requested_reinforcepp_dac9_coef_sweep() -> None:
    specs = default_run_specs()

    assert [spec.run_key for spec in specs] == [
        "reinforcepp_dac9_grpo_coef1",
        "reinforcepp_dac9_grpo_coef01",
        "reinforcepp_dac9_grpo_coef05",
    ]
    assert [spec.coef for spec in specs] == [1.0, 0.1, 0.5]
    assert all(spec.algo_tag == "reinforcepp_dac9" for spec in specs)
    assert [spec.config_path for spec in specs] == [
        REPO_ROOT / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef1.yaml",
        REPO_ROOT / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef01.yaml",
        REPO_ROOT / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef05.yaml",
    ]


def test_promoted_ckpt_names_are_partitioned_by_run_key_without_duplicate_algo_tag(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "20260426_reinforcepp_dac9_grpo_coef1_ver01_latest.ckpt").write_text("a", encoding="utf-8")
    (ckpt_dir / "20260426_reinforcepp_dac9_grpo_coef01_ver07_latest.ckpt").write_text("b", encoding="utf-8")

    assert detect_next_version(ckpt_dir=ckpt_dir, run_key_prefix="reinforcepp_dac9_grpo") == 8
    assert (
        build_promoted_ckpt_name(
            date_tag="20260426",
            run_key="reinforcepp_dac9_grpo_coef05",
            version=8,
        )
        == "20260426_reinforcepp_dac9_grpo_coef05_ver08_latest.ckpt"
    )
