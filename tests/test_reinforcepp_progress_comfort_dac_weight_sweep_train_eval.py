from __future__ import annotations

from pathlib import Path

from tools.run_reinforcepp_progress_comfort_dac_weight_sweep_train_eval import default_run_specs


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_run_specs_target_dac_weight_coef01_and_coef08() -> None:
    specs = default_run_specs()

    assert [spec.run_key for spec in specs] == [
        "reinforcepp_progress_comfort_dac_weight_grpo_coef01",
        "reinforcepp_progress_comfort_dac_weight_grpo_coef08",
    ]
    assert [spec.coef for spec in specs] == [0.1, 0.8]
    assert [spec.config_path for spec in specs] == [
        REPO_ROOT
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_progress_comfort_dac_weight_grpo_coef01.yaml",
        REPO_ROOT
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_progress_comfort_dac_weight_grpo_coef08.yaml",
    ]
