from __future__ import annotations

from pathlib import Path

from tools.run_reinforcepp_progress_comfort_grpo_dac_ttc_coef08_train_eval import default_run_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_run_spec_targets_progress_comfort_dac_ttc_coef08_and_two_eval_repeats() -> None:
    spec = default_run_spec()

    assert spec.run_key == "reinforcepp_progress_comfort_grpo_dac_ttc_coef08"
    assert spec.coef == 0.8
    assert spec.eval_repeats == 2
    assert spec.config_path == (
        REPO_ROOT
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_progress_comfort_grpo_dac_ttc_coef08.yaml"
    )
