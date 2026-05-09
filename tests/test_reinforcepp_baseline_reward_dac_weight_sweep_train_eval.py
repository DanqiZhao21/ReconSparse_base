from __future__ import annotations

from pathlib import Path

from tools.run_reinforcepp_baseline_reward_dac_weight_sweep_train_eval import default_run_specs


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_run_specs_target_baseline_reward_dac_weight_coef003_and_coef01() -> None:
    specs = default_run_specs()

    assert [spec.run_key for spec in specs] == [
        "reinforcepp_baseline_reward_dac_weight_grpo_coef003",
        "reinforcepp_baseline_reward_dac_weight_grpo_coef01",
    ]
    assert [spec.coef for spec in specs] == [0.03, 0.1]
    assert [spec.config_path for spec in specs] == [
        REPO_ROOT
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef003.yaml",
        REPO_ROOT
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef01.yaml",
    ]
