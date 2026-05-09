from __future__ import annotations

from pathlib import Path

from tools.run_reinforcepp_minimal_grpo_ablation_train_eval import default_run_specs


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_run_specs_target_coef001_gate_on_and_coef003_gate_off() -> None:
    specs = default_run_specs()

    assert [spec.run_key for spec in specs] == [
        "reinforcepp_baseline_reward_dac_weight_grpo_coef001",
        "reinforcepp_baseline_reward_dac_weight_grpo_coef003_gateoff",
    ]
    assert [spec.coef for spec in specs] == [0.01, 0.03]
    assert [spec.config_path for spec in specs] == [
        REPO_ROOT
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef001.yaml",
        REPO_ROOT
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef003_gateoff.yaml",
    ]
