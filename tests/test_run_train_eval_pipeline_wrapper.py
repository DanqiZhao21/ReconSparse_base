from __future__ import annotations

from pathlib import Path


def test_run_train_eval_pipeline_defaults_to_20260525_fullpara_config() -> None:
    script = (Path(__file__).resolve().parents[1] / "script/run_train_eval_pipeline.sh").read_text(
        encoding="utf-8"
    )

    assert "202605250049_reinforcepp_closed_loop_sparsedrive_v2_craft_closeCloseloop_openGRPOCraft-FullPara.yaml" in script
    assert "--reinforcepp-config" in script


def test_run_train_eval_pipeline_documents_hugsim_ori_eval_defaults() -> None:
    script = (Path(__file__).resolve().parents[1] / "script/run_train_eval_pipeline.sh").read_text(
        encoding="utf-8"
    )

    assert "HUGSIM_RANDOM_SEED=\"${HUGSIM_RANDOM_SEED:-0}\"" in script
    assert "--slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7" in script
    assert "--max-scenes 88" in script
    assert "--repeat-evals 2" in script


def test_hugsim_ori_train_eval_pipeline_defaults_to_hugsim_shard_config() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "script/run_train_eval_pipeline_hugsim_ori.sh"
    ).read_text(encoding="utf-8")

    assert "202605251610_HUGSM_reinforcepp_closed_loop_closeCloselopop_openGRPOCraft-try.yaml" in script
    assert "--reinforcepp-config" in script
    assert "--reinforcepp-algo-tag hugsim_ori_reinforcepp_craft_grpo" in script
    assert "--slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7" in script
    assert "--max-scenes 88" in script
    assert "--repeat-evals 2" in script
