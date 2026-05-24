from pathlib import Path

import yaml


def test_hugsim_ori_sparsedrive_v2_smoke_config_selects_hugsim_backend() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "script/configs/sparsedrive_v2/hugsim_ori_sparsedrive_v2_smoke.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert cfg["env"]["backend"] == "hugsim_ori"
    assert cfg["env"]["hugsim"]["launch_mode"] == "fifo"
    assert cfg["env"]["hugsim"]["pixi_cmd"] == "pixi"
    assert cfg["env"]["hugsim"]["fifo_timeout_s"] == 300
    assert cfg["env"]["hugsim"]["substeps_per_rl_step"] == 2
    assert cfg["env"]["hugsim"]["scenes"] == ["scene-0013"]
    assert cfg["train"]["actor_learner"]["actor_horizon"] == 8
