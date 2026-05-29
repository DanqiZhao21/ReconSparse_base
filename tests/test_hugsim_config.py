from pathlib import Path

import yaml


def test_hugsim_ori_sparsedrive_v2_smoke_config_selects_hugsim_backend() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "script/configs/sparsedrive_v2/202605291041_HUGSM_reinforcepp_closed_loop_closeCloseloop_openGRPOCraft.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert cfg["env"]["backend"] == "hugsim_ori"
    assert cfg["env"]["hugsim"]["launch_mode"] == "fifo"
    assert cfg["env"]["hugsim"]["pixi_cmd"] == "pixi"
    assert cfg["env"]["hugsim"]["fifo_timeout_s"] == 300
    assert "substeps_per_rl_step" not in cfg["env"]["hugsim"]
    assert cfg["env"]["use_all_scenes"] is True
    assert cfg["env"]["hugsim"]["repo"] == "third_party/HUGSIM-ORI"
    assert cfg["env"]["hugsim"]["scenario_dir"] == "third_party/HUGSIM-ORI/configs/scenarios/nuscenes"
    assert cfg["train"]["actor_learner"]["actor_horizon"] > 0
