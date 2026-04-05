import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "egoADs" / "SparseDriveV2"))

from navsim.agents.sparsedrive.sparsedrive_paths import (
    backbone_model_kwargs,
    normalize_checkpoint_state_dict,
)
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig, sparsedrive_root


def test_sparsedrive_config_defaults_resolve_under_egoads():
    root = sparsedrive_root()
    cfg = SparseDriveConfig()

    assert cfg.bkb_path == str(root / "ckpt" / "resnet34.a1_in1k" / "pytorch_model.bin")
    assert cfg.path_anchor == str(root / "ckpt" / "kmeans" / "path_1024.npy")
    assert cfg.velocity_anchor == str(root / "ckpt" / "kmeans" / "velocity_256.npy")
    assert cfg.trajectory_anchor == str(root / "ckpt" / "kmeans" / "trajectory_1024_256.npz")


def test_backbone_model_kwargs_disable_pretrained_when_local_file_missing(tmp_path):
    kwargs = backbone_model_kwargs(num_levels=4, bkb_path=str(tmp_path / "missing.bin"))

    assert kwargs["pretrained"] is False
    assert "pretrained_cfg_overlay" not in kwargs
    assert kwargs["out_indices"] == (1, 2, 3, 4)


def test_backbone_model_kwargs_use_local_pretrained_file_when_present(tmp_path):
    bkb_file = tmp_path / "resnet.bin"
    bkb_file.write_bytes(b"test")
    kwargs = backbone_model_kwargs(num_levels=2, bkb_path=str(bkb_file))

    assert kwargs["pretrained"] is True
    assert kwargs["pretrained_cfg_overlay"] == {"file": str(bkb_file)}
    assert kwargs["out_indices"] == (3, 4)


def test_normalize_checkpoint_state_dict_prefixes_legacy_agent_keys():
    state_dict = {
        "agent._backbone.img_backbone.conv1.weight": 1,
        "agent._status_encoding.weight": 2,
        "agent._trajectory_head.decoder.layers.0.weight": 3,
    }

    normalized = normalize_checkpoint_state_dict(state_dict)

    assert normalized == {
        "_sparsedrive_model._backbone.img_backbone.conv1.weight": 1,
        "_sparsedrive_model._status_encoding.weight": 2,
        "_sparsedrive_model._trajectory_head.decoder.layers.0.weight": 3,
    }


def test_normalize_checkpoint_state_dict_keeps_current_model_prefix():
    state_dict = {
        "agent._sparsedrive_model._backbone.img_backbone.conv1.weight": 1,
    }

    normalized = normalize_checkpoint_state_dict(state_dict)

    assert normalized == {
        "_sparsedrive_model._backbone.img_backbone.conv1.weight": 1,
    }
