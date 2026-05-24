import json
from pathlib import Path

from framework.env_wrapper.hugsim_scene_index import HUGSIMSceneIndex


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_scene_index_maps_official_scene_to_recon_id(tmp_path: Path):
    nusc = tmp_path / "nuscenes" / "v1.0-trainval"
    recon_info = tmp_path / "assets" / "nus" / "information" / "frame2token"

    _write_json(nusc / "scene.json", [{"name": "scene-0013", "token": "scene-token"}])
    _write_json(
        nusc / "sample.json",
        [
            {"token": "tok0", "scene_token": "scene-token", "timestamp": 1000000},
            {"token": "tok1", "scene_token": "scene-token", "timestamp": 1500000},
        ],
    )
    _write_json(recon_info / "012.json", {"tok0": 0, "tok1": 5})

    index = HUGSIMSceneIndex(
        nuscenes_root=nusc,
        frame2token_dir=recon_info,
    )

    assert index.recon_scene_id_for_official_scene("scene-0013") == 12
    assert index.frame_for_sample_token(12, "tok1") == 5


def test_scene_index_maps_hugsim_time_to_nearest_recon_frame(tmp_path: Path):
    nusc = tmp_path / "nuscenes" / "v1.0-trainval"
    recon_info = tmp_path / "assets" / "nus" / "information" / "frame2token"

    _write_json(nusc / "scene.json", [{"name": "scene-0013", "token": "scene-token"}])
    _write_json(
        nusc / "sample.json",
        [
            {"token": "tok0", "scene_token": "scene-token", "timestamp": 1000000},
            {"token": "tok1", "scene_token": "scene-token", "timestamp": 1500000},
            {"token": "tok2", "scene_token": "scene-token", "timestamp": 2000000},
        ],
    )
    _write_json(recon_info / "012.json", {"tok0": 0, "tok1": 5, "tok2": 10})

    index = HUGSIMSceneIndex(nuscenes_root=nusc, frame2token_dir=recon_info)

    mapped = index.map_time("scene-0013", relative_time_s=0.49)
    assert mapped.recon_scene_id == 12
    assert mapped.frame_idx == 5
    assert mapped.sample_token == "tok1"
