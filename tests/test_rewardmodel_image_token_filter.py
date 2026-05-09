from __future__ import annotations

from pathlib import Path

from framework.rewardmodel.data.build_navsim_image_token_filter import (
    CameraFrameRecord,
    select_valid_image_tokens,
    write_scene_filter_yaml,
    write_token_list,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpg")


def test_select_valid_image_tokens_requires_all_current_frame_cameras(tmp_path: Path) -> None:
    sensor_root = tmp_path / "sensor_blobs" / "trainval"
    valid = CameraFrameRecord(
        token="token-valid",
        log_name="log-a",
        camera_paths={
            "CAM_L0": Path("log-a/CAM_L0/valid-l0.jpg"),
            "CAM_F0": Path("log-a/CAM_F0/valid-f0.jpg"),
            "CAM_R0": Path("log-a/CAM_R0/valid-r0.jpg"),
        },
    )
    missing_right = CameraFrameRecord(
        token="token-missing",
        log_name="log-a",
        camera_paths={
            "CAM_L0": Path("log-a/CAM_L0/missing-l0.jpg"),
            "CAM_F0": Path("log-a/CAM_F0/missing-f0.jpg"),
            "CAM_R0": Path("log-a/CAM_R0/missing-r0.jpg"),
        },
    )
    for rel_path in valid.camera_paths.values():
        _touch(sensor_root / rel_path)
    _touch(sensor_root / missing_right.camera_paths["CAM_L0"])
    _touch(sensor_root / missing_right.camera_paths["CAM_F0"])

    selected = select_valid_image_tokens(
        [valid, missing_right],
        sensor_root=sensor_root,
        camera_names=("CAM_L0", "CAM_F0", "CAM_R0"),
        max_tokens=20_000,
    )

    assert selected == [valid]


def test_select_valid_image_tokens_caps_output_count(tmp_path: Path) -> None:
    sensor_root = tmp_path / "sensor_blobs" / "trainval"
    records = []
    for idx in range(3):
        record = CameraFrameRecord(
            token=f"token-{idx}",
            log_name="log-a",
            camera_paths={
                "CAM_F0": Path(f"log-a/CAM_F0/{idx}.jpg"),
            },
        )
        _touch(sensor_root / record.camera_paths["CAM_F0"])
        records.append(record)

    selected = select_valid_image_tokens(records, sensor_root=sensor_root, camera_names=("CAM_F0",), max_tokens=2)

    assert [record.token for record in selected] == ["token-0", "token-1"]


def test_writers_emit_token_list_and_hydra_scene_filter_yaml(tmp_path: Path) -> None:
    selected = [
        CameraFrameRecord(token="token-b", log_name="log-b", camera_paths={}),
        CameraFrameRecord(token="token-a", log_name="log-a", camera_paths={}),
    ]
    token_path = tmp_path / "tokens.txt"
    yaml_path = tmp_path / "scene_filter.yaml"

    write_token_list(selected, token_path)
    write_scene_filter_yaml(
        selected,
        yaml_path,
        num_history_frames=4,
        num_future_frames=8,
        frame_interval=1,
        has_route=True,
    )

    assert token_path.read_text().splitlines() == ["token-b", "token-a"]
    text = yaml_path.read_text()
    assert "_target_: navsim.common.dataclasses.SceneFilter" in text
    assert "num_history_frames: 4" in text
    assert "num_future_frames: 8" in text
    assert "frame_interval: 1" in text
    assert "log_names:" in text
    assert "  - log-a" in text
    assert "  - log-b" in text
    assert "tokens:" in text
    assert "  - token-a" in text
    assert "  - token-b" in text
