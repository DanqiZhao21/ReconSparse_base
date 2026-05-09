from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CameraFrameRecord:
    token: str
    log_name: str
    camera_paths: dict[str, Path]


def _normalize_camera_name(camera_name: str) -> str:
    name = str(camera_name)
    return name.upper() if name.upper().startswith("CAM_") else name.replace("cam_", "CAM_").upper()


def iter_camera_frame_records(
    *,
    navsim_log_path: Path,
    num_history_frames: int,
    num_future_frames: int,
    frame_interval: int,
    has_route: bool,
) -> Iterable[CameraFrameRecord]:
    num_frames = int(num_history_frames) + int(num_future_frames)
    current_frame_index = int(num_history_frames) - 1
    for log_path in sorted(Path(navsim_log_path).glob("*.pkl")):
        with log_path.open("rb") as f:
            frames = pickle.load(f)
        for start in range(0, len(frames), int(frame_interval)):
            frame_list = frames[start : start + num_frames]
            if len(frame_list) < num_frames:
                continue
            current = frame_list[current_frame_index]
            if bool(has_route) and len(current.get("roadblock_ids", [])) == 0:
                continue
            cams = current.get("cams", {})
            camera_paths = {
                str(camera_name): Path(camera_info["data_path"])
                for camera_name, camera_info in cams.items()
                if isinstance(camera_info, dict) and "data_path" in camera_info
            }
            yield CameraFrameRecord(
                token=str(current["token"]),
                log_name=str(current["log_name"]),
                camera_paths=camera_paths,
            )


def select_valid_image_tokens(
    records: Iterable[CameraFrameRecord],
    *,
    sensor_root: Path,
    camera_names: Sequence[str],
    max_tokens: int,
) -> list[CameraFrameRecord]:
    required_cameras = tuple(_normalize_camera_name(camera_name) for camera_name in camera_names)
    selected: list[CameraFrameRecord] = []
    for record in records:
        valid = True
        for camera_name in required_cameras:
            rel_path = record.camera_paths.get(camera_name)
            if rel_path is None or not (Path(sensor_root) / rel_path).is_file():
                valid = False
                break
        if valid:
            selected.append(record)
            if len(selected) >= int(max_tokens):
                break
    return selected


def write_token_list(records: Sequence[CameraFrameRecord], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(record.token for record in records) + ("\n" if records else ""))


def _yaml_list(values: Sequence[str]) -> str:
    return "\n".join(f"  - {value}" for value in values)


def write_scene_filter_yaml(
    records: Sequence[CameraFrameRecord],
    output_path: Path,
    *,
    num_history_frames: int,
    num_future_frames: int,
    frame_interval: int,
    has_route: bool,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_names = sorted({record.log_name for record in records})
    tokens = sorted({record.token for record in records})
    text = "\n".join(
        [
            "_target_: navsim.common.dataclasses.SceneFilter",
            "_convert_: 'all'",
            "",
            f"num_history_frames: {int(num_history_frames)}",
            f"num_future_frames: {int(num_future_frames)}",
            f"frame_interval: {int(frame_interval)}",
            f"has_route: {str(bool(has_route)).lower()}",
            "max_scenes: null",
            "log_names:",
            _yaml_list(log_names),
            "tokens:",
            _yaml_list(tokens),
            "include_synthetic_scenes: false",
            "all_mapping: null",
            "synthetic_scene_tokens: null",
            "reactive_synthetic_initial_tokens: null",
            "non_reactive_synthetic_initial_tokens: null",
            "",
        ]
    )
    output_path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a NavSIM token filter requiring current-frame camera images")
    parser.add_argument("--navsim-root", default="/OpenDataset/navsim/dataset")
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--output-token-list", required=True)
    parser.add_argument("--output-scene-filter", required=True)
    parser.add_argument("--camera-names", nargs="+", default=["cam_l0", "cam_f0", "cam_r0"])
    parser.add_argument("--max-tokens", type=int, default=20_000)
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=8)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--has-route", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    navsim_root = Path(args.navsim_root)
    records = iter_camera_frame_records(
        navsim_log_path=navsim_root / "navsim_logs" / str(args.split),
        num_history_frames=int(args.num_history_frames),
        num_future_frames=int(args.num_future_frames),
        frame_interval=int(args.frame_interval),
        has_route=bool(args.has_route),
    )
    selected = select_valid_image_tokens(
        records,
        sensor_root=navsim_root / "sensor_blobs" / str(args.split),
        camera_names=args.camera_names,
        max_tokens=int(args.max_tokens),
    )
    write_token_list(selected, Path(args.output_token_list))
    write_scene_filter_yaml(
        selected,
        Path(args.output_scene_filter),
        num_history_frames=int(args.num_history_frames),
        num_future_frames=int(args.num_future_frames),
        frame_interval=int(args.frame_interval),
        has_route=bool(args.has_route),
    )
    print(
        f"[rewardmodel-token-filter] selected={len(selected)} max_tokens={int(args.max_tokens)} "
        f"token_list={args.output_token_list} scene_filter={args.output_scene_filter}"
    )


if __name__ == "__main__":
    main()
