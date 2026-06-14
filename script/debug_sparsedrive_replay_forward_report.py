from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be a mapping: {path}")
    return payload


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_config(run_dir: Path | None, config: str | Path | None) -> Path:
    if config is not None:
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return path
    if run_dir is None:
        raise ValueError("Either --run-dir or --config is required")
    configs = sorted(run_dir.glob("*.yaml"))
    if len(configs) != 1:
        raise FileNotFoundError(f"Expected exactly one YAML config in {run_dir}, found {len(configs)}")
    return configs[0]


def find_latest_manifest(run_dir: Path | None, manifest: str | Path | None) -> Path:
    if manifest is not None:
        path = Path(manifest)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path
    if run_dir is None:
        raise ValueError("Either --run-dir or --manifest is required")
    manifests = sorted((run_dir / "debug_retention").glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No debug_retention manifest found under {run_dir}")

    def sort_key(path: Path) -> tuple[int, float, str]:
        try:
            payload = _load_json(path)
            return (int(payload.get("update_index", -1)), float(payload.get("created_time", 0.0)), str(path))
        except Exception:
            return (-1, 0.0, str(path))

    return sorted(manifests, key=sort_key)[-1]


def resolve_ckpt_history_dir(run_dir: Path | None, ckpt_history_dir: str | Path | None) -> Path | None:
    if ckpt_history_dir is not None:
        path = Path(ckpt_history_dir)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint history dir not found: {path}")
        return path
    if run_dir is None:
        return None
    path = run_dir / "weights" / "history"
    return path if path.exists() else None


def infer_weights_version(manifest_path: Path, override: int | None) -> int | None:
    if override is not None:
        return int(override)
    payload = _load_json(manifest_path)
    shards = payload.get("shards", [])
    if isinstance(shards, list):
        for shard in shards:
            if isinstance(shard, dict) and shard.get("weights_version", None) is not None:
                return int(shard["weights_version"])
    return None


def checkpoint_for_version(
    *,
    weights_version: int | None,
    ckpt_history_dir: Path | None,
    version0_ckpt: str | Path | None,
) -> Path | None:
    if weights_version is None:
        return None
    if int(weights_version) == 0:
        return Path(version0_ckpt) if version0_ckpt is not None else None
    if ckpt_history_dir is None:
        return None
    return ckpt_history_dir / f"version_{int(weights_version):06d}.ckpt"


def prepare_candidate_shard_dir(manifest: Path, *, weights_version: int | None, target_dir: Path) -> Path:
    payload = _load_json(manifest)
    shards = payload.get("shards", [])
    if not isinstance(shards, list):
        raise ValueError(f"Manifest field 'shards' must be a list: {manifest}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for old_path in target_dir.glob("*.pt"):
        old_path.unlink()

    selected = 0
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        if weights_version is not None and int(shard.get("weights_version", -1)) != int(weights_version):
            continue
        archive_path = shard.get("archive_path", shard.get("path", None))
        if archive_path is None:
            continue
        source = Path(archive_path)
        if not source.exists():
            raise FileNotFoundError(f"Manifest shard not found: {source}")
        link = target_dir / f"{selected:06d}_{source.name}"
        link.symlink_to(source)
        selected += 1
    if selected == 0:
        raise FileNotFoundError(f"No candidate shards selected from {manifest} for weights_version={weights_version}")
    return target_dir


def build_child_command(
    script_path: str,
    *,
    config: Path,
    manifest: Path,
    ckpt_history_dir: Path | None,
    version0_ckpt: str | Path | None,
    weights_version: int | None,
    batch_size: int,
    target_index: int | None,
    out_dir: Path,
    device: str,
    tolerance: float,
    disable_tf32: bool,
    deterministic: bool,
    extra_args: Sequence[str] = (),
) -> List[str]:
    command = [
        sys.executable,
        script_path,
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        str(device),
        "--tolerance",
        str(float(tolerance)),
        "--out-dir",
        str(out_dir),
    ]
    if ckpt_history_dir is not None:
        command.extend(["--ckpt-history-dir", str(ckpt_history_dir)])
    if version0_ckpt is not None:
        command.extend(["--version0-ckpt", str(version0_ckpt)])
    if weights_version is not None:
        command.extend(["--weights-version", str(int(weights_version))])
    if target_index is not None:
        command.extend(["--target-index", str(int(target_index))])
    if disable_tf32:
        command.append("--disable-tf32")
    if deterministic:
        command.append("--deterministic")
    command.extend(list(extra_args))
    return command


def choose_target_index(candidate_rows: Path, first_failures: Path, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    for row in _read_csv_rows(candidate_rows):
        if row.get("batch_present") == "0" and row.get("single_present_when_missing") == "1":
            return int(row["index"])
    failures = _read_csv_rows(first_failures)
    if failures:
        return int(failures[0]["sample_index"])
    return 0


def load_stage_summary(path: Path) -> Dict[str, Any]:
    rows = _read_csv_rows(path)
    failed = [row for row in rows if int(row.get("failed", "0") or 0) > 0]
    first = failed[0] if failed else None
    return {
        "failed_stage_count": len(failed),
        "first_failed_stage": None if first is None else first.get("stage"),
        "first_failed_max_abs_error": None if first is None else first.get("max_abs_error"),
        "first_failed_mismatch_count": None if first is None else first.get("mismatch_count"),
    }


def load_p_deform_summary(path: Path) -> Dict[str, Any]:
    rows = _read_csv_rows(path)
    failed = [row for row in rows if int(row.get("failed", "0") or 0) > 0]
    first = failed[0] if failed else None
    return {
        "failed_stage_count": len(failed),
        "first_failed_case": None if first is None else first.get("case"),
        "first_failed_stage": None if first is None else first.get("stage"),
        "first_failed_max_abs_error": None if first is None else first.get("max_abs_error"),
        "first_failed_mismatch_count": None if first is None else first.get("mismatch_count"),
    }


def run_command(command: Sequence[str], *, dry_run: bool) -> None:
    print("[replay-report] run " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# SparseDriveV2 Replay Forward Debug Report",
        "",
        f"- run_dir: `{payload.get('run_dir')}`",
        f"- config: `{payload.get('config')}`",
        f"- manifest: `{payload.get('manifest')}`",
        f"- weights_version: `{payload.get('weights_version')}`",
        f"- batch_size: `{payload.get('batch_size')}`",
        f"- target_index: `{payload.get('target_index')}`",
        "",
        "## Full Forward Trace",
        "",
        f"- first_failed_stage: `{payload['forward_trace'].get('first_failed_stage')}`",
        f"- failed_stage_count: `{payload['forward_trace'].get('failed_stage_count')}`",
        f"- first_failed_max_abs_error: `{payload['forward_trace'].get('first_failed_max_abs_error')}`",
        f"- first_failed_mismatch_count: `{payload['forward_trace'].get('first_failed_mismatch_count')}`",
        "",
        "## Decoder0 p_deform_model Trace",
        "",
        f"- first_failed_case: `{payload['p_deform'].get('first_failed_case')}`",
        f"- first_failed_stage: `{payload['p_deform'].get('first_failed_stage')}`",
        f"- failed_stage_count: `{payload['p_deform'].get('failed_stage_count')}`",
        f"- first_failed_max_abs_error: `{payload['p_deform'].get('first_failed_max_abs_error')}`",
        f"- first_failed_mismatch_count: `{payload['p_deform'].get('first_failed_mismatch_count')}`",
        "",
        "## Artifacts",
        "",
        f"- forward_trace: `{payload['artifacts'].get('forward_trace_dir')}`",
        f"- p_deform: `{payload['artifacts'].get('p_deform_dir')}`",
    ]
    if payload["artifacts"].get("candidates_dir"):
        lines.append(f"- candidates: `{payload['artifacts'].get('candidates_dir')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a unified SparseDriveV2 replay single/batch forward report from retained debug shards."
    )
    parser.add_argument("--run-dir", default=None, help="Actor-learner run directory with config, weights, debug_retention")
    parser.add_argument("--config", default=None, help="Override config YAML")
    parser.add_argument("--manifest", default=None, help="Override debug_retention manifest.json")
    parser.add_argument("--ckpt-history-dir", default=None, help="Override weights/history directory")
    parser.add_argument("--version0-ckpt", default=None, help="Initial checkpoint for weights_version=0")
    parser.add_argument("--weights-version", type=int, default=None, help="Replay weights version to select")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-samples", type=int, default=None, help="Forward trace sample count; defaults to batch-size")
    parser.add_argument("--target-index", type=int, default=None, help="Replay index inside selected manifest/version samples")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-candidates", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print child commands without executing model forward")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir is not None else None
    config = find_config(run_dir, args.config)
    manifest = find_latest_manifest(run_dir, args.manifest)
    ckpt_history_dir = resolve_ckpt_history_dir(run_dir, args.ckpt_history_dir)
    weights_version = infer_weights_version(manifest, args.weights_version)
    selected_ckpt = checkpoint_for_version(
        weights_version=weights_version,
        ckpt_history_dir=ckpt_history_dir,
        version0_ckpt=args.version0_ckpt,
    )
    if selected_ckpt is not None and not selected_ckpt.exists():
        raise FileNotFoundError(f"Selected checkpoint not found: {selected_ckpt}")

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = Path(args.out_dir or Path("outputs") / "replay_forward_report" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    forward_dir = out_dir / "forward_trace"
    p_deform_dir = out_dir / "p_deform"
    candidates_dir = out_dir / "candidates"

    sample_count = int(args.limit_samples or args.batch_size)
    forward_command = build_child_command(
        "script/debug_sparsedrive_batch_invariance.py",
        config=config,
        manifest=manifest,
        ckpt_history_dir=ckpt_history_dir,
        version0_ckpt=args.version0_ckpt,
        weights_version=weights_version,
        batch_size=int(args.batch_size),
        target_index=None,
        out_dir=forward_dir,
        device=str(args.device),
        tolerance=float(args.tolerance),
        disable_tf32=bool(args.disable_tf32),
        deterministic=bool(args.deterministic),
        extra_args=["--limit-samples", str(sample_count)],
    )
    run_command(forward_command, dry_run=bool(args.dry_run))

    candidate_rows = candidates_dir / "rows.csv"
    if not bool(args.skip_candidates):
        if selected_ckpt is None:
            print("[replay-report] skip candidates: selected checkpoint could not be resolved")
        else:
            candidates_dir.mkdir(parents=True, exist_ok=True)
            candidate_shard_dir = prepare_candidate_shard_dir(
                manifest,
                weights_version=weights_version,
                target_dir=out_dir / "candidate_shards",
            )
            candidate_command = [
                sys.executable,
                "script/debug_sparsedrive_replay_batch_candidates.py",
                "--config",
                str(config),
                "--ckpt",
                str(selected_ckpt),
                "--shard-dir",
                str(candidate_shard_dir),
                "--device",
                str(args.device if args.device != "auto" else "cuda:0"),
                "--batch-size",
                str(int(args.batch_size)),
                "--limit-samples",
                str(sample_count),
                "--out-dir",
                str(candidates_dir),
            ]
            run_command(candidate_command, dry_run=bool(args.dry_run))

    target_index = choose_target_index(
        candidate_rows,
        forward_dir / "first_failures.csv",
        explicit=args.target_index,
    )
    p_deform_command = build_child_command(
        "script/debug_sparsedrive_deformable_batch.py",
        config=config,
        manifest=manifest,
        ckpt_history_dir=ckpt_history_dir,
        version0_ckpt=args.version0_ckpt,
        weights_version=weights_version,
        batch_size=int(args.batch_size),
        target_index=int(target_index),
        out_dir=p_deform_dir,
        device=str(args.device),
        tolerance=float(args.tolerance),
        disable_tf32=bool(args.disable_tf32),
        deterministic=bool(args.deterministic),
    )
    run_command(p_deform_command, dry_run=bool(args.dry_run))

    payload = {
        "run_dir": None if run_dir is None else str(run_dir),
        "config": str(config),
        "manifest": str(manifest),
        "ckpt_history_dir": None if ckpt_history_dir is None else str(ckpt_history_dir),
        "selected_ckpt": None if selected_ckpt is None else str(selected_ckpt),
        "weights_version": weights_version,
        "batch_size": int(args.batch_size),
        "sample_count": int(sample_count),
        "target_index": int(target_index),
        "tolerance": float(args.tolerance),
        "forward_trace": load_stage_summary(forward_dir / "stage_summary.csv") if not args.dry_run else {},
        "p_deform": load_p_deform_summary(p_deform_dir / "p_deform_summary.csv") if not args.dry_run else {},
        "artifacts": {
            "forward_trace_dir": str(forward_dir),
            "p_deform_dir": str(p_deform_dir),
            "candidates_dir": None if bool(args.skip_candidates) else str(candidates_dir),
        },
    }
    _write_json(out_dir / "summary.json", payload)
    _write_report(out_dir / "report.md", payload)
    print(f"[replay-report] wrote {out_dir / 'summary.json'}")
    print(f"[replay-report] wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
