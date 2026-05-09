from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
HUGSIM_ROOT = Path("/root/clone/HUGSIM-ORI")
TRAIN_PYTHON = Path(os.environ.get("TRAIN_PYTHON", "/root/miniconda3/envs/recondreamerNew-rl/bin/python"))
SPARSEDRIVE_CKPT_DIR = REPO_ROOT / "egoADs" / "SparseDriveV2" / "ckpt"
DEFAULT_SCENARIO_DIR = HUGSIM_ROOT / "configs" / "scenarios" / "nuscenes"
DEFAULT_EVAL_OUTPUT_ROOT = HUGSIM_ROOT / "outputs" / "evaluate-auto"


@dataclass(frozen=True)
class RunSpec:
    run_key: str
    coef: float
    eval_repeats: int
    config_path: Path


def default_run_spec() -> RunSpec:
    return RunSpec(
        run_key="reinforcepp_dac9_grpo_coef08",
        coef=0.8,
        eval_repeats=2,
        config_path=REPO_ROOT
        / "script"
        / "configs"
        / "sparsedrive_v2"
        / "reinforcepp_closed_loop_sparsedrive_v2_dac9_grpo_coef08.yaml",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected mapping config: {path}")
    return payload


def _resolve_buffer_dir(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    train_cfg = cfg.get("train", {}) or {}
    actor_learner_cfg = train_cfg.get("actor_learner", {}) or {}
    return (REPO_ROOT / str(actor_learner_cfg.get("buffer_dir", "outputs/actor_learner"))).resolve()


def _matching_eval_processes() -> list[str]:
    out = subprocess.run(
        ["ps", "-eo", "pid,ppid,sid,stat,etime,cmd"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    matches = []
    current_pid = str(os.getpid())
    for line in out:
        if current_pid in line:
            continue
        if "evaluate_existing_sparsedrive_v2_ckpts.py" in line or "/root/clone/HUGSIM-ORI/closed_loop.py" in line:
            matches.append(line.strip())
    return matches


def wait_for_current_eval(*, wait_log: Path, poll_s: float, dry_run: bool) -> None:
    wait_log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        wait_log.write_text("[dry-run] would wait for active evaluation processes\n", encoding="utf-8")
        return
    while True:
        matches = _matching_eval_processes()
        with wait_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}] active_eval_processes={len(matches)}\n")
            for match in matches:
                handle.write(f"  {match}\n")
        if not matches:
            return
        time.sleep(float(poll_s))


def _run_command(*, cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[cwd] {cwd}\n")
        handle.write(f"[cmd] {' '.join(cmd)}\n")
        handle.flush()
        if dry_run:
            return
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code={result.returncode}: {' '.join(cmd)}; see {log_path}")


def detect_next_version(*, ckpt_dir: Path, run_key: str) -> int:
    import re

    pattern = re.compile(rf"^\d{{8}}_{re.escape(run_key)}_ver(\d+)_latest\.ckpt$")
    max_ver = 0
    for path in ckpt_dir.glob(f"*_{run_key}_ver*_latest.ckpt"):
        match = pattern.match(path.name)
        if match is None:
            continue
        max_ver = max(max_ver, int(match.group(1)))
    return max_ver + 1


def build_promoted_ckpt_name(*, date_tag: str, run_key: str, version: int) -> str:
    return f"{date_tag}_{run_key}_ver{int(version):02d}_latest.ckpt"


def run_training(*, spec: RunSpec, run_root: Path, dry_run: bool) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    cmd = [
        str(TRAIN_PYTHON),
        "-u",
        str(REPO_ROOT / "script" / "train_actor_learner_v2.py"),
        "--role",
        "orchestrator",
        "--config",
        str(spec.config_path),
    ]
    _run_command(cmd=cmd, cwd=REPO_ROOT, env=env, log_path=run_root / spec.run_key / "train.log", dry_run=dry_run)
    latest_ckpt = _resolve_buffer_dir(spec.config_path) / "weights" / "latest.ckpt"
    if not dry_run and not latest_ckpt.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {latest_ckpt}")
    return latest_ckpt


def promote_checkpoint(*, src_ckpt: Path, spec: RunSpec, date_tag: str, dry_run: bool) -> tuple[Path, int]:
    version = detect_next_version(ckpt_dir=SPARSEDRIVE_CKPT_DIR, run_key=spec.run_key)
    target = SPARSEDRIVE_CKPT_DIR / build_promoted_ckpt_name(date_tag=date_tag, run_key=spec.run_key, version=version)
    if not dry_run:
        SPARSEDRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_ckpt, target)
    return target, version


def run_eval(
    *,
    ckpt_path: Path,
    spec: RunSpec,
    run_id: str,
    run_root: Path,
    scenario_dir: Path,
    eval_output_root: Path,
    slots: list[str],
    max_scenes: int,
    dry_run: bool,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    cmd = [
        str(TRAIN_PYTHON),
        "-u",
        str(REPO_ROOT / "tools" / "evaluate_existing_sparsedrive_v2_ckpts.py"),
        "--ckpts",
        str(ckpt_path),
        "--scenario-dir",
        str(scenario_dir),
        "--eval-output-root",
        str(eval_output_root),
        "--run-name",
        f"eval_{run_id}_nusc{int(max_scenes)}_2x",
        "--repeat-evals",
        str(spec.eval_repeats),
        "--slots",
        *slots,
        "--max-scenes",
        str(int(max_scenes)),
    ]
    _run_command(cmd=cmd, cwd=REPO_ROOT, env=env, log_path=run_root / "eval.log", dry_run=dry_run)


def write_manifest(*, run_root: Path, data: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Reinforce++ DAC9-only GRPO coef=0.8 and two HUGSIM eval repeats.")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "outputs" / "reinforcepp_dac9_grpo_coef08_train_eval")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--eval-output-root", type=Path, default=DEFAULT_EVAL_OUTPUT_ROOT)
    parser.add_argument("--slots", nargs="+", default=["0:0", "1:1", "2:2", "3:3"])
    parser.add_argument("--max-scenes", type=int, default=88)
    parser.add_argument("--wait-poll-s", type=float, default=60.0)
    parser.add_argument("--date-tag", type=str, default=time.strftime("%Y%m%d", time.gmtime()))
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = default_run_spec()
    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{spec.run_key}_2eval"
    run_root = args.run_root.resolve() / run_id
    scenario_dir = args.scenario_dir.resolve()
    eval_output_root = args.eval_output_root.resolve()
    scenario_count = len(sorted(scenario_dir.glob("*.yaml")))
    if scenario_count < 1:
        raise RuntimeError(f"No scenario YAMLs found under {scenario_dir}")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "date_tag": str(args.date_tag),
        "run_key": spec.run_key,
        "coef": spec.coef,
        "config_path": str(spec.config_path),
        "buffer_dir": str(_resolve_buffer_dir(spec.config_path)),
        "scenario_dir": str(scenario_dir),
        "scenario_count": scenario_count,
        "max_scenes": int(args.max_scenes),
        "eval_repeats": spec.eval_repeats,
        "eval_output_root": str(eval_output_root),
        "slots": list(args.slots),
    }
    write_manifest(run_root=run_root, data=manifest)

    if not args.no_wait:
        wait_for_current_eval(wait_log=run_root / "wait.log", poll_s=float(args.wait_poll_s), dry_run=bool(args.dry_run))

    latest_ckpt = run_training(spec=spec, run_root=run_root, dry_run=bool(args.dry_run))
    promoted_ckpt, version = promote_checkpoint(
        src_ckpt=latest_ckpt,
        spec=spec,
        date_tag=str(args.date_tag),
        dry_run=bool(args.dry_run),
    )
    promotion = {
        "latest_ckpt": str(latest_ckpt),
        "promoted_ckpt": str(promoted_ckpt),
        "version": int(version),
    }
    manifest["promotion"] = promotion
    write_manifest(run_root=run_root, data=manifest)
    (run_root / spec.run_key).mkdir(parents=True, exist_ok=True)
    (run_root / spec.run_key / "promotion.json").write_text(json.dumps(promotion, indent=2, sort_keys=True), encoding="utf-8")

    run_eval(
        ckpt_path=promoted_ckpt,
        spec=spec,
        run_id=run_id,
        run_root=run_root,
        scenario_dir=scenario_dir,
        eval_output_root=eval_output_root,
        slots=list(args.slots),
        max_scenes=int(args.max_scenes),
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
