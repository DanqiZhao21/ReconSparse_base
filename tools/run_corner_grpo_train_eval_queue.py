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
TRAIN_PYTHON = Path("/root/miniconda3/envs/recondreamerNew-rl/bin/python")
SPARSEDRIVE_CKPT_DIR = REPO_ROOT / "egoADs" / "SparseDriveV2" / "ckpt"
DEFAULT_SCENARIO_DIR = HUGSIM_ROOT / "configs" / "scenarios" / "nuscenes"
DEFAULT_EVAL_OUTPUT_ROOT = HUGSIM_ROOT / "outputs" / "evaluate-auto"
DEFAULT_HUGSIM_TEMPLATE = HUGSIM_ROOT / "configs" / "sim" / "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml"


@dataclass(frozen=True)
class RunSpec:
    run_key: str
    algo_tag: str
    config_path: Path


def default_run_specs() -> list[RunSpec]:
    cfg_root = REPO_ROOT / "script" / "configs" / "sparsedrive_v2"
    return [
        RunSpec(
            run_key="ppo_corner_baseline",
            algo_tag="ppo",
            config_path=cfg_root / "ppo_closed_loop_sparsedrive_v2_corner_baseline.yaml",
        ),
        RunSpec(
            run_key="ppo_corner_grpo003",
            algo_tag="ppo",
            config_path=cfg_root / "ppo_closed_loop_sparsedrive_v2_corner_grpo003.yaml",
        ),
        RunSpec(
            run_key="reinforcepp_corner_baseline",
            algo_tag="reinforcepp",
            config_path=cfg_root / "reinforcepp_closed_loop_sparsedrive_v2_corner_baseline.yaml",
        ),
        RunSpec(
            run_key="reinforcepp_corner_grpo003",
            algo_tag="reinforcepp",
            config_path=cfg_root / "reinforcepp_closed_loop_sparsedrive_v2_corner_grpo003.yaml",
        ),
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected mapping config: {path}")
    return payload


def _resolve_buffer_dir(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    train_cfg = cfg.get("train", {}) or {}
    actor_learner_cfg = train_cfg.get("actor_learner", {}) or {}
    buffer_dir = actor_learner_cfg.get("buffer_dir", "outputs/actor_learner")
    return (REPO_ROOT / str(buffer_dir)).resolve()


def _latest_training_ckpt(buffer_dir: Path) -> Path:
    ckpt_path = buffer_dir / "weights" / "latest.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {ckpt_path}")
    return ckpt_path


def _run_command(*, cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[cwd] {cwd}\n")
        handle.write(f"[cmd] {' '.join(cmd)}\n")
        handle.flush()
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


def _matching_eval_processes() -> list[str]:
    out = subprocess.run(
        ["ps", "-eo", "pid,cmd"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    matches: list[str] = []
    for line in out:
        if "evaluate_existing_sparsedrive_v2_ckpts.py" in line:
            matches.append(line.strip())
            continue
        if "/root/clone/HUGSIM-ORI/closed_loop.py" in line and "evaluate-auto/eval_20260422_123722_sparsedrive_v2_no_grpo_2x88x2" in line:
            matches.append(line.strip())
    return matches


def wait_for_current_eval(*, poll_s: float, wait_log: Path, dry_run: bool) -> None:
    wait_log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        wait_log.write_text("[dry-run] would wait for existing HUGSIM evaluation batch to finish\n", encoding="utf-8")
        return
    while True:
        matches = _matching_eval_processes()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        with wait_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] active_eval_processes={len(matches)}\n")
            for line in matches:
                handle.write(f"  {line}\n")
        if not matches:
            return
        time.sleep(float(poll_s))


def detect_next_version(*, ckpt_dir: Path, algo_tag: str) -> int:
    import re

    pattern = re.compile(rf"^\d{{8}}_{re.escape(algo_tag)}_ver(\d+)_latest\.ckpt$")
    max_ver = 0
    for path in ckpt_dir.glob(f"*_{algo_tag}_ver*_latest.ckpt"):
        match = pattern.match(path.name)
        if match is None:
            continue
        max_ver = max(max_ver, int(match.group(1)))
    return max_ver + 1


def build_promoted_ckpt_name(*, date_tag: str, algo_tag: str, version: int) -> str:
    return f"{date_tag}_{algo_tag}_ver{int(version):02d}_latest.ckpt"


def promote_checkpoint(*, src_ckpt: Path, algo_tag: str, date_tag: str) -> tuple[Path, int]:
    SPARSEDRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    version = detect_next_version(ckpt_dir=SPARSEDRIVE_CKPT_DIR, algo_tag=algo_tag)
    target_path = SPARSEDRIVE_CKPT_DIR / build_promoted_ckpt_name(date_tag=date_tag, algo_tag=algo_tag, version=version)
    shutil.copy2(src_ckpt, target_path)
    return target_path, version


def run_training(*, spec: RunSpec, date_tag: str, run_root: Path, dry_run: bool) -> Path:
    buffer_dir = _resolve_buffer_dir(spec.config_path)
    latest_ckpt = buffer_dir / "weights" / "latest.ckpt"
    if dry_run:
        return latest_ckpt
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
    log_path = run_root / spec.run_key / "train.log"
    _run_command(cmd=cmd, cwd=REPO_ROOT, env=env, log_path=log_path)
    return _latest_training_ckpt(buffer_dir)


def run_eval(
    *,
    promoted_ckpt: Path,
    spec: RunSpec,
    run_root: Path,
    scenario_dir: Path,
    eval_output_root: Path,
    max_scenes: int,
    dry_run: bool,
) -> None:
    cmd = [
        str(TRAIN_PYTHON),
        "-u",
        str(REPO_ROOT / "tools" / "evaluate_existing_sparsedrive_v2_ckpts.py"),
        "--ckpts",
        str(promoted_ckpt),
        "--scenario-dir",
        str(scenario_dir),
        "--eval-output-root",
        str(eval_output_root),
        "--run-name",
        f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{spec.run_key}",
        "--repeat-evals",
        "1",
        "--slots",
        "0:0",
        "1:1",
        "2:2",
        "3:3",
        "--max-scenes",
        str(int(max_scenes)),
    ]
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    log_path = run_root / spec.run_key / "eval.log"
    _run_command(cmd=cmd, cwd=REPO_ROOT, env=env, log_path=log_path)


def write_manifest(*, run_root: Path, data: dict[str, Any]) -> None:
    manifest_path = run_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue 4 corner-GRPO train/eval runs after current evaluation finishes.")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "outputs" / "corner_grpo_train_eval_queue")
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--eval-output-root", type=Path, default=DEFAULT_EVAL_OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=88)
    parser.add_argument("--wait-poll-s", type=float, default=60.0)
    parser.add_argument("--date-tag", type=str, default=time.strftime("%Y%m%d", time.gmtime()))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve() / str(args.date_tag)
    scenario_dir = args.scenario_dir.resolve()
    eval_output_root = args.eval_output_root.resolve()
    run_specs = default_run_specs()

    write_manifest(
        run_root=run_root,
        data={
            "date_tag": str(args.date_tag),
            "scenario_dir": str(scenario_dir),
            "eval_output_root": str(eval_output_root),
            "max_scenes": int(args.max_scenes),
            "run_specs": [
                {
                    "run_key": spec.run_key,
                    "algo_tag": spec.algo_tag,
                    "config_path": str(spec.config_path),
                    "buffer_dir": str(_resolve_buffer_dir(spec.config_path)),
                }
                for spec in run_specs
            ],
        },
    )

    if args.dry_run:
        print("Would wait for current evaluation batch to finish.", flush=True)
        for spec in run_specs:
            print(
                json.dumps(
                    {
                        "run_key": spec.run_key,
                        "algo_tag": spec.algo_tag,
                        "config_path": str(spec.config_path),
                        "buffer_dir": str(_resolve_buffer_dir(spec.config_path)),
                        "grpo_coef": _load_yaml(spec.config_path)["train"]["grpo"]["coef"],
                        "max_scenes": int(args.max_scenes),
                        "repeat_evals": 1,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return 0

    wait_for_current_eval(
        poll_s=float(args.wait_poll_s),
        wait_log=run_root / "wait.log",
        dry_run=False,
    )

    for spec in run_specs:
        latest_ckpt = run_training(spec=spec, date_tag=str(args.date_tag), run_root=run_root, dry_run=False)
        promoted_ckpt, version = promote_checkpoint(
            src_ckpt=latest_ckpt,
            algo_tag=spec.algo_tag,
            date_tag=str(args.date_tag),
        )
        (run_root / spec.run_key / "promotion.json").write_text(
            json.dumps(
                {
                    "latest_ckpt": str(latest_ckpt),
                    "promoted_ckpt": str(promoted_ckpt),
                    "version": int(version),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        run_eval(
            promoted_ckpt=promoted_ckpt,
            spec=spec,
            run_root=run_root,
            scenario_dir=scenario_dir,
            eval_output_root=eval_output_root,
            max_scenes=int(args.max_scenes),
            dry_run=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
