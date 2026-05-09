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
    config_path: Path


def default_run_specs() -> list[RunSpec]:
    cfg_root = REPO_ROOT / "script" / "configs" / "sparsedrive_v2"
    return [
        RunSpec(
            run_key="reinforcepp_baseline_reward_dac_weight_grpo_coef001",
            coef=0.01,
            config_path=cfg_root
            / "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef001.yaml",
        ),
        RunSpec(
            run_key="reinforcepp_baseline_reward_dac_weight_grpo_coef003_gateoff",
            coef=0.03,
            config_path=cfg_root
            / "reinforcepp_closed_loop_sparsedrive_v2_baseline_reward_dac_weight_grpo_coef003_gateoff.yaml",
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


def detect_next_version(*, ckpt_dir: Path, run_key_prefix: str) -> int:
    import re

    pattern = re.compile(rf"^\d{{8}}_{re.escape(run_key_prefix)}.*_ver(\d+)_latest\.ckpt$")
    max_ver = 0
    for path in ckpt_dir.glob(f"*_{run_key_prefix}*_ver*_latest.ckpt"):
        match = pattern.match(path.name)
        if match is None:
            continue
        max_ver = max(max_ver, int(match.group(1)))
    return max_ver + 1


def build_promoted_ckpt_name(*, date_tag: str, run_key: str, version: int) -> str:
    return f"{date_tag}_{run_key}_ver{int(version):02d}_latest.ckpt"


def promote_checkpoint(*, src_ckpt: Path, spec: RunSpec, date_tag: str) -> tuple[Path, int]:
    SPARSEDRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    version = detect_next_version(
        ckpt_dir=SPARSEDRIVE_CKPT_DIR,
        run_key_prefix=spec.run_key,
    )
    target_path = SPARSEDRIVE_CKPT_DIR / build_promoted_ckpt_name(
        date_tag=date_tag,
        version=version,
        run_key=spec.run_key,
    )
    shutil.copy2(src_ckpt, target_path)
    return target_path, version


def run_training(*, spec: RunSpec, run_root: Path, dry_run: bool) -> Path:
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
    _run_command(cmd=cmd, cwd=REPO_ROOT, env=env, log_path=run_root / spec.run_key / "train.log")
    return _latest_training_ckpt(buffer_dir)


def run_eval(
    *,
    promoted_ckpts: list[Path],
    run_root: Path,
    scenario_dir: Path,
    eval_output_root: Path,
    run_name: str,
    repeat_evals: int,
    slots: list[str],
    max_scenes: int | None,
    dry_run: bool,
) -> None:
    cmd = [
        str(TRAIN_PYTHON),
        "-u",
        str(REPO_ROOT / "tools" / "evaluate_existing_sparsedrive_v2_ckpts.py"),
        "--ckpts",
        *[str(path) for path in promoted_ckpts],
        "--scenario-dir",
        str(scenario_dir),
        "--eval-output-root",
        str(eval_output_root),
        "--run-name",
        run_name,
        "--repeat-evals",
        str(int(repeat_evals)),
        "--slots",
        *slots,
    ]
    if max_scenes is not None:
        cmd.extend(["--max-scenes", str(int(max_scenes))])
    if dry_run:
        print("[dry-run] " + " ".join(cmd), flush=True)
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    _run_command(cmd=cmd, cwd=REPO_ROOT, env=env, log_path=run_root / "eval.log")


def write_manifest(*, run_root: Path, data: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal GRPO ablations: coef=0.01 gate-on and coef=0.03 gate-off, then HUGSIM-ORI eval."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "reinforcepp_minimal_grpo_ablation",
    )
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--eval-output-root", type=Path, default=DEFAULT_EVAL_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--repeat-evals", type=int, default=2)
    parser.add_argument("--slots", nargs="+", default=["0:0", "1:1", "2:2", "3:3"])
    parser.add_argument("--max-scenes", type=int, default=88)
    parser.add_argument("--date-tag", type=str, default=time.strftime("%Y%m%d", time.gmtime()))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_minimal_grpo_ablation"
    run_root = args.run_root.resolve() / run_id
    scenario_dir = args.scenario_dir.resolve()
    eval_output_root = args.eval_output_root.resolve()
    specs = default_run_specs()
    scenario_count = len(sorted(scenario_dir.glob("*.yaml")))
    max_scenes = int(args.max_scenes) if args.max_scenes is not None else None

    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
    if scenario_count < 1:
        raise RuntimeError(f"No scenario YAMLs found under {scenario_dir}")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "date_tag": str(args.date_tag),
        "scenario_dir": str(scenario_dir),
        "scenario_count": scenario_count,
        "max_scenes": max_scenes,
        "eval_output_root": str(eval_output_root),
        "repeat_evals": int(args.repeat_evals),
        "slots": list(args.slots),
        "run_specs": [
            {
                "run_key": spec.run_key,
                "coef": spec.coef,
                "config_path": str(spec.config_path),
                "buffer_dir": str(_resolve_buffer_dir(spec.config_path)),
            }
            for spec in specs
        ],
        "promotions": [],
    }
    write_manifest(run_root=run_root, data=manifest)

    promoted_ckpts: list[Path] = []
    for spec in specs:
        latest_ckpt = run_training(spec=spec, run_root=run_root, dry_run=bool(args.dry_run))
        if args.dry_run:
            promoted_ckpt = SPARSEDRIVE_CKPT_DIR / build_promoted_ckpt_name(
                date_tag=str(args.date_tag),
                version=detect_next_version(
                    ckpt_dir=SPARSEDRIVE_CKPT_DIR,
                    run_key_prefix=spec.run_key,
                ),
                run_key=spec.run_key,
            )
            version = 0
        else:
            promoted_ckpt, version = promote_checkpoint(src_ckpt=latest_ckpt, spec=spec, date_tag=str(args.date_tag))
        promoted_ckpts.append(promoted_ckpt)
        promotion = {
            "run_key": spec.run_key,
            "coef": spec.coef,
            "latest_ckpt": str(latest_ckpt),
            "promoted_ckpt": str(promoted_ckpt),
            "version": int(version),
        }
        manifest["promotions"].append(promotion)
        (run_root / spec.run_key).mkdir(parents=True, exist_ok=True)
        (run_root / spec.run_key / "promotion.json").write_text(
            json.dumps(promotion, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_manifest(run_root=run_root, data=manifest)

    run_eval(
        promoted_ckpts=promoted_ckpts,
        run_root=run_root,
        scenario_dir=scenario_dir,
        eval_output_root=eval_output_root,
        run_name=f"eval_{run_id}_nusc{max_scenes or scenario_count}",
        repeat_evals=int(args.repeat_evals),
        slots=list(args.slots),
        max_scenes=max_scenes,
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
