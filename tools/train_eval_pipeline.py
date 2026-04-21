from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
HUGSIM_ROOT = Path("/root/clone/HUGSIM-ORI")
SPARSEDRIVE_CKPT_DIR = REPO_ROOT / "egoADs" / "SparseDriveV2" / "ckpt"
DEFAULT_TRAIN_PYTHON = Path("/root/miniconda3/envs/recondreamerNew-rl/bin/python")
DEFAULT_HUGSIM_TEMPLATE = HUGSIM_ROOT / "configs" / "sim" / "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml"
DEFAULT_EVAL_OUTPUT_ROOT = HUGSIM_ROOT / "outputs" / "evaluate-auto"
DEFAULT_PPO_CONFIG = REPO_ROOT / "script" / "configs" / "sparsedrive_v2" / "ppo_closed_loop_sparsedrive_v2.yaml"
DEFAULT_REINFORCEPP_CONFIG = REPO_ROOT / "script" / "configs" / "sparsedrive_v2" / "reinforcepp_closed_loop_sparsedrive_v2.yaml"


@dataclass(frozen=True)
class TrainSpec:
    algo_tag: str
    config_path: Path


def detect_next_version(*, ckpt_dir: Path, algo_tag: str) -> int:
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


def build_eval_run_name(*, algo_tag: str, ckpt_stem: str, repeat_idx: int) -> str:
    return f"{algo_tag}/{ckpt_stem}/repeat_{int(repeat_idx):02d}"


def write_run_manifest(*, manifest_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_training_summary(
    *,
    algo_tag: str,
    config_path: Path,
    buffer_dir: Path,
    latest_ckpt: Path,
    promoted_ckpt: Path,
    version: int,
) -> str:
    return (
        f"[train-summary] algo={algo_tag} "
        f"version=ver{int(version):02d} "
        f"config={config_path} "
        f"buffer_dir={buffer_dir} "
        f"latest_ckpt={latest_ckpt} "
        f"promoted_ckpt={promoted_ckpt}"
    )


def rewrite_hugsim_eval_config(
    *,
    template_config: Path,
    output_config: Path,
    ckpt_path: Path,
    output_prefix: str,
) -> None:
    payload = yaml.safe_load(template_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected yaml mapping in {template_config}")
    payload["sparsedrive_v2_ckpt"] = str(ckpt_path)
    payload["sparsedrive_v2_pretrain_ckpt"] = str(ckpt_path)
    payload["output_dir"] = str(output_prefix)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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


def _promote_checkpoint(*, src_ckpt: Path, algo_tag: str, date_tag: str) -> tuple[Path, int]:
    SPARSEDRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    version = detect_next_version(ckpt_dir=SPARSEDRIVE_CKPT_DIR, algo_tag=algo_tag)
    target_name = build_promoted_ckpt_name(date_tag=date_tag, algo_tag=algo_tag, version=version)
    target_path = SPARSEDRIVE_CKPT_DIR / target_name
    shutil.copy2(src_ckpt, target_path)
    return target_path, version


def _run_training(*, spec: TrainSpec, python_bin: Path, date_tag: str, run_root: Path) -> Path:
    buffer_dir = _resolve_buffer_dir(spec.config_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    cmd = [
        str(python_bin),
        "-u",
        str(REPO_ROOT / "script" / "train_actor_learner_v2.py"),
        "--role",
        "orchestrator",
        "--config",
        str(spec.config_path),
    ]
    log_path = run_root / spec.algo_tag / date_tag / "train.log"
    _run_command(cmd=cmd, cwd=REPO_ROOT, env=env, log_path=log_path)
    return _latest_training_ckpt(buffer_dir)


def _run_eval_repeat(
    *,
    algo_tag: str,
    promoted_ckpt: Path,
    repeat_idx: int,
    hugsim_template: Path,
    eval_output_root: Path,
    slots: list[str],
    scenario_dir: Path | None,
    max_scenes: int | None,
    retry_count: int,
) -> None:
    ckpt_stem = promoted_ckpt.stem
    run_name = build_eval_run_name(algo_tag=algo_tag, ckpt_stem=ckpt_stem, repeat_idx=repeat_idx)
    run_root = eval_output_root / run_name
    generated_cfg = run_root / "generated_configs" / "sparsedrive_v2_eval.yaml"
    rewrite_hugsim_eval_config(
        template_config=hugsim_template,
        output_config=generated_cfg,
        ckpt_path=promoted_ckpt,
        output_prefix=str(run_root / "results" / "nusc_"),
    )

    cmd = [
        "pixi",
        "run",
        "python",
        str(HUGSIM_ROOT / "utils" / "nuscenes_eval_runner.py"),
        "--repo-root",
        str(HUGSIM_ROOT),
        "--output-root",
        str(run_root / "results"),
        "--summary-root",
        str(run_root / "summary"),
        "--repeats",
        "1",
        "--retry-count",
        str(int(retry_count)),
        "--models",
        "sparsedrive_ppo_grpo_ver14",
        "--slots",
        *slots,
    ]
    if scenario_dir is not None:
        cmd.extend(["--scenario-dir", str(scenario_dir)])
    if max_scenes is not None:
        cmd.extend(["--max-scenes", str(int(max_scenes))])

    log_path = run_root / "eval.log"
    env = os.environ.copy()
    _run_command(cmd=cmd, cwd=HUGSIM_ROOT, env=env, log_path=log_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/Reinforce++ and auto-evaluate on HUGSIM-ORI.")
    parser.add_argument("--train-python", type=Path, default=DEFAULT_TRAIN_PYTHON)
    parser.add_argument("--ppo-config", type=Path, default=DEFAULT_PPO_CONFIG)
    parser.add_argument("--reinforcepp-config", type=Path, default=DEFAULT_REINFORCEPP_CONFIG)
    parser.add_argument("--hugsim-template", type=Path, default=DEFAULT_HUGSIM_TEMPLATE)
    parser.add_argument("--eval-output-root", type=Path, default=DEFAULT_EVAL_OUTPUT_ROOT)
    parser.add_argument("--repeat-evals", type=int, default=2)
    parser.add_argument("--slots", nargs="+", default=["0:0", "1:1", "2:2", "3:3"])
    parser.add_argument("--scenario-dir", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--skip-reinforcepp", action="store_true")
    parser.add_argument("--date-tag", type=str, default=time.strftime("%Y%m%d", time.gmtime()))
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "outputs" / "train_eval_pipeline")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_specs: list[TrainSpec] = []
    if not args.skip_ppo:
        run_specs.append(TrainSpec(algo_tag="ppo", config_path=args.ppo_config.resolve()))
    if not args.skip_reinforcepp:
        run_specs.append(TrainSpec(algo_tag="reinforcepp", config_path=args.reinforcepp_config.resolve()))

    for spec in run_specs:
        buffer_dir = _resolve_buffer_dir(spec.config_path)
        latest_ckpt = _run_training(
            spec=spec,
            python_bin=args.train_python.resolve(),
            date_tag=str(args.date_tag),
            run_root=args.run_root.resolve(),
        )
        promoted_ckpt, version = _promote_checkpoint(
            src_ckpt=latest_ckpt,
            algo_tag=spec.algo_tag,
            date_tag=str(args.date_tag),
        )
        manifest_path = args.run_root.resolve() / spec.algo_tag / str(args.date_tag) / "manifest.json"
        write_run_manifest(
            manifest_path=manifest_path,
            data={
                "algo_tag": spec.algo_tag,
                "date_tag": str(args.date_tag),
                "version": int(version),
                "config_path": str(spec.config_path),
                "buffer_dir": str(buffer_dir),
                "latest_ckpt": str(latest_ckpt),
                "promoted_ckpt": str(promoted_ckpt),
                "repeat_evals": int(args.repeat_evals),
                "eval_output_root": str(args.eval_output_root.resolve()),
                "hugsim_template": str(args.hugsim_template.resolve()),
                "slots": list(args.slots),
                "scenario_dir": str(args.scenario_dir.resolve()) if args.scenario_dir is not None else None,
                "max_scenes": args.max_scenes,
            },
        )
        print(
            build_training_summary(
                algo_tag=spec.algo_tag,
                config_path=spec.config_path,
                buffer_dir=buffer_dir,
                latest_ckpt=latest_ckpt,
                promoted_ckpt=promoted_ckpt,
                version=version,
            ),
            flush=True,
        )
        for repeat_idx in range(1, int(args.repeat_evals) + 1):
            _run_eval_repeat(
                algo_tag=spec.algo_tag,
                promoted_ckpt=promoted_ckpt,
                repeat_idx=repeat_idx,
                hugsim_template=args.hugsim_template.resolve(),
                eval_output_root=args.eval_output_root.resolve(),
                slots=list(args.slots),
                scenario_dir=(args.scenario_dir.resolve() if args.scenario_dir is not None else None),
                max_scenes=args.max_scenes,
                retry_count=int(args.retry_count),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
