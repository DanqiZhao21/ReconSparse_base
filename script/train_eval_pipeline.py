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

'''
#####################
默认只跑 reinforcepp
只有显式加 --ppo 才跑 ppo
#####################
cd /root/clone/ReconDreamer-RL

PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_eval_pipeline.py \
  --reinforcepp-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202605211155_reinforcepp_closed_loop_sparsedrive_v2_craft_closeNo_openGRPO.yaml

#####################
PPO版本
#####################
PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_eval_pipeline.py \
  --ppo \
  --ppo-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/xxx.yaml \
  --reinforcepp-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202605211155_reinforcepp_closed_loop_sparsedrive_v2_craft_closeNo_openGRPO.yaml
'''

REPO_ROOT = Path(__file__).resolve().parents[1]
HUGSIM_ROOT = Path(os.environ.get("HUGSIM_ROOT", "/root/clone/HUGSIM-ORI")).resolve()
SPARSEDRIVE_CKPT_DIR = REPO_ROOT / "egoADs" / "SparseDriveV2" / "ckpt"
DEFAULT_TRAIN_PYTHON = Path("/root/miniconda3/envs/recondreamerNew-rl/bin/python")
DEFAULT_HUGSIM_TEMPLATE = HUGSIM_ROOT / "configs" / "sim" / "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml"
DEFAULT_SCENARIO_DIR = HUGSIM_ROOT / "configs" / "scenarios" / "nuscenes"
DEFAULT_EVAL_OUTPUT_ROOT = HUGSIM_ROOT / "outputs" / "evaluate-auto"
DEFAULT_PPO_CONFIG = REPO_ROOT / "script" / "configs" / "sparsedrive_v2" / "ppo_closed_loop_sparsedrive_v2.yaml"
DEFAULT_REINFORCEPP_CONFIG = REPO_ROOT / "script" / "configs" / "sparsedrive_v2" / "reinforcepp_closed_loop_sparsedrive_v2.yaml"
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs" / "TrainEvaluationAuto"
DEFAULT_EVAL_SLOTS = ["0:0", "1:1", "2:2", "3:3", "4:4", "5:5", "6:6", "7:7"]
DEFAULT_MAX_SCENES = 88


@dataclass(frozen=True)
class TrainSpec:
    algo_tag: str
    config_path: Path


@dataclass(frozen=True)
class PreparedTrainingConfig:
    config_path: Path
    buffer_dir: Path


def build_train_specs(
    *,
    ppo_config: Path,
    reinforcepp_config: Path,
    ppo_algo_tag: str,
    reinforcepp_algo_tag: str,
    run_ppo: bool = False,
    run_reinforcepp: bool = True,
) -> list[TrainSpec]:
    run_specs: list[TrainSpec] = []
    if run_ppo:
        run_specs.append(TrainSpec(algo_tag=str(ppo_algo_tag), config_path=ppo_config))
    if run_reinforcepp:
        run_specs.append(TrainSpec(algo_tag=str(reinforcepp_algo_tag), config_path=reinforcepp_config))
    return run_specs


def detect_next_version(*, ckpt_dir: Path, algo_tag: str) -> int:
    patterns = [
        re.compile(rf"^\d{{8}}_{re.escape(algo_tag)}_ver(\d+)_latest\.ckpt$"),
        re.compile(rf"^{re.escape(algo_tag)}_ver(\d+)_latest\.ckpt$"),
    ]
    max_ver = 0
    for path in ckpt_dir.glob(f"*{algo_tag}_ver*_latest.ckpt"):
        for pattern in patterns:
            match = pattern.match(path.name)
            if match is None:
                continue
            max_ver = max(max_ver, int(match.group(1)))
            break
    return max_ver + 1


def build_promoted_ckpt_name(*, date_tag: str, algo_tag: str, version: int) -> str:
    if str(algo_tag).startswith(f"{date_tag}_"):
        return f"{algo_tag}_ver{int(version):02d}_latest.ckpt"
    return f"{date_tag}_{algo_tag}_ver{int(version):02d}_latest.ckpt"


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.-")
    return cleaned or "run"


def build_run_tags(*, config: dict[str, Any], algo_tag: str) -> list[str]:
    train_cfg = config.get("train", {}) or {}
    env_cfg = config.get("env", {}) or {}
    reward_cfg = env_cfg.get("reward", {}) or {}
    agent_cfg = config.get("agent", {}) or {}
    tags: list[str] = []

    craft_cfg = train_cfg.get("CRAFT", {}) or {}
    env_craft_cfg = reward_cfg.get("CRAFT", {}) or {}
    craft_enabled = bool(craft_cfg.get("enable", False)) or bool(env_craft_cfg.get("enable", False))
    tags.append("Craft" if craft_enabled else "NoCraft")

    algo = str(train_cfg.get("algo", algo_tag)).strip().lower()
    if algo == "ppo":
        tags.append("PPO")
    elif algo == "grpo_only":
        tags.append("GRPOOnly")
    elif algo == "reinforcepp":
        tags.append("ReinforcePP")
    else:
        tags.append(_safe_name(algo))

    grpo_cfg = train_cfg.get("grpo", {}) or {}
    grpo_enabled = bool(grpo_cfg.get("enable", False)) and float(grpo_cfg.get("coef", 1.0) or 0.0) != 0.0
    tags.append("GRPO" if grpo_enabled else "NoGRPO")

    scorer_cfg = agent_cfg.get("nuscenes_scorer", {}) or {}
    if bool(scorer_cfg.get("ea_gate_enabled", False)) or "ea" in str(algo_tag).lower():
        tags.append("EA")

    algo_tag_l = str(algo_tag).lower()
    if "safety" in algo_tag_l or "safety" in str(scorer_cfg).lower():
        tags.append("safety")
    if "dac" in algo_tag_l:
        match = re.search(r"dac\d*", algo_tag_l)
        tags.append(match.group(0) if match is not None else "dac")
    if "coef" in algo_tag_l:
        match = re.search(r"coef[0-9]+", algo_tag_l)
        if match is not None:
            tags.append(match.group(0))

    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        safe = _safe_name(tag)
        key = safe.lower()
        if key not in seen:
            out.append(safe)
            seen.add(key)
    return out


def build_run_id(*, timestamp: str, config: dict[str, Any], algo_tag: str) -> str:
    tags = build_run_tags(config=config, algo_tag=algo_tag)
    return "_".join([_safe_name(timestamp), *tags])


def build_eval_run_name(*, algo_tag: str, ckpt_stem: str, repeat_idx: int) -> str:
    return f"{algo_tag}/{ckpt_stem}/repeat_{int(repeat_idx):02d}"


def write_run_manifest(*, manifest_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def prepare_training_config(*, source_config: Path, run_dir: Path, run_id: str) -> PreparedTrainingConfig:
    payload = _load_yaml(source_config)
    train_cfg = payload.setdefault("train", {})
    if not isinstance(train_cfg, dict):
        raise RuntimeError(f"Expected train mapping in {source_config}")
    actor_learner_cfg = train_cfg.setdefault("actor_learner", {})
    if not isinstance(actor_learner_cfg, dict):
        raise RuntimeError(f"Expected train.actor_learner mapping in {source_config}")

    buffer_dir = (run_dir / "actor_learner").resolve()
    actor_learner_cfg["buffer_dir"] = str(buffer_dir)
    actor_learner_cfg["timestamp_buffer_dir"] = False
    actor_learner_cfg["resolved_from_config"] = str(source_config)
    train_cfg["actor_learner"] = actor_learner_cfg
    payload["train"] = train_cfg

    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    out_path = config_dir / f"{_safe_name(run_id)}_{source_config.name}"
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return PreparedTrainingConfig(config_path=out_path.resolve(), buffer_dir=buffer_dir)


def cleanup_training_artifacts(*, run_dir: Path, buffer_dir: Path) -> None:
    for name in ["buffer", "weights", "actors"]:
        path = buffer_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in ["STOP", "TRAINING_LOCK"]:
        path = buffer_dir / name
        if path.exists():
            path.unlink()
    for name in ["train.log", "promoted_ckpt.txt"]:
        path = run_dir / name
        if path.exists():
            path.unlink()
    if buffer_dir.exists():
        try:
            buffer_dir.rmdir()
        except OSError:
            pass


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


def build_eval_environment(
    *,
    base_env: dict[str, str] | None = None,
    hugsim_random_seed: str | int | None = None,
    default_eval_seed: bool = True,
) -> dict[str, str]:
    env = dict(base_env or {})
    if env.get("HUGSIM_DISABLE_DEFAULT_EVAL_SEED") == "1":
        return env
    seed_value = None if hugsim_random_seed is None else str(hugsim_random_seed)
    if seed_value is None and default_eval_seed:
        seed_value = "0"
    if seed_value is not None:
        env["HUGSIM_RANDOM_SEED"] = seed_value
        env["PYTHONHASHSEED"] = seed_value
    return env


def build_eval_existing_ckpt_command(
    *,
    python_bin: Path,
    ckpt_path: Path,
    hugsim_template: Path,
    eval_output_root: Path,
    run_name: str,
    repeat_evals: int,
    slots: list[str],
    scenario_dir: Path | None,
    max_scenes: int | None,
    retry_count: int,
) -> list[str]:
    cmd = [
        str(python_bin),
        "-u",
        str(REPO_ROOT / "tools" / "evaluate_existing_sparsedrive_v2_ckpts.py"),
        "--ckpts",
        str(ckpt_path),
        "--hugsim-template",
        str(hugsim_template),
        "--eval-output-root",
        str(eval_output_root),
        "--run-name",
        str(run_name),
        "--repeat-evals",
        str(int(repeat_evals)),
        "--retry-count",
        str(int(retry_count)),
    ]
    if scenario_dir is not None:
        cmd.extend(["--scenario-dir", str(scenario_dir)])
    if max_scenes is not None:
        cmd.extend(["--max-scenes", str(int(max_scenes))])
    cmd.extend(["--slots", *slots])
    return cmd


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


def _run_training(*, spec: TrainSpec, python_bin: Path, date_tag: str, run_root: Path, flat_log: bool = False) -> Path:
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
    log_path = run_root / "train.log" if bool(flat_log) else run_root / spec.algo_tag / date_tag / "train.log"
    _run_command(cmd=cmd, cwd=REPO_ROOT, env=env, log_path=log_path)
    return _latest_training_ckpt(buffer_dir)


def _run_eval_existing_ckpt(
    *,
    algo_tag: str,
    promoted_ckpt: Path,
    python_bin: Path,
    repeat_evals: int,
    hugsim_template: Path,
    eval_output_root: Path,
    slots: list[str],
    scenario_dir: Path | None,
    max_scenes: int | None,
    retry_count: int,
    default_eval_seed: bool,
    hugsim_random_seed: str | int | None,
) -> None:
    ckpt_stem = promoted_ckpt.stem
    run_name = f"{algo_tag}/{ckpt_stem}"
    run_root = eval_output_root / run_name
    cmd = build_eval_existing_ckpt_command(
        python_bin=python_bin,
        ckpt_path=promoted_ckpt,
        hugsim_template=hugsim_template,
        eval_output_root=eval_output_root,
        run_name=run_name,
        repeat_evals=repeat_evals,
        slots=slots,
        scenario_dir=scenario_dir,
        max_scenes=max_scenes,
        retry_count=retry_count,
    )
    log_path = run_root / "eval.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if not default_eval_seed:
        env["HUGSIM_DISABLE_DEFAULT_EVAL_SEED"] = "1"
    env = build_eval_environment(
        base_env=env,
        hugsim_random_seed=hugsim_random_seed,
        default_eval_seed=default_eval_seed,
    )
    _run_command(cmd=cmd, cwd=REPO_ROOT, env=env, log_path=log_path)


def parse_train_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/Reinforce++ and auto-evaluate on HUGSIM-ORI.")
    parser.add_argument("--train-python", type=Path, default=DEFAULT_TRAIN_PYTHON)
    parser.add_argument("--ppo-config", type=Path, default=DEFAULT_PPO_CONFIG)
    parser.add_argument("--reinforcepp-config", type=Path, default=DEFAULT_REINFORCEPP_CONFIG)
    parser.add_argument("--ppo-algo-tag", type=str, default="ppo")
    parser.add_argument("--reinforcepp-algo-tag", type=str, default="reinforcepp")
    parser.add_argument("--hugsim-template", type=Path, default=DEFAULT_HUGSIM_TEMPLATE)
    parser.add_argument("--eval-output-root", type=Path, default=DEFAULT_EVAL_OUTPUT_ROOT)
    parser.add_argument("--repeat-evals", type=int, default=2)
    parser.add_argument("--slots", nargs="+", default=list(DEFAULT_EVAL_SLOTS))
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--max-scenes", type=int, default=DEFAULT_MAX_SCENES)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--ppo", action="store_true", help="Run PPO in addition to the default ReinforcePP run")
    parser.add_argument("--skip-reinforcepp", action="store_true", help="Skip the default ReinforcePP run")
    parser.add_argument("--date-tag", type=str, default=time.strftime("%Y%m%d", time.gmtime()))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--no-default-eval-seed", action="store_true")
    parser.add_argument("--hugsim-random-seed", type=str, default=None)
    return parser.parse_args(argv)


def _parse_args() -> argparse.Namespace:
    return parse_train_eval_args()


def main() -> int:
    args = _parse_args()
    run_specs = build_train_specs(
        ppo_config=args.ppo_config.resolve(),
        reinforcepp_config=args.reinforcepp_config.resolve(),
        ppo_algo_tag=str(args.ppo_algo_tag),
        reinforcepp_algo_tag=str(args.reinforcepp_algo_tag),
        run_ppo=bool(args.ppo),
        run_reinforcepp=not bool(args.skip_reinforcepp),
    )

    for spec in run_specs:
        source_config = spec.config_path.resolve()
        source_cfg = _load_yaml(source_config)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        run_id = build_run_id(timestamp=timestamp, config=source_cfg, algo_tag=spec.algo_tag)
        run_dir = args.run_root.resolve() / run_id
        prepared = prepare_training_config(
            source_config=source_config,
            run_dir=run_dir,
            run_id=run_id,
        )
        prepared_spec = TrainSpec(algo_tag=spec.algo_tag, config_path=prepared.config_path)
        buffer_dir = prepared.buffer_dir
        latest_ckpt = _run_training(
            spec=prepared_spec,
            python_bin=args.train_python.resolve(),
            date_tag=str(args.date_tag),
            run_root=run_dir,
            flat_log=True,
        )
        promoted_ckpt, version = _promote_checkpoint(
            src_ckpt=latest_ckpt,
            algo_tag=run_id,
            date_tag=str(args.date_tag),
        )
        manifest_path = run_dir / "manifest.json"
        write_run_manifest(
            manifest_path=manifest_path,
            data={
                "algo_tag": spec.algo_tag,
                "date_tag": str(args.date_tag),
                "run_id": str(run_id),
                "run_dir": str(run_dir),
                "version": int(version),
                "source_config_path": str(source_config),
                "config_path": str(prepared.config_path),
                "buffer_dir": str(buffer_dir),
                "latest_ckpt": str(latest_ckpt),
                "promoted_ckpt": str(promoted_ckpt),
                "repeat_evals": int(args.repeat_evals),
                "eval_output_root": str(args.eval_output_root.resolve()),
                "hugsim_template": str(args.hugsim_template.resolve()),
                "slots": list(args.slots),
                "scenario_dir": str(args.scenario_dir.resolve()) if args.scenario_dir is not None else None,
                "max_scenes": args.max_scenes,
                "cleanup_after_successful_eval": True,
            },
        )
        print(
            build_training_summary(
                algo_tag=run_id,
                config_path=prepared.config_path,
                buffer_dir=buffer_dir,
                latest_ckpt=latest_ckpt,
                promoted_ckpt=promoted_ckpt,
                version=version,
            ),
            flush=True,
        )
        _run_eval_existing_ckpt(
            algo_tag=run_id,
            promoted_ckpt=promoted_ckpt,
            python_bin=args.train_python.resolve(),
            repeat_evals=int(args.repeat_evals),
            hugsim_template=args.hugsim_template.resolve(),
            eval_output_root=args.eval_output_root.resolve(),
            slots=list(args.slots),
            scenario_dir=(args.scenario_dir.resolve() if args.scenario_dir is not None else None),
            max_scenes=args.max_scenes,
            retry_count=int(args.retry_count),
            default_eval_seed=not bool(args.no_default_eval_seed),
            hugsim_random_seed=args.hugsim_random_seed,
        )
        cleanup_training_artifacts(run_dir=run_dir, buffer_dir=buffer_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
