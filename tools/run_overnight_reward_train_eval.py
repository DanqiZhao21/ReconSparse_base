from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
HUGSIM_ROOT = Path("/root/clone/HUGSIM-ORI")
TRAIN_PYTHON = Path("/root/miniconda3/envs/recondreamerNew-rl/bin/python")
SPARSEDRIVE_CKPT_DIR = REPO_ROOT / "egoADs" / "SparseDriveV2" / "ckpt"
DEFAULT_SCENARIO_DIR = HUGSIM_ROOT / "configs" / "scenarios" / "nuscenes"
DEFAULT_EVAL_OUTPUT_ROOT = HUGSIM_ROOT / "outputs" / "evaluate-auto"
PRIMARY_CONFIG = REPO_ROOT / "script" / "configs" / "sparsedrive_v2" / "reinforcepp_closed_loop_sparsedrive_v2_overnight_safe.yaml"
FALLBACK_CONFIG = REPO_ROOT / "script" / "configs" / "sparsedrive_v2" / "reinforcepp_closed_loop_sparsedrive_v2_overnight_fallback.yaml"

UPDATE_RE = re.compile(r"\[learner\] update=(?P<update>\d+) .* metrics=(?P<metrics>\{.*\})")
REWARD_SUMMARY_RE = re.compile(r"\[learner\] reward_summary update=(?P<update>\d+) summary=(?P<summary>\{.*\})")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected yaml mapping: {path}")
    return payload


def _buffer_dir(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    buffer_dir = (((cfg.get("train", {}) or {}).get("actor_learner", {}) or {}).get("buffer_dir", "outputs/actor_learner"))
    return (REPO_ROOT / str(buffer_dir)).resolve()


def _max_updates(config_path: Path) -> int:
    cfg = _load_yaml(config_path)
    return int((((cfg.get("train", {}) or {}).get("actor_learner", {}) or {}).get("max_updates", 0)))


def _matching_active_processes() -> list[str]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    matches: list[str] = []
    for line in proc.stdout.splitlines():
        if "run_overnight_reward_train_eval.py" in line:
            continue
        if "train_actor_learner_v2.py" in line:
            matches.append(line.strip())
        elif "evaluate_existing_sparsedrive_v2_ckpts.py" in line:
            matches.append(line.strip())
        elif "/root/clone/HUGSIM-ORI/closed_loop.py" in line:
            matches.append(line.strip())
    return matches


def wait_for_current_work(*, wait_log: Path, poll_s: float) -> None:
    wait_log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        matches = _matching_active_processes()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        with wait_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] active_processes={len(matches)}\n")
            for item in matches:
                handle.write(f"  {item}\n")
        if not matches:
            return
        time.sleep(float(poll_s))


def _safe_dict(text: str) -> dict[str, float]:
    try:
        raw = ast.literal_eval(text)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def _latest_metrics(log_path: Path) -> tuple[int, dict[str, float], dict[str, float]]:
    latest_update = -1
    latest_metrics: dict[str, float] = {}
    reward_summary_by_update: dict[int, dict[str, float]] = {}
    if not log_path.exists():
        return latest_update, latest_metrics, {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        summary_match = REWARD_SUMMARY_RE.search(line)
        if summary_match:
            reward_summary_by_update[int(summary_match.group("update"))] = _safe_dict(summary_match.group("summary"))
            continue
        update_match = UPDATE_RE.search(line)
        if update_match:
            latest_update = int(update_match.group("update"))
            latest_metrics = _safe_dict(update_match.group("metrics"))
    return latest_update, latest_metrics, reward_summary_by_update.get(latest_update, {})


def _anomaly_reason(*, update: int, max_updates: int, metrics: dict[str, float], summary: dict[str, float]) -> str | None:
    if update < 0:
        return None
    approx_kl = float(metrics.get("approx_kl", 0.0))
    approx_kl_max = float(metrics.get("approx_kl_max", 0.0))
    if update >= 1 and approx_kl > 0.12:
        return f"approx_kl too high at update={update}: {approx_kl:.4f}"
    if update >= 1 and approx_kl_max > 8.0:
        return f"approx_kl_max too high at update={update}: {approx_kl_max:.4f}"
    gate_rate = float(summary.get("safety_gate_rate", 0.0))
    terminal_failure_rate = float(summary.get("terminal_failure_rate", 0.0))
    if update >= 2 and gate_rate > 0.75:
        return f"safety_gate_rate too high at update={update}: {gate_rate:.4f}"
    if update >= 2 and terminal_failure_rate > 0.65:
        return f"terminal_failure_rate too high at update={update}: {terminal_failure_rate:.4f}"
    if max_updates > 0 and update >= max_updates:
        return None
    return None


def _terminate_tree(pid: int) -> None:
    try:
        subprocess.run(["pkill", "-TERM", "-P", str(pid)], check=False)
    except Exception:
        pass
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        pass
    time.sleep(10.0)
    try:
        subprocess.run(["pkill", "-KILL", "-P", str(pid)], check=False)
    except Exception:
        pass
    try:
        os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass


def run_training_monitored(*, label: str, config_path: Path, run_dir: Path, monitor_poll_s: float) -> tuple[bool, str | None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    cmd = [
        str(TRAIN_PYTHON),
        "-u",
        str(REPO_ROOT / "script" / "train_actor_learner_v2.py"),
        "--role",
        "orchestrator",
        "--config",
        str(config_path),
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[label] {label}\n")
        handle.write(f"[cmd] {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    max_updates = _max_updates(config_path)
    anomaly_path = run_dir / "anomaly.json"
    try:
        while proc.poll() is None:
            time.sleep(float(monitor_poll_s))
            latest_update, metrics, summary = _latest_metrics(log_path)
            reason = _anomaly_reason(update=latest_update, max_updates=max_updates, metrics=metrics, summary=summary)
            if reason is None:
                continue
            anomaly_path.write_text(
                json.dumps(
                    {
                        "label": label,
                        "reason": reason,
                        "latest_update": int(latest_update),
                        "metrics": metrics,
                        "reward_summary": summary,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            _terminate_tree(proc.pid)
            return False, reason
        return_code = int(proc.returncode or 0)
        if return_code != 0:
            return False, f"training process exited with code {return_code}"
        return True, None
    finally:
        if proc.poll() is None:
            _terminate_tree(proc.pid)


def _promote_checkpoint(*, config_path: Path, ckpt_name: str, date_tag: str) -> Path:
    latest = _buffer_dir(config_path) / "weights" / "latest.ckpt"
    if not latest.exists():
        raise FileNotFoundError(f"Missing training checkpoint: {latest}")
    SPARSEDRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    target = SPARSEDRIVE_CKPT_DIR / f"{date_tag}_{ckpt_name}_latest.ckpt"
    shutil.copy2(latest, target)
    return target


def run_eval(*, ckpt_path: Path, run_name: str, run_dir: Path, scenario_dir: Path, eval_output_root: Path, repeat_evals: int, max_scenes: int, slots: list[str]) -> None:
    log_path = run_dir / "eval.log"
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
        str(run_name),
        "--repeat-evals",
        str(int(repeat_evals)),
        "--slots",
        *slots,
        "--max-scenes",
        str(int(max_scenes)),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[cmd] {' '.join(cmd)}\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"evaluation failed with code={result.returncode}; see {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for current work, then train/evaluate layered-reward Reinforce++ overnight.")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "outputs" / "overnight_reward_train_eval")
    parser.add_argument("--primary-config", type=Path, default=PRIMARY_CONFIG)
    parser.add_argument("--fallback-config", type=Path, default=FALLBACK_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--eval-output-root", type=Path, default=DEFAULT_EVAL_OUTPUT_ROOT)
    parser.add_argument("--wait-poll-s", type=float, default=60.0)
    parser.add_argument("--monitor-poll-s", type=float, default=45.0)
    parser.add_argument("--repeat-evals", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=88)
    parser.add_argument("--slots", nargs="+", default=["0:0", "1:1", "2:2", "3:3"])
    parser.add_argument("--date-tag", type=str, default=time.strftime("%Y%m%d", time.gmtime()))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_layered_reward"
    run_root = args.run_root.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "primary_config": str(args.primary_config.resolve()),
        "fallback_config": str(args.fallback_config.resolve()),
        "scenario_dir": str(args.scenario_dir.resolve()),
        "eval_output_root": str(args.eval_output_root.resolve()),
        "repeat_evals": int(args.repeat_evals),
        "max_scenes": int(args.max_scenes),
        "slots": list(args.slots),
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    wait_for_current_work(wait_log=run_root / "wait_current_work.log", poll_s=float(args.wait_poll_s))

    ok, reason = run_training_monitored(
        label="primary",
        config_path=args.primary_config.resolve(),
        run_dir=run_root / "primary",
        monitor_poll_s=float(args.monitor_poll_s),
    )
    chosen_label = "primary"
    chosen_config = args.primary_config.resolve()
    if not ok:
        (run_root / "fallback_reason.txt").write_text(str(reason), encoding="utf-8")
        ok, fallback_reason = run_training_monitored(
            label="fallback",
            config_path=args.fallback_config.resolve(),
            run_dir=run_root / "fallback",
            monitor_poll_s=float(args.monitor_poll_s),
        )
        chosen_label = "fallback"
        chosen_config = args.fallback_config.resolve()
        if not ok:
            raise RuntimeError(f"fallback training failed: {fallback_reason}")

    ckpt = _promote_checkpoint(
        config_path=chosen_config,
        ckpt_name=f"reinforce_layered_reward_{chosen_label}",
        date_tag=str(args.date_tag),
    )
    (run_root / "promoted_ckpt.txt").write_text(str(ckpt), encoding="utf-8")
    run_eval(
        ckpt_path=ckpt,
        run_name=f"eval_{run_id}_{chosen_label}_88",
        run_dir=run_root / "eval",
        scenario_dir=args.scenario_dir.resolve(),
        eval_output_root=args.eval_output_root.resolve(),
        repeat_evals=int(args.repeat_evals),
        max_scenes=int(args.max_scenes),
        slots=list(args.slots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
