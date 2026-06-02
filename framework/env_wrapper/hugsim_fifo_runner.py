from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.env_wrapper.fifo_io import read_fifo_payload, write_fifo_payload


def ensure_hugsim_import_paths(hugsim_repo: str | Path | None = None) -> None:
    repo = Path(hugsim_repo) if hugsim_repo is not None else Path.cwd()
    for path in (repo, repo / "sim"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HUGSIM env-only FIFO backend.")
    parser.add_argument("--scenario_path", required=True)
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--camera_path", required=True)
    parser.add_argument("--kinematic_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ad", default="sparsedrive_v2")
    parser.add_argument("--substeps_per_rl_step", type=int, default=2)
    parser.add_argument("--fifo_timeout_s", type=float, default=300.0)
    parser.add_argument("--fifo_poll_interval_s", type=float, default=0.2)
    return parser


def execute_control_substeps(
    *,
    env: Any,
    plan: np.ndarray,
    initial_info: dict[str, Any],
    traj2control: Any,
    substeps_per_rl_step: int,
    status_fn: Any | None = None,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    substeps = max(1, int(substeps_per_rl_step))
    start_ts = _safe_float(initial_info.get("timestamp", None))
    acc, steer_rate = traj2control(np.asarray(plan, dtype=np.float32), dict(initial_info))
    action = {"acc": float(acc), "steer_rate": float(steer_rate)}

    obs = None
    info: dict[str, Any] = dict(initial_info)
    total_reward = 0.0
    terminated = False
    truncated = False
    executed = 0
    substep_rewards: list[float] = []
    for _idx in range(substeps):
        if callable(status_fn):
            status_fn(state="executing_substep", substep_idx=int(_idx), substeps_per_rl_step=int(substeps))
        obs, reward, terminated, truncated, info = env.step(dict(action))
        info = dict(info)
        reward_f = float(reward)
        total_reward += reward_f
        substep_rewards.append(reward_f)
        executed += 1
        if callable(status_fn):
            status_fn(
                state="executed_substep",
                substep_idx=int(_idx),
                substeps_per_rl_step=int(substeps),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        if bool(terminated or truncated):
            break

    end_ts = _safe_float(info.get("timestamp", None))
    info["reward"] = float(total_reward)
    info["terminated"] = bool(terminated)
    info["truncated"] = bool(truncated)
    info["hugsim_substeps_per_rl_step"] = int(substeps)
    info["hugsim_executed_substeps"] = int(executed)
    info["hugsim_substep_rewards"] = substep_rewards
    if start_ts is not None:
        info["hugsim_rl_step_start_timestamp"] = float(start_ts)
    if end_ts is not None:
        info["hugsim_rl_step_end_timestamp"] = float(end_ts)
    if start_ts is not None and end_ts is not None:
        info["hugsim_rl_step_delta_s"] = float(end_ts - start_ts)
    return obs, float(total_reward), bool(terminated), bool(truncated), info


def run_fifo_env(args: argparse.Namespace) -> None:
    ensure_hugsim_import_paths()

    import gymnasium  # type: ignore
    import hugsim_env  # noqa: F401
    from sim.utils.config_loader import load_closed_loop_cfg  # type: ignore
    from sim.utils.sim_utils import traj2control  # type: ignore

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_pipe = output_dir / "obs_pipe"
    plan_pipe = output_dir / "plan_pipe"
    for pipe in (obs_pipe, plan_pipe):
        if pipe.exists():
            pipe.unlink()
        os.mkfifo(pipe)

    _write_status(output_dir, state="starting", pid=os.getpid(), started_at=time.time())
    env = None
    try:
        cfg, _output = load_closed_loop_cfg(
            scenario_path=str(args.scenario_path),
            base_path=str(args.base_path),
            camera_path=str(args.camera_path),
            kinematic_path=str(args.kinematic_path),
            ad=str(args.ad),
        )
        _write_status(output_dir, state="making_env", pid=os.getpid(), started_at=time.time())
        env = gymnasium.make("hugsim_env/HUGSim-v0", cfg=cfg, output=str(output_dir))
        _write_status(output_dir, state="resetting", pid=os.getpid(), started_at=time.time())
        obs, info = env.reset()
        substeps = max(1, int(args.substeps_per_rl_step))
        print(f"[hugsim_fifo_runner] substeps_per_rl_step={substeps}", flush=True)
        _write_status(
            output_dir,
            state="running",
            pid=os.getpid(),
            started_at=time.time(),
            substeps_per_rl_step=int(substeps),
        )

        while True:
            _write_status(output_dir, state="writing_obs", pid=os.getpid(), updated_at=time.time())
            write_fifo_payload(
                obs_pipe,
                (obs, info),
                timeout_s=float(args.fifo_timeout_s),
                poll_interval_s=float(args.fifo_poll_interval_s),
            )
            _write_status(output_dir, state="waiting_plan", pid=os.getpid(), updated_at=time.time())
            plan = read_fifo_payload(
                plan_pipe,
                timeout_s=float(args.fifo_timeout_s),
                poll_interval_s=float(args.fifo_poll_interval_s),
            )
            if plan is None or (isinstance(plan, str) and plan == "STOP"):
                _write_status(output_dir, state="stopped", pid=os.getpid(), completed_at=time.time())
                return

            obs, reward, terminated, truncated, info = execute_control_substeps(
                env=env,
                plan=np.asarray(plan, dtype=np.float32),
                initial_info=dict(info),
                traj2control=traj2control,
                substeps_per_rl_step=int(substeps),
                status_fn=lambda **payload: _write_status(
                    output_dir,
                    pid=os.getpid(),
                    updated_at=time.time(),
                    **payload,
                ),
            )
            info = dict(info)
            if bool(terminated or truncated):
                write_fifo_payload(
                    obs_pipe,
                    (obs, info),
                    timeout_s=float(args.fifo_timeout_s),
                    poll_interval_s=float(args.fifo_poll_interval_s),
                )
                _write_status(
                    output_dir,
                    state="completed",
                    pid=os.getpid(),
                    completed_at=time.time(),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                return
    except BaseException as exc:
        _write_status(
            output_dir,
            state="error",
            pid=os.getpid(),
            error=repr(exc),
            traceback=traceback.format_exc(),
            completed_at=time.time(),
        )
        raise
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_fifo_env(args)


def _write_status(output_dir: Path, **payload: Any) -> None:
    status_path = output_dir / "status.json"
    status_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":
    main()
