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
    parser.add_argument("--fifo_timeout_s", type=float, default=300.0)
    parser.add_argument("--fifo_poll_interval_s", type=float, default=0.2)
    return parser


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
        env = gymnasium.make("hugsim_env/HUGSim-v0", cfg=cfg, output=str(output_dir))
        obs, info = env.reset()
        _write_status(output_dir, state="running", pid=os.getpid(), started_at=time.time())

        while True:
            write_fifo_payload(
                obs_pipe,
                (obs, info),
                timeout_s=float(args.fifo_timeout_s),
                poll_interval_s=float(args.fifo_poll_interval_s),
            )
            plan = read_fifo_payload(
                plan_pipe,
                timeout_s=float(args.fifo_timeout_s),
                poll_interval_s=float(args.fifo_poll_interval_s),
            )
            if plan is None or (isinstance(plan, str) and plan == "STOP"):
                _write_status(output_dir, state="stopped", pid=os.getpid(), completed_at=time.time())
                return

            acc, steer_rate = traj2control(np.asarray(plan, dtype=np.float32), dict(info))
            obs, reward, terminated, truncated, info = env.step(
                {"acc": float(acc), "steer_rate": float(steer_rate)}
            )
            info = dict(info)
            info["reward"] = float(reward)
            info["terminated"] = bool(terminated)
            info["truncated"] = bool(truncated)
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


if __name__ == "__main__":
    main()
