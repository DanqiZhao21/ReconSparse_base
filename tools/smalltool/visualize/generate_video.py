#!/usr/bin/env python3
"""Generate a closed-loop rollout MP4 for a *single* scene.

目标：把脚本保持到最简单：选择一个场景（scene id），跑完（直到 done）或按指定
duration 跑一段，然后把 6 个相机视角拼成网格写成 MP4。

示例：
    python tools/smalltool/visualize/generate_video.py --scene 12 --duration-s 18
    python tools/smalltool/visualize/generate_video.py --scene 12 --out outputs/visualize/scene12.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch
import imageio


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def _lazy_import_runtime() -> tuple[Any, Any]:
    """Import modules that require the full RL runtime (gymnasium, simulator, etc.).

    We keep this lazy so `--help` and simple argument checks work even when
    optional runtime deps aren't installed in the current environment.
    """

    try:
        from framework.env_wrapper import RLReconEnv  # type: ignore
        from framework.agent.policy_diffusiondrivev2 import DiffusionDriveV2Policy  # type: ignore

        return RLReconEnv, DiffusionDriveV2Policy
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Missing runtime dependency for video rollout. "
            f"Import failed on module: {missing}. "
            "Activate the project conda env / install deps (see environment.yml), "
            "then retry."
        ) from e


DEFAULT_CKPT = os.path.join(_REPO_ROOT, "outputs", "weight", "20260129_ppo_ver27_latest.ckpt")

def _grid_frame(observation: Dict[str, np.ndarray]) -> np.ndarray:
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    imgs = [observation[k] for k in keys]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:6], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    return grid


def _ensure_parent(p: str) -> None:
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a closed-loop MP4 video for one scene (minimal version).")

    ap.add_argument("--scene", type=int, required=True, help="Scene id (int)")
    ap.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Rollout duration in seconds. If omitted, runs until env done.",
    )

    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help=f"DDV2-RL checkpoint path (default: {DEFAULT_CKPT})")
    ap.add_argument("--out", type=str, default=None, help="Output mp4 path. Default: outputs/visualize/sceneXXX_<ts>.mp4")

    ap.add_argument("--cuda", type=int, default=0, help="CUDA device index")
    ap.add_argument("--debug", action="store_true", help="Enable simulator debug mode")

    ap.add_argument("--step-frames", type=int, default=None, help="Override env step_frames (e.g., 5 means 0.5s/step when frame_dt=0.1)")
    ap.add_argument("--frame-dt-s", type=float, default=0.1, help="Seconds per raw dataset frame (default: 0.1)")
    ap.add_argument("--fps", type=float, default=None, help="Video playback FPS (default: realtime, i.e. 1/step_dt)")

    ap.add_argument("--render-w", type=int, default=None)
    ap.add_argument("--render-h", type=int, default=None)

    args = ap.parse_args()

    try:
        RLReconEnv, DiffusionDriveV2Policy = _lazy_import_runtime()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    ckpt_path = str(args.ckpt)
    if not ckpt_path:
        raise RuntimeError("Empty --ckpt")

    cuda = int(args.cuda)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    try:
        step_frames = int(args.step_frames) if args.step_frames is not None else None
    except Exception:
        step_frames = None

    frame_dt_s = float(args.frame_dt_s)
    if not (frame_dt_s > 0.0):
        raise RuntimeError("--frame-dt-s must be > 0")

    render_w = args.render_w
    render_h = args.render_h
    try:
        render_w = int(render_w) if render_w is not None else None
    except Exception:
        render_w = None
    try:
        render_h = int(render_h) if render_h is not None else None
    except Exception:
        render_h = None

    scene_id = int(args.scene)
    ts = time.strftime("%Y%m%d-%H%M%S")
    default_out = os.path.join(_REPO_ROOT, "outputs", "visualize", f"scene{scene_id:03d}_{ts}.mp4")
    out_path = str(args.out) if args.out else default_out
    _ensure_parent(out_path)

    print("==== generate_video (single-scene) ====")
    print(f"scene={scene_id} ckpt={ckpt_path}")
    print(f"device={device} cuda={cuda} debug={bool(args.debug)}")
    print(f"out={out_path}")

    env = RLReconEnv(
        cuda=cuda,
        scene=scene_id,
        reward_cfg={},
        debug=bool(args.debug),
        render_w=render_w,
        render_h=render_h,
    )

    obs, _info = env.reset(scene=scene_id, start_frame=0, step_frames=step_frames)
    sim = getattr(env, "env", None)

    sf_eff = step_frames
    try:
        if sf_eff is None and sim is not None:
            sf_eff = int(getattr(sim, "step_frames", 1))
    except Exception:
        sf_eff = step_frames
    if sf_eff is None:
        sf_eff = 1

    step_dt_s = float(sf_eff) * float(frame_dt_s)
    if not (step_dt_s > 0.0):
        raise RuntimeError("Invalid step dt computed")

    if args.duration_s is not None:
        dur = float(args.duration_s)
        if not (dur > 0.0):
            raise RuntimeError("--duration-s must be > 0")
        max_steps = max(1, int(round(dur / step_dt_s)))
        effective_dur = float(max_steps) * float(step_dt_s)
        print(f"duration_s={dur:.3f} -> steps={max_steps} (effective {effective_dur:.3f}s, step_dt={step_dt_s:.3f}s)")
    else:
        max_steps = None
        print(f"duration_s=until_done (step_dt={step_dt_s:.3f}s)")

    fps_used = float(args.fps) if args.fps is not None else (1.0 / float(step_dt_s))
    if not (fps_used > 0.0):
        raise RuntimeError("--fps must be > 0")
    print(f"fps={fps_used:.3f}")

    x_anchor = getattr(env.env, "x_anchor", 61)
    y_anchor = getattr(env.env, "y_anchor", 61)
    policy = DiffusionDriveV2Policy(
        x_anchor=x_anchor,
        y_anchor=y_anchor,
        ckpt_path=ckpt_path,
        device=str(device),
        rl_lr=1e-5,
        reinforce_baseline_beta=0.98,
    )

    # Note: imageio pipes rawvideo frames to ffmpeg. If we don't specify input framerate,
    # ffmpeg may warn: "not enough frames to estimate rate".
    writer = imageio.get_writer(
        out_path,
        mode="I",
        fps=float(fps_used),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_log_level="error",
        input_params=["-framerate", str(float(fps_used))],
        output_params=["-pix_fmt", "yuv420p"],
    )
    frames_written = 0

    def _append(frame: np.ndarray) -> None:
        nonlocal frames_written
        writer.append_data(frame)
        frames_written += 1

    _append(_grid_frame(obs))

    done = False
    steps = 0
    while True:
        if max_steps is not None and steps >= int(max_steps):
            break

        action, _logp, _replay = policy.sample_ddv2rl_with_replay(
            obs,
            eta=1.0,
            mode_idx=-1,
            mode_select="greedy",
        )
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)
        _append(_grid_frame(obs))

        steps += 1
        if done:
            break

        # Safety: avoid infinite episodes if duration not specified.
        if max_steps is None and steps >= 5000:
            print("warning: hit internal safety cap (5000 steps); stopping.", file=sys.stderr)
            break

    writer.close()
    sim_s = float(steps) * float(step_dt_s)
    # We write the initial frame (t=0) + one frame per step.
    video_s = float(frames_written) / float(fps_used)
    video_span_s = float(max(0, frames_written - 1)) / float(fps_used)
    print(
        f"done={done} steps={steps} sim_seconds~{sim_s:.2f}s frames={frames_written} "
        f"video_seconds~{video_s:.2f}s (span~{video_span_s:.2f}s)"
    )
    print("==== all done ====")


if __name__ == "__main__":
    main()
