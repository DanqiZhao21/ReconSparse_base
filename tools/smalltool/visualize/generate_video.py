#!/usr/bin/env python3
"""Generate closed-loop rollout videos for one or more scenes.

This tool is intentionally lightweight and reuses the same environment + policy
stack used by the training/eval scripts.

Typical usage:
  python tools/smalltool/visualize/generate_video.py --num-scenes 3 --frames 36

By default it uses the requested ckpt:
    outputs/weight/20260129_ppo_ver27_latest.ckpt
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
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
DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "script", "configs", "ppo_closed_loop.yaml")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _list_scene_ids_from_assets() -> List[int]:
    """Best-effort scene listing from assets/nus/data/XXX folders."""
    try:
        from reconsimulator.envs import nus_config as nus_cfg

        base_dir = os.path.join(_REPO_ROOT, str(nus_cfg.BASE_DATA_DIR))
        if not os.path.isdir(base_dir):
            return []
        out: List[int] = []
        for name in os.listdir(base_dir):
            p = os.path.join(base_dir, name)
            if not os.path.isdir(p):
                continue
            try:
                out.append(int(name))
            except Exception:
                continue
        out = sorted(set(out))
        return out
    except Exception:
        return []


def _grid_frame(observation: Dict[str, np.ndarray], info: Optional[Dict[str, Any]], *, draw_overlay: bool) -> np.ndarray:
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    imgs = [observation[k] for k in keys]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:6], axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    if not draw_overlay:
        return grid

    try:
        import cv2

        gh, gw = grid.shape[:2]
        box_w, box_h = 360, 140
        margin = 10
        x0, y0 = margin, gh - box_h - margin

        roi_bg = grid[y0 : y0 + box_h, x0 : x0 + box_w].copy()
        overlay = roi_bg.copy()
        cv2.rectangle(overlay, (0, 0), (box_w - 1, box_h - 1), (32, 32, 32), thickness=-1)
        blended = cv2.addWeighted(overlay, 0.45, roi_bg, 0.55, 0)
        grid[y0 : y0 + box_h, x0 : x0 + box_w] = blended

        scene_id = None
        now_frame = None
        xz_err_m = None
        yaw_err_deg = None
        done_reason = None

        if isinstance(info, dict):
            scene_id = info.get("scene_id", None)
            now_frame = info.get("now_frame", None)
            xz_err_m = info.get("xz_err_m", None)
            yaw_err_deg = info.get("yaw_err_deg", None)
            done_reason = info.get("done_reason", None)

        line0 = None
        try:
            if scene_id is not None or now_frame is not None:
                s = "?" if scene_id is None else f"{int(scene_id):03d}"
                f = "?" if now_frame is None else str(int(now_frame))
                line0 = f"scene={s} frame={f}"
        except Exception:
            line0 = None

        line1 = None
        try:
            if xz_err_m is not None and yaw_err_deg is not None:
                line1 = f"err: xz={float(xz_err_m):.3f}m, yaw={float(yaw_err_deg):.2f}deg"
        except Exception:
            line1 = None

        line2 = None
        if done_reason is not None:
            line2 = f"done_reason: {str(done_reason)}"

        base_x = x0 + 10
        base_y = y0 + 24
        if line0:
            cv2.putText(grid, line0, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1, cv2.LINE_AA)
            base_y += 26
        if line1:
            cv2.putText(grid, line1, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            base_y += 24
        if line2:
            cv2.putText(grid, line2, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 200), 1, cv2.LINE_AA)

    except Exception:
        return grid

    return grid


def _interp_pose(p0: np.ndarray, p1: np.ndarray, alpha: float) -> np.ndarray:
    """SE(3) interpolation: linear translation + SLERP rotation."""
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

    # Lazy import to avoid hard dependency unless rerender interpolation is used.
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    t0 = p0[:3, 3]
    t1 = p1[:3, 3]
    t = (1.0 - alpha) * t0 + alpha * t1

    rkey = R.from_matrix(np.stack([p0[:3, :3], p1[:3, :3]], axis=0))
    slerp = Slerp([0.0, 1.0], rkey)
    rr = slerp([alpha]).as_matrix()[0]

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rr
    out[:3, 3] = t
    return out


def _render_obs_from_pose_and_time(sim_env: Any, *, ego_pose: np.ndarray, normed_time: float) -> Dict[str, np.ndarray]:
    """Render 6 camera RGBs from an arbitrary ego pose, without stepping the env.

    This relies on the underlying simulator implementation having fields similar
    to ReconSimulator (trainer/all_cams/all_images/cam2ego).
    """

    # Lazy import to keep the default path lightweight.
    from framework.env_wrapper import get_sky_view, move_to_device

    device = getattr(sim_env, "device", "cuda")
    w = int(getattr(sim_env, "w", 800))
    h = int(getattr(sim_env, "h", 450))
    transform = getattr(sim_env, "_transform_matrix", None)

    out_imgs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(6):
            cam_info = move_to_device(copy.deepcopy(sim_env.all_cams[i]), device)
            cam2ego = sim_env.cam2ego[i]
            cam_to_world = torch.tensor(ego_pose @ cam2ego, device=device, dtype=torch.float32)
            if transform is not None:
                cam_to_world = cam_to_world @ transform
            cam_info["camera_to_world"] = cam_to_world

            img_info = move_to_device(copy.deepcopy(sim_env.all_images[i]), device)
            img_info["origins"], img_info["viewdirs"], img_info["direction_norm"] = get_sky_view(
                cam_info["camera_to_world"], cam_info["intrinsics"], device, h, w
            )
            img_info["normed_time"] = torch.tensor(float(normed_time), device=device)

            results = sim_env.trainer(img_info, cam_info)
            rgb = results["rgb"].clamp(0, 1).detach().cpu().numpy()
            out_imgs.append((rgb * 255).astype(np.uint8))

    return {
        "front": out_imgs[0],
        "front_left": out_imgs[1],
        "front_right": out_imgs[2],
        "back_left": out_imgs[3],
        "back_right": out_imgs[4],
        "back": out_imgs[5],
    }


@dataclass(frozen=True)
class RunSpec:
    scene_id: int
    start_frame: Optional[int]


def _parse_scene_list(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(list(range(lo, hi + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def _pick_scenes(
    *,
    available: Sequence[int],
    num_scenes: int,
    scene_sampling: str,
    seed: int,
    scene_start: int,
) -> List[int]:
    if len(available) == 0:
        # Fallback: assume scenes are [0, 1, 2, ...]
        return [int(scene_start + i) for i in range(int(num_scenes))]

    mode = str(scene_sampling).strip().lower()
    num_scenes = max(1, int(num_scenes))

    if mode in {"seq", "sequential"}:
        # Find start position; if not present, start from closest insertion point.
        avail = list(sorted(set(int(x) for x in available)))
        # binary search
        import bisect

        idx = bisect.bisect_left(avail, int(scene_start))
        out = []
        for i in range(num_scenes):
            out.append(avail[(idx + i) % len(avail)])
        return out

    # random sampling (with replacement if needed)
    rng = np.random.RandomState(int(seed))
    avail = np.asarray(list(sorted(set(int(x) for x in available))), dtype=np.int64)
    if num_scenes <= len(avail):
        choice = rng.choice(avail, size=num_scenes, replace=False)
    else:
        choice = rng.choice(avail, size=num_scenes, replace=True)
    return [int(x) for x in choice.tolist()]


def _sample_start_frame(
    env: Any,
    *,
    scene_id: int,
    step_frames: Optional[int],
    start_mode: str,
    start_frame: Optional[int],
    seed: int,
    allow_short_tail: bool,
    max_steps: int,
) -> Optional[int]:
    if start_frame is not None:
        return int(start_frame)

    mode = str(start_mode).strip().lower()
    if mode in {"none", "keep"}:
        return None

    if mode in {"zero", "0"}:
        return 0

    # random within valid range
    try:
        final_frame = int(getattr(env.env, "final_frame", 0))
    except Exception:
        final_frame = 0
    if final_frame <= 0:
        return None

    sf_step = int(step_frames) if step_frames is not None else int(getattr(env.env, "step_frames", 1) or 1)
    sf_step = max(1, sf_step)

    # ensure we can run max_steps without exceeding final_frame unless allow_short_tail
    if allow_short_tail:
        hi = max(0, final_frame - 1)
    else:
        hi = max(0, final_frame - 1 - int(max_steps) * sf_step)

    lo = 0
    if hi <= lo:
        return 0

    rng = np.random.RandomState(int(seed) + int(scene_id) * 97)
    sf = int(rng.randint(lo, hi + 1))
    sf = (sf // sf_step) * sf_step
    return int(sf)


def _ensure_parent(p: str) -> None:
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate closed-loop MP4 videos for multiple scenes.")

    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help=f"DDV2-RL checkpoint path (default: {DEFAULT_CKPT})")
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Config yaml for env/reward defaults")

    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="Enable simulator debug mode")

    ap.add_argument(
        "--frames",
        "--max-steps",
        dest="max_steps",
        type=int,
        default=36,
        help="How many env steps to run (default: 36). In the default mode, 1 env.step = 1 video frame.",
    )
    ap.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Specify rollout duration in seconds. Must be a multiple of env step dt (typically 0.5s when step_frames=5). Overrides --frames.",
    )
    ap.add_argument(
        "--frame-dt-s",
        type=float,
        default=0.1,
        help="Seconds per raw dataset frame. Default 0.1 so step_frames=5 => 0.5s/step.",
    )
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--draw-overlay", action="store_true", help="Draw scene/frame/error overlay")

    ap.add_argument(
        "--interp-method",
        type=str,
        default="none",
        choices=["none", "blend", "rerender"],
        help="Optional interpolation inside each env step. none: 1 frame/step; blend: pixel blend; rerender: pose+time interpolation and re-render.",
    )
    ap.add_argument(
        "--interp-frames-per-step",
        type=int,
        default=None,
        help="How many frames to generate within one env step when interp-method != none. Default: step_frames (if available) else 1.",
    )

    ap.add_argument("--step-frames", type=int, default=None, help="Env step_frames (e.g., 5 means 0.5s per step). Default: from config/env or env default")
    ap.add_argument("--render-w", type=int, default=None)
    ap.add_argument("--render-h", type=int, default=None)

    ap.add_argument("--scene-list", type=str, default=None, help='Explicit scene ids, e.g. "0,1,2" or "10-19"')
    ap.add_argument("--num-scenes", type=int, default=1, help="Number of scenes to render if --scene-list not provided")
    ap.add_argument("--scene-sampling", type=str, default="random", choices=["random", "sequential"], help="How to pick scenes")
    ap.add_argument("--scene-start", type=int, default=0, help="Start scene id for sequential mode")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--start-mode", type=str, default="random", choices=["random", "zero", "none"], help="How to pick start_frame per scene")
    ap.add_argument("--start-frame", type=int, default=None, help="Force start_frame (overrides --start-mode)")
    ap.add_argument("--allow-short-tail", action="store_true", help="Allow starting near end even if fewer than --frames remain")

    ap.add_argument("--ddv2-eta", type=float, default=1.0)
    ap.add_argument("--mode-idx", type=int, default=-1, help="Trajectory mode index; -1 uses mode_select")
    ap.add_argument("--mode-select", type=str, default="greedy", choices=["sample", "greedy"])

    ap.add_argument("--outdir", type=str, default=os.path.join(_REPO_ROOT, "outputs", "visualize"), help="Output directory")
    ap.add_argument("--prefix", type=str, default=None, help="Filename prefix (default: ckpt basename)")

    args = ap.parse_args()

    try:
        RLReconEnv, DiffusionDriveV2Policy = _lazy_import_runtime()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    ckpt_path = str(args.ckpt)
    if not ckpt_path:
        raise RuntimeError("Empty --ckpt")

    cfg: Dict[str, Any] = {}
    try:
        if args.config and os.path.exists(str(args.config)):
            cfg = _load_yaml(str(args.config)) or {}
    except Exception:
        cfg = {}

    env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
    reward_cfg = (env_cfg.get("reward", {}) or {}) if isinstance(env_cfg, dict) else {}

    cuda = int(args.cuda)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    # Resolve env knobs: CLI overrides config.
    step_frames = args.step_frames
    if step_frames is None:
        try:
            step_frames = env_cfg.get("step_frames", None)
        except Exception:
            step_frames = None
    try:
        step_frames = int(step_frames) if step_frames is not None else None
    except Exception:
        step_frames = None

    frame_dt_s = float(args.frame_dt_s)
    if not (frame_dt_s > 0.0):
        raise RuntimeError("--frame-dt-s must be > 0")

    render_w = args.render_w
    render_h = args.render_h
    if render_w is None:
        try:
            render_w = env_cfg.get("render_w", None)
        except Exception:
            render_w = None
    if render_h is None:
        try:
            render_h = env_cfg.get("render_h", None)
        except Exception:
            render_h = None
    try:
        render_w = int(render_w) if render_w is not None else None
    except Exception:
        render_w = None
    try:
        render_h = int(render_h) if render_h is not None else None
    except Exception:
        render_h = None

    # Scene selection
    available = _list_scene_ids_from_assets()
    if args.scene_list:
        scenes = _parse_scene_list(str(args.scene_list))
    else:
        scenes = _pick_scenes(
            available=available,
            num_scenes=int(args.num_scenes),
            scene_sampling=str(args.scene_sampling),
            seed=int(args.seed),
            scene_start=int(args.scene_start),
        )

    outdir = str(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    prefix = str(args.prefix) if args.prefix is not None else os.path.splitext(os.path.basename(ckpt_path))[0]
    ts = time.strftime("%Y%m%d-%H%M%S")

    print("==== generate_video ====")
    print(f"ckpt={ckpt_path}")
    print(f"device={device}")
    print(f"scenes={scenes} (available={len(available)})")
    # We will compute step_dt_s after env reset (step_frames may come from env).
    print(f"max_steps={int(args.max_steps)} fps={int(args.fps)} step_frames={step_frames}")
    print(f"render_w={render_w} render_h={render_h} debug={bool(args.debug)}")
    print(f"outdir={outdir} prefix={prefix} ts={ts}")

    for idx, sid in enumerate(scenes):
        # Create env per scene to avoid stale caches
        env = RLReconEnv(
            cuda=cuda,
            scene=int(sid),
            reward_cfg=reward_cfg,
            debug=bool(args.debug or bool(env_cfg.get("debug", False) if isinstance(env_cfg, dict) else False)),
            render_w=render_w,
            render_h=render_h,
        )

        sf = _sample_start_frame(
            env,
            scene_id=int(sid),
            step_frames=step_frames,
            start_mode=str(args.start_mode),
            start_frame=args.start_frame,
            seed=int(args.seed),
            allow_short_tail=bool(args.allow_short_tail),
            max_steps=int(args.max_steps),
        )

        obs, info = env.reset(scene=int(sid), start_frame=sf, step_frames=step_frames)

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

        # Duration control: user wants length in multiples of 0.5s (i.e., multiples of step_dt_s).
        if args.duration_s is not None:
            dur = float(args.duration_s)
            if not (dur > 0.0):
                raise RuntimeError("--duration-s must be > 0")
            steps_f = dur / step_dt_s
            steps_i = int(round(steps_f))
            if steps_i <= 0:
                raise RuntimeError("--duration-s too small")
            # Must be an (almost) integer multiple.
            if abs(steps_f - float(steps_i)) > 1e-6:
                raise RuntimeError(
                    f"--duration-s={dur} is not a multiple of step_dt_s={step_dt_s:.6f}. "
                    "Pick a multiple (e.g., 18.0, 17.5, 1.0, ...) or adjust --frame-dt-s/--step-frames."
                )
            args.max_steps = steps_i

        # DDV2 planning semantics reminder: 8 points * step_dt_s -> 4s horizon when step_dt_s=0.5s.
        ddv2_points = 8
        ddv2_horizon_s = float(ddv2_points) * step_dt_s
        total_rollout_s = float(int(args.max_steps)) * step_dt_s

        interp_method = str(args.interp_method).strip().lower()
        if interp_method not in {"none", "blend", "rerender"}:
            interp_method = "none"
        interp_n = int(args.interp_frames_per_step) if args.interp_frames_per_step is not None else int(max(1, int(sf_eff)))
        interp_n = max(1, int(interp_n))

        x_anchor = getattr(env.env, "x_anchor", 61)
        y_anchor = getattr(env.env, "y_anchor", 61)

        policy = DiffusionDriveV2Policy(
            x_anchor=x_anchor,
            y_anchor=y_anchor,
            ckpt_path=ckpt_path,
            device=str(device),
            rl_lr=float(cfg.get("train", {}).get("ddv2_lr", 1e-5) if isinstance(cfg, dict) else 1e-5),
            reinforce_baseline_beta=float(cfg.get("train", {}).get("ddv2_baseline_beta", 0.98) if isinstance(cfg, dict) else 0.98),
        )

        out_path = os.path.join(outdir, f"{prefix}_scene{int(sid):03d}_{ts}.mp4")
        _ensure_parent(out_path)
        writer = imageio.get_writer(out_path, mode="I", fps=int(args.fps), macro_block_size=1)

        info0 = dict(info or {})
        try:
            info0["scene_id"] = int(getattr(env.env, "scene", int(sid)))
            info0["now_frame"] = int(getattr(env.env, "now_frame", 0))
        except Exception:
            pass
        writer.append_data(_grid_frame(obs, info0, draw_overlay=bool(args.draw_overlay)))

        print(f"-- scene[{idx+1}/{len(scenes)}]={int(sid)} start_frame={sf} out={out_path}")
        try:
            sf_print = "None" if sf is None else str(int(sf))
            sf_s = None if sf is None else float(int(sf)) * float(frame_dt_s)
            sf_s_print = "None" if sf_s is None else f"{sf_s:.2f}s"
            print(
                f"   step_frames={int(sf_eff)} => step_dt={step_dt_s:.3f}s; "
                f"DDV2 predicts {ddv2_points} pts (~{ddv2_horizon_s:.1f}s) but executes 1 pt then replans; "
                f"rollout={int(args.max_steps)} steps (~{total_rollout_s:.1f}s); start_frame={sf_print} (~{sf_s_print})"
            )
            if sim is not None and sf is None and int(args.max_steps) * int(sf_eff) >= int(getattr(sim, "final_frame", 0)):
                print("   note: requested rollout nearly spans the scene, so start_frame often must be near 0.")
        except Exception:
            pass

        done = False
        prev_grid = _grid_frame(obs, info0, draw_overlay=bool(args.draw_overlay))
        for t in range(int(args.max_steps)):
            prev_pose = None
            ts0 = None
            if interp_method == "rerender":
                if sim is None:
                    raise RuntimeError("interp_method=rerender requires env.env simulator")
                prev_pose = np.array(getattr(sim, "start_ego"), dtype=np.float64, copy=True)
                prev_frame = int(getattr(sim, "now_frame", 0))
                try:
                    ts0 = float(sim.trainer.normalized_timestamps[prev_frame].item())
                except Exception:
                    ts0 = float(prev_frame)

            action, _logp, _replay = policy.sample_ddv2rl_with_replay(
                obs,
                eta=float(args.ddv2_eta),
                mode_idx=int(args.mode_idx),
                mode_select=str(args.mode_select),
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            info_i = dict(info or {})
            try:
                info_i["scene_id"] = int(getattr(env.env, "scene", int(sid)))
                info_i["now_frame"] = int(getattr(env.env, "now_frame", 0))
            except Exception:
                pass

            now_grid = _grid_frame(obs, info_i, draw_overlay=bool(args.draw_overlay))

            if interp_method == "none":
                writer.append_data(now_grid)
            elif interp_method == "blend":
                # Linear blend in pixel space between prev_grid and now_grid.
                for j in range(1, int(interp_n) + 1):
                    a = float(j) / float(max(1, int(interp_n)))
                    blended = ((1.0 - a) * prev_grid.astype(np.float32) + a * now_grid.astype(np.float32)).astype(np.uint8)
                    writer.append_data(blended)
            else:
                # Pose interpolation + re-render.
                if sim is None:
                    raise RuntimeError("interp_method=rerender requires env.env simulator")
                next_pose = np.array(getattr(sim, "start_ego"), dtype=np.float64, copy=True)
                next_frame = int(getattr(sim, "now_frame", 0))
                try:
                    ts1 = float(sim.trainer.normalized_timestamps[next_frame].item())
                except Exception:
                    ts1 = float(next_frame)
                if prev_pose is None or ts0 is None:
                    raise RuntimeError("rerender interpolation missing prev pose/time")
                for j in range(1, int(interp_n) + 1):
                    a = float(j) / float(max(1, int(interp_n)))
                    pose_j = _interp_pose(prev_pose, next_pose, a)
                    t_j = (1.0 - a) * float(ts0) + a * float(ts1)
                    obs_j = _render_obs_from_pose_and_time(sim, ego_pose=pose_j, normed_time=t_j)
                    writer.append_data(_grid_frame(obs_j, info_i, draw_overlay=False))

            prev_grid = now_grid
            if done:
                break

        writer.close()
        print(f"   done={done} steps={t+1} saved={out_path}")

    print("==== all done ====")


if __name__ == "__main__":
    main()
