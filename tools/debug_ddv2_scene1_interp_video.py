import argparse
import copy
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import imageio
import torch
import yaml

from scipy.spatial.transform import Slerp, Rotation as R


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconsimulator.envs.rl_wrapper import RLReconEnv
from reconsimulator.envs.tool import get_sky_view, move_to_device
from rl.policy_diffusiondrivev2 import DiffusionDriveV2Policy


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _grid_frame(observation: Dict[str, np.ndarray]) -> np.ndarray:
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    imgs = [observation[k] for k in keys]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:6], axis=1)
    return np.concatenate([row1, row2], axis=0)


def _interp_pose(p0: np.ndarray, p1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

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


def _render_obs_from_pose_and_time(sim_env, *, ego_pose: np.ndarray, normed_time: float) -> Dict[str, np.ndarray]:
    """Render 6 camera RGBs from an arbitrary ego pose, without stepping the env."""
    device = getattr(sim_env, "device", "cuda")
    w = int(getattr(sim_env, "w", 800))
    h = int(getattr(sim_env, "h", 450))
    transform = getattr(sim_env, "_transform_matrix", None)

    out_imgs: list[np.ndarray] = []
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Debug: run DDV2 ckpt on scene1 and export two videos (raw vs. interpolated rerender)."
    )
    ap.add_argument("--config", type=str, default=os.path.join(_REPO_ROOT, "script", "configs", "ppo_closed_loop.yaml"))
    ap.add_argument("--scene", type=int, default=1)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--step-frames", type=int, default=5, help="Env step_frames; set 5 to align with DDV2 0.5s")

    ap.add_argument("--ckpt", type=str, default=None, help="Override DDV2 ckpt path")
    ap.add_argument("--ddv2-eta", type=float, default=1.0)
    ap.add_argument("--mode-idx", type=int, default=-1)
    ap.add_argument("--mode-select", type=str, default="greedy", choices=["sample", "greedy"])

    ap.add_argument("--raw-fps", type=int, default=2, help="FPS for raw video (1 frame per env step)")
    ap.add_argument("--interp-fps", type=int, default=10, help="FPS for interpolated video")
    ap.add_argument("--interp-frames-per-step", type=int, default=None, help="How many frames to render inside one env step")

    ap.add_argument(
        "--interp-method",
        type=str,
        default="rerender",
        choices=["rerender", "blend"],
        help="rerender: interpolate pose and re-render; blend: linear blend between step frames",
    )

    ap.add_argument("--out-dir", type=str, default=os.path.join(_REPO_ROOT, "outputs", "ddv2_scene_debug"))
    ap.add_argument("--debug", action="store_true", help="Enable simulator debug mode (expert-driven); normally keep off")

    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("env", {}) or {}
    reward_cfg = (env_cfg.get("reward", {}) or {})
    agent_cfg = cfg.get("agent", {}) or {}

    ckpt_default = agent_cfg.get("ckpt", os.path.join(_REPO_ROOT, "DiffusionDriveV2", "ckpt", "diffusiondrivev2_rl.ckpt"))
    ckpt_path = str(args.ckpt) if args.ckpt is not None else str(ckpt_default)
    if not ckpt_path:
        raise RuntimeError("Empty ckpt path. Provide --ckpt or set agent.ckpt in config.")

    device = torch.device(f"cuda:{int(args.cuda)}" if torch.cuda.is_available() else "cpu")

    env = RLReconEnv(
        cuda=int(args.cuda),
        scene=int(args.scene),
        reward_cfg=reward_cfg,
        debug=bool(args.debug),
        render_w=int(env_cfg.get("render_w", 800)) if env_cfg.get("render_w", None) is not None else None,
        render_h=int(env_cfg.get("render_h", 450)) if env_cfg.get("render_h", None) is not None else None,
    )
    obs, info = env.reset(scene=int(args.scene), start_frame=args.start_frame, step_frames=int(args.step_frames))

    sim = env.env

    x_anchor = getattr(sim, "x_anchor", 61)
    y_anchor = getattr(sim, "y_anchor", 61)

    policy = DiffusionDriveV2Policy(
        x_anchor=x_anchor,
        y_anchor=y_anchor,
        ckpt_path=ckpt_path,
        device=str(device),
        rl_lr=float(cfg.get("train", {}).get("ddv2_lr", 1e-5) or 1e-5),
        reinforce_baseline_beta=float(cfg.get("train", {}).get("ddv2_baseline_beta", 0.98) or 0.98),
    )

    step_frames = int(getattr(sim, "step_frames", int(args.step_frames)))
    interp_n = int(args.interp_frames_per_step) if args.interp_frames_per_step is not None else max(1, step_frames)

    ts = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(str(args.out_dir), exist_ok=True)
    raw_path = os.path.join(str(args.out_dir), f"scene{int(args.scene):03d}_raw_{ts}.mp4")
    interp_path = os.path.join(str(args.out_dir), f"scene{int(args.scene):03d}_interp_{ts}.mp4")

    raw_writer = imageio.get_writer(raw_path, mode="I", fps=int(args.raw_fps), macro_block_size=1)
    interp_writer = imageio.get_writer(interp_path, mode="I", fps=int(args.interp_fps), macro_block_size=1)

    print("==== ddv2 scene debug (raw + interp) ====")
    print(f"scene={int(args.scene)} cuda={int(args.cuda)} device={device}")
    print(f"ckpt={ckpt_path}")
    print(f"start_frame={args.start_frame} step_frames={step_frames} max_steps={int(args.max_steps)}")
    print(f"ddv2_eta={float(args.ddv2_eta)} mode_idx={int(args.mode_idx)} mode_select={str(args.mode_select)}")
    print(f"interp_method={str(args.interp_method)} interp_frames_per_step={interp_n} raw_fps={int(args.raw_fps)} interp_fps={int(args.interp_fps)}")
    print(f"raw_out={raw_path}")
    print(f"interp_out={interp_path}")

    # Initial frame
    raw_writer.append_data(_grid_frame(obs))
    interp_writer.append_data(_grid_frame(obs))

    # Cached previous rendered frame for blend fallback
    prev_grid = _grid_frame(obs)

    for t in range(int(args.max_steps) if int(args.max_steps) >= 0 else 1000000):
        prev_pose = np.array(getattr(sim, "start_ego"), dtype=np.float64, copy=True)
        prev_frame = int(getattr(sim, "now_frame", 0))
        try:
            ts0 = float(sim.trainer.normalized_timestamps[prev_frame].item())
        except Exception:
            ts0 = float(prev_frame)

        action, logp, replay = policy.sample_ddv2rl_with_replay(
            obs,
            eta=float(args.ddv2_eta),
            mode_idx=int(args.mode_idx),
            mode_select=str(args.mode_select),
        )

        obs, reward, terminated, truncated, info = env.step(action)

        next_pose = np.array(getattr(sim, "start_ego"), dtype=np.float64, copy=True)
        next_frame = int(getattr(sim, "now_frame", prev_frame + step_frames))
        try:
            ts1 = float(sim.trainer.normalized_timestamps[next_frame].item())
        except Exception:
            ts1 = float(next_frame)

        now_grid = _grid_frame(obs)
        raw_writer.append_data(now_grid)

        sel_mi = None
        try:
            if isinstance(replay, dict) and replay.get("mode_idx") is not None:
                sel_mi = int(replay.get("mode_idx"))
        except Exception:
            sel_mi = None

        print(
            f"t={t:03d} frame={next_frame} action={action} logp={float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp):.4f} "
            f"reward={float(reward):.4f} mode_idx={sel_mi} terminated={bool(terminated)} truncated={bool(truncated)}"
        )

        # Interpolated frames inside this env step
        if str(args.interp_method).strip().lower() == "blend":
            # Linear blend in pixel space between prev_grid and now_grid.
            for j in range(1, int(interp_n) + 1):
                a = float(j) / float(max(1, int(interp_n)))
                blended = ((1.0 - a) * prev_grid.astype(np.float32) + a * now_grid.astype(np.float32)).astype(np.uint8)
                interp_writer.append_data(blended)
        else:
            # Pose interpolation + re-render.
            for j in range(1, int(interp_n) + 1):
                a = float(j) / float(max(1, int(interp_n)))
                pose_j = _interp_pose(prev_pose, next_pose, a)
                t_j = (1.0 - a) * ts0 + a * ts1
                obs_j = _render_obs_from_pose_and_time(sim, ego_pose=pose_j, normed_time=t_j)
                interp_writer.append_data(_grid_frame(obs_j))

        prev_grid = now_grid

        if bool(terminated or truncated):
            break

    raw_writer.close()
    interp_writer.close()

    print("==== done ====")
    print(f"saved raw:   {raw_path}")
    print(f"saved interp:{interp_path}")


if __name__ == "__main__":
    main()
