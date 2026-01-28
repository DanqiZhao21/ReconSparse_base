import argparse
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import yaml
import numpy as np
import torch
import imageio


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconsimulator.envs.rl_wrapper import RLReconEnv
from rl.policy_diffusiondrivev2 import DiffusionDriveV2Policy


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _grid_frame(observation: Dict[str, np.ndarray], info: Optional[Dict[str, Any]], *, draw_traj_overlay: bool) -> np.ndarray:
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    imgs = [observation[k] for k in keys]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:6], axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    if not draw_traj_overlay:
        return grid

    try:
        import cv2
    except Exception:
        return grid

    try:
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
        exp_pos = None
        act_pos = None
        exp_yaw_deg = None
        act_yaw_deg = None
        done_reason = None
        yaw_thr = None
        xz_thr = None

        if isinstance(info, dict):
            scene_id = info.get("scene_id", None)
            now_frame = info.get("now_frame", None)
            xz_err_m = info.get("xz_err_m", None)
            yaw_err_deg = info.get("yaw_err_deg", None)
            exp_pos = info.get("exp_pos", None)
            act_pos = info.get("act_pos", None)
            exp_yaw_deg = info.get("exp_yaw_deg", None)
            act_yaw_deg = info.get("act_yaw_deg", None)
            done_reason = info.get("done_reason", None)
            yaw_thr = info.get("yaw_err_deg_max", None)
            xz_thr = info.get("xz_err_m_max", None)

        line0 = None
        try:
            if scene_id is not None or now_frame is not None:
                s = "?" if scene_id is None else f"{int(scene_id):03d}"
                f = "?" if now_frame is None else str(int(now_frame))
                line0 = f"scene={s} frame={f}"
        except Exception:
            line0 = None

        def _fmt_pose(tag: str, pos: Any, yaw: Any) -> str:
            try:
                if pos is None or yaw is None:
                    return f"{tag}: (x=?, z=?) yaw=?"
                return f"{tag}: x={float(pos[0]):.3f}, z={float(pos[2]):.3f}, yaw={float(yaw):.2f}deg"
            except Exception:
                return f"{tag}: (x=?, z=?) yaw=?"

        line1 = _fmt_pose("EXP", exp_pos, exp_yaw_deg)
        line2 = _fmt_pose("ACT", act_pos, act_yaw_deg)
        line3 = None
        try:
            if xz_err_m is not None and yaw_err_deg is not None:
                thr_s = ""
                if xz_thr is not None or yaw_thr is not None:
                    thr_s = f" (thr xz={xz_thr}, yaw={yaw_thr})"
                line3 = f"err: xz={float(xz_err_m):.3f}m, yaw={float(yaw_err_deg):.2f}deg{thr_s}"
        except Exception:
            line3 = None

        line4 = None
        if done_reason is not None:
            line4 = f"done_reason: {str(done_reason)}"

        base_x = x0 + 10
        base_y = y0 + 24
        if line0:
            cv2.putText(grid, line0, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1, cv2.LINE_AA)
            base_y += 24
        cv2.putText(grid, line1, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(grid, line2, (base_x, base_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 0, 0), 1, cv2.LINE_AA)
        if line3:
            cv2.putText(grid, line3, (base_x, base_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        if line4:
            cv2.putText(grid, line4, (base_x, base_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

    except Exception:
        return grid

    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug a single scene: interact with DDV2 and save a video with yaw/xz overlay.")
    ap.add_argument("--config", type=str, default=os.path.join(_REPO_ROOT, "script", "configs", "ppo_closed_loop.yaml"))
    ap.add_argument("--scene", type=int, required=True, help="Scene id to debug (e.g., 122)")
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=64,
        help="Max env steps to run; use -1 to run until env termination/truncation.",
    )
    ap.add_argument("--debug", action="store_true", help="Enable simulator debug mode")

    ap.add_argument(
        "--disable-threshold-termination",
        action="store_true",
        help="Disable reward_cfg.terminal threshold-based early termination (yaw/xz/collision).",
    )

    ap.add_argument("--ckpt", type=str, default=None, help="Override agent.ckpt path")
    ap.add_argument("--ddv2-eta", type=float, default=None)
    ap.add_argument("--mode-idx", type=int, default=None)
    ap.add_argument(
        "--mode-select",
        type=str,
        default="greedy",
        choices=["sample", "greedy"],
        help="When mode_idx<0: select trajectory mode by sampling or greedy argmax (default: greedy for debug)",
    )

    ap.add_argument("--out", type=str, default=None, help="Output mp4 path (default: outputs/yaw_debug/sceneXXX_*.mp4)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--draw-traj-overlay", action="store_true")

    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("env", {}) or {}
    reward_cfg = env_cfg.get("reward", {}) or {}

    # Optionally disable threshold-based early termination.
    if bool(args.disable_threshold_termination):
        try:
            if isinstance(reward_cfg, dict):
                term_cfg = reward_cfg.get("terminal", {})
                if not isinstance(term_cfg, dict):
                    term_cfg = {}
                term_cfg["enable"] = False
                reward_cfg["terminal"] = term_cfg
        except Exception:
            pass
    train_cfg = cfg.get("train", {}) or {}
    agent_cfg = cfg.get("agent", {}) or {}

    ckpt_path = str(args.ckpt if args.ckpt is not None else agent_cfg.get("ckpt", ""))
    if not ckpt_path:
        raise RuntimeError("Empty ckpt path. Provide --ckpt or set agent.ckpt in config.")

    device = torch.device(f"cuda:{int(args.cuda)}" if torch.cuda.is_available() else "cpu")

    env = RLReconEnv(cuda=int(args.cuda), scene=int(args.scene), reward_cfg=reward_cfg, debug=bool(args.debug or env_cfg.get("debug", False)))
    obs, info = env.reset(scene=int(args.scene), start_frame=args.start_frame)

    # Best-effort: print the *actual* scene loaded and a simple signature from ego_pose.
    try:
        from reconsimulator.envs import nus_config as nus_cfg

        actual_scene = int(getattr(env.env, "scene", int(args.scene)))
        actual_frame = int(getattr(env.env, "now_frame", 0))
        ego_pose_path = os.path.join(
            _REPO_ROOT,
            str(nus_cfg.BASE_DATA_DIR),
            f"{actual_scene:03d}",
            "ego_pose",
            f"{actual_frame:03d}.txt",
        )
        sig = None
        if os.path.exists(ego_pose_path):
            try:
                m = np.loadtxt(ego_pose_path)
                sig = float(m.sum())
            except Exception:
                sig = None
        print(f"actual_env_scene={actual_scene:03d} actual_env_frame={actual_frame} ego_pose_path={ego_pose_path} sig_sum={sig}")
    except Exception:
        pass

    x_anchor = getattr(env.env, "x_anchor", 61)
    y_anchor = getattr(env.env, "y_anchor", 61)

    policy = DiffusionDriveV2Policy(
        x_anchor=x_anchor,
        y_anchor=y_anchor,
        ckpt_path=ckpt_path,
        device=str(device),
        rl_lr=float(train_cfg.get("ddv2_lr", 1e-5)),
        reinforce_baseline_beta=float(train_cfg.get("ddv2_baseline_beta", 0.98)),
    )

    eta = float(args.ddv2_eta if args.ddv2_eta is not None else train_cfg.get("ddv2_eta", 1.0))
    mode_idx = int(args.mode_idx if args.mode_idx is not None else train_cfg.get("ddv2_mode_idx", -1))

    ts = time.strftime("%Y%m%d-%H%M%S")
    if args.out is None:
        out_dir = os.path.join(_REPO_ROOT, "outputs", "yaw_debug")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"scene{int(args.scene):03d}_{ts}.mp4")
    else:
        out_path = str(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    writer = imageio.get_writer(out_path, mode="I", fps=int(args.fps), macro_block_size=1)

    print("==== yaw debug ====")
    print(f"scene={int(args.scene)} cuda={int(args.cuda)} device={device}")
    print(f"ckpt={ckpt_path}")
    print(f"start_frame={args.start_frame} max_steps={int(args.max_steps)}")
    print(f"ddv2_eta={eta} mode_idx={mode_idx}")
    print(f"mode_select={str(args.mode_select)}")
    print(f"disable_threshold_termination={bool(args.disable_threshold_termination)}")
    print(f"video_out={out_path}")

    info0: Dict[str, Any] = dict(info or {})
    try:
        info0["scene_id"] = int(getattr(env.env, "scene", int(args.scene)))
        info0["now_frame"] = int(getattr(env.env, "now_frame", 0))
    except Exception:
        pass
    writer.append_data(_grid_frame(obs, info0, draw_traj_overlay=bool(args.draw_traj_overlay)))

    done_reason = None
    last_info: Dict[str, Any] = dict(info or {})

    # If max_steps < 0: run until env ends (with a safe upper bound).
    if int(args.max_steps) < 0:
        try:
            final_frame = int(getattr(env.env, "final_frame", 0))
            now_frame0 = int(getattr(env.env, "now_frame", 0))
            step_frames = int(getattr(env.env, "step_frames", 1))
            remaining_frames = max(0, final_frame - 1 - now_frame0)
            auto_steps = max(1, int(np.ceil(remaining_frames / max(1, step_frames))))
            max_steps = auto_steps + 5
        except Exception:
            max_steps = 10000
    else:
        max_steps = int(args.max_steps)

    for t in range(int(max_steps)):
        # Use the replay-capable sampler because it supports explicit/auto mode selection via mode_idx.
        action, logp, _replay = policy.sample_ddv2rl_with_replay(
            obs,
            eta=float(eta),
            mode_idx=int(mode_idx),
            mode_select=str(args.mode_select),
        )
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info or {})
        try:
            last_info["scene_id"] = int(getattr(env.env, "scene", int(args.scene)))
            last_info["now_frame"] = int(getattr(env.env, "now_frame", None))
        except Exception:
            pass

        try:
            now_frame = getattr(env.env, "now_frame", None)
        except Exception:
            now_frame = None

        yaw_err = last_info.get("yaw_err_deg", None)
        xz_err = last_info.get("xz_err_m", None)
        exp_yaw = last_info.get("exp_yaw_deg", None)
        act_yaw = last_info.get("act_yaw_deg", None)
        sel_mi = None
        try:
            if isinstance(_replay, dict) and _replay.get("mode_idx") is not None:
                sel_mi = int(_replay.get("mode_idx"))
        except Exception:
            sel_mi = None

        print(
            f"t={t:03d} frame={now_frame} action={action} logp={float(logp.detach().cpu().item()) if torch.is_tensor(logp) else logp:.4f} "
            f"reward={float(reward):.4f} mode_idx={sel_mi} xz_err_m={xz_err} yaw_err_deg={yaw_err} exp_yaw={exp_yaw} act_yaw={act_yaw}"
        )

        done = bool(terminated or truncated)
        if done:
            done_reason = last_info.get("done_reason", None)

        writer.append_data(_grid_frame(obs, last_info, draw_traj_overlay=bool(args.draw_traj_overlay)))

        if done:
            break

    writer.close()
    print("==== done ====")
    print(f"done={bool(done_reason is not None)} done_reason={done_reason}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
