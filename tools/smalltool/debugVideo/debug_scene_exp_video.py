import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import imageio


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconsimulator.envs.nus import ReconSimulator


def _grid_frame(
    observation: Dict[str, np.ndarray],
    info: Optional[Dict[str, Any]],
    *,
    draw_traj_overlay: bool,
) -> np.ndarray:
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

        if isinstance(info, dict):
            scene_id = info.get("scene_id", None)
            now_frame = info.get("now_frame", None)
            xz_err_m = info.get("xz_err_m", None)
            yaw_err_deg = info.get("yaw_err_deg", None)
            exp_pos = info.get("exp_pos", None)
            act_pos = info.get("act_pos", None)
            exp_yaw_deg = info.get("exp_yaw_deg", None)
            act_yaw_deg = info.get("act_yaw_deg", None)

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
                line3 = f"err: xz={float(xz_err_m):.3f}m, yaw={float(yaw_err_deg):.2f}deg"
        except Exception:
            line3 = None

        base_x = x0 + 10
        base_y = y0 + 24
        if line0:
            cv2.putText(grid, line0, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1, cv2.LINE_AA)
            base_y += 24
        cv2.putText(grid, line1, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(grid, line2, (base_x, base_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 0, 0), 1, cv2.LINE_AA)
        if line3:
            cv2.putText(grid, line3, (base_x, base_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    except Exception:
        return grid

    return grid


def _auto_max_steps(env: ReconSimulator) -> int:
    try:
        final_frame = int(getattr(env, "final_frame", 0))
        now_frame0 = int(getattr(env, "now_frame", 0))
        step_frames = int(getattr(env, "step_frames", 1))
        remaining = max(0, (final_frame - 1) - now_frame0)
        # number of steps needed to reach final_frame-1
        auto_steps = int(np.ceil(remaining / max(1, step_frames)))
        return max(1, auto_steps + 1)
    except Exception:
        return 10000


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a single-scene video driven by expert trajectory (no agent).")
    ap.add_argument("--scene", type=int, required=True, help="Scene id (e.g., 122)")
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--step-frames", type=int, default=None)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max env steps; -1 runs until env termination (expert-driven).",
    )
    ap.add_argument("--out", type=str, default=None, help="Output mp4 path (default: outputs/exp_debug/sceneXXX_*.mp4)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--draw-traj-overlay", action="store_true")

    args = ap.parse_args()

    device_desc = f"cuda:{int(args.cuda)}" if int(args.cuda) >= 0 else "cuda"

    # debug=True forces expert-driven advancement inside ReconSimulator.step()
    env = ReconSimulator(cuda=int(args.cuda), scene=int(args.scene), debug=True)

    options: Dict[str, Any] = {}
    if args.start_frame is not None:
        options["start_frame"] = int(args.start_frame)
    if args.step_frames is not None:
        options["step_frames"] = int(args.step_frames)

    obs, info = env.reset(seed=int(args.scene), options=options if len(options) else None)

    ts = time.strftime("%Y%m%d-%H%M%S")
    if args.out is None:
        out_dir = os.path.join(_REPO_ROOT, "outputs", "exp_debug")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"scene{int(args.scene):03d}_{ts}.mp4")
    else:
        out_path = str(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    max_steps = int(args.max_steps)
    if max_steps < 0:
        max_steps = _auto_max_steps(env)

    print("==== expert video ====")
    print(f"scene={int(args.scene)} cuda={int(args.cuda)} device={device_desc} debug=True")
    print(f"start_frame={args.start_frame} step_frames={args.step_frames} max_steps={int(max_steps)}")
    print(f"video_out={out_path}")

    writer = imageio.get_writer(out_path, mode="I", fps=int(args.fps), macro_block_size=1)

    info0: Dict[str, Any] = dict(info or {})
    info0["scene_id"] = int(getattr(env, "scene", int(args.scene)))
    info0["now_frame"] = int(getattr(env, "now_frame", 0))
    writer.append_data(_grid_frame(obs, info0, draw_traj_overlay=bool(args.draw_traj_overlay)))

    last_info: Dict[str, Any] = dict(info or {})
    for t in range(int(max_steps)):
        # Action is irrelevant in debug=True; keep a valid triplet.
        obs, terminated, truncated, info = env.step((0, 0, 0))
        last_info = dict(info or {})
        last_info["scene_id"] = int(getattr(env, "scene", int(args.scene)))
        last_info["now_frame"] = int(getattr(env, "now_frame", -1))

        print(
            f"t={t:03d} frame={last_info.get('now_frame')} terminated={bool(terminated)} truncated={bool(truncated)} "
            f"xz_err_m={last_info.get('xz_err_m')} yaw_err_deg={last_info.get('yaw_err_deg')}"
        )

        writer.append_data(_grid_frame(obs, last_info, draw_traj_overlay=bool(args.draw_traj_overlay)))

        if bool(terminated or truncated):
            break

    writer.close()
    print("==== done ====")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
