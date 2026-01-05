import os
import sys
import time
from typing import Any, Dict

import yaml
import torch
import numpy as np
import imageio

# Ensure project root is on sys.path before importing internal packages
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconsimulator.envs.rl_wrapper import RLReconEnv
from rl.ppo import PPOAgent
from rl.policy_diffusiondrivev2 import DiffusionDriveV2Policy


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "ppo_closed_loop.yaml"))

    env_cfg = cfg.get("env", {})
    reward_cfg = env_cfg.get("reward", {})
    cuda = int(env_cfg.get("cuda", 0))
    scene = int(env_cfg.get("scene", 0))
    debug = bool(env_cfg.get("debug", False))

    env = RLReconEnv(cuda=cuda, scene=scene, reward_cfg=reward_cfg, debug=debug)
    obs, info = env.reset(scene=scene)

    # Anchor sizes (from env attributes)
    x_anchor = getattr(env.env, "x_anchor", 61)
    y_anchor = getattr(env.env, "y_anchor", 61)
    agent_cfg = cfg.get("agent", {})
    ckpt_path = agent_cfg.get("ckpt", "/root/clone/ReconDreamer-RL/DiffusionDriveV2/ckpt/diffusiondrivev2_sel.ckpt")
    use_ddv2 = bool(agent_cfg.get("use_ddv2", True))

    if use_ddv2:
        agent = DiffusionDriveV2Policy(x_anchor=x_anchor, y_anchor=y_anchor, ckpt_path=ckpt_path, device=f"cuda:{cuda}")
    else:
        agent = PPOAgent(x_anchor=x_anchor, y_anchor=y_anchor)

    max_steps = int(env_cfg.get("max_steps", 200))

    ep_reward = 0.0
    
#ADD START
    # ---- Video saving config ----
    train_cfg = cfg.get("train", {})
    save_video = bool(train_cfg.get("save_video", False))
    video_path = str(train_cfg.get("video_path", os.path.join("outputs/ppo_closed_loop", "episode.mp4")))
    fps = int(train_cfg.get("fps", 10))
    draw_traj_overlay = bool(train_cfg.get("draw_traj_overlay", False))
    writer = None
    final_video_path = video_path

    exp_hist: list[tuple[float, float]] = []
    act_hist: list[tuple[float, float]] = []

    def _grid_frame(observation: Dict[str, np.ndarray], info: Dict[str, Any] | None = None) -> np.ndarray:
        """Stack 6 views to 2x3 grid (H*2 x W*3 x 3), optionally draw trajectory overlay bottom-left."""
        keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
        imgs = [observation[k] for k in keys]
        h, w = imgs[0].shape[:2]
        row1 = np.concatenate(imgs[:3], axis=1)
        row2 = np.concatenate(imgs[3:6], axis=1)
        grid = np.concatenate([row1, row2], axis=0)

        # Append positions to history from info
        if info is not None:
            exp_pos = info.get("exp_pos", None)
            act_pos = info.get("act_pos", None)
            if exp_pos is not None and act_pos is not None:
                try:
                    exp_hist.append((float(exp_pos[0]), float(exp_pos[2])))
                    act_hist.append((float(act_pos[0]), float(act_pos[2])))
                except Exception:
                    pass

        if draw_traj_overlay:
            try:
                gh, gw = grid.shape[:2]
                box_w, box_h = 320, 120
                margin = 10
                x0, y0 = margin, gh - box_h - margin
                # Blend gray transparent rectangle
                roi_bg = grid[y0:y0+box_h, x0:x0+box_w].copy()
                overlay = roi_bg.copy()
                import cv2
                cv2.rectangle(overlay, (0, 0), (box_w - 1, box_h - 1), (128, 128, 128), thickness=-1)
                blended = cv2.addWeighted(overlay, 0.4, roi_bg, 0.6, 0)
                grid[y0:y0+box_h, x0:x0+box_w] = blended

                # Compose text lines from info
                exp_pos = info.get("exp_pos") if info else None
                act_pos = info.get("act_pos") if info else None
                exp_yaw_deg = info.get("exp_yaw_deg") if info else None
                act_yaw_deg = info.get("act_yaw_deg") if info else None
                xz_err_m = info.get("xz_err_m") if info else None
                yaw_err_deg = info.get("yaw_err_deg") if info else None

                def fmt_pose(tag, pos, yaw):
                    if pos is None or yaw is None:
                        return f"{tag}: (x=?, y=?) yaw=?"
                    return f"{tag}: x={pos[0]:.3f}, y={pos[1]:.3f}, yaw={float(yaw):.2f}deg"

                line1 = fmt_pose("EXP", exp_pos, exp_yaw_deg)
                line2 = fmt_pose("ACT", act_pos, act_yaw_deg)
                line3 = None
                if xz_err_m is not None and yaw_err_deg is not None:
                    line3 = f"err: xz={float(xz_err_m):.3f}m, yaw={float(yaw_err_deg):.2f}deg"

                # Draw text
                base_x = x0 + 8
                base_y = y0 + 22
                cv2.putText(grid, line1, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(grid, line2, (base_x, base_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                if line3:
                    cv2.putText(grid, line3, (base_x, base_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass
        return grid

    if save_video:
        # Append timestamp to avoid overwrite, e.g., episode_20260104-153045.mp4
        ts = time.strftime("%Y%m%d-%H%M%S")
        base_dir = os.path.dirname(video_path)
        base_name = os.path.basename(video_path)
        name, ext = os.path.splitext(base_name)
        final_video_path = os.path.join(base_dir, f"{name}_{ts}{ext}")
        os.makedirs(os.path.dirname(final_video_path), exist_ok=True)
        writer = imageio.get_writer(final_video_path, mode="I", fps=fps)
        writer.append_data(_grid_frame(obs, None))
#ADD END

    #NOTE: 与环境进行交互
    # import ipdb; ipdb.set_trace()
    for t in range(max_steps):
        # 训练阶段：使用 forward_train（保留梯度）+ 量化为环境动作
        if isinstance(agent, DiffusionDriveV2Policy) and hasattr(agent, "forward_train"):
            pred = agent.forward_train(obs)
            traj = pred["trajectory"]  # (1,8,3)
            traj_np = traj.squeeze(0).detach().cpu().numpy()
            x, y = float(traj_np[0, 0]), float(traj_np[0, 1])
            
            action = agent.quantize_xy_to_action(x, y)
            #PRINT
            # print(f"🐅forward_train ： Step {t}: Predicted (x, y) = ({x:.3f}, {y:.3f})，action is{action}")
        else:
            action = agent.act(obs)
            #PRINT
            print(f"🐅act ： Step {t}: Predicted action = {action}")
            if isinstance(action, tuple):
                action = torch.tensor(action)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        if save_video:
            writer.append_data(_grid_frame(obs, info))
        if terminated or truncated:
            break
    
    out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run.log"), "a", encoding="utf-8") as f:
        f.write(f"{time.time():.0f}\tep_reward={ep_reward:.4f}\tsteps={t+1}\n")

    if writer is not None:
        writer.close()
        print(f"Saved video to: {final_video_path}")

    print(f"Episode finished: reward={ep_reward:.4f}, steps={t+1}")


if __name__ == "__main__":
    main()
