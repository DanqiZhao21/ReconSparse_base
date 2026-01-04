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
    writer = None
    final_video_path = video_path

    def _grid_frame(observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack 6 views to 2x3 grid (H*2 x W*3 x 3)."""
        keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
        imgs = [observation[k] for k in keys]
        h, w = imgs[0].shape[:2]
        row1 = np.concatenate(imgs[:3], axis=1)
        row2 = np.concatenate(imgs[3:6], axis=1)
        grid = np.concatenate([row1, row2], axis=0)
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
        writer.append_data(_grid_frame(obs))
#ADD END

    #NOTE: 与环境进行交互
    # import ipdb; ipdb.set_trace()
    for t in range(max_steps):
        action = agent.act(obs)#action(tensor([48, 31,  0])) action torch.Size([3])
        if isinstance(action, tuple):
            action = torch.tensor(action)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        if save_video:
            writer.append_data(_grid_frame(obs))
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
