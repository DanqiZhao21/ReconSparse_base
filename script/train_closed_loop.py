import os
import sys
import time
from typing import Any, Dict

import yaml

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
    #NOTE: 与环境进行交互
    for t in range(max_steps):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        if terminated or truncated:
            break
    
    out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run.log"), "a", encoding="utf-8") as f:
        f.write(f"{time.time():.0f}\tep_reward={ep_reward:.4f}\tsteps={t+1}\n")

    print(f"Episode finished: reward={ep_reward:.4f}, steps={t+1}")


if __name__ == "__main__":
    main()
