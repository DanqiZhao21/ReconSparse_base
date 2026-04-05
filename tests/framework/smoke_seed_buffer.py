from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Any, Dict

import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.io.buffer import BufferPaths, atomic_torch_save, ensure_buffer_layout, write_int
from framework.runner.agent_factory import build_agent


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def _clear_path(path: pathlib.Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _make_ppo_shard(agent: Any) -> Dict[str, Any]:
    obs = torch.zeros((2, 18, 64, 64), dtype=torch.float32)
    obs[0, 0, 0, 0] = 1.0
    obs[1, 0, 0, 0] = 2.0
    next_obs = torch.zeros((18, 64, 64), dtype=torch.float32)
    next_obs[0, 0, 0] = 3.0
    replay = [{"feature": 1.0}, {"feature": 2.0}]
    old_logp = agent.logp_from_replay_batch(replay).detach().cpu()
    return {
        "obs": obs,
        "old_logp": old_logp,
        "reward": torch.tensor([1.0, 0.5], dtype=torch.float32),
        "done": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "terminated": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "next_obs": next_obs,
        "done_last": torch.tensor(1.0, dtype=torch.float32),
        "terminated_last": torch.tensor(1.0, dtype=torch.float32),
        "replay": replay,
    }


def _make_reinforce_shard(agent: Any) -> Dict[str, Any]:
    replay = [{"feature": 1.0}, {"feature": 2.0}]
    old_logp = agent.logp_from_replay_batch(replay).detach().cpu()
    return {
        "old_logp": old_logp,
        "reward": torch.tensor([1.0, 0.5], dtype=torch.float32),
        "done": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "replay": replay,
    }


def seed_buffer(config_path: pathlib.Path) -> BufferPaths:
    cfg = load_yaml(config_path)
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    agent_cfg = cfg.get("agent", {}) or {}

    paths = BufferPaths(root=str(al_cfg.get("buffer_dir")))
    root_path = pathlib.Path(paths.root)
    ckpt_path = pathlib.Path(str(agent_cfg.get("ckpt")))

    _clear_path(root_path)
    ensure_buffer_layout(paths)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.parent != root_path and ckpt_path.parent.exists():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    agent = build_agent(cfg, device=torch.device("cpu"))
    agent.save_checkpoint(str(ckpt_path))
    agent.save_checkpoint(paths.latest_ckpt)
    write_int(paths.version_file, 1)

    algo_key = str(train_cfg.get("algo", "ppo")).strip().lower()
    if algo_key.startswith("ppo"):
        shard = _make_ppo_shard(agent)
    else:
        shard = _make_reinforce_shard(agent)
    shard_path = pathlib.Path(paths.shards_dir) / "actor0_e0_v1_t0_smoke.pt"
    atomic_torch_save(shard, str(shard_path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed a tiny actor-learner smoke buffer")
    parser.add_argument("--config", required=True, help="Path to a smoke YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = seed_buffer(pathlib.Path(args.config))
    print(f"Seeded smoke buffer at {paths.root}")


if __name__ == "__main__":
    main()
