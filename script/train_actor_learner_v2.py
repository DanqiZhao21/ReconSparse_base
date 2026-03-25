import argparse
import os
import sys
from typing import Any, Dict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actor-learner training entrypoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--role",
        choices=["orchestrator", "actor", "learner"],
        default="orchestrator",
        help="Runtime role for this process",
    )
    parser.add_argument("--actor-id", type=int, default=0, help="Actor id for actor role")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU id for actor role; defaults to runner assignment",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=None,
        help="Total actor count for actor role; used for scene sharding",
    )
    parser.add_argument(
        "--learner-rank",
        type=int,
        default=0,
        help="Learner rank index used by launcher scripts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    from framework.runner.actor_learner import actor_main, learner_main, orchestrator_main
    from framework.runner.factories import normalize_actor_learner_cfg

    normalize_actor_learner_cfg(cfg)

    if args.role == "orchestrator":
        orchestrator_main(cfg, config_path=args.config)
        return
    if args.role == "actor":
        actor_main(
            cfg,
            actor_id=args.actor_id,
            gpu_id=args.gpu_id,
            total_actors=args.num_actors,
        )
        return
    learner_main(cfg, learner_rank=args.learner_rank)


if __name__ == "__main__":
    main()
    
    
# python script/train_actor_learner_v2.py --role orchestrator --config script/configs/ppo_closed_loop_sparsedrive_v2.yaml