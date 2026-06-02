import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml

'''
cd /root/clone/ReconDreamer-RL

PYTHONPATH=/root/clone/ReconDreamer-RL python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202606011200_HUGSM_reinforcepp_closed_loop_reward-close_loop_NoGRPOCraft_substeps1.yaml

#如果一直启动不起来卡在gsplat编译:
rm -rf /root/clone/ReconDreamer-RL/.cache/torch_extensions/gsplat_cuda_legacy

'''
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


def load_runner_entrypoints() -> Dict[str, Any]:
    from framework.runner.actor_runtime import actor_main
    from framework.runner.config_normalization import normalize_actor_learner_cfg
    from framework.runner.learner_runtime import learner_main
    from framework.runner.orchestrator import orchestrator_main

    return {
        "actor_main": actor_main,
        "learner_main": learner_main,
        "normalize_actor_learner_cfg": normalize_actor_learner_cfg,
        "orchestrator_main": orchestrator_main,
    }


def materialize_orchestrator_config(
    cfg: Dict[str, Any],
    *,
    config_path: str,
    timestamp: str | None = None,
    generated_config_dir: str | os.PathLike[str] | None = None,
) -> str:
    from framework.runner.config_normalization import timestamp_actor_learner_buffer_dir

    run_timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    resolved_buffer_dir = timestamp_actor_learner_buffer_dir(cfg, timestamp=run_timestamp)
    if resolved_buffer_dir is None:
        return str(config_path)

    root = Path(generated_config_dir) if generated_config_dir is not None else Path(resolved_buffer_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_path = root / f"{run_timestamp}_{Path(config_path).stem}.yaml"
    cfg.setdefault("train", {}).setdefault("actor_learner", {})["resolved_from_config"] = str(config_path)
    cfg["train"]["actor_learner"]["run_timestamp"] = str(run_timestamp)
    cfg["train"]["actor_learner"]["resolved_buffer_dir"] = str(resolved_buffer_dir)
    cfg["train"]["actor_learner"]["resolved_config_path"] = str(resolved_path)
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return str(resolved_path)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    cfg = load_yaml(args.config)
    entrypoints = load_runner_entrypoints()
    entrypoints["normalize_actor_learner_cfg"](cfg)

    if args.role == "orchestrator":
        config_path = materialize_orchestrator_config(cfg, config_path=args.config)
        entrypoints["orchestrator_main"](cfg, config_path=config_path)
        return
    if args.role == "actor":
        entrypoints["actor_main"](
            cfg,
            actor_id=args.actor_id,
            gpu_id=args.gpu_id,
            total_actors=args.num_actors,
        )
        return
    entrypoints["learner_main"](cfg, learner_rank=args.learner_rank)


if __name__ == "__main__":
    main()
    
    