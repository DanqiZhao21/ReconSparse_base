from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore

from framework.rewardmodel.config import ObservationRewardModelConfig, RewardLossConfig
from framework.rewardmodel.data.cached_dataset import CachedRewardModelDataset, reward_model_collate
from framework.rewardmodel.models.reward_model import ObservationTrajectoryRewardModel
from framework.rewardmodel.training.losses import reward_model_bce_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train observation-conditioned reward model")
    parser.add_argument("--data-root", required=True, help="Directory containing cached .pt samples")
    parser.add_argument("--output", required=True, help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-2)
    parser.add_argument("--observation-channels", type=int, default=18)
    parser.add_argument("--ego-state-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--query-dim", type=int, default=64)
    parser.add_argument("--trajectory-hidden-dim", type=int, default=128)
    parser.add_argument("--num-horizons", type=int, default=8)
    parser.add_argument("--num-observation-queries", type=int, default=32)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-enable", action="store_true")
    parser.add_argument("--wandb-project", default="ReconDreamer-RL-rewardmodel")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="baseline")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"))
    return parser.parse_args()


def _init_wandb(args: argparse.Namespace, cfg: ObservationRewardModelConfig, dataset_size: int) -> bool:
    if not bool(args.wandb_enable):
        return False
    if wandb is None:
        print("[rewardmodel][wandb] wandb is not available; continuing without logging")
        return False
    init_kwargs = {
        "project": str(args.wandb_project),
        "group": str(args.wandb_group),
        "name": args.wandb_name,
        "entity": args.wandb_entity,
        "config": {
            **vars(args),
            "dataset_size": int(dataset_size),
            "model_config": cfg.to_dict(),
        },
    }
    if args.wandb_mode:
        init_kwargs["mode"] = str(args.wandb_mode)
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    try:
        wandb.init(**init_kwargs)
        return True
    except Exception as exc:
        print(f"[rewardmodel][wandb] init failed: {exc}; continuing without logging")
        return False


def _distributed_context() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return distributed, rank, local_rank, world_size


def _reduce_mean(value: float, *, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / float(dist.get_world_size())
    return float(tensor.detach().cpu())


def train(args: argparse.Namespace) -> None:
    distributed, rank, local_rank, world_size = _distributed_context()
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    image_size = None
    if args.image_height is not None or args.image_width is not None:
        if args.image_height is None or args.image_width is None:
            raise ValueError("--image-height and --image-width must be provided together")
        image_size = (int(args.image_height), int(args.image_width))
    dataset = CachedRewardModelDataset(args.data_root, image_size=image_size)
    cfg = ObservationRewardModelConfig(
        observation_channels=int(args.observation_channels),
        ego_state_dim=int(args.ego_state_dim),
        hidden_dim=int(args.hidden_dim),
        query_dim=int(args.query_dim),
        trajectory_hidden_dim=int(args.trajectory_hidden_dim),
        num_horizons=int(args.num_horizons),
        num_observation_queries=int(args.num_observation_queries),
        num_attention_heads=int(args.num_attention_heads),
        attention_dropout=float(args.attention_dropout),
    )
    model = ObservationTrajectoryRewardModel(cfg).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        collate_fn=reward_model_collate,
        pin_memory=device.type == "cuda",
    )
    loss_cfg = RewardLossConfig()
    wandb_enabled = _init_wandb(args, cfg, len(dataset)) if rank == 0 else False

    model.train()
    global_step = 0
    for epoch in range(int(args.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(
                observations=batch["observations"].to(device=device, dtype=torch.float32),
                ego_states=batch["ego_states"].to(device=device, dtype=torch.float32),
                candidate_trajectories=batch["candidate_trajectories"].to(device=device, dtype=torch.float32),
            )
            loss = reward_model_bce_loss(
                out.metric_logits,
                batch["targets"].to(device=device, dtype=torch.float32),
                valid_mask=batch.get("valid_mask", None).to(device=device) if "valid_mask" in batch else None,
                config=loss_cfg,
            )
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            total_loss += loss_value
            count += 1
            global_step += 1
            if wandb_enabled:
                wandb.log(
                    {
                        "train/loss": loss_value,
                        "train/epoch": int(epoch + 1),
                        "train/global_step": int(global_step),
                        "train/batch_size": int(batch["observations"].shape[0]),
                        "train/world_size": int(world_size),
                    },
                    step=global_step,
                )
        local_mean_loss = total_loss / float(max(1, count))
        mean_loss = _reduce_mean(local_mean_loss, device=device, distributed=distributed)
        if rank == 0:
            print(f"[rewardmodel] epoch={epoch + 1}/{int(args.epochs)} loss={mean_loss:.6f}")
        if wandb_enabled:
            wandb.log({"train/epoch_loss": mean_loss, "train/epoch": int(epoch + 1)}, step=global_step)

    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        torch.save({"model_config": cfg.to_dict(), "state_dict": state_dict}, output)
        print(f"[rewardmodel] saved checkpoint: {output}")
        if wandb_enabled:
            wandb.log({"artifact/checkpoint_saved": 1, "train/final_loss": mean_loss}, step=global_step)
            wandb.finish()
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
