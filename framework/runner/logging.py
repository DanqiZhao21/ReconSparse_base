from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Dict, Optional

import torch

try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


def stage(msg: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    prefix = f"[{time.strftime('%H:%M:%S')}]"
    if world > 1:
        prefix = f"{prefix} [rank {rank}]"
    print(f"{prefix} {msg}", flush=True)


def _format_gib(num_bytes: int | float) -> str:
    try:
        return f"{float(num_bytes) / float(1024 ** 3):.2f} GiB"
    except Exception:
        return "n/a"


def _exception_is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def _nvidia_smi_compute_apps() -> list[str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except Exception as exc:
        return [f"nvidia-smi unavailable: {exc}"]

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return [f"nvidia-smi failed rc={proc.returncode}: {err or 'unknown error'}"]

    lines = [line.strip() for line in str(proc.stdout).splitlines() if line.strip()]
    if len(lines) == 0:
        return ["nvidia-smi compute apps: none"]
    return [f"nvidia-smi app: {line}" for line in lines]


def cuda_memory_snapshot_lines(*, label: str) -> list[str]:
    lines = [f"[cuda] snapshot: {label}"]
    if not torch.cuda.is_available():
        lines.append("[cuda] not available")
        return lines

    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        current_device = int(torch.cuda.current_device())
        device_count = int(torch.cuda.device_count())
        lines.append(
            f"[cuda] visible_devices={visible if visible is not None else '<unset>'} "
            f"device_count={device_count} current_device={current_device}"
        )
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            allocated = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            max_allocated = torch.cuda.max_memory_allocated(idx)
            max_reserved = torch.cuda.max_memory_reserved(idx)
            marker = "*" if idx == current_device else "-"
            lines.append(
                f"[cuda] {marker} gpu={idx} name={props.name} "
                f"free={_format_gib(free_bytes)} total={_format_gib(total_bytes)} "
                f"allocated={_format_gib(allocated)} reserved={_format_gib(reserved)} "
                f"max_allocated={_format_gib(max_allocated)} max_reserved={_format_gib(max_reserved)}"
            )
    except Exception as exc:
        lines.append(f"[cuda] torch snapshot failed: {exc}")

    lines.extend(_nvidia_smi_compute_apps())
    return lines


def log_cuda_memory_snapshot(*, label: str, log_fn: Any = stage) -> None:
    writer = log_fn if callable(log_fn) else stage
    for line in cuda_memory_snapshot_lines(label=label):
        writer(line)


def get_train_wandb_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {}) or {}
    wb_cfg = train_cfg.get("wandb", {}) or {}
    return wb_cfg if isinstance(wb_cfg, dict) else {}


def wandb_init_if_enabled(
    cfg: Dict[str, Any],
    *,
    role: str,
    ddp_enabled: bool,
    rank: int,
    actor_id: Optional[int] = None,
) -> bool:
    if not _WANDB_AVAILABLE:
        return False
    wb_cfg = get_train_wandb_cfg(cfg)
    if not bool(wb_cfg.get("enabled", False)):
        return False
    if ddp_enabled and int(rank) != 0:
        return False

    project = str(wb_cfg.get("project", "ReconDreamer-RL"))
    entity = wb_cfg.get("entity", None)
    group = wb_cfg.get("group", None)
    mode = wb_cfg.get("mode", None)
    run_id = wb_cfg.get("id", None)
    resume = wb_cfg.get("resume", None)
    wb_dir = wb_cfg.get("dir", None)
    tags = wb_cfg.get("tags", None)
    name = wb_cfg.get("name", None)
    if name is None or str(name).strip() == "":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{role}_{stamp}" if actor_id is None else f"{role}{int(actor_id)}_{stamp}"

    init_kwargs: Dict[str, Any] = {"project": project, "name": str(name), "config": cfg}
    entity_str = None if entity is None else str(entity).strip()
    if entity_str:
        if entity_str.isdigit():
            stage("[wandb] train.wandb.entity looks like a numeric account id; set a team/workspace slug instead")
        else:
            init_kwargs["entity"] = entity_str
    if group is not None:
        init_kwargs["group"] = str(group)
    if mode is not None:
        init_kwargs["mode"] = str(mode)
    if run_id is not None:
        init_kwargs["id"] = str(run_id)
    if resume is not None:
        init_kwargs["resume"] = str(resume)
    if wb_dir is not None:
        init_kwargs["dir"] = str(wb_dir)
    if tags is not None:
        init_kwargs["tags"] = tags

    try:
        wandb.init(**init_kwargs)
        wandb.define_metric("update")
        wandb.define_metric("global_step")
        wandb.define_metric("global_sample_step")
        wandb.define_metric("global_train_seen_sample_step")
        wandb.define_metric("train_update/*", step_metric="update")
        wandb.define_metric("train_seen_samples/*", step_metric="global_train_seen_sample_step")
        for key in [
            "loss_pi",
            "loss_v",
            "approx_kl",
            "approx_kl_max",
            "ratio_mean",
            "adv_mean",
            "clip_frac",
            "value_clip_frac",
            "explained_variance",
            "collect_time_s",
            "train_time_s",
            "update_time_s",
            "reward_mean",
            "reward_sum",
            "done_rate",
            "ret_mean",
            "ret_std",
            "adv_std",
            "samples",
            "shards",
            "weights_version",
            "num_minibatches",
        ]:
            wandb.define_metric(key, summary="last")
        return True
    except Exception as exc:
        stage(f"[wandb] init failed: {exc}")
        return False


__all__ = [
    "_exception_is_cuda_oom",
    "cuda_memory_snapshot_lines",
    "get_train_wandb_cfg",
    "log_cuda_memory_snapshot",
    "stage",
    "wandb_init_if_enabled",
]
