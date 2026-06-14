from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import torch


@dataclass
class LogpDiagnostics:
    count: int
    summary: Dict[str, float]
    rows: List[Dict[str, float | int | str]]
    top_anomalies: List[Dict[str, float | int | str]]


def collect_shard_paths(
    *,
    shards: List[str | Path] | None = None,
    buffer_dir: str | Path | None = None,
    include_consumed: bool = False,
    pattern: str = "*.pt",
    limit_shards: int | None = None,
) -> List[Path]:
    if shards:
        out = [Path(path) for path in shards]
    elif buffer_dir is not None:
        root = Path(buffer_dir)
        search_dirs = [root / "buffer" / "shards"]
        if bool(include_consumed):
            search_dirs.append(root / "buffer" / "consumed")
        out = []
        for directory in search_dirs:
            if directory.exists():
                out.extend(directory.glob(str(pattern)))
        out = sorted(out)
    else:
        raise ValueError("provide either shards or buffer_dir")

    if limit_shards is not None:
        out = out[: max(0, int(limit_shards))]
    return out


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _std(values: torch.Tensor) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return _scalar(values.std(unbiased=False))


def compute_logp_diagnostics(
    *,
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    clip_eps: float,
    top_k: int = 20,
    labels: List[str] | None = None,
) -> LogpDiagnostics:
    old = old_logp.detach().to(dtype=torch.float32, device="cpu").view(-1)
    new = new_logp.detach().to(dtype=torch.float32, device="cpu").view(-1)
    if int(old.numel()) != int(new.numel()):
        raise ValueError(f"old_logp/new_logp length mismatch: old={int(old.numel())} new={int(new.numel())}")
    count = int(old.numel())
    if labels is not None and len(labels) != count:
        raise ValueError(f"labels length mismatch: labels={len(labels)} count={count}")
    if count == 0:
        return LogpDiagnostics(count=0, summary={}, rows=[], top_anomalies=[])

    delta = new - old
    ratio = torch.exp(delta)
    approx_kl_term = (ratio - 1.0) - delta
    clipped = (ratio - 1.0).abs() > float(clip_eps)

    summary = {
        "old_logp_mean": _scalar(old.mean()),
        "old_logp_std": _std(old),
        "old_logp_min": _scalar(old.min()),
        "old_logp_max": _scalar(old.max()),
        "new_logp_mean": _scalar(new.mean()),
        "new_logp_std": _std(new),
        "new_logp_min": _scalar(new.min()),
        "new_logp_max": _scalar(new.max()),
        "delta_logp_mean": _scalar(delta.mean()),
        "delta_logp_std": _std(delta),
        "delta_logp_min": _scalar(delta.min()),
        "delta_logp_max": _scalar(delta.max()),
        "ratio_mean": _scalar(ratio.mean()),
        "ratio_std": _std(ratio),
        "ratio_min": _scalar(ratio.min()),
        "ratio_max": _scalar(ratio.max()),
        "approx_kl": _scalar(approx_kl_term.mean()),
        "approx_kl_max": _scalar(approx_kl_term.max()),
        "clip_frac": _scalar(clipped.to(dtype=torch.float32).mean()),
    }

    rows: List[Dict[str, float | int | str]] = []
    for idx in range(count):
        row: Dict[str, float | int | str] = {
            "index": idx,
            "old_logp": float(old[idx].item()),
            "new_logp": float(new[idx].item()),
            "delta_logp": float(delta[idx].item()),
            "ratio": float(ratio[idx].item()),
            "approx_kl_term": float(approx_kl_term[idx].item()),
            "is_clipped": int(bool(clipped[idx].item())),
        }
        if labels is not None:
            row["label"] = labels[idx]
        rows.append(row)

    top_n = max(0, min(int(top_k), count))
    if top_n == 0:
        top_anomalies: List[Dict[str, float | int | str]] = []
    else:
        order = torch.argsort(approx_kl_term, descending=True)[:top_n].tolist()
        top_anomalies = [rows[int(idx)] for idx in order]

    return LogpDiagnostics(count=count, summary=summary, rows=rows, top_anomalies=top_anomalies)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) == 0:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def summarize_rows_by_key(rows: List[Mapping[str, Any]], *, key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        group_value = str(row.get(key, ""))
        groups.setdefault(group_value, []).append(row)

    out: List[Dict[str, Any]] = []
    for group_value, group_rows in groups.items():
        approx = [float(row.get("approx_kl_term", 0.0) or 0.0) for row in group_rows]
        clipped = [float(row.get("is_clipped", 0.0) or 0.0) for row in group_rows]
        delta = [float(row.get("delta_logp", 0.0) or 0.0) for row in group_rows]
        item: Dict[str, Any] = {
            key: group_value,
            "count": int(len(group_rows)),
            "approx_kl": _mean(approx),
            "approx_kl_max": max(approx) if len(approx) else 0.0,
            "clip_frac": _mean(clipped),
            "delta_logp_mean": _mean(delta),
            "abs_delta_logp_max": max((abs(value) for value in delta), default=0.0),
        }
        for passthrough in ("actor_id", "env_id", "weights_version", "num_steps", "shard_name"):
            if passthrough in group_rows[0]:
                item[passthrough] = group_rows[0].get(passthrough)
        out.append(item)

    return sorted(out, key=lambda item: float(item.get("approx_kl_max", 0.0)), reverse=True)


def compute_recompute_error_summary(
    stored_logp: torch.Tensor,
    recomputed_logp: torch.Tensor,
    *,
    tolerance: float = 1.0e-5,
) -> Dict[str, float]:
    stored = stored_logp.detach().to(dtype=torch.float32, device="cpu").view(-1)
    recomputed = recomputed_logp.detach().to(dtype=torch.float32, device="cpu").view(-1)
    if int(stored.numel()) != int(recomputed.numel()):
        raise ValueError(
            f"stored/recomputed logp length mismatch: stored={int(stored.numel())} "
            f"recomputed={int(recomputed.numel())}"
        )
    if int(stored.numel()) == 0:
        return {
            "count": 0.0,
            "mean_abs_error": 0.0,
            "max_abs_error": 0.0,
            "mismatch_count": 0.0,
            "tolerance": float(tolerance),
            "pass": 1.0,
        }
    abs_error = (recomputed - stored).abs()
    mismatch = abs_error > float(tolerance)
    return {
        "count": float(stored.numel()),
        "mean_abs_error": _scalar(abs_error.mean()),
        "max_abs_error": _scalar(abs_error.max()),
        "mismatch_count": float(mismatch.to(dtype=torch.float32).sum().item()),
        "tolerance": float(tolerance),
        "pass": 1.0 if bool((~mismatch).all().item()) else 0.0,
    }


def _summarize_tensor(prefix: str, tensor: torch.Tensor, out: Dict[str, Any]) -> None:
    value = tensor.detach().to(device="cpu")
    out[f"{prefix}.shape"] = list(value.shape)
    out[f"{prefix}.dtype"] = str(value.dtype)
    if int(value.numel()) == 0:
        out[f"{prefix}.numel"] = 0
        out[f"{prefix}.finite_count"] = 0
        out[f"{prefix}.nan_count"] = 0
        out[f"{prefix}.inf_count"] = 0
        return
    numeric = value.to(dtype=torch.float32)
    finite = torch.isfinite(numeric)
    out[f"{prefix}.numel"] = int(numeric.numel())
    out[f"{prefix}.finite_count"] = int(finite.sum().item())
    out[f"{prefix}.nan_count"] = int(torch.isnan(numeric).sum().item())
    out[f"{prefix}.inf_count"] = int(torch.isinf(numeric).sum().item())
    if bool(finite.any().item()):
        finite_values = numeric[finite]
        out[f"{prefix}.mean"] = float(finite_values.mean().item())
        out[f"{prefix}.std"] = float(finite_values.std(unbiased=False).item())
        out[f"{prefix}.min"] = float(finite_values.min().item())
        out[f"{prefix}.max"] = float(finite_values.max().item())


def summarize_replay_entry(replay: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "global_mode_idx",
        "selected_path_idx",
        "selected_vel_idx",
        "execute_mode",
        "sample_token",
        "scene_id",
        "frame_idx",
    ):
        if key in replay:
            value = replay[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
    for key, value in replay.items():
        if torch.is_tensor(value):
            _summarize_tensor(str(key), value, out)
        elif isinstance(value, Mapping):
            for subkey, subvalue in value.items():
                if torch.is_tensor(subvalue):
                    _summarize_tensor(f"{key}.{subkey}", subvalue, out)
    return out


def summarize_tensor_payload(prefix: str, payload: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if torch.is_tensor(payload):
        _summarize_tensor(prefix, payload, out)
    else:
        out[f"{prefix}.present"] = False
    return out
