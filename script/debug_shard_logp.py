from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml

from framework.debug.logp_diagnostics import (
    collect_shard_paths,
    compute_recompute_error_summary,
    compute_logp_diagnostics,
    summarize_replay_entry,
    summarize_rows_by_key,
    summarize_tensor_payload,
)
from framework.runner.agent_factory import build_agent
from framework.runner.config_normalization import normalize_actor_learner_cfg
from framework.utils.repo_paths import resolve_repo_path


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def _device_from_arg(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _default_buffer_dir(cfg: Dict[str, Any]) -> str | None:
    actor_learner = ((cfg.get("train", {}) or {}).get("actor_learner", {}) or {})
    return actor_learner.get("resolved_buffer_dir") or actor_learner.get("buffer_dir")


def _default_new_ckpt_path(cfg: Dict[str, Any], *, buffer_dir: str | Path | None) -> Path | None:
    root = Path(buffer_dir) if buffer_dir is not None else None
    if root is None:
        default_buffer = _default_buffer_dir(cfg)
        root = Path(default_buffer) if default_buffer else None
    if root is None:
        return None
    return root / "weights" / "latest.ckpt"


def _history_ckpt_path(history_dir: str | Path, *, version: int) -> Path:
    return Path(history_dir) / f"version_{int(version):06d}.ckpt"


def _version0_ckpt_path(cfg: Dict[str, Any], override: str | Path | None = None) -> Path | None:
    if override is not None:
        return Path(override)
    agent_cfg = cfg.get("agent", {}) or {}
    ckpt = agent_cfg.get("ckpt", None)
    if ckpt is None:
        return None
    return Path(resolve_repo_path(str(ckpt)))


def _load_manifest(path: str | Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return payload


def _manifest_shard_paths(manifest: Dict[str, Any]) -> List[str]:
    shards = manifest.get("shards", [])
    if not isinstance(shards, list):
        raise ValueError("Manifest field 'shards' must be a list")
    out: List[str] = []
    for item in shards:
        if not isinstance(item, dict):
            continue
        archive_path = item.get("archive_path", None)
        if archive_path:
            out.append(str(archive_path))
    return out


def _clip_eps_from_config(cfg: Dict[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    return float((cfg.get("train", {}) or {}).get("clip_eps", 0.2))


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _load_records(
    shard_paths: Sequence[Path],
    *,
    limit_samples: int | None,
) -> tuple[torch.Tensor, List[Dict[str, Any]], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    old_logps: List[torch.Tensor] = []
    replays: List[Dict[str, Any]] = []
    labels: List[str] = []
    metadata_rows: List[Dict[str, Any]] = []
    shard_summary: Dict[str, Any] = {}
    remaining = None if limit_samples is None else max(0, int(limit_samples))

    for shard_path in shard_paths:
        if remaining == 0:
            break
        shard = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard, dict):
            raise ValueError(f"Shard must be a mapping: {shard_path}")
        old_logp = shard.get("old_logp", None)
        replay = shard.get("replay", None)
        if not torch.is_tensor(old_logp):
            raise ValueError(f"Shard missing tensor old_logp: {shard_path}")
        if not isinstance(replay, list):
            raise ValueError(f"Shard missing replay list: {shard_path}")
        old_logp = old_logp.to(dtype=torch.float32, device="cpu").view(-1)
        sample_count = min(int(old_logp.numel()), len(replay))
        if sample_count != int(old_logp.numel()) or sample_count != len(replay):
            raise ValueError(
                f"Shard old_logp/replay length mismatch: path={shard_path} "
                f"old_logp={int(old_logp.numel())} replay={len(replay)}"
            )
        if remaining is not None:
            sample_count = min(sample_count, remaining)

        rewards = shard.get("reward", None)
        dones = shard.get("done", None)
        terminated = shard.get("terminated", None)
        truncated = shard.get("truncated", None)
        meta = shard.get("meta", {}) or {}
        shard_key = str(shard_path)
        shard_summary[shard_key] = {
            "samples": int(sample_count),
            "actor_id": meta.get("actor_id"),
            "env_id": meta.get("env_id"),
            "weights_version": meta.get("weights_version"),
            "num_steps": meta.get("num_steps"),
            "obs_present": torch.is_tensor(shard.get("obs", None)),
            "next_obs_present": torch.is_tensor(shard.get("next_obs", None)),
        }

        for local_idx in range(sample_count):
            labels.append(f"{shard_path.name}:{local_idx}")
            old_logps.append(old_logp[local_idx].view(()))
            replays.append(replay[local_idx])
            row: Dict[str, Any] = {
                "shard_path": shard_key,
                "shard_name": shard_path.name,
                "local_index": int(local_idx),
                "actor_id": meta.get("actor_id"),
                "env_id": meta.get("env_id"),
                "weights_version": meta.get("weights_version"),
                "num_steps": meta.get("num_steps"),
            }
            for key, tensor in (
                ("reward", rewards),
                ("done", dones),
                ("terminated", terminated),
                ("truncated", truncated),
            ):
                if torch.is_tensor(tensor) and int(tensor.numel()) > local_idx:
                    row[key] = float(tensor.view(-1)[local_idx].item())
            if isinstance(replay[local_idx], dict):
                for key in (
                    "global_mode_idx",
                    "selected_path_idx",
                    "selected_vel_idx",
                    "execute_mode",
                    "sample_token",
                    "scene_id",
                    "frame_idx",
                ):
                    if key in replay[local_idx]:
                        row[f"replay_{key}"] = replay[local_idx][key]
            metadata_rows.append(row)
        if remaining is not None:
            remaining -= sample_count

    if len(old_logps) == 0:
        raise ValueError("No samples found in selected shards")
    return torch.stack(old_logps, dim=0), replays, labels, metadata_rows, shard_summary


def _recompute_logp(
    agent: Any,
    replays: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
    eta: float,
) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(replays), max(1, int(batch_size))):
            batch = list(replays[start : start + max(1, int(batch_size))])
            logp = agent.logp_from_replay_batch(batch, eta=float(eta))
            parts.append(logp.detach().to(device="cpu", dtype=torch.float32).view(-1))
    return torch.cat(parts, dim=0) if len(parts) else torch.empty((0,), dtype=torch.float32)


def _load_agent_for_logp(cfg: Dict[str, Any], *, device: torch.device, ckpt_path: Path) -> Any:
    agent = build_agent(cfg, device=device)
    agent.load_checkpoint(str(ckpt_path), strict=False)
    return agent


def _recompute_old_logp(
    *,
    cfg: Dict[str, Any],
    device: torch.device,
    replays: Sequence[Dict[str, Any]],
    metadata_rows: Sequence[Dict[str, Any]],
    batch_size: int,
    eta: float,
    old_ckpt_path: Path | None,
    ckpt_history_dir: str | Path | None,
    version0_ckpt_path: Path | None,
) -> torch.Tensor | None:
    if old_ckpt_path is not None:
        old_agent = _load_agent_for_logp(cfg, device=device, ckpt_path=old_ckpt_path)
        try:
            return _recompute_logp(old_agent, replays, batch_size=int(batch_size), eta=float(eta))
        finally:
            del old_agent

    if ckpt_history_dir is None:
        return None

    by_version: Dict[int, List[int]] = {}
    for idx, row in enumerate(metadata_rows):
        version = row.get("weights_version", None)
        if version is None:
            raise ValueError("Cannot use --ckpt-history-dir because at least one shard is missing meta.weights_version")
        by_version.setdefault(int(version), []).append(int(idx))

    out = torch.empty((len(replays),), dtype=torch.float32)
    for version, indices in sorted(by_version.items()):
        if int(version) == 0:
            ckpt_path = version0_ckpt_path
            if ckpt_path is None:
                raise FileNotFoundError(
                    "Shard weights_version=0 requires the initial config checkpoint; "
                    "provide agent.ckpt in config or pass --version0-ckpt"
                )
        else:
            ckpt_path = _history_ckpt_path(ckpt_history_dir, version=int(version))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing old checkpoint for shard weights_version={version}: {ckpt_path}")
        agent = _load_agent_for_logp(cfg, device=device, ckpt_path=ckpt_path)
        try:
            subset = [replays[idx] for idx in indices]
            values = _recompute_logp(agent, subset, batch_size=int(batch_size), eta=float(eta))
            for offset, idx in enumerate(indices):
                out[int(idx)] = values[int(offset)]
        finally:
            del agent
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _plot_outputs(out_dir: Path, rows: Sequence[Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping plots: {exc}")
        return []

    old = [float(row["old_logp"]) for row in rows]
    new = [float(row["new_logp"]) for row in rows]
    delta = [float(row["delta_logp"]) for row in rows]
    ratio = [min(float(row["ratio"]), 20.0) for row in rows]
    kl_terms = [float(row["approx_kl_term"]) for row in rows]

    outputs: List[str] = []

    def save_hist(values: List[float], title: str, filename: str, xlabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=80)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        fig.tight_layout()
        target = out_dir / filename
        fig.savefig(target, dpi=160)
        plt.close(fig)
        outputs.append(str(target))

    save_hist(old, "old_logp distribution", "old_logp_hist.png", "old_logp")
    save_hist(new, "new_logp distribution", "new_logp_hist.png", "new_logp")
    save_hist(delta, "new_logp - old_logp distribution", "delta_logp_hist.png", "delta_logp")
    save_hist(ratio, "ratio distribution clipped for display at 20", "ratio_hist.png", "ratio")
    save_hist(kl_terms, "per-sample approx_kl term distribution", "approx_kl_term_hist.png", "approx_kl_term")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(old, bins=80, alpha=0.55, density=True, label="stored_old_logp")
    ax.hist(new, bins=80, alpha=0.55, density=True, label="new_logp")
    ax.set_title("stored old_logp vs new_logp density histogram")
    ax.set_xlabel("logp")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    target = out_dir / "old_new_logp_density_overlay.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    outputs.append(str(target))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(old, new, s=8, alpha=0.45)
    lo = min(min(old), min(new))
    hi = max(max(old), max(new))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title("old_logp vs new_logp")
    ax.set_xlabel("old_logp")
    ax.set_ylabel("new_logp")
    fig.tight_layout()
    target = out_dir / "old_vs_new_logp.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    outputs.append(str(target))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline shard replay log-prob diagnostic")
    parser.add_argument("--config", required=True, help="Training YAML config used to build the agent")
    parser.add_argument("--ckpt", default=None, help="Alias for --new-ckpt; kept for compatibility")
    parser.add_argument("--new-ckpt", default=None, help="Updated checkpoint used to recompute new_logp")
    parser.add_argument("--old-ckpt", default=None, help="Original actor checkpoint used for zero-error old_logp sanity check")
    parser.add_argument("--manifest", default=None, help="Debug retention manifest.json; uses archived shards by default")
    parser.add_argument("--ckpt-history-dir", default=None, help="Directory containing version_000001.ckpt style history files")
    parser.add_argument("--version0-ckpt", default=None, help="Initial actor checkpoint for shards with weights_version=0")
    parser.add_argument("--buffer-dir", default=None, help="Actor-learner run buffer dir; defaults from config")
    parser.add_argument("--shards", nargs="*", default=None, help="Explicit shard .pt files to inspect")
    parser.add_argument("--include-consumed", action="store_true", help="Also inspect buffer/consumed when using --buffer-dir")
    parser.add_argument("--pattern", default="*.pt", help="Shard glob pattern under buffer directories")
    parser.add_argument("--limit-shards", type=int, default=None, help="Limit number of shard files")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit total samples after shard loading")
    parser.add_argument("--batch-size", type=int, default=32, help="Replay logp recompute batch size")
    parser.add_argument("--top-k", type=int, default=20, help="Number of worst approx_kl samples to dump")
    parser.add_argument("--clip-eps", type=float, default=None, help="Clip epsilon; defaults to train.clip_eps")
    parser.add_argument("--sanity-tolerance", type=float, default=1.0e-5, help="Tolerance for old-ckpt recompute sanity check")
    parser.add_argument("--eta", type=float, default=1.0, help="Passed to agent.logp_from_replay_batch")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--out-dir", default=None, help="Output directory for CSV/JSON/plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plot generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    normalize_actor_learner_cfg(cfg)
    manifest = _load_manifest(args.manifest)

    buffer_dir = args.buffer_dir or _default_buffer_dir(cfg)
    manifest_shards = _manifest_shard_paths(manifest) if manifest is not None else None
    shard_paths = collect_shard_paths(
        shards=args.shards if args.shards else manifest_shards,
        buffer_dir=buffer_dir,
        include_consumed=bool(args.include_consumed),
        pattern=str(args.pattern),
        limit_shards=args.limit_shards,
    )
    if len(shard_paths) == 0:
        raise FileNotFoundError("No shard files matched the requested inputs")

    if args.new_ckpt or args.ckpt:
        new_ckpt_path = Path(args.new_ckpt or args.ckpt)
    elif manifest is not None and args.ckpt_history_dir is not None and manifest.get("new_version", None) is not None:
        new_ckpt_path = _history_ckpt_path(args.ckpt_history_dir, version=int(manifest["new_version"]))
    else:
        new_ckpt_path = _default_new_ckpt_path(cfg, buffer_dir=buffer_dir)
    if new_ckpt_path is None:
        raise ValueError("No checkpoint provided and no buffer_dir/default weights/latest.ckpt could be resolved")
    if not new_ckpt_path.exists():
        raise FileNotFoundError(f"New checkpoint not found: {new_ckpt_path}")
    old_ckpt_path = Path(args.old_ckpt) if args.old_ckpt else None
    if old_ckpt_path is not None and not old_ckpt_path.exists():
        raise FileNotFoundError(f"Old checkpoint not found: {old_ckpt_path}")

    out_dir = Path(args.out_dir or Path("outputs") / "logp_debug" / time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[logp-debug] config={args.config}")
    if manifest is not None:
        print(f"[logp-debug] manifest={args.manifest}")
    print(f"[logp-debug] new_ckpt={new_ckpt_path}")
    if old_ckpt_path is not None:
        print(f"[logp-debug] old_ckpt={old_ckpt_path}")
    elif args.ckpt_history_dir is not None:
        print(f"[logp-debug] old_ckpt=<history:{args.ckpt_history_dir}>")
    else:
        print("[logp-debug] old_ckpt=<none>; skipping stored old_logp zero-error sanity check")
    print(f"[logp-debug] shards={len(shard_paths)} out_dir={out_dir}")

    old_logp, replays, labels, metadata_rows, shard_summary = _load_records(
        shard_paths,
        limit_samples=args.limit_samples,
    )

    device = _device_from_arg(str(args.device))
    recomputed_old_logp: torch.Tensor | None = None
    sanity_summary: Dict[str, float] | None = None
    recomputed_old_logp = _recompute_old_logp(
        cfg=cfg,
        device=device,
        replays=replays,
        metadata_rows=metadata_rows,
        batch_size=int(args.batch_size),
        eta=float(args.eta),
        old_ckpt_path=old_ckpt_path,
        ckpt_history_dir=args.ckpt_history_dir,
        version0_ckpt_path=_version0_ckpt_path(cfg, args.version0_ckpt),
    )
    if recomputed_old_logp is not None:
        sanity_summary = compute_recompute_error_summary(
            old_logp,
            recomputed_old_logp,
            tolerance=float(args.sanity_tolerance),
        )

    new_agent = _load_agent_for_logp(cfg, device=device, ckpt_path=new_ckpt_path)
    new_logp = _recompute_logp(new_agent, replays, batch_size=int(args.batch_size), eta=float(args.eta))

    clip_eps = _clip_eps_from_config(cfg, args.clip_eps)
    diagnostics = compute_logp_diagnostics(
        old_logp=old_logp,
        new_logp=new_logp,
        clip_eps=clip_eps,
        top_k=int(args.top_k),
        labels=labels,
    )

    rows: List[Dict[str, Any]] = []
    for idx, (row, metadata) in enumerate(zip(diagnostics.rows, metadata_rows)):
        merged = dict(metadata)
        merged.update(row)
        merged["stored_old_logp"] = merged["old_logp"]
        if recomputed_old_logp is not None:
            recomputed_value = float(recomputed_old_logp[idx].item())
            merged["recomputed_old_logp"] = recomputed_value
            merged["old_recompute_delta"] = recomputed_value - float(merged["stored_old_logp"])
            merged["old_recompute_abs_error"] = abs(float(merged["old_recompute_delta"]))
        rows.append(merged)

    top_rows: List[Dict[str, Any]] = []
    for anomaly in diagnostics.top_anomalies:
        idx = int(anomaly["index"])
        enriched = dict(rows[idx])
        replay = replays[idx]
        if isinstance(replay, dict):
            replay_summary = summarize_replay_entry(replay)
            enriched.update({f"replay_summary.{key}": value for key, value in replay_summary.items()})
        top_rows.append(enriched)

    top_abs_delta_rows: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: abs(float(item.get("delta_logp", 0.0))), reverse=True)[: max(0, int(args.top_k))]:
        idx = int(row["index"])
        enriched = dict(row)
        replay = replays[idx]
        if isinstance(replay, dict):
            replay_summary = summarize_replay_entry(replay)
            enriched.update({f"replay_summary.{key}": value for key, value in replay_summary.items()})
        top_abs_delta_rows.append(enriched)

    per_shard_rows = summarize_rows_by_key(rows, key="shard_path")

    csv_path = out_dir / "logp_rows.csv"
    top_csv_path = out_dir / "top_anomalies.csv"
    top_abs_delta_csv_path = out_dir / "top_abs_delta_logp.csv"
    per_shard_csv_path = out_dir / "per_shard_summary.csv"
    summary_path = out_dir / "summary.json"
    _write_csv(csv_path, rows)
    _write_csv(top_csv_path, top_rows)
    _write_csv(top_abs_delta_csv_path, top_abs_delta_rows)
    _write_csv(per_shard_csv_path, per_shard_rows)

    payload = {
        "config": str(args.config),
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "new_ckpt": str(new_ckpt_path),
        "old_ckpt": str(old_ckpt_path) if old_ckpt_path is not None else None,
        "ckpt_history_dir": str(args.ckpt_history_dir) if args.ckpt_history_dir is not None else None,
        "version0_ckpt": str(_version0_ckpt_path(cfg, args.version0_ckpt)),
        "buffer_dir": str(buffer_dir) if buffer_dir is not None else None,
        "clip_eps": float(clip_eps),
        "sanity_tolerance": float(args.sanity_tolerance),
        "old_recompute_sanity": sanity_summary,
        "count": int(diagnostics.count),
        "summary": diagnostics.summary,
        "per_shard_summary": per_shard_rows,
        "shards": shard_summary,
        "top_anomalies": top_rows,
        "top_abs_delta_logp": top_abs_delta_rows,
    }
    for shard_path in shard_paths[: min(5, len(shard_paths))]:
        shard = torch.load(shard_path, map_location="cpu")
        if isinstance(shard, dict):
            key = str(shard_path)
            payload["shards"][key]["obs_summary"] = summarize_tensor_payload("obs", shard.get("obs", None))
            payload["shards"][key]["next_obs_summary"] = summarize_tensor_payload("next_obs", shard.get("next_obs", None))
    _write_json(summary_path, payload)

    plot_paths: List[str] = []
    if not bool(args.no_plots):
        plot_paths = _plot_outputs(out_dir, rows)

    if sanity_summary is not None:
        status = "PASS" if float(sanity_summary.get("pass", 0.0)) >= 1.0 else "FAIL"
        print(f"[logp-debug] old_logp sanity={status}")
        for key in sorted(sanity_summary.keys()):
            print(f"  sanity/{key}: {sanity_summary[key]:.6g}")
    print("[logp-debug] summary")
    for key in sorted(diagnostics.summary.keys()):
        value = diagnostics.summary[key]
        print(f"  {key}: {value:.6g}" if _safe_float(value) is not None else f"  {key}: {value}")
    print(f"[logp-debug] wrote {csv_path}")
    print(f"[logp-debug] wrote {top_csv_path}")
    print(f"[logp-debug] wrote {top_abs_delta_csv_path}")
    print(f"[logp-debug] wrote {per_shard_csv_path}")
    print(f"[logp-debug] wrote {summary_path}")
    for path in plot_paths:
        print(f"[logp-debug] wrote {path}")


if __name__ == "__main__":
    main()
