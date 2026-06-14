from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from framework.runner.agent_factory import build_agent


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def _find_config(run_dir: Path) -> Path:
    configs = sorted(run_dir.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No config yaml found in {run_dir}")
    return configs[-1]


def _find_ckpt(run_dir: Path) -> Path:
    history = run_dir / "weights" / "history"
    ckpts = sorted(history.glob("version_*.ckpt"))
    if ckpts:
        return ckpts[-1]
    latest = run_dir / "weights" / "latest.ckpt"
    if latest.exists():
        return latest
    raise FileNotFoundError(f"No checkpoint found under {run_dir / 'weights'}")


def _load_replays(shard_dir: Path, *, limit_samples: int) -> tuple[List[Dict[str, Any]], List[str]]:
    replays: List[Dict[str, Any]] = []
    labels: List[str] = []
    for shard_path in sorted(shard_dir.glob("*.pt")):
        shard = torch.load(shard_path, map_location="cpu")
        shard_replays = shard.get("replay", [])
        if not isinstance(shard_replays, list):
            continue
        for local_idx, replay in enumerate(shard_replays):
            if not isinstance(replay, dict):
                continue
            replays.append(replay)
            labels.append(f"{shard_path.name}:{local_idx}")
            if len(replays) >= int(limit_samples):
                return replays, labels
    return replays, labels


def _candidate_global_indices(agent: Any, replays: Sequence[Dict[str, Any]], *, device: torch.device) -> torch.Tensor:
    model = agent._unwrap_model(agent._model)
    model.eval()
    features = agent._batched_replay_features(replays)
    features_dev = agent._to_device_features(features, device)
    out = agent._forward_policy_on_model(model, features_dev)
    return agent._candidate_identity_from_outputs(out)["candidate_global_indices"].detach().to(device="cpu", dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SparseDriveV2 replay candidate batch invariance.")
    parser.add_argument("--run-dir", default=None, help="Actor-learner run directory; can infer config, ckpt, and shard dir")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-samples", type=int, default=128)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    config_path = Path(args.config) if args.config else _find_config(run_dir)  # type: ignore[arg-type]
    ckpt_path = Path(args.ckpt) if args.ckpt else _find_ckpt(run_dir)  # type: ignore[arg-type]
    shard_dir = Path(args.shard_dir) if args.shard_dir else (run_dir / "buffer" / "shards")  # type: ignore[operator]
    device = torch.device(args.device)

    cfg = _load_yaml(config_path)
    replays, labels = _load_replays(shard_dir, limit_samples=int(args.limit_samples))
    if not replays:
        raise ValueError(f"No replay entries found in {shard_dir}")

    agent = build_agent(cfg, device=device)
    agent.load_checkpoint(str(ckpt_path), strict=False)
    rows: List[Dict[str, Any]] = []
    batch_missing = 0
    single_recovered = 0
    single_missing = 0

    with torch.inference_mode():
        for start in range(0, len(replays), max(1, int(args.batch_size))):
            chunk = replays[start : start + max(1, int(args.batch_size))]
            chunk_labels = labels[start : start + max(1, int(args.batch_size))]
            batch_ids = _candidate_global_indices(agent, chunk, device=device)
            for local_idx, replay in enumerate(chunk):
                target = int(replay["global_mode_idx"])
                present = bool((batch_ids[local_idx] == target).any().item())
                single_present = None
                if not present:
                    batch_missing += 1
                    single_ids = _candidate_global_indices(agent, [replay], device=device)[0]
                    single_present = bool((single_ids == target).any().item())
                    if single_present:
                        single_recovered += 1
                    else:
                        single_missing += 1
                rows.append(
                    {
                        "index": start + local_idx,
                        "label": chunk_labels[local_idx],
                        "target_global_mode_idx": target,
                        "selected_path_idx": int(replay["selected_path_idx"]),
                        "selected_vel_idx": int(replay["selected_vel_idx"]),
                        "batch_present": int(present),
                        "single_present_when_missing": "" if single_present is None else int(single_present),
                    }
                )

    summary = {
        "config": str(config_path),
        "ckpt": str(ckpt_path),
        "shard_dir": str(shard_dir),
        "batch_size": int(args.batch_size),
        "samples": len(replays),
        "batch_missing": int(batch_missing),
        "single_recovered": int(single_recovered),
        "single_missing": int(single_missing),
    }
    print("[sparsedrive-batch-candidates] summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        with (out_dir / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[sparsedrive-batch-candidates] wrote {out_dir / 'summary.json'}")
        print(f"[sparsedrive-batch-candidates] wrote {out_dir / 'rows.csv'}")


if __name__ == "__main__":
    main()
