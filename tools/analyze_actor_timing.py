from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def summarize_shards(paths: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for path in paths:
        shard = torch.load(path, map_location="cpu")
        meta = shard.get("meta", {}) or {}
        timing = meta.get("timing", {}) or {}
        if not isinstance(timing, dict):
            continue
        row = {
            "path": str(path),
            "basename": os.path.basename(path),
            "actor_id": meta.get("actor_id", None),
        }
        for key, value in timing.items():
            row[str(key)] = value
            metric_value = _as_float(value)
            if metric_value is None:
                continue
            totals[str(key)] += float(metric_value)
            counts[str(key)] += 1
        rows.append(row)

    summary: Dict[str, Any] = {"num_shards": int(len(rows))}
    if "collect_shard_s" in totals:
        summary["collect_shard_avg_s"] = float(totals["collect_shard_s"] / float(max(1, counts["collect_shard_s"])))
    for key in ["closed_loop_step_avg_s", "act_avg_s", "env_step_avg_s", "render_avg_s"]:
        if key in totals:
            summary[key] = float(totals[key] / float(max(1, counts[key])))
    return rows, summary


def format_summary(summary: Dict[str, Any]) -> str:
    parts = [f"shards={int(summary.get('num_shards', 0))}"]
    key_order = [
        ("collect_avg", "collect_shard_avg_s"),
        ("step_avg", "closed_loop_step_avg_s"),
        ("act_avg", "act_avg_s"),
        ("env_avg", "env_step_avg_s"),
        ("render_avg", "render_avg_s"),
    ]
    for label, key in key_order:
        value = _as_float(summary.get(key, None))
        if value is not None:
            parts.append(f"{label}={value:.2f}s")
    return " ".join(parts)


def _expand_inputs(inputs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in inputs:
        expanded = sorted(glob.glob(str(item)))
        if expanded:
            out.extend(expanded)
        elif os.path.isdir(item):
            out.extend(sorted(glob.glob(os.path.join(str(item), "*.pt"))))
        else:
            out.append(str(item))
    return [path for path in out if os.path.isfile(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize actor rollout timing from shard meta.")
    parser.add_argument("inputs", nargs="+", help="Shard files, globs, or directories.")
    parser.add_argument("--limit", type=int, default=20, help="How many per-shard rows to print.")
    args = parser.parse_args()

    paths = _expand_inputs(args.inputs)
    if len(paths) == 0:
        raise SystemExit("No shard files found.")

    rows, summary = summarize_shards(paths)
    print(format_summary(summary))
    for row in rows[: max(0, int(args.limit))]:
        print(
            f"{row['basename']}: "
            f"collect={_as_float(row.get('collect_shard_s', 0.0)):.2f}s "
            f"step_avg={_as_float(row.get('closed_loop_step_avg_s', 0.0)):.2f}s "
            f"act_avg={_as_float(row.get('act_avg_s', 0.0)):.2f}s "
            f"env_avg={_as_float(row.get('env_step_avg_s', 0.0)):.2f}s "
            f"render_avg={_as_float(row.get('render_avg_s', 0.0)):.2f}s"
        )


if __name__ == "__main__":
    main()
