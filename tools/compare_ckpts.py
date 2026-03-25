#!/usr/bin/env python3
"""Compare PyTorch checkpoints (keys/shapes and optional tensor diffs).

Supports common formats:
- {"state_dict": {...}} (Lightning-style)
- raw state_dict (dict[str, Tensor])

Typical DDV2 wrappers in this repo may prefix keys with "agent.".
This script can optionally strip prefixes for a fairer comparison.

Examples:
  python tools/compare_ckpts.py \
    DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt \
    DiffusionDriveV2/ckpt/diffusiondrivev2_sel.ckpt \
    outputs/weight/20260207reinforceplusplus.ckpt \
    outputs/weight/20260129_ppo_ver27_latest.ckptb \
    --strip-prefix agent.

  # Compare baseline to others and compute simple numeric stats:
  python tools/compare_ckpts.py a.ckpt b.ckpt c.ckpt --baseline a.ckpt --stats

Exit codes:
- 0: success
- 2: usage error
- 3: load/parse error
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


def _load_checkpoint(path: str) -> Any:
    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        raise FileNotFoundError(abspath)
    try:
        return torch.load(abspath, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"torch.load failed for {abspath}: {e}")


def _extract_state_dict(obj: Any) -> Mapping[str, Any]:
    # Lightning / our wrappers often store {"state_dict": ...}
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    # Some checkpoints store nested dicts like {"model": {"state_dict": ...}}
    if isinstance(obj, dict):
        for k in ("model", "module", "net", "agent"):
            v = obj.get(k, None)
            if isinstance(v, dict) and "state_dict" in v and isinstance(v["state_dict"], dict):
                return v["state_dict"]
    # Raw state_dict
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        # Heuristic: if values look like tensors/parameters
        return obj
    raise ValueError(f"Unsupported checkpoint structure (type={type(obj)!r})")


def _strip_prefix(key: str, prefixes: Sequence[str]) -> str:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p) :]
    return key


def _canonicalize_state_dict(sd: Mapping[str, Any], *, strip_prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        kk = _strip_prefix(k, strip_prefixes)
        # Accept tensors and Parameters
        if torch.is_tensor(v):
            out[kk] = v.detach().cpu()
        else:
            try:
                if hasattr(v, "data") and torch.is_tensor(v.data):
                    out[kk] = v.data.detach().cpu()
            except Exception:
                pass
    return out


def _shape_str(t: torch.Tensor) -> str:
    return "x".join(str(int(x)) for x in t.shape) if t.ndim else "scalar"


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _group_prefix(key: str) -> str:
    # group by first segment (before '.')
    return key.split(".", 1)[0] if "." in key else key


@dataclass
class TensorDiff:
    key: str
    same_shape: bool
    same_dtype: bool
    a_shape: str
    b_shape: str
    a_dtype: str
    b_dtype: str
    max_abs: Optional[float] = None
    mean_abs: Optional[float] = None
    rmse: Optional[float] = None


def _tensor_diff(a: torch.Tensor, b: torch.Tensor, *, key: str) -> TensorDiff:
    same_shape = tuple(a.shape) == tuple(b.shape)
    same_dtype = a.dtype == b.dtype
    td = TensorDiff(
        key=key,
        same_shape=same_shape,
        same_dtype=same_dtype,
        a_shape=_shape_str(a),
        b_shape=_shape_str(b),
        a_dtype=_dtype_str(a),
        b_dtype=_dtype_str(b),
    )
    if not same_shape:
        return td

    # Compute stats in float32 to avoid overflow / dtype mismatch.
    aa = a.float()
    bb = b.float()
    diff = aa - bb
    td.max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    td.mean_abs = float(diff.abs().mean().item()) if diff.numel() else 0.0
    td.rmse = float(torch.sqrt((diff * diff).mean()).item()) if diff.numel() else 0.0
    return td


def _summarize_keys(sd: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    prefixes = Counter(_group_prefix(k) for k in sd.keys())
    return {
        "num_keys": int(len(sd)),
        "prefix_counts": dict(prefixes.most_common()),
    }


def compare(
    a_path: str,
    b_path: str,
    *,
    strip_prefixes: Sequence[str],
    max_list: int = 50,
    stats: bool = False,
) -> Dict[str, Any]:
    a_obj = _load_checkpoint(a_path)
    b_obj = _load_checkpoint(b_path)

    a_sd_raw = _extract_state_dict(a_obj)
    b_sd_raw = _extract_state_dict(b_obj)

    a_sd = _canonicalize_state_dict(a_sd_raw, strip_prefixes=strip_prefixes)
    b_sd = _canonicalize_state_dict(b_sd_raw, strip_prefixes=strip_prefixes)

    a_keys = set(a_sd.keys())
    b_keys = set(b_sd.keys())

    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)
    common = sorted(a_keys & b_keys)

    # shape/dtype overview on common keys
    mismatch_shape: List[str] = []
    mismatch_dtype: List[str] = []
    for k in common:
        ta, tb = a_sd[k], b_sd[k]
        if tuple(ta.shape) != tuple(tb.shape):
            mismatch_shape.append(k)
        if ta.dtype != tb.dtype:
            mismatch_dtype.append(k)

    out: Dict[str, Any] = {
        "a": {"path": os.path.abspath(a_path), "summary": _summarize_keys(a_sd)},
        "b": {"path": os.path.abspath(b_path), "summary": _summarize_keys(b_sd)},
        "diff": {
            "only_in_a": {"count": int(len(only_a)), "sample": only_a[:max_list]},
            "only_in_b": {"count": int(len(only_b)), "sample": only_b[:max_list]},
            "common": {"count": int(len(common))},
            "mismatch_shape": {"count": int(len(mismatch_shape)), "sample": mismatch_shape[:max_list]},
            "mismatch_dtype": {"count": int(len(mismatch_dtype)), "sample": mismatch_dtype[:max_list]},
        },
    }

    if stats:
        diffs: List[TensorDiff] = []
        for k in common:
            ta, tb = a_sd[k], b_sd[k]
            # Only compute for reasonably sized tensors to avoid huge runtime.
            # (You can still use --max-list and post-process if needed.)
            if ta.numel() > 50_000_000:
                continue
            diffs.append(_tensor_diff(ta, tb, key=k))

        # Sort by max_abs (None last)
        diffs_sorted = sorted(
            diffs,
            key=lambda d: (-1.0 if d.max_abs is None else -float(d.max_abs), d.key),
        )
        out["tensor_stats"] = {
            "computed": int(len(diffs_sorted)),
            "top_by_max_abs": [d.__dict__ for d in diffs_sorted[:max_list]],
        }

    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="Checkpoint paths (2+)")
    p.add_argument("--baseline", default=None, help="Baseline checkpoint; compare baseline vs each other")
    p.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Strip this prefix from all keys before comparing (repeatable), e.g. --strip-prefix agent.",
    )
    p.add_argument("--max-list", type=int, default=50, help="Max keys to print/sample per category")
    p.add_argument("--stats", action="store_true", help="Compute numeric tensor diff stats for common keys")
    p.add_argument("--report", default=None, help="Write JSON report to this path")

    args = p.parse_args(argv)

    paths = list(dict.fromkeys(args.paths))  # preserve order, de-dup
    if len(paths) < 2:
        p.error("Need at least 2 checkpoints")

    strip_prefixes: List[str] = list(args.strip_prefix)

    # Determine comparisons
    if args.baseline is not None:
        base = args.baseline
        if base not in paths:
            paths = [base] + paths
        pairs = [(base, x) for x in paths if x != base]
    else:
        # Compare first vs each other
        base = paths[0]
        pairs = [(base, x) for x in paths[1:]]

    report: Dict[str, Any] = {
        "strip_prefixes": strip_prefixes,
        "pairs": [],
    }

    for a_path, b_path in pairs:
        try:
            res = compare(
                a_path,
                b_path,
                strip_prefixes=strip_prefixes,
                max_list=int(args.max_list),
                stats=bool(args.stats),
            )
        except Exception as e:
            raise SystemExit(3) from e

        report["pairs"].append(res)

        # Human-readable summary
        da = res["diff"]
        print("=" * 80)
        print(f"A: {res['a']['path']}")
        print(f"B: {res['b']['path']}")
        print(
            "keys: "
            f"common={da['common']['count']} "
            f"onlyA={da['only_in_a']['count']} "
            f"onlyB={da['only_in_b']['count']} "
            f"shape_mismatch={da['mismatch_shape']['count']} "
            f"dtype_mismatch={da['mismatch_dtype']['count']}"
        )
        if da["only_in_a"]["count"]:
            print("only in A (sample):")
            for k in da["only_in_a"]["sample"]:
                print("  ", k)
        if da["only_in_b"]["count"]:
            print("only in B (sample):")
            for k in da["only_in_b"]["sample"]:
                print("  ", k)
        if da["mismatch_shape"]["count"]:
            print("shape mismatch (sample):")
            for k in da["mismatch_shape"]["sample"]:
                print("  ", k)
        if da["mismatch_dtype"]["count"]:
            print("dtype mismatch (sample):")
            for k in da["mismatch_dtype"]["sample"]:
                print("  ", k)

        if args.stats and "tensor_stats" in res:
            ts = res["tensor_stats"]
            print(f"tensor stats computed: {ts['computed']}")
            for row in ts["top_by_max_abs"][: min(10, len(ts["top_by_max_abs"]))]:
                print(
                    f"  {row['key']}: max_abs={row['max_abs']} mean_abs={row['mean_abs']} rmse={row['rmse']} "
                    f"shape {row['a_shape']} dtype {row['a_dtype']}"
                )

    if args.report:
        out_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("=" * 80)
        print(f"Wrote report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
