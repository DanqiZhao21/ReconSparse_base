# tools/smalltool/inspect_ckpt.py
import argparse
import json
import os
import re
from typing import Dict, Any, List, Optional

import torch


DEFAULT_CKPTS = [
    "/root/clone/ReconDreamer-RL/outputs/weight/20260209ppolatest_v24_with_anchor_encoder.ckpt",         # PPO fine-tune
    "/root/clone/ReconDreamer-RL/outputs/weight/20260207reinforceplusplus_with_anchor_encoder.ckpt",     # Reinforce++ fine-tune
    "DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt",    # DDV2 RL base
    "DiffusionDriveV2/ckpt/diffusiondrivev2_sel.ckpt",   # DDV2 SEL base
]


def load_state_dict(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    # Try common structures
    for key in ["state_dict", "model", "weights", "params"]:
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            return obj[key]
    if isinstance(obj, dict):
        # Heuristic: looks like raw state_dict (keys ~ .weight/.bias)
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def normalize_key(k: str, *, strip_prefixes: Optional[List[str]] = None) -> str:
    """Normalize state_dict keys across different wrapper formats.

    In this repo, the same underlying Transfuser/DDV2 model can be saved under
    different top-level namespaces:
      - PPO/Reinforce++ snapshots often save raw modules: _backbone.*, _tf_decoder.* ...
      - DDV2 ckpts often save under: _transfuser_model._backbone.*, ...

    This function strips common wrappers and also flattens `_transfuser_model.`.
    """

    strip_prefixes = strip_prefixes or []

    # Repeatedly strip common wrappers (they can be nested).
    wrappers = ("module.", "agent.", "_agent.", "ddp.", "model.")
    changed = True
    while changed:
        changed = False
        for pfx in wrappers:
            if k.startswith(pfx):
                k = k[len(pfx) :]
                changed = True

    # Crucial: flatten DDV2 wrapper namespace so keys align with PPO snapshots.
    if k.startswith("_transfuser_model."):
        k = k[len("_transfuser_model.") :]

    # Optional user-provided prefixes to strip.
    for pfx in strip_prefixes:
        if k.startswith(pfx):
            k = k[len(pfx) :]

    return k


def guess_model_type(keys: List[str]) -> str:
    """Heuristic RL vs SEL classifier.

    IMPORTANT: DDV2-RL ckpts may contain a small `plan_anchor_scorer_encoder`.
    That alone is NOT sufficient to classify as SEL. SEL typically contains a
    large scorer stack and multiple head parameters.
    """

    lower = [k.lower() for k in keys]
    scorer_cnt = sum(("scorer" in k) for k in lower)
    has_coarse_fine = any(("coarse" in k or "fine" in k) for k in lower)

    # Avoid false positives like "status_encoding" (contains "nc" inside "encoding").
    # Require separator-bounded head tokens.
    head_token = re.compile(r"(?:^|[._])(nc|ep|dac|ttc|comfort)(?:$|[._])", flags=re.IGNORECASE)
    has_multi_heads = any(head_token.search(k) for k in keys)

    if has_multi_heads:
        return "SEL"
    if has_coarse_fine and scorer_cnt >= 10:
        return "SEL"
    if scorer_cnt >= 50:
        return "SEL"
    return "RL"


def group_key(k: str) -> str:
    kl = k.lower()

    # Trajectory head
    if "trajectory_head" in kl or k.startswith("_trajectory_head."):
        return "trajectory_head"

    # Transformer decoder
    if "tf_decoder" in kl or "decoder" in kl:
        return "transformer_decoder"

    # Backbone / transfuser core
    if k.startswith("_backbone.") or "image_encoder" in kl or "backbone" in kl:
        return "transfuser_core"

    # BEV blocks
    if "bev" in kl or k.startswith("_bev_") or k.startswith("bev_"):
        return "bev_blocks"

    # SEL extras
    if "scorer" in kl:
        return "sel_scorer"
    head_token = re.compile(r"(?:^|[._])(nc|ep|dac|ttc|comfort)(?:$|[._])", flags=re.IGNORECASE)
    if head_token.search(k):
        return "sel_heads"

    return "other"


def tensor_numel(v: Any) -> int:
    try:
        return int(v.numel())
    except Exception:
        return 0


def analyze_ckpt(path: str, *, strip_prefixes: Optional[List[str]] = None) -> Dict[str, Any]:
    sd_raw = load_state_dict(path)
    sd = {normalize_key(k, strip_prefixes=strip_prefixes): v for k, v in sd_raw.items()}
    keys = list(sd.keys())
    mtype = guess_model_type(keys)

    groups: Dict[str, Dict[str, Any]] = {}
    for k, v in sd.items():
        g = group_key(k)
        if g not in groups:
            groups[g] = {"count": 0, "params": 0, "sample_keys": []}
        groups[g]["count"] += 1
        groups[g]["params"] += tensor_numel(v)
        if len(groups[g]["sample_keys"]) < 8:
            groups[g]["sample_keys"].append(k)

    total_params = sum(tensor_numel(v) for v in sd.values())
    result = {
        "path": path,
        "exists": os.path.exists(path),
        "model_type": mtype,
        "total_params": total_params,
        "groups": groups,
        "all_keys": keys,
    }
    return result


def compare_sets(a: List[str], b: List[str]) -> Dict[str, int]:
    sa, sb = set(a), set(b)
    return {
        "only_a": len(sa - sb),
        "only_b": len(sb - sa),
        "intersection": len(sa & sb),
        "union": len(sa | sb),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=DEFAULT_CKPTS, help="Checkpoint paths")
    ap.add_argument("--json", type=str, default="", help="Optional JSON output path")
    ap.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Strip this prefix from all keys before comparing (repeatable), e.g. --strip-prefix agent.",
    )
    args = ap.parse_args()

    reports: List[Dict[str, Any]] = []
    for p in args.ckpt:
        if not os.path.exists(p):
            print(f"[WARN] missing: {p}")
            continue
        rep = analyze_ckpt(p, strip_prefixes=list(args.strip_prefix))
        reports.append(rep)

    # Print summaries
    for rep in reports:
        print("=" * 80)
        print(f"CKPT: {rep['path']}")
        print(f"Type: {rep['model_type']}  Total params: {rep['total_params']:,}")
        print("Groups:")
        for g, info in sorted(rep["groups"].items(), key=lambda x: -x[1]["params"]):
            print(f"  - {g:22s}  tensors={info['count']:6d}  params={info['params']:,}")
            if info["sample_keys"]:
                print(f"    samples: ")
                for sk in info["sample_keys"]:
                    print(f"      {sk}")

    # Pairwise comparisons
    print("\nPairwise key-set comparisons:")
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            ci, cj = reports[i], reports[j]
            stats = compare_sets(ci["all_keys"], cj["all_keys"])
            print(f"  [{os.path.basename(ci['path'])}] vs [{os.path.basename(cj['path'])}]: {stats}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON to: {args.json}")


if __name__ == "__main__":
    main()