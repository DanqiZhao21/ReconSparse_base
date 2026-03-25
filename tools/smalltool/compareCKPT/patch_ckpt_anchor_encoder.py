#!/usr/bin/env python3
"""Copy DDV2 plan_anchor_scorer_encoder weights into a fine-tuned PPO/Reinforce++ checkpoint.

Use case
--------
After this repo's changes, RL model definition now includes
`_trajectory_head.plan_anchor_scorer_encoder` to match some upstream DDV2-RL
checkpoints.

However, existing fine-tuned checkpoints created *before* the module existed in
code will not contain those 6 tensors. This script patches them by copying the
encoder tensors from a source checkpoint (e.g. DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt).

It only edits the checkpoint file. It does not retrain.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import torch


def _load_obj(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    return {"state_dict": obj}


def _extract_sd(obj: Any) -> Dict[str, torch.Tensor]:
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint does not contain a state_dict mapping")
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[str(k)] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Fine-tuned ckpt to patch.")
    ap.add_argument("--source", required=True, help="Checkpoint containing anchor scorer encoder weights.")
    ap.add_argument("--out", required=True, help="Output path for patched ckpt.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite if base already has those keys.")
    args = ap.parse_args()

    base_obj = _load_obj(args.base)
    src_obj = _load_obj(args.source)

    base_sd = _extract_sd(base_obj)
    src_sd = _extract_sd(src_obj)

    copied = 0
    skipped = 0
    for k, v in src_sd.items():
        if "plan_anchor_scorer_encoder" not in k:
            continue

        kk = k
        if kk.startswith("agent."):
            kk = kk[len("agent.") :]
        if kk.startswith("_transfuser_model."):
            kk = kk[len("_transfuser_model.") :]

        target_key = "agent." + kk
        if (target_key in base_sd) and (not args.overwrite):
            skipped += 1
            continue
        base_sd[target_key] = v.detach().cpu()
        copied += 1

    base_obj["state_dict"] = base_sd
    torch.save(base_obj, args.out)

    print(f"Wrote: {args.out}")
    print(f"Copied: {copied} tensors")
    if skipped:
        print(f"Skipped: {skipped} existing (use --overwrite to force)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
