#!/usr/bin/env python3
"""
Script to read and display the contents of a shard file.
"""

import torch
import json
from pathlib import Path


def read_shard_file(shard_path: str) -> None:
    """Read and display the shard file contents in markdown format."""
    shard = torch.load(shard_path, map_location="cpu")

    print("# Shard File Contents")
    print()
    print(f"**File:** `{shard_path}`")
    print()

    # Print top-level keys
    print("## Top-Level Keys")
    print()
    for key in shard.keys():
        value = shard[key]
        if isinstance(value, torch.Tensor):
            print(f"- `{key}`: Tensor with shape `{list(value.shape)}`, dtype `{value.dtype}`")
        elif isinstance(value, np.ndarray):
            print(f"- `{key}`: ndarray with shape `{list(value.shape)}`, dtype `{value.dtype}`")
        else:
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:97] + "..."
            print(f"- `{key}`: {value_str}")
    print()

    # Detailed inspection of each key
    for key, value in shard.items():
        print(f"## `{key}`")
        print()

        if isinstance(value, torch.Tensor):
            print(f"- **Type:** Tensor")
            print(f"- **Shape:** `{list(value.shape)}`")
            print(f"- **Dtype:** `{value.dtype}`")
            print(f"- **Device:** `{value.device}`")

            # Print statistics for numeric tensors
            if torch.is_floating_point(value):
                print(f"- **Min:** `{value.min().item():.6f}`")
                print(f"- **Max:** `{value.max().item():.6f}`")
                print(f"- **Mean:** `{value.mean().item():.6f}`")
                print(f"- **Std:** `{value.std().item():.6f}`")

            # Print first few values
            flat = value.flatten()
            if flat.numel() > 0:
                print(f"- **First 10 values:** `{flat[:10].tolist()}`")

        elif isinstance(value, np.ndarray):
            print(f"- **Type:** ndarray")
            print(f"- **Shape:** `{list(value.shape)}`")
            print(f"- **Dtype:** `{value.dtype}`")

            # Print statistics for numeric arrays
            if np.issubdtype(value.dtype, np.number):
                print(f"- **Min:** `{value.min():.6f}`")
                print(f"- **Max:** `{value.max():.6f}`")
                print(f"- **Mean:** `{value.mean():.6f}`")
                print(f"- **Std:** `{value.std():.6f}`")

            # Print first few values
            flat = value.flatten()
            if flat.size > 0:
                print(f"- **First 10 values:** `{flat[:10].tolist()}`")

        elif isinstance(value, (list, tuple)):
            print(f"- **Type:** {type(value).__name__}")
            print(f"- **Length:** `{len(value)}`")
            if len(value) > 0:
                first = value[0]
                if isinstance(first, torch.Tensor):
                    print(f"- **First element:** Tensor with shape `{list(first.shape)}`, dtype `{first.dtype}`")
                elif isinstance(first, np.ndarray):
                    print(f"- **First element:** ndarray with shape `{list(first.shape)}`, dtype `{first.dtype}`")
                elif isinstance(first, dict):
                    print(f"- **First element keys:** `{list(first.keys())}`")
                else:
                    print(f"- **First element:** `{str(first)[:100]}`")

        elif isinstance(value, dict):
            print(f"- **Type:** dict")
            print(f"- **Keys:** `{list(value.keys())}`")

            # Recursively inspect first few dict items
            for i, (sub_key, sub_value) in enumerate(value.items()):
                if i >= 3:
                    print(f"- *(remaining {len(value) - 3} keys omitted)*")
                    break

                if isinstance(sub_value, torch.Tensor):
                    print(f"  - `{sub_key}`: Tensor shape `{list(sub_value.shape)}`, dtype `{sub_value.dtype}`")
                elif isinstance(sub_value, np.ndarray):
                    print(f"  - `{sub_key}`: ndarray shape `{list(sub_value.shape)}`, dtype `{sub_value.dtype}`")
                else:
                    sub_str = str(sub_value)
                    if len(sub_str) > 80:
                        sub_str = sub_str[:77] + "..."
                    print(f"  - `{sub_key}`: {sub_str}")

        else:
            print(f"- **Type:** `{type(value).__name__}`")
            value_str = str(value)
            if len(value_str) > 500:
                value_str = value_str[:497] + "..."
            print(f"- **Value:**")
            print(f"  ```")
            print(f"  {value_str}")
            print(f"  ```")

        print()


if __name__ == "__main__":
    import numpy as np  # Import for type checking

    shard_path = "/root/clone/ReconDreamer-RL/checkpoints/actor_learner/20260603_053040_HUGSIM_CloseCloseloop_OpenGRPOCraft_FullPara/buffer/consumed/actor3_e0_v1_t1780464897_7546896b.pt"

    if not Path(shard_path).exists():
        print(f"Error: Shard file not found: {shard_path}")
        exit(1)

    read_shard_file(shard_path)