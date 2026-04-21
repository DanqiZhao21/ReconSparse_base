from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config at {path} must be a mapping")
    return data


def _iter_sample_tokens(token2vad_path: Path) -> list[str]:
    import pickle

    with token2vad_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected dict in {token2vad_path}, got {type(payload)!r}")
    return [str(token) for token in payload.keys()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute persisted NuScenes PDM sample context cache")
    parser.add_argument("--config", required=True, help="Training YAML config used to build scorer settings")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of sample tokens to precompute",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="0-based worker index for strided token partitioning",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of workers participating in strided token partitioning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_workers) <= 0:
        raise RuntimeError(f"--num-workers must be >= 1, got {args.num_workers!r}")
    if int(args.worker_index) < 0 or int(args.worker_index) >= int(args.num_workers):
        raise RuntimeError(
            f"--worker-index must be in [0, {int(args.num_workers) - 1}], got {args.worker_index!r}"
        )
    cfg = _load_yaml(Path(args.config))

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer
    from reconsimulator.envs import nus_config as nus_cfg

    agent_cfg = dict(cfg.get("agent", {}) or {})
    scorer_cfg = dict(agent_cfg.get("nuscenes_scorer", {}) or {})
    scorer_cfg.pop("backend", None)

    scorer = NuScenesPDMScorer(
        token2vad_path=Path(nus_cfg.TOKEN2VAD_FILE),
        **scorer_cfg,
    )

    sample_tokens = _iter_sample_tokens(Path(nus_cfg.TOKEN2VAD_FILE))
    if int(args.num_workers) > 1:
        sample_tokens = sample_tokens[int(args.worker_index) :: int(args.num_workers)]
    if args.limit is not None:
        sample_tokens = sample_tokens[: max(0, int(args.limit))]

    total = len(sample_tokens)
    print(
        "precomputing NuScenes PDM cache "
        f"for {total} samples (worker_index={int(args.worker_index)}/{int(args.num_workers)})"
    )
    for idx, sample_token in enumerate(sample_tokens, start=1):
        scorer._build_sample_context({"sample_token": sample_token}, patch_radius=20.0)
        if idx == 1 or idx % 100 == 0 or idx == total:
            print(f"[{idx}/{total}] cached sample_token={sample_token}")

    print(f"done: persisted cache root = {scorer._derived_context_cache_root}")


if __name__ == "__main__":
    main()
