from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from framework.runner.agent_factory import build_agent
from framework.utils.repo_paths import resolve_repo_path


TensorTree = Dict[str, torch.Tensor]


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def _load_manifest(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest at {path} must be a mapping")
    return data


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


def _device_from_arg(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _load_agent(cfg: Dict[str, Any], *, device: torch.device, ckpt_path: Path) -> Any:
    agent = build_agent(cfg, device=device)
    agent.load_checkpoint(str(ckpt_path), strict=False)
    return agent


def _select_replays_from_manifest(
    manifest: Dict[str, Any],
    *,
    weights_version: int | None,
    limit_samples: int,
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    replays: List[Dict[str, Any]] = []
    labels: List[str] = []
    selected_version: int | None = weights_version
    shards = manifest.get("shards", [])
    if not isinstance(shards, list):
        raise ValueError("Manifest must contain a list field 'shards'")

    for shard_info in shards:
        if not isinstance(shard_info, dict):
            continue
        version = int(shard_info.get("weights_version", -1))
        if selected_version is None:
            selected_version = int(version)
        if int(version) != int(selected_version):
            continue
        archive_path = shard_info.get("archive_path", shard_info.get("path", None))
        if archive_path is None:
            continue
        shard_path = Path(archive_path)
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
                break
        if len(replays) >= int(limit_samples):
            break

    if selected_version is None:
        raise ValueError("Could not infer a weights_version from manifest")
    if len(replays) == 0:
        raise ValueError(f"No replay entries selected for weights_version={selected_version}")
    return replays, labels, int(selected_version)


def _batch_replay_features(agent: Any, replays: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    helper = getattr(agent, "_batched_replay_features", None)
    if callable(helper):
        return helper(replays)
    camera_keys = list(replays[0]["camera_feature"].keys())
    batched_camera = {
        key: torch.cat([rep["camera_feature"][key] for rep in replays], dim=0)
        for key in camera_keys
    }
    return {
        "camera_feature": batched_camera,
        "status_feature": torch.cat([rep["status_feature"] for rep in replays], dim=0),
    }


def _to_device_features(agent: Any, features: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    helper = getattr(agent, "_to_device_features", None)
    if callable(helper):
        return helper(features, device)
    out: Dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, dict):
            out[key] = {
                sub_key: sub_value.to(device=device, dtype=torch.float32)
                for sub_key, sub_value in value.items()
                if torch.is_tensor(sub_value)
            }
        elif torch.is_tensor(value):
            out[key] = value.to(device=device, dtype=torch.float32)
    return out


def _capture_tensor(out: TensorTree, name: str, value: torch.Tensor) -> None:
    out[name] = value.detach().to(device="cpu", dtype=torch.float32).clone()


def _capture_index(out: TensorTree, name: str, value: torch.Tensor) -> None:
    out[name] = value.detach().to(device="cpu", dtype=torch.long).clone()


def _trace_sparsedrive_forward(agent: Any, replays: Sequence[Dict[str, Any]], *, device: torch.device) -> TensorTree:
    model = agent._model.module if hasattr(agent._model, "module") else agent._model
    model.eval()
    features = _batch_replay_features(agent, replays)
    features_dev = _to_device_features(agent, features, device)
    trace: TensorTree = {}

    camera_feature = features_dev["camera_feature"]
    status_feature = features_dev["status_feature"]
    _capture_tensor(trace, "input/status_feature", status_feature)
    _capture_tensor(trace, "input/imgs", camera_feature["imgs"])

    status_encoding = model._status_encoding(status_feature)
    _capture_tensor(trace, "model/status_encoding", status_encoding)

    feature_maps = model._backbone(camera_feature["imgs"])
    for idx, feature_map in enumerate(feature_maps):
        _capture_tensor(trace, f"backbone/feature_map_{idx}", feature_map)
    camera_feature = dict(camera_feature)
    camera_feature["feature_maps"] = feature_maps

    head = model._trajectory_head
    batch_size = int(status_encoding.shape[0])
    path_vocab = head.path_vocab.data[None].repeat(batch_size, 1, 1, 1)
    vel_vocab = head.vel_vocab.data[None].repeat(batch_size, 1, 1)
    traj_vocab = head.traj_vocab.data[None].repeat(batch_size, 1, 1, 1, 1)
    traj_mask = head.traj_mask.data[None].repeat(batch_size, 1, 1, 1)
    path_embed = head.path_pos_embed(path_vocab.flatten(-2, -1))
    vel_embed = head.vel_pos_embed(vel_vocab)
    _capture_tensor(trace, "head/path_embed_initial", path_embed)
    _capture_tensor(trace, "head/vel_embed_initial", vel_embed)

    from navsim.agents.sparsedrive.ops import deformable_format

    for layer_idx, layer in enumerate(head.decoder.layers):
        prefix = f"decoder{layer_idx}"
        num_path = int(path_embed.shape[1])
        num_vel = int(vel_embed.shape[1])
        img_value = camera_feature["feature_maps"][-1].permute(0, 1, 3, 4, 2).flatten(1, 3)
        deform_value = deformable_format(camera_feature["feature_maps"])

        path_embed = path_embed + status_encoding.unsqueeze(1)
        vel_embed = vel_embed + status_encoding.unsqueeze(1)
        _capture_tensor(trace, f"{prefix}/path_embed_plus_status", path_embed)
        _capture_tensor(trace, f"{prefix}/vel_embed_plus_status", vel_embed)

        path_vocab_flat = path_vocab[..., :2].flatten(-2)
        path_embed = layer.p_deform_model(
            path_embed,
            path_vocab_flat,
            None,
            deform_value,
            camera_feature,
            None,
        )
        _capture_tensor(trace, f"{prefix}/path_embed_after_p_deform", path_embed)
        path_embed = path_embed + layer.p_dropout1(layer.p_attention(path_embed, path_embed, path_embed)[0])
        path_embed = layer.p_norm1(path_embed)
        path_embed = path_embed + layer.p_dropout2(layer.p_ffn(path_embed))
        path_embed = layer.p_norm2(path_embed)
        path_scores = layer.path_mlp(path_embed).squeeze(-1)
        _capture_tensor(trace, f"{prefix}/path_scores", path_scores)

        vel_embed = vel_embed + layer.v_img_attention(vel_embed, img_value, img_value)[0]
        _capture_tensor(trace, f"{prefix}/vel_embed_after_img_attention", vel_embed)
        vel_embed = vel_embed + layer.v_dropout1(layer.v_attention(vel_embed, vel_embed, vel_embed)[0])
        vel_embed = layer.v_norm1(vel_embed)
        vel_embed = vel_embed + layer.v_dropout2(layer.v_ffn(vel_embed))
        vel_embed = layer.v_norm2(vel_embed)
        vel_scores = layer.vel_mlp(vel_embed).squeeze(-1)
        _capture_tensor(trace, f"{prefix}/vel_scores", vel_scores)

        filter_traj_vocab = traj_vocab.clone()
        filter_traj_mask = traj_mask.clone()
        if num_path > layer._config.path_filter_num[layer.decoder_idx]:
            _scores, topk_path_indices = torch.topk(
                path_scores,
                layer._config.path_filter_num[layer.decoder_idx],
                dim=1,
            )
            _capture_index(trace, f"{prefix}/topk_path_indices", topk_path_indices)
            filter_path_embed = torch.gather(
                path_embed,
                1,
                topk_path_indices.unsqueeze(-1).expand(-1, -1, path_embed.shape[-1]),
            )
            filter_path_vocab = torch.gather(
                path_vocab,
                1,
                topk_path_indices.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, path_vocab.shape[-2], path_vocab.shape[-1]
                ),
            )
            filter_traj_vocab = torch.gather(
                filter_traj_vocab,
                1,
                topk_path_indices[:, :, None, None, None].expand(
                    -1,
                    -1,
                    filter_traj_vocab.shape[-3],
                    filter_traj_vocab.shape[-2],
                    filter_traj_vocab.shape[-1],
                ),
            )
            filter_traj_mask = torch.gather(
                filter_traj_mask,
                1,
                topk_path_indices[:, :, None, None].expand(
                    -1,
                    -1,
                    filter_traj_mask.shape[-2],
                    filter_traj_mask.shape[-1],
                ),
            )
        else:
            filter_path_embed = path_embed
            filter_path_vocab = path_vocab

        if num_vel > layer._config.velocity_filter_num[layer.decoder_idx]:
            _scores, topk_vel_indices = torch.topk(
                vel_scores,
                layer._config.velocity_filter_num[layer.decoder_idx],
                dim=1,
            )
            _capture_index(trace, f"{prefix}/topk_vel_indices", topk_vel_indices)
            filter_vel_embed = torch.gather(
                vel_embed,
                1,
                topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_embed.shape[-1]),
            )
            filter_vel_vocab = torch.gather(
                vel_vocab,
                1,
                topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_vocab.shape[-1]),
            )
            filter_traj_vocab = torch.gather(
                filter_traj_vocab,
                2,
                topk_vel_indices[:, None, :, None, None].expand(
                    -1,
                    filter_traj_vocab.shape[-4],
                    -1,
                    filter_traj_vocab.shape[-2],
                    filter_traj_vocab.shape[-1],
                ),
            )
            filter_traj_mask = torch.gather(
                filter_traj_mask,
                2,
                topk_vel_indices[:, None, :, None].expand(
                    -1,
                    filter_traj_mask.shape[-3],
                    -1,
                    filter_traj_mask.shape[-1],
                ),
            )
        else:
            filter_vel_embed = vel_embed
            filter_vel_vocab = vel_vocab

        path_embed, vel_embed = filter_path_embed, filter_vel_embed
        path_vocab, vel_vocab = filter_path_vocab, filter_vel_vocab
        traj_vocab, traj_mask = filter_traj_vocab, filter_traj_mask
        _capture_tensor(trace, f"{prefix}/filter_traj_vocab", filter_traj_vocab)

        if layer.decoder_idx == layer._config.decoder_num_layers - 1:
            traj_embed = filter_path_embed.unsqueeze(2) + filter_vel_embed.unsqueeze(1)
            traj_embed = traj_embed.flatten(1, 2)
            _capture_tensor(trace, f"{prefix}/traj_embed_initial", traj_embed)
            filter_traj_vocab_flat = filter_traj_vocab[..., :2].flatten(1, 2).flatten(-2)
            traj_embed = layer.t_deform_model(
                traj_embed,
                filter_traj_vocab_flat,
                None,
                deform_value,
                camera_feature,
                None,
            )
            _capture_tensor(trace, f"{prefix}/traj_embed_after_t_deform", traj_embed)
            traj_embed = traj_embed + layer.t_dropout1(layer.t_attention(traj_embed, traj_embed, traj_embed)[0])
            traj_embed = layer.t_norm1(traj_embed)
            traj_embed = traj_embed + layer.t_dropout2(layer.t_ffn(traj_embed))
            traj_embed = layer.t_norm2(traj_embed)
            traj_scores = layer.traj_mlp(traj_embed).squeeze(-1)
            _capture_tensor(trace, f"{prefix}/traj_scores", traj_scores)
            metric_logit: Dict[str, torch.Tensor] = {}
            for metric in layer._config.metrics:
                metric_logit[metric] = layer.metric_heads[metric](traj_embed).squeeze(-1)
                _capture_tensor(trace, f"{prefix}/metric_logits/{metric}", metric_logit[metric])

            candidate_trajectories = filter_traj_vocab.flatten(1, 2)
            _capture_tensor(trace, "output/candidate_trajectories", candidate_trajectories)
            if layer._config.dataset_version == "v1":
                scores = (
                    metric_logit["no_at_fault_collisions"].sigmoid()
                    * metric_logit["drivable_area_compliance"].sigmoid()
                ) * (
                    5 * metric_logit["time_to_collision_within_bound"].sigmoid()
                    + 5 * metric_logit["ego_progress"].sigmoid()
                    + 2 * metric_logit["comfort"].sigmoid()
                )
            elif layer._config.dataset_version == "v2":
                scores = (
                    metric_logit["no_at_fault_collisions"].sigmoid()
                    * metric_logit["drivable_area_compliance"].sigmoid()
                    * metric_logit["driving_direction_compliance"].sigmoid()
                    * metric_logit["traffic_light_compliance"].sigmoid()
                ) * (
                    5 * metric_logit["time_to_collision_within_bound"].sigmoid()
                    + 5 * metric_logit["ego_progress"].sigmoid()
                    + 2 * metric_logit["lane_keeping"].sigmoid()
                    + 2 * metric_logit["history_comfort"].sigmoid()
                )
            else:
                raise ValueError(f"Unsupported SparseDrive dataset_version={layer._config.dataset_version!r}")
            _capture_tensor(trace, "output/candidate_scores", scores)

    return trace


def _select_sample(tensor: torch.Tensor, index: int) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    return tensor[int(index)]


def _compare_trace(
    *,
    single_trace: TensorTree,
    batch_trace: TensorTree,
    sample_index: int,
    label: str,
    tolerance: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stage in single_trace.keys():
        if stage not in batch_trace:
            continue
        single_value = single_trace[stage]
        batch_value = _select_sample(batch_trace[stage], sample_index)
        if single_value.ndim > 0 and single_value.shape[0] == 1:
            single_value = single_value[0]
        if tuple(single_value.shape) != tuple(batch_value.shape):
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "label": label,
                    "stage": stage,
                    "shape_single": list(single_value.shape),
                    "shape_batch": list(batch_value.shape),
                    "kind": "shape_mismatch",
                    "max_abs_error": "",
                    "mean_abs_error": "",
                    "mismatch_count": "",
                    "pass": 0,
                }
            )
            continue
        if single_value.dtype == torch.long or batch_value.dtype == torch.long:
            neq = single_value.to(dtype=torch.long) != batch_value.to(dtype=torch.long)
            mismatch_count = int(neq.sum().item())
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "label": label,
                    "stage": stage,
                    "shape_single": list(single_value.shape),
                    "shape_batch": list(batch_value.shape),
                    "kind": "index",
                    "max_abs_error": int(mismatch_count),
                    "mean_abs_error": "",
                    "mismatch_count": int(mismatch_count),
                    "pass": 1 if mismatch_count == 0 else 0,
                }
            )
            continue
        diff = (single_value.to(dtype=torch.float32) - batch_value.to(dtype=torch.float32)).abs()
        max_abs = float(diff.max().item()) if int(diff.numel()) else 0.0
        mean_abs = float(diff.mean().item()) if int(diff.numel()) else 0.0
        mismatch_count = int((diff > float(tolerance)).sum().item()) if int(diff.numel()) else 0
        rows.append(
            {
                "sample_index": int(sample_index),
                "label": label,
                "stage": stage,
                "shape_single": list(single_value.shape),
                "shape_batch": list(batch_value.shape),
                "kind": "tensor",
                "max_abs_error": max_abs,
                "mean_abs_error": mean_abs,
                "mismatch_count": int(mismatch_count),
                "pass": 1 if mismatch_count == 0 else 0,
            }
        )
    return rows


def _first_failures(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_sample: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if int(row.get("pass", 0)) == 1:
            continue
        sample_index = int(row["sample_index"])
        if sample_index not in by_sample:
            by_sample[sample_index] = row
    return list(by_sample.values())


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _stage_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        stage = str(row["stage"])
        item = grouped.setdefault(
            stage,
            {
                "stage": stage,
                "count": 0,
                "failed": 0,
                "max_abs_error": 0.0,
                "mismatch_count": 0,
            },
        )
        item["count"] += 1
        if int(row.get("pass", 0)) == 0:
            item["failed"] += 1
        try:
            item["max_abs_error"] = max(float(item["max_abs_error"]), float(row.get("max_abs_error") or 0.0))
        except Exception:
            pass
        try:
            item["mismatch_count"] += int(row.get("mismatch_count") or 0)
        except Exception:
            pass
    return sorted(grouped.values(), key=lambda x: (int(x["failed"]) == 0, str(x["stage"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SparseDriveV2 single-sample and batched forward traces")
    parser.add_argument("--config", required=True, help="Training config used to build SparseDriveV2 agent")
    parser.add_argument("--manifest", required=True, help="Debug retention manifest.json")
    parser.add_argument("--ckpt-history-dir", default=None, help="Directory containing version_XXXXXX.ckpt")
    parser.add_argument("--version0-ckpt", default=None, help="Checkpoint to use for weights_version=0")
    parser.add_argument("--weights-version", type=int, default=None, help="Only select replay entries from this version")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of replay entries to compare")
    parser.add_argument("--limit-samples", type=int, default=None, help="Alias/override for --batch-size")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--disable-tf32", action="store_true", help="Disable TF32 matmul/cuDNN before model construction")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic algorithms where PyTorch supports them")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if bool(args.disable_tf32):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        print("[batch-invariance] disabled TF32 matmul/cuDNN")
    if bool(args.deterministic):
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        print("[batch-invariance] enabled deterministic algorithms warn_only=True")

    cfg = _load_yaml(args.config)
    manifest = _load_manifest(args.manifest)
    sample_count = int(args.limit_samples or args.batch_size)
    replays, labels, version = _select_replays_from_manifest(
        manifest,
        weights_version=args.weights_version,
        limit_samples=sample_count,
    )

    if int(version) == 0:
        ckpt_path = _version0_ckpt_path(cfg, args.version0_ckpt)
        if ckpt_path is None:
            raise FileNotFoundError("weights_version=0 requires --version0-ckpt or agent.ckpt in config")
    else:
        if args.ckpt_history_dir is None:
            raise ValueError("Nonzero weights_version requires --ckpt-history-dir")
        ckpt_path = _history_ckpt_path(args.ckpt_history_dir, version=int(version))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for weights_version={version}: {ckpt_path}")

    device = _device_from_arg(args.device)
    print(f"[batch-invariance] config={args.config}")
    print(f"[batch-invariance] manifest={args.manifest}")
    print(f"[batch-invariance] weights_version={version} ckpt={ckpt_path}")
    print(f"[batch-invariance] samples={len(replays)} device={device} out_dir={args.out_dir}")

    agent = _load_agent(cfg, device=device, ckpt_path=ckpt_path)
    rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        batch_trace = _trace_sparsedrive_forward(agent, replays, device=device)
        for sample_index, replay in enumerate(replays):
            single_trace = _trace_sparsedrive_forward(agent, [replay], device=device)
            rows.extend(
                _compare_trace(
                    single_trace=single_trace,
                    batch_trace=batch_trace,
                    sample_index=int(sample_index),
                    label=labels[int(sample_index)],
                    tolerance=float(args.tolerance),
                )
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "layer_compare.csv", rows)
    summary = {
        "config": str(args.config),
        "manifest": str(args.manifest),
        "weights_version": int(version),
        "ckpt": str(ckpt_path),
        "sample_count": int(len(replays)),
        "tolerance": float(args.tolerance),
        "first_failures": _first_failures(rows),
        "stage_summary": _stage_summary(rows),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_csv(out_dir / "stage_summary.csv", summary["stage_summary"])
    _write_csv(out_dir / "first_failures.csv", summary["first_failures"])

    print(f"[batch-invariance] wrote {out_dir / 'layer_compare.csv'}")
    print(f"[batch-invariance] wrote {out_dir / 'stage_summary.csv'}")
    print(f"[batch-invariance] wrote {out_dir / 'first_failures.csv'}")
    if summary["first_failures"]:
        first = summary["first_failures"][0]
        print(
            "[batch-invariance] first failure "
            f"sample={first['sample_index']} stage={first['stage']} "
            f"kind={first['kind']} max_abs_error={first['max_abs_error']}"
        )
    else:
        print("[batch-invariance] all compared stages passed")


if __name__ == "__main__":
    main()
