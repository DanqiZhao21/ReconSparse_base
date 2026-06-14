from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from script.debug_sparsedrive_batch_invariance import (  # noqa: E402
    _device_from_arg,
    _history_ckpt_path,
    _load_agent,
    _load_manifest,
    _load_yaml,
    _select_replays_from_manifest,
    _to_device_features,
    _version0_ckpt_path,
)


Trace = Dict[str, Tuple[torch.Tensor, bool]]


def _capture(trace: Trace, name: str, value: torch.Tensor, *, batch_dim: bool = True) -> None:
    trace[name] = (value.detach().to(device="cpu").clone(), bool(batch_dim))


def _prepare_features(agent: Any, replays: Sequence[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    features = agent._batched_replay_features(replays)
    return _to_device_features(agent, features, device)


def _trace_decoder0_p_deform(
    agent: Any,
    replays: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    forced_instance_feature: torch.Tensor | None = None,
) -> Trace:
    model = agent._model.module if hasattr(agent._model, "module") else agent._model
    model.eval()
    features_dev = _prepare_features(agent, replays, device)
    camera_feature = dict(features_dev["camera_feature"])
    status_feature = features_dev["status_feature"]
    trace: Trace = {}

    _capture(trace, "input/status_feature", status_feature)
    _capture(trace, "input/imgs", camera_feature["imgs"])

    status_encoding = model._status_encoding(status_feature)
    _capture(trace, "model/status_encoding", status_encoding)

    feature_maps = model._backbone(camera_feature["imgs"])
    for level, feature_map in enumerate(feature_maps):
        _capture(trace, f"backbone/feature_map_{level}", feature_map)
    camera_feature["feature_maps"] = feature_maps
    for meta_key in ("projection_mat", "image_wh"):
        meta_value = camera_feature.get(meta_key, None)
        if torch.is_tensor(meta_value):
            _capture(trace, f"metas/{meta_key}", meta_value)

    head = model._trajectory_head
    layer = head.decoder.layers[0]
    p_deform = layer.p_deform_model
    batch_size = int(status_encoding.shape[0])

    path_vocab = head.path_vocab.data[None].repeat(batch_size, 1, 1, 1)
    path_embed = head.path_pos_embed(path_vocab.flatten(-2, -1))
    path_embed = path_embed + status_encoding.unsqueeze(1)
    if forced_instance_feature is not None:
        path_embed = forced_instance_feature.to(device=path_embed.device, dtype=path_embed.dtype)
        if tuple(path_embed.shape[:2]) != (batch_size, int(path_vocab.shape[1])):
            raise RuntimeError(
                "forced_instance_feature must have shape "
                f"({batch_size}, {int(path_vocab.shape[1])}, embed_dim); got {tuple(path_embed.shape)}"
            )
    anchor = path_vocab[..., :2].flatten(-2)

    _capture(trace, "decoder0/path_vocab", path_vocab)
    _capture(trace, "decoder0/p_deform_input_instance_feature", path_embed)
    _capture(trace, "decoder0/p_deform_input_anchor", anchor)

    from navsim.agents.sparsedrive.ops import deformable_format
    from navsim.agents.sparsedrive.blocks import DAF

    deform_value = deformable_format(camera_feature["feature_maps"])
    _capture(trace, "decoder0/deform_value/feat_flatten", deform_value[0])
    _capture(trace, "decoder0/deform_value/spatial_shapes", deform_value[1], batch_dim=False)
    _capture(trace, "decoder0/deform_value/level_start_index", deform_value[2], batch_dim=False)

    instance_feature = path_embed
    key_points = p_deform.kps_generator(anchor, instance_feature)
    _capture(trace, "decoder0/p_deform/key_points", key_points)

    points_2d, depth, mask = p_deform.project_points(
        key_points,
        camera_feature["projection_mat"],
        camera_feature.get("image_wh"),
    )
    _capture(trace, "decoder0/p_deform/points_2d_projected", points_2d)
    _capture(trace, "decoder0/p_deform/depth", depth)
    _capture(trace, "decoder0/p_deform/mask", mask.to(dtype=torch.long))

    weights = p_deform._get_weights(instance_feature, None, camera_feature, mask)
    _capture(trace, "decoder0/p_deform/weights_softmax", weights)

    bs, num_anchor = instance_feature.shape[:2]
    points_2d_daf = points_2d.permute(0, 2, 3, 1, 4).reshape(bs, num_anchor * p_deform.num_pts, -1, 2)
    weights_daf = (
        weights.permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .reshape(bs, num_anchor * p_deform.num_pts, p_deform.num_cams, p_deform.num_levels, p_deform.num_groups)
    )
    _capture(trace, "decoder0/p_deform/points_2d_daf", points_2d_daf)
    _capture(trace, "decoder0/p_deform/weights_daf", weights_daf)

    if not bool(p_deform.use_deformable_func):
        raise RuntimeError("This diagnostic currently expects p_deform_model.use_deformable_func=True")

    daf_raw = DAF(*deform_value, points_2d_daf, weights_daf)
    _capture(trace, "decoder0/p_deform/daf_raw", daf_raw)

    features = daf_raw.reshape(bs, num_anchor, p_deform.num_pts, p_deform.embed_dims)
    _capture(trace, "decoder0/p_deform/features_before_sum", features)
    features = features.sum(dim=2)
    _capture(trace, "decoder0/p_deform/features_after_sum", features)

    projected = p_deform.proj_drop(p_deform.output_proj(features))
    _capture(trace, "decoder0/p_deform/output_proj", projected)
    if p_deform.residual_mode == "add":
        output = projected + instance_feature
    elif p_deform.residual_mode == "cat":
        output = torch.cat([projected, instance_feature], dim=-1)
    else:
        output = projected
    _capture(trace, "decoder0/path_embed_after_p_deform", output)

    path_after = output + layer.p_dropout1(layer.p_attention(output, output, output)[0])
    path_after = layer.p_norm1(path_after)
    path_after = path_after + layer.p_dropout2(layer.p_ffn(path_after))
    path_after = layer.p_norm2(path_after)
    path_scores = layer.path_mlp(path_after).squeeze(-1)
    _capture(trace, "decoder0/path_scores", path_scores)
    return trace


def _select_value(trace: Trace, stage: str, sample_index: int) -> torch.Tensor:
    value, has_batch_dim = trace[stage]
    if has_batch_dim and value.ndim > 0:
        return value[int(sample_index)]
    return value


def _compare_traces(
    *,
    case: str,
    single_trace: Trace,
    batch_trace: Trace,
    sample_index: int,
    label: str,
    tolerance: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stage in single_trace.keys():
        if stage not in batch_trace:
            continue
        single_value = _select_value(single_trace, stage, 0)
        batch_value = _select_value(batch_trace, stage, sample_index)
        row: Dict[str, Any] = {
            "case": case,
            "sample_index": int(sample_index),
            "label": label,
            "stage": stage,
            "shape_single": list(single_value.shape),
            "shape_batch": list(batch_value.shape),
        }
        if tuple(single_value.shape) != tuple(batch_value.shape):
            row.update({"kind": "shape_mismatch", "max_abs_error": "", "mean_abs_error": "", "mismatch_count": "", "pass": 0})
            rows.append(row)
            continue
        if single_value.dtype == torch.long or batch_value.dtype == torch.long or single_value.dtype == torch.bool or batch_value.dtype == torch.bool:
            neq = single_value.to(dtype=torch.long) != batch_value.to(dtype=torch.long)
            mismatch_count = int(neq.sum().item())
            row.update(
                {
                    "kind": "index",
                    "max_abs_error": int(mismatch_count),
                    "mean_abs_error": "",
                    "mismatch_count": int(mismatch_count),
                    "pass": 1 if mismatch_count == 0 else 0,
                }
            )
            rows.append(row)
            continue
        diff = (single_value.to(dtype=torch.float32) - batch_value.to(dtype=torch.float32)).abs()
        max_abs = float(diff.max().item()) if int(diff.numel()) else 0.0
        mean_abs = float(diff.mean().item()) if int(diff.numel()) else 0.0
        mismatch_count = int((diff > float(tolerance)).sum().item()) if int(diff.numel()) else 0
        row.update(
            {
                "kind": "tensor",
                "max_abs_error": max_abs,
                "mean_abs_error": mean_abs,
                "mismatch_count": int(mismatch_count),
                "pass": 1 if mismatch_count == 0 else 0,
            }
        )
        rows.append(row)
    return rows


def _summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["case"]), str(row["stage"]))
        item = grouped.setdefault(
            key,
            {
                "case": row["case"],
                "stage": row["stage"],
                "count": 0,
                "failed": 0,
                "max_abs_error": 0.0,
                "mismatch_count": 0,
            },
        )
        item["count"] = int(item["count"]) + 1
        item["failed"] = int(item["failed"]) + (0 if int(row.get("pass", 0)) else 1)
        try:
            item["max_abs_error"] = max(float(item["max_abs_error"]), float(row.get("max_abs_error", 0.0)))
        except Exception:
            pass
        try:
            item["mismatch_count"] = int(item["mismatch_count"]) + int(row.get("mismatch_count", 0) or 0)
        except Exception:
            pass
    return sorted(grouped.values(), key=lambda x: (str(x["case"]), int(x["failed"]) == 0, str(x["stage"])))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_permuted(replays: Sequence[Dict[str, Any]], target_index: int, target_position: int) -> Tuple[List[Dict[str, Any]], int]:
    out = list(replays)
    target = out.pop(int(target_index))
    target_position = max(0, min(int(target_position), len(out)))
    out.insert(target_position, target)
    return out, target_position


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace SparseDriveV2 decoder0 p_deform_model single/batch invariance")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ckpt-history-dir", default=None)
    parser.add_argument("--version0-ckpt", default=None)
    parser.add_argument("--weights-version", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-index", type=int, default=0, help="Replay index inside selected samples to compare against single")
    parser.add_argument("--permutation-index", type=int, default=1, help="Position for target replay in the permuted batch")
    parser.add_argument(
        "--case",
        choices=["real", "duplicate", "duplicate_forced_instance", "permuted", "all"],
        default="all",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if bool(args.disable_tf32):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        print("[deform-batch] disabled TF32 matmul/cuDNN")
    if bool(args.deterministic):
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        print("[deform-batch] enabled deterministic algorithms warn_only=True")

    cfg = _load_yaml(args.config)
    manifest = _load_manifest(args.manifest)
    replays, labels, version = _select_replays_from_manifest(
        manifest,
        weights_version=args.weights_version,
        limit_samples=max(int(args.batch_size), int(args.target_index) + 1),
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

    target_index = int(args.target_index)
    if target_index < 0 or target_index >= len(replays):
        raise ValueError(f"--target-index {target_index} outside selected replay range 0..{len(replays) - 1}")

    device = _device_from_arg(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[deform-batch] config={args.config}")
    print(f"[deform-batch] manifest={args.manifest}")
    print(f"[deform-batch] weights_version={version} ckpt={ckpt_path}")
    print(f"[deform-batch] samples={len(replays)} target_index={target_index} device={device} out_dir={out_dir}")

    agent = _load_agent(cfg, device=device, ckpt_path=ckpt_path)
    selected_cases = ["real", "duplicate", "duplicate_forced_instance", "permuted"] if args.case == "all" else [str(args.case)]
    rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        target_replay = replays[target_index]
        single_trace = _trace_decoder0_p_deform(agent, [target_replay], device=device)
        single_instance_feature = single_trace["decoder0/p_deform_input_instance_feature"][0]
        for case in selected_cases:
            forced_instance_feature = None
            if case == "real":
                batch_replays = list(replays)
                compare_index = target_index
                label = labels[target_index]
            elif case == "duplicate":
                batch_replays = [target_replay for _ in range(len(replays))]
                compare_index = 0
                label = f"{labels[target_index]}:duplicated_row0"
            elif case == "duplicate_forced_instance":
                batch_replays = [target_replay for _ in range(len(replays))]
                compare_index = 0
                label = f"{labels[target_index]}:duplicated_forced_instance_row0"
                forced_instance_feature = single_instance_feature.repeat(len(batch_replays), 1, 1)
            elif case == "permuted":
                batch_replays, compare_index = _make_permuted(replays, target_index, int(args.permutation_index))
                label = f"{labels[target_index]}:permuted_row{compare_index}"
            else:
                raise AssertionError(case)
            batch_trace = _trace_decoder0_p_deform(
                agent,
                batch_replays,
                device=device,
                forced_instance_feature=forced_instance_feature,
            )
            rows.extend(
                _compare_traces(
                    case=case,
                    single_trace=single_trace,
                    batch_trace=batch_trace,
                    sample_index=compare_index,
                    label=label,
                    tolerance=float(args.tolerance),
                )
            )

    summary = _summarize(rows)
    _write_csv(out_dir / "p_deform_compare.csv", rows)
    _write_csv(out_dir / "p_deform_summary.csv", summary)
    metadata = {
        "config": str(args.config),
        "manifest": str(args.manifest),
        "weights_version": int(version),
        "ckpt": str(ckpt_path),
        "batch_size": int(args.batch_size),
        "target_index": int(target_index),
        "target_label": labels[target_index],
        "target_global_mode_idx": int(target_replay.get("global_mode_idx", -1)),
        "case": str(args.case),
        "tolerance": float(args.tolerance),
    }
    (out_dir / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[deform-batch] wrote {out_dir / 'p_deform_compare.csv'}")
    print(f"[deform-batch] wrote {out_dir / 'p_deform_summary.csv'}")
    print(f"[deform-batch] wrote {out_dir / 'summary.json'}")
    failing = [row for row in summary if int(row.get("failed", 0)) > 0]
    for row in failing[:12]:
        print(
            "[deform-batch] fail "
            f"case={row['case']} stage={row['stage']} "
            f"max_abs_error={row['max_abs_error']} mismatch_count={row['mismatch_count']}"
        )


if __name__ == "__main__":
    main()
