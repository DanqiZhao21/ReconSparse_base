from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def extract_env_timing(info: Any) -> Dict[str, float]:
    if not isinstance(info, Mapping):
        return {}
    timing = info.get("timing", None)
    if not isinstance(timing, Mapping):
        return {}

    out: Dict[str, float] = {}
    for key in [
        "render_s",
        "render_camera_total_s",
        "render_camera_avg_s",
        "render_camera_max_s",
        "camera_prepare_s",
        "camera_template_copy_s",
        "image_template_copy_s",
        "device_transfer_s",
        "camera_pose_update_s",
        "normed_time_s",
        "camera_setup_s",
        "get_sky_view_s",
        "obs_pipeline_s",
    ]:
        value = _as_float(timing.get(key, None))
        if value is not None:
            out[str(key)] = float(value)
    return out


def build_rollout_timing(
    *,
    horizon: int,
    step_records: Sequence[Mapping[str, Any]],
    collect_shard_s: float | None = None,
    next_value_feature_s: float | None = None,
    counters: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    maxes: dict[str, float] = {}

    for raw_record in step_records:
        record = dict(raw_record)
        act_s = _as_float(record.get("act_s", None))
        env_step_s = _as_float(record.get("env_step_s", None))
        if act_s is not None and env_step_s is not None:
            record.setdefault("closed_loop_step_s", float(act_s) + float(env_step_s))

        for key, value in record.items():
            metric_value = _as_float(value)
            if metric_value is None:
                continue
            metric = str(key)
            if metric.endswith("_s"):
                metric = metric[:-2]
            totals[metric] += float(metric_value)
            counts[metric] += 1
            prev_max = maxes.get(metric, None)
            if prev_max is None or float(metric_value) > float(prev_max):
                maxes[metric] = float(metric_value)

    out: Dict[str, Any] = {
        "horizon": int(horizon),
        "num_steps": int(len(step_records)),
    }
    for metric in sorted(totals.keys()):
        out[f"{metric}_total_s"] = float(totals[metric])
        out[f"{metric}_avg_s"] = float(totals[metric] / float(max(1, counts[metric])))
        out[f"{metric}_max_s"] = float(maxes[metric])

    if collect_shard_s is not None:
        out["collect_shard_s"] = float(collect_shard_s)
    if next_value_feature_s is not None:
        out["next_value_feature_s"] = float(next_value_feature_s)

    for key, value in dict(counters or {}).items():
        try:
            out[str(key)] = int(value)
        except Exception:
            out[str(key)] = value
    return out


def format_rollout_timing_summary(timing: Mapping[str, Any]) -> str:
    parts: list[str] = []
    key_order = [
        ("collect", "collect_shard_s"),
        ("step_avg", "closed_loop_step_avg_s"),
        ("act_avg", "act_avg_s"),
        ("env_avg", "env_step_avg_s"),
        ("render_avg", "render_avg_s"),
        ("backpressure", "backpressure_wait_s"),
        ("save", "save_shard_s"),
    ]
    for label, key in key_order:
        value = _as_float(timing.get(key, None))
        if value is not None:
            parts.append(f"{label}={value:.2f}s")

    reset_count = timing.get("reset_count", None)
    reset_total = _as_float(timing.get("env_reset_total_s", None))
    if reset_count is not None or reset_total is not None:
        try:
            count_text = str(int(reset_count))
        except Exception:
            count_text = str(reset_count if reset_count is not None else 0)
        total_text = f"{reset_total:.2f}s" if reset_total is not None else "n/a"
        parts.append(f"reset={count_text}/{total_text}")

    done_count = timing.get("done_count", None)
    if done_count is not None:
        try:
            parts.append(f"done={int(done_count)}")
        except Exception:
            parts.append(f"done={done_count}")

    return " ".join(parts)


__all__ = [
    "build_rollout_timing",
    "extract_env_timing",
    "format_rollout_timing_summary",
]
