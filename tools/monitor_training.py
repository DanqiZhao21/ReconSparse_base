from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LEARNER_START_RE = re.compile(
    r"\[learner\] start algo=(?P<algo>\S+) device=(?P<device>\S+) "
    r"weights_version=(?P<weights_version>\d+) max_updates=(?P<max_updates>\d+)"
)
ACTOR_SHARD_RE = re.compile(
    r"\[actor(?P<actor_id>\d+)\] wrote shard (?P<shard_idx>\d+) horizon=(?P<horizon>\d+) "
    r"ver=(?P<version>\d+) collect=(?P<collect>[0-9.]+)s step_avg=(?P<step_avg>[0-9.]+)s "
    r"act_avg=(?P<act_avg>[0-9.]+)s env_avg=(?P<env_avg>[0-9.]+)s render_avg=(?P<render_avg>[0-9.]+)s "
    r"backpressure=(?P<backpressure>[0-9.]+)s save=(?P<save>[0-9.]+)s reset=(?P<reset>\S+) done=(?P<done>\d+)"
)
UPDATE_METRICS_RE = re.compile(
    r"\[learner\] update=(?P<update>\d+) shards=(?P<shards>\d+) samples=(?P<samples>\d+) "
    r"ver=(?P<version>\d+) metrics=(?P<metrics>\{.*\})"
)
UPDATE_TIMING_RE = re.compile(
    r"\[learner\] timing update=(?P<update>\d+) collect=(?P<collect>[0-9.]+)s "
    r"load=(?P<load>[0-9.]+)s prepare=(?P<prepare>[0-9.]+)s train=(?P<train>[0-9.]+)s "
    r"save=(?P<save>[0-9.]+)s update=(?P<update_time>[0-9.]+)s time_per_shard=(?P<time_per_shard>[0-9.]+)s"
)
STEP_TIMING_RE = re.compile(r"\[learner\] step_timing update=(?P<update>\d+) parts=(?P<parts>\{.*\})")
SLOW_STEP_RE = re.compile(
    r"\[learner\] slow_step update=(?P<update>\d+) batch_idx=(?P<batch_idx>\d+) "
    r"part=(?P<part>\S+) took=(?P<seconds>[0-9.]+)s"
)
SHARD_DIR_RE = re.compile(r"dir=(?P<shards_dir>\S+)")


@dataclass
class ActorShardEvent:
    actor_id: int
    shard_idx: int
    horizon: int
    version: int
    collect_s: float
    step_avg_s: float
    act_avg_s: float
    env_avg_s: float
    render_avg_s: float
    backpressure_s: float
    save_s: float
    reset_text: str
    done: int


@dataclass
class SlowStepEvent:
    update: int
    batch_idx: int
    part: str
    seconds: float


@dataclass
class UpdateTiming:
    update: int
    collect_s: float
    load_s: float
    prepare_s: float
    train_s: float
    save_s: float
    update_time_s: float
    time_per_shard_s: float


@dataclass
class ParsedLog:
    algo: str | None = None
    device: str | None = None
    max_updates: int | None = None
    weight_version: int | None = None
    latest_update: int | None = None
    actor_shards: list[ActorShardEvent] = field(default_factory=list)
    slow_steps: list[SlowStepEvent] = field(default_factory=list)
    update_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    update_timings: dict[int, UpdateTiming] = field(default_factory=dict)
    step_timing_parts: dict[int, dict[str, float]] = field(default_factory=dict)
    shard_dir: str | None = None
    anomalies: list[str] = field(default_factory=list)

    @property
    def num_completed_updates(self) -> int:
        return len(self.update_timings)


def _safe_literal_dict(text: str) -> dict[str, float]:
    try:
        raw = ast.literal_eval(text)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def parse_log_text(text: str) -> ParsedLog:
    parsed = ParsedLog()
    for line in text.splitlines():
        if parsed.max_updates is None:
            start_match = LEARNER_START_RE.search(line)
            if start_match:
                parsed.algo = start_match.group("algo")
                parsed.device = start_match.group("device")
                parsed.weight_version = int(start_match.group("weights_version"))
                parsed.max_updates = int(start_match.group("max_updates"))
                continue

        if parsed.shard_dir is None and "stage1 collect" in line and "dir=" in line:
            shard_match = SHARD_DIR_RE.search(line)
            if shard_match:
                parsed.shard_dir = shard_match.group("shards_dir")

        actor_match = ACTOR_SHARD_RE.search(line)
        if actor_match:
            parsed.actor_shards.append(
                ActorShardEvent(
                    actor_id=int(actor_match.group("actor_id")),
                    shard_idx=int(actor_match.group("shard_idx")),
                    horizon=int(actor_match.group("horizon")),
                    version=int(actor_match.group("version")),
                    collect_s=float(actor_match.group("collect")),
                    step_avg_s=float(actor_match.group("step_avg")),
                    act_avg_s=float(actor_match.group("act_avg")),
                    env_avg_s=float(actor_match.group("env_avg")),
                    render_avg_s=float(actor_match.group("render_avg")),
                    backpressure_s=float(actor_match.group("backpressure")),
                    save_s=float(actor_match.group("save")),
                    reset_text=actor_match.group("reset"),
                    done=int(actor_match.group("done")),
                )
            )
            continue

        metrics_match = UPDATE_METRICS_RE.search(line)
        if metrics_match:
            update = int(metrics_match.group("update"))
            parsed.latest_update = update
            parsed.weight_version = int(metrics_match.group("version"))
            parsed.update_metrics[update] = _safe_literal_dict(metrics_match.group("metrics"))
            continue

        timing_match = UPDATE_TIMING_RE.search(line)
        if timing_match:
            update = int(timing_match.group("update"))
            parsed.latest_update = update
            parsed.update_timings[update] = UpdateTiming(
                update=update,
                collect_s=float(timing_match.group("collect")),
                load_s=float(timing_match.group("load")),
                prepare_s=float(timing_match.group("prepare")),
                train_s=float(timing_match.group("train")),
                save_s=float(timing_match.group("save")),
                update_time_s=float(timing_match.group("update_time")),
                time_per_shard_s=float(timing_match.group("time_per_shard")),
            )
            continue

        step_timing_match = STEP_TIMING_RE.search(line)
        if step_timing_match:
            parsed.step_timing_parts[int(step_timing_match.group("update"))] = _safe_literal_dict(
                step_timing_match.group("parts")
            )
            continue

        slow_match = SLOW_STEP_RE.search(line)
        if slow_match:
            parsed.slow_steps.append(
                SlowStepEvent(
                    update=int(slow_match.group("update")),
                    batch_idx=int(slow_match.group("batch_idx")),
                    part=slow_match.group("part"),
                    seconds=float(slow_match.group("seconds")),
                )
            )

    parsed.anomalies.extend(_detect_anomalies(parsed))
    return parsed


def _detect_anomalies(parsed: ParsedLog) -> list[str]:
    issues: list[str] = []
    for item in parsed.slow_steps:
        if item.seconds >= 5.0:
            issues.append(
                f"slow_step: update {item.update} batch {item.batch_idx} part={item.part} took {item.seconds:.2f}s"
            )
    if parsed.update_timings:
        latest = parsed.update_timings[max(parsed.update_timings)]
        if latest.update_time_s >= 180.0:
            issues.append(
                f"long_update: update {latest.update} total {latest.update_time_s:.2f}s "
                f"(collect {latest.collect_s:.2f}s, train {latest.train_s:.2f}s)"
            )
        if latest.collect_s > latest.train_s * 1.5:
            issues.append(
                f"collect_dominates: update {latest.update} collect {latest.collect_s:.2f}s vs train {latest.train_s:.2f}s"
            )
    elif parsed.actor_shards:
        collect_values = [item.collect_s for item in parsed.actor_shards]
        if statistics.mean(collect_values) >= 60.0:
            issues.append(f"slow_actor_collect: mean shard collect {statistics.mean(collect_values):.2f}s before first update")
    return issues


def find_latest_train_log(search_root: Path) -> Path:
    candidates = list(search_root.rglob("train.log"))
    if not candidates:
        raise FileNotFoundError(f"No train.log found under {search_root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def infer_buffer_dir(log_path: Path, parsed: ParsedLog) -> Path | None:
    if parsed.shard_dir:
        shard_path = Path(parsed.shard_dir)
        if not shard_path.is_absolute():
            shard_path = (Path.cwd() / shard_path).resolve()
        return shard_path.parent.parent
    return None


def _series_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "min": round(min(values), 2),
        "mean": round(statistics.mean(values), 2),
        "max": round(max(values), 2),
        "p95": round(ordered[p95_index], 2),
    }


def build_snapshot(*, log_path: Path, buffer_dir: Path | None = None) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log_text(text)
    resolved_buffer_dir = buffer_dir or infer_buffer_dir(log_path, parsed)

    pending_shards = 0
    consumed_shards = 0
    training_lock_present = False
    weights_version = parsed.weight_version
    if resolved_buffer_dir is not None:
        pending_shards = len(list((resolved_buffer_dir / "buffer" / "shards").glob("*.pt")))
        consumed_shards = len(list((resolved_buffer_dir / "buffer" / "consumed").glob("*.pt")))
        training_lock_present = (resolved_buffer_dir / "TRAINING_LOCK").exists()
        version_file = resolved_buffer_dir / "weights" / "version.txt"
        if version_file.exists():
            try:
                weights_version = int(version_file.read_text(encoding="utf-8").strip())
            except Exception:
                pass

    actor_collect = [item.collect_s for item in parsed.actor_shards]
    actor_step = [item.step_avg_s for item in parsed.actor_shards]
    actor_env = [item.env_avg_s for item in parsed.actor_shards]
    actor_render = [item.render_avg_s for item in parsed.actor_shards]
    actor_backpressure = [item.backpressure_s for item in parsed.actor_shards]
    actor_save = [item.save_s for item in parsed.actor_shards]

    latest_update_idx = max(parsed.update_timings) if parsed.update_timings else None
    latest_update_timing = None
    latest_update_metrics: dict[str, float] | None = None
    latest_step_timing: dict[str, float] | None = None
    if latest_update_idx is not None:
        latest_update_timing = parsed.update_timings[latest_update_idx]
        latest_update_metrics = parsed.update_metrics.get(latest_update_idx, {})
        latest_step_timing = parsed.step_timing_parts.get(latest_update_idx, {})

    max_updates = parsed.max_updates or 0
    completed_updates = parsed.num_completed_updates
    progress_pct = round((completed_updates / max_updates) * 100.0, 2) if max_updates else 0.0

    return {
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "log_path": str(log_path),
        "run_name": log_path.parent.name,
        "algo": parsed.algo,
        "device": parsed.device,
        "progress": {
            "completed_updates": completed_updates,
            "latest_update": latest_update_idx,
            "max_updates": parsed.max_updates,
            "progress_pct": progress_pct,
            "weights_version": weights_version,
        },
        "buffer": {
            "buffer_dir": str(resolved_buffer_dir) if resolved_buffer_dir is not None else None,
            "pending_shards": pending_shards,
            "consumed_shards": consumed_shards,
            "training_lock_present": training_lock_present,
        },
        "latest_update": {
            "timing": vars(latest_update_timing) if latest_update_timing is not None else None,
            "metrics": latest_update_metrics,
            "step_timing_parts": latest_step_timing,
        },
        "actor_stats": {
            "shards_logged": len(parsed.actor_shards),
            "collect_s": _series_stats(actor_collect),
            "step_avg_s": _series_stats(actor_step),
            "env_avg_s": _series_stats(actor_env),
            "render_avg_s": _series_stats(actor_render),
            "backpressure_s": _series_stats(actor_backpressure),
            "save_s": _series_stats(actor_save),
        },
        "slow_steps": [
            {"update": item.update, "batch_idx": item.batch_idx, "part": item.part, "seconds": item.seconds}
            for item in parsed.slow_steps[-20:]
        ],
        "anomalies": parsed.anomalies,
        "curves": {
            "update_time_s": [
                {"update": idx, "value": parsed.update_timings[idx].update_time_s}
                for idx in sorted(parsed.update_timings)
            ],
            "collect_time_s": [
                {"update": idx, "value": parsed.update_timings[idx].collect_s}
                for idx in sorted(parsed.update_timings)
            ],
            "train_time_s": [
                {"update": idx, "value": parsed.update_timings[idx].train_s}
                for idx in sorted(parsed.update_timings)
            ],
            "actor_collect_s": [
                {"point": idx, "value": item.collect_s}
                for idx, item in enumerate(parsed.actor_shards)
            ],
            "slow_step_s": [
                {"point": idx, "value": item.seconds}
                for idx, item in enumerate(parsed.slow_steps)
            ],
        },
    }


def _format_stats_block(name: str, stats: dict[str, float] | None) -> str:
    if not stats:
        return f"- {name}: n/a"
    return (
        f"- {name}: mean {stats['mean']:.2f}s, p95 {stats['p95']:.2f}s, "
        f"min {stats['min']:.2f}s, max {stats['max']:.2f}s"
    )


def render_markdown(snapshot: dict[str, Any]) -> str:
    progress = snapshot["progress"]
    latest_update = snapshot["latest_update"]
    buffer_state = snapshot["buffer"]
    lines = [
        f"# Training Monitor: {snapshot['run_name']}",
        "",
        f"- Generated: {snapshot['generated_at_utc']}",
        f"- Log: `{snapshot['log_path']}`",
        f"- Algo/Device: `{snapshot.get('algo')}` on `{snapshot.get('device')}`",
        (
            f"- Progress: {progress['completed_updates']}/{progress['max_updates']} updates "
            f"({progress['progress_pct']:.2f}%), weights version {progress['weights_version']}"
        ),
        (
            f"- Buffer: pending shards {buffer_state['pending_shards']}, consumed shards {buffer_state['consumed_shards']}, "
            f"training lock {'present' if buffer_state['training_lock_present'] else 'absent'}"
        ),
        "",
        "## Current Status",
    ]
    if latest_update["timing"] is None:
        lines.append("- No completed learner update yet; run is still collecting or training the first update.")
    else:
        timing = latest_update["timing"]
        lines.extend(
            [
                (
                    f"- Latest completed update {timing['update']}: update {timing['update_time_s']:.2f}s, "
                    f"collect {timing['collect_s']:.2f}s, train {timing['train_s']:.2f}s, "
                    f"load {timing['load_s']:.2f}s, prepare {timing['prepare_s']:.2f}s, save {timing['save_s']:.2f}s"
                ),
                f"- Latest metrics: `{json.dumps(latest_update['metrics'] or {}, ensure_ascii=True, sort_keys=True)}`",
                f"- Latest step timing: `{json.dumps(latest_update['step_timing_parts'] or {}, ensure_ascii=True, sort_keys=True)}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Actor Timing",
            f"- Logged actor shards: {snapshot['actor_stats']['shards_logged']}",
            _format_stats_block("collect", snapshot["actor_stats"]["collect_s"]),
            _format_stats_block("step_avg", snapshot["actor_stats"]["step_avg_s"]),
            _format_stats_block("env_avg", snapshot["actor_stats"]["env_avg_s"]),
            _format_stats_block("render_avg", snapshot["actor_stats"]["render_avg_s"]),
            _format_stats_block("save", snapshot["actor_stats"]["save_s"]),
            _format_stats_block("backpressure", snapshot["actor_stats"]["backpressure_s"]),
            "",
            "## Anomalies",
        ]
    )
    anomalies = snapshot["anomalies"]
    if anomalies:
        lines.extend(f"- {item}" for item in anomalies)
    else:
        lines.append("- No anomaly detected by the current heuristics.")

    if snapshot["slow_steps"]:
        lines.extend(["", "## Recent Slow Steps"])
        for item in snapshot["slow_steps"][-10:]:
            lines.append(
                f"- update {item['update']} batch {item['batch_idx']} part={item['part']} took {item['seconds']:.2f}s"
            )
    return "\n".join(lines) + "\n"


def _svg_path(points: list[tuple[float, float]], width: int, height: int, margin: int) -> str:
    if not points:
        return ""
    x_values = [x for x, _ in points]
    y_values = [y for _, y in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0

    coords: list[str] = []
    for x, y in points:
        sx = margin + ((x - x_min) / (x_max - x_min)) * (width - 2 * margin)
        sy = height - margin - ((y - y_min) / (y_max - y_min)) * (height - 2 * margin)
        coords.append(f"{sx:.1f},{sy:.1f}")
    return " ".join(coords)


def _render_panel(title: str, points: list[tuple[float, float]], width: int, height: int, y_label: str) -> str:
    polyline = _svg_path(points, width, height, margin=36)
    if not points:
        return (
            f"<g><rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='#d0d7de'/>"
            f"<text x='16' y='24' font-size='16' font-family='monospace'>{title}</text>"
            f"<text x='16' y='52' font-size='13' font-family='monospace' fill='#57606a'>no data yet</text></g>"
        )
    y_values = [value for _, value in points]
    return (
        f"<g>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='#d0d7de'/>"
        f"<text x='16' y='24' font-size='16' font-family='monospace'>{title}</text>"
        f"<text x='16' y='44' font-size='12' font-family='monospace' fill='#57606a'>{y_label}</text>"
        f"<text x='{width - 160}' y='24' font-size='12' font-family='monospace' fill='#57606a'>"
        f"min {min(y_values):.2f} | max {max(y_values):.2f}"
        f"</text>"
        f"<line x1='36' y1='{height - 36}' x2='{width - 12}' y2='{height - 36}' stroke='#d0d7de'/>"
        f"<line x1='36' y1='12' x2='36' y2='{height - 36}' stroke='#d0d7de'/>"
        f"<polyline fill='none' stroke='#0969da' stroke-width='2' points='{polyline}'/>"
        f"</g>"
    )


def render_svg(snapshot: dict[str, Any]) -> str:
    width = 900
    panel_height = 220
    spacing = 24
    panels = [
        (
            "Update Time",
            [(float(item["update"]), float(item["value"])) for item in snapshot["curves"]["update_time_s"]],
            "seconds per completed update",
        ),
        (
            "Actor Collect Time",
            [(float(item["point"]), float(item["value"])) for item in snapshot["curves"]["actor_collect_s"]],
            "seconds per shard",
        ),
        (
            "Slow Step Duration",
            [(float(item["point"]), float(item["value"])) for item in snapshot["curves"]["slow_step_s"]],
            "seconds per flagged slow step",
        ),
    ]
    total_height = spacing + len(panels) * (panel_height + spacing)
    chunks = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{total_height}' viewBox='0 0 {width} {total_height}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='#f6f8fa'/>",
    ]
    y_offset = spacing
    for title, points, y_label in panels:
        chunks.append(f"<g transform='translate(0,{y_offset})'>{_render_panel(title, points, width, panel_height, y_label)}</g>")
        y_offset += panel_height + spacing
    chunks.append("</svg>")
    return "".join(chunks)


def write_report(snapshot: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=True), encoding="utf-8")
    (output_dir / "latest_summary.md").write_text(render_markdown(snapshot), encoding="utf-8")
    (output_dir / "curves.svg").write_text(render_svg(snapshot), encoding="utf-8")

    history_path = output_dir / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor the latest ReconDreamer training run.")
    parser.add_argument("--search-root", default="outputs/ops_logs", help="Where to search for train.log files.")
    parser.add_argument("--log-path", default=None, help="Use an explicit train.log path instead of auto-discovery.")
    parser.add_argument("--buffer-dir", default=None, help="Override actor-learner buffer root.")
    parser.add_argument("--output-dir", default=None, help="Directory for monitor artifacts.")
    parser.add_argument("--interval-s", type=float, default=0.0, help="Refresh interval in seconds. Zero means one-shot.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of refreshes to run when interval-s > 0.")
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    log_path = Path(args.log_path).resolve() if args.log_path else find_latest_train_log(Path(args.search_root).resolve())
    buffer_dir = Path(args.buffer_dir).resolve() if args.buffer_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (log_path.parent / "monitor")
    return log_path, buffer_dir, output_dir


def main() -> None:
    args = parse_args()
    log_path, buffer_dir, output_dir = _resolve_paths(args)
    iterations = max(1, int(args.iterations))

    for index in range(iterations):
        snapshot = build_snapshot(log_path=log_path, buffer_dir=buffer_dir)
        write_report(snapshot, output_dir)
        print(
            json.dumps(
                {
                    "generated_at_utc": snapshot["generated_at_utc"],
                    "run_name": snapshot["run_name"],
                    "completed_updates": snapshot["progress"]["completed_updates"],
                    "max_updates": snapshot["progress"]["max_updates"],
                    "weights_version": snapshot["progress"]["weights_version"],
                    "anomalies": snapshot["anomalies"][:3],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        if float(args.interval_s) <= 0.0 or index >= iterations - 1:
            break
        time.sleep(float(args.interval_s))


if __name__ == "__main__":
    main()
