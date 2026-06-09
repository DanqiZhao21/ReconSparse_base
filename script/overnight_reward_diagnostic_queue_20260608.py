from __future__ import annotations

import ast
import json
import os
import re
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
PYTHON = Path("/root/miniconda3/envs/recondreamerNew-rl/bin/python")
PIPELINE = REPO_ROOT / "script" / "run_train_eval_pipeline_hugsim_ori.sh"
BASE_CONFIG = (
    REPO_ROOT
    / "script/configs/sparsedrive_v2/"
    / "202606081145_HUGSM_reinforcepp_closed_loop_steppath_hd_collision_only_extreme_NoGRPOCraft_substeps1_epoch2_update64_oneScene.yaml"
)
OUT_ROOT = REPO_ROOT / "outputs" / "reward_diagnostics" / "20260608_overnight"
CONFIG_ROOT = REPO_ROOT / "script" / "configs" / "sparsedrive_v2" / "generated_20260608_reward_diag"
QUEUE_LOG_ROOT = REPO_ROOT / "outputs" / "train_eval_queues" / "20260608_reward_diag"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _deepcopy_yaml(payload: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(payload, sort_keys=False))


def _processes_in_session(session_id: int) -> list[int]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,sid"], text=True)
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            sid = int(parts[1])
        except ValueError:
            continue
        if sid == int(session_id) and pid != os.getpid():
            pids.append(pid)
    return pids


def _cleanup_session(session_id: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _processes_in_session(session_id)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if pids:
            time.sleep(10 if sig == signal.SIGTERM else 2)


def _run_command_until_success(cmd: list[str], *, cwd: Path, log_prefix: str, retry: int = 0) -> Path:
    QUEUE_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        log_path = QUEUE_LOG_ROOT / f"{log_prefix}_attempt{attempt}_{_stamp()}.log"
        print(f"[{_now()}] start prefix={log_prefix} attempt={attempt} log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"[cwd] {cwd}\n")
            handle.write(f"[cmd] {' '.join(cmd)}\n")
            handle.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            try:
                code = proc.wait()
            except BaseException:
                _cleanup_session(proc.pid)
                raise
        _cleanup_session(proc.pid)
        if code == 0:
            print(f"[{_now()}] success prefix={log_prefix} attempt={attempt} log={log_path}", flush=True)
            return log_path
        print(f"[{_now()}] failed prefix={log_prefix} attempt={attempt} code={code} log={log_path}", flush=True)
        if attempt > int(retry):
            raise RuntimeError(f"Command failed after {attempt} attempt(s): {' '.join(cmd)}; see {log_path}")
        attempt += 1
        time.sleep(60)


def _set_common_train_controls(
    cfg: dict[str, Any],
    *,
    tag: str,
    buffer_dir: Path,
    max_updates: int,
    actors_per_gpu: int,
    shards_per_update: int,
    samples_per_update: int,
    minibatch_size: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    train = cfg.setdefault("train", {})
    env = cfg.setdefault("env", {})
    reward = env.setdefault("reward", {})
    al = train.setdefault("actor_learner", {})
    wandb = train.setdefault("wandb", {})

    wandb["enabled"] = False
    wandb["group"] = tag
    wandb["log_minibatch_metrics"] = True
    wandb["log_legacy_raw_metrics"] = True

    al["buffer_dir"] = str(buffer_dir)
    al["timestamp_buffer_dir"] = False
    al["actors_per_gpu"] = int(actors_per_gpu)
    al["actor_gpu_pool"] = [4, 5, 6, 7]
    al["learner_gpu_ids"] = [0, 1, 2, 3]
    al["learner_gpu_id"] = 0
    al["auto_max_inflight_per_actor"] = False
    al["max_inflight_per_actor"] = 1
    al["shards_per_update"] = int(shards_per_update)
    al["samples_per_update"] = int(samples_per_update)
    al["max_updates"] = int(max_updates)
    al["actor_horizon"] = 32
    al["num_envs_per_actor"] = 1
    al["vec_env_mode"] = "serial"
    al["allow_partial_updates_after_timeout"] = True
    al["shard_collect_timeout_s"] = 240
    al["actor_shard_stall_timeout_s"] = 900
    al["restart_failed_actors"] = True
    al["max_actor_restarts"] = 2

    train["minibatch_size"] = int(minibatch_size)
    train.setdefault("ddp", {})["grad_accum_steps"] = int(grad_accum_steps)
    train["policy_lr"] = float(train.get("policy_lr", 3.0e-6))
    train["max_grad_norm"] = float(train.get("max_grad_norm", 0.5))
    reward.setdefault("logging", {}).setdefault("wandb", {})["enable"] = False
    for mode_key in ("step_path", "craft_close_loop", "craft_sparse_loop"):
        mode_cfg = reward.get(mode_key, None)
        if isinstance(mode_cfg, dict):
            mode_cfg.setdefault("logging", {}).setdefault("wandb", {})["enable"] = False
    return cfg


def _set_step_path_collision(
    cfg: dict[str, Any],
    *,
    static_weight: float,
    dynamic_weight: float,
    terminal_penalty: float = 0.0,
    progress_weight: float = 0.0,
    safety_enabled: bool = False,
) -> None:
    reward = cfg["env"]["reward"]
    reward["mode"] = "step_path"
    step_path = reward.setdefault("step_path", {})
    step_path.setdefault("CRAFT", {})["enable"] = False
    path_cfg = step_path.setdefault("path", {})
    path_cfg["w_progress"] = float(progress_weight)
    path_cfg.setdefault("progress_forward_cap_m", 2.0)
    path_cfg.setdefault("progress_backward_cap_m", 0.5)
    path_cfg.setdefault("w_lateral", 0.0)
    path_cfg.setdefault("w_yaw", 0.0)
    collision = step_path.setdefault("collision", {})
    collision["mode"] = "dense_penalty"
    collision["w_static"] = float(static_weight)
    collision["w_dynamic"] = float(dynamic_weight)
    safety = step_path.setdefault("safety", {})
    safety["enable"] = bool(safety_enabled)
    if bool(safety_enabled):
        safety["lookahead_m"] = 15.0
        safety["corridor_half_width_m"] = 2.5
        safety["safe_gap_m"] = 8.0
        safety["safe_ttc_s"] = 3.0
        safety["w_clearance"] = 2.0
        safety["w_ttc"] = 2.0
        safety["progress_gate_strength"] = 1.0
        safety["min_progress_gate"] = 0.2
    terminal = step_path.setdefault("terminal", {})
    terminal["enable"] = True
    terminal["penalty"] = float(terminal_penalty)
    terminal["apply_on_failure"] = True
    terminal["apply_on_timeout"] = False
    terminal["apply_on_env_done"] = False


def _set_craft_closed_loop(cfg: dict[str, Any], *, term_collision: float, grpo: bool, aux: bool) -> None:
    reward = cfg["env"]["reward"]
    reward["mode"] = "craft_close_loop"
    craft_mode = reward.setdefault("craft_close_loop", {})
    craft = craft_mode.setdefault("CRAFT", {})
    craft["enable"] = True
    craft["real_reward_model"] = "close loop"
    craft["term_collision"] = float(term_collision)
    craft["progress_weight"] = 10.0
    craft["progress_max_m"] = 1.2
    craft["progress_min_m"] = 0.0
    craft["w_g"] = 2.0
    craft["w_c"] = 0.0
    craft["w_h"] = 1.5
    craft["efficiency_floor"] = 0.15
    craft["cost_off_global_route"] = 3.0
    craft["cost_off_road"] = 4.0
    craft["cost_opposite_lane"] = 0.5
    craft["collision_cost_static"] = None
    craft["collision_cost_dynamic"] = None
    craft_mode.setdefault("terminal", {})["penalty"] = 0.0

    train = cfg.setdefault("train", {})
    train["algo"] = "reinforcepp"
    train["minibatch_size"] = 4 if grpo else 32
    train.setdefault("ddp", {})["grad_accum_steps"] = 8 if grpo else 1
    train["grpo"] = {
        "enable": bool(grpo),
        "config_path": None,
        "coef": 2.0 if grpo else 0.0,
        "num_candidates": 8,
        "candidate_select": "topk",
        "objective": "expected_prob",
        "temperature": 1.0,
        "norm_eps": 1.0e-6,
        "use_rank_adv": False,
        "score_clip": None,
        "debug_visualize": False,
        "debug_dir": f"outputs/visualize/{train.get('wandb', {}).get('group', 'reward_diag')}",
        "debug_max_batches": 24,
        "debug_top_k": 8,
    }
    train["reinforcepp"] = {
        "norm_eps": 1.0e-8,
        "kl_coef": 0.0,
        "epochs": 1 if grpo else 2,
        "policy_grad_weight": 0.5 if grpo else 1.0,
        "forward_kl_coef": 0.0,
        "reverse_kl_coef": 0.5 if grpo else 0.0,
        "distill_temperature": 1.0,
    }
    if aux:
        train["auxiliary_objectives"] = {
            "risk_decel": {
                "enable": True,
                "coef": 0.5,
                "dt_s": 0.5,
                "high_risk_gap_m": 10.0,
                "high_risk_ttc_s": 2.5,
                "lateral_m": 2.5,
                "speed_margin_mps": 0.15,
                "eps": 1.0e-6,
            }
        }
    else:
        train.pop("auxiliary_objectives", None)

    scorer = cfg.setdefault("agent", {}).setdefault("nuscenes_scorer", {})
    scorer["backend"] = "craft_carl"
    carl = scorer.setdefault("carl", {})
    carl["term_collision"] = float(term_collision)
    carl["w_prog"] = 10.0
    carl["w_g"] = 2.0
    carl["w_c"] = 0.0
    carl["w_h"] = 1.5


def _set_sac_dense(cfg: dict[str, Any]) -> None:
    train = cfg.setdefault("train", {})
    train["algo"] = "sac"
    train["policy_lr"] = 5.0e-7
    train["minibatch_size"] = 4
    train.setdefault("ddp", {})["grad_accum_steps"] = 8
    train["grpo"] = {
        "enable": False,
        "config_path": None,
        "coef": 0.0,
        "num_candidates": 8,
        "candidate_select": "topk",
        "objective": "expected_prob",
        "temperature": 1.0,
        "norm_eps": 1.0e-6,
        "use_rank_adv": False,
        "score_clip": None,
        "debug_visualize": False,
        "debug_dir": "outputs/visualize/reward_diag_sac",
        "debug_max_batches": 0,
        "debug_top_k": 8,
    }
    train["sac"] = {
        "entropy_coef": 0.01,
        "kl_coef": 0.02,
        "epochs": 1,
        "norm_eps": 1.0e-8,
        "policy_grad_weight": 0.8,
        "forward_kl_coef": 0.0,
        "reverse_kl_coef": 0.0,
        "distill_temperature": 1.0,
    }


def _diagnostic_config(base: dict[str, Any], *, tag: str, collision_weight: float) -> Path:
    cfg = _deepcopy_yaml(base)
    buffer_dir = OUT_ROOT / "actor_learner" / tag
    shutil.rmtree(buffer_dir, ignore_errors=True)
    _set_common_train_controls(
        cfg,
        tag=tag,
        buffer_dir=buffer_dir,
        max_updates=2,
        actors_per_gpu=2,
        shards_per_update=8,
        samples_per_update=256,
        minibatch_size=32,
        grad_accum_steps=1,
    )
    _set_step_path_collision(
        cfg,
        static_weight=collision_weight,
        dynamic_weight=collision_weight,
        terminal_penalty=0.0,
        progress_weight=0.0,
        safety_enabled=False,
    )
    cfg["train"]["grpo"] = {**cfg["train"].get("grpo", {}), "enable": False, "coef": 0.0}
    cfg["train"]["reinforcepp"] = {
        "norm_eps": 1.0e-8,
        "kl_coef": 0.0,
        "epochs": 1,
        "policy_grad_weight": 1.0,
        "forward_kl_coef": 0.0,
        "reverse_kl_coef": 0.0,
        "distill_temperature": 1.0,
    }
    return _write_yaml(CONFIG_ROOT / f"{tag}.yaml", cfg)


def _run_diagnostic(config_path: Path, *, tag: str) -> Path:
    cmd = [
        str(PYTHON),
        "-u",
        str(REPO_ROOT / "script" / "train_actor_learner_v2.py"),
        "--role",
        "orchestrator",
        "--config",
        str(config_path),
    ]
    return _run_command_until_success(cmd, cwd=REPO_ROOT, log_prefix=tag, retry=0)


def _parse_metrics_from_log(log_path: Path) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    reward_summaries: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if " metrics=" in line and "[learner] update=" in line:
            match = re.search(r"metrics=(\{.*\})", line)
            if match:
                try:
                    updates.append(ast.literal_eval(match.group(1)))
                except Exception:
                    pass
        if " reward_summary " in line and "summary=" in line:
            match = re.search(r"summary=(\{.*\})", line)
            if match:
                try:
                    reward_summaries.append(ast.literal_eval(match.group(1)))
                except Exception:
                    pass
    return {
        "updates": updates,
        "last_update_metrics": updates[-1] if updates else {},
        "reward_summaries": reward_summaries,
        "last_reward_summary": reward_summaries[-1] if reward_summaries else {},
    }


def _analyze_buffer(config_path: Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    buffer_dir = Path(cfg["train"]["actor_learner"]["buffer_dir"])
    shard_paths = sorted((buffer_dir / "buffer" / "consumed").glob("*.pt"))
    if not shard_paths:
        shard_paths = sorted((buffer_dir / "buffer" / "shards").glob("*.pt"))

    reward_values: list[float] = []
    done_values: list[float] = []
    reward_summary: dict[str, float] = {}
    for path in shard_paths:
        try:
            shard = torch.load(path, map_location="cpu")
        except Exception:
            continue
        reward = shard.get("reward", None)
        if torch.is_tensor(reward):
            reward_values.extend(float(x) for x in reward.detach().view(-1).tolist())
        done = shard.get("done", None)
        if torch.is_tensor(done):
            done_values.extend(float(x) for x in done.detach().view(-1).tolist())
        meta = shard.get("meta", {}) or {}
        rs = meta.get("reward_summary", {}) or {}
        if isinstance(rs, dict):
            for key, value in rs.items():
                try:
                    reward_summary[str(key)] = reward_summary.get(str(key), 0.0) + float(value)
                except Exception:
                    pass

    n = len(reward_values)
    if n == 0:
        return {
            "buffer_dir": str(buffer_dir),
            "shards": len(shard_paths),
            "samples": 0,
            "error": "no rewards found",
        }
    reward_t = torch.as_tensor(reward_values, dtype=torch.float32)
    done_t = torch.as_tensor(done_values, dtype=torch.float32) if done_values else torch.empty((0,), dtype=torch.float32)
    steps = max(1.0, float(reward_summary.get("step_count", n)))
    return {
        "buffer_dir": str(buffer_dir),
        "shards": len(shard_paths),
        "samples": int(n),
        "reward_mean": float(reward_t.mean().item()),
        "reward_std": float(reward_t.std(unbiased=False).item()),
        "reward_min": float(reward_t.min().item()),
        "reward_max": float(reward_t.max().item()),
        "done_rate": float(done_t.mean().item()) if int(done_t.numel()) else 0.0,
        "cost_reward_mean": float(reward_summary.get("cost_reward_sum", 0.0)) / steps,
        "positive_reward_mean": float(reward_summary.get("positive_reward_sum", 0.0)) / steps,
        "collision_gate_rate": float(reward_summary.get("collision_gate_count", 0.0)) / steps,
        "terminal_failure_count": float(reward_summary.get("terminal_failure_count", 0.0)),
        "terminal_timeout_count": float(reward_summary.get("terminal_timeout_count", 0.0)),
        "terminal_env_done_count": float(reward_summary.get("terminal_env_done_count", 0.0)),
        "reward_summary": reward_summary,
    }


def _make_pipeline_config(
    base: dict[str, Any],
    *,
    name: str,
    tag: str,
    max_updates: int,
    variant: str,
) -> Path:
    cfg = _deepcopy_yaml(base)
    buffer_dir = OUT_ROOT / "actor_learner" / tag
    shutil.rmtree(buffer_dir, ignore_errors=True)
    _set_common_train_controls(
        cfg,
        tag=tag,
        buffer_dir=buffer_dir,
        max_updates=max_updates,
        actors_per_gpu=4,
        shards_per_update=32,
        samples_per_update=512,
        minibatch_size=32,
        grad_accum_steps=1,
    )
    cfg["train"]["wandb"]["enabled"] = True
    cfg["train"]["wandb"]["group"] = f"20260608_reward_diag_{tag}"
    cfg["train"]["wandb"]["log_minibatch_metrics"] = True
    cfg["train"]["wandb"]["log_legacy_raw_metrics"] = True

    if variant == "step_path_dense":
        _set_step_path_collision(
            cfg,
            static_weight=500.0,
            dynamic_weight=500.0,
            terminal_penalty=-100.0,
            progress_weight=2.0,
            safety_enabled=True,
        )
        cfg["train"]["algo"] = "reinforcepp"
        cfg["train"]["reinforcepp"] = {
            "norm_eps": 1.0e-8,
            "kl_coef": 0.0,
            "epochs": 2,
            "policy_grad_weight": 1.0,
            "forward_kl_coef": 0.0,
            "reverse_kl_coef": 0.0,
            "distill_temperature": 1.0,
        }
        cfg["train"]["grpo"] = {**cfg["train"].get("grpo", {}), "enable": False, "coef": 0.0}
    elif variant == "craft_grpo":
        _set_craft_closed_loop(cfg, term_collision=100.0, grpo=True, aux=False)
    elif variant == "craft_grpo_aux":
        _set_craft_closed_loop(cfg, term_collision=100.0, grpo=True, aux=True)
    elif variant == "sac_dense":
        _set_step_path_collision(
            cfg,
            static_weight=500.0,
            dynamic_weight=500.0,
            terminal_penalty=-100.0,
            progress_weight=1.0,
            safety_enabled=True,
        )
        _set_sac_dense(cfg)
    else:
        raise ValueError(f"unknown variant: {variant}")

    return _write_yaml(CONFIG_ROOT / name, cfg)


def _generate_experiments(base: dict[str, Any], diagnostics: dict[str, Any]) -> list[tuple[Path, str]]:
    low = diagnostics.get("diag_collision100", {}).get("buffer", {})
    high = diagnostics.get("diag_collision1000", {}).get("buffer", {})
    low_std = float(low.get("reward_std", 0.0) or 0.0)
    high_std = float(high.get("reward_std", 0.0) or 0.0)
    high_cost = float(high.get("cost_reward_mean", 0.0) or 0.0)
    low_cost = float(low.get("cost_reward_mean", 0.0) or 0.0)
    cost_changed = abs(high_cost - low_cost) > max(1.0, 2.0 * abs(low_cost))
    sparse_collision_signal = high_std <= max(1.0, low_std * 1.5) and not cost_changed

    max_updates = 16
    experiments: list[tuple[Path, str]] = []
    experiments.append(
        (
            _make_pipeline_config(
                base,
                name="20260608_rewarddiag_step_path_dense_progress_safety_reinforcepp.yaml",
                tag="rewarddiag_step_dense_rpp",
                max_updates=max_updates,
                variant="step_path_dense",
            ),
            "rewarddiag_step_dense_rpp",
        )
    )
    experiments.append(
        (
            _make_pipeline_config(
                base,
                name="20260608_rewarddiag_craft_close_loop_grpo.yaml",
                tag="rewarddiag_craft_grpo",
                max_updates=max_updates,
                variant="craft_grpo",
            ),
            "rewarddiag_craft_grpo",
        )
    )
    if sparse_collision_signal:
        experiments.append(
            (
                _make_pipeline_config(
                    base,
                    name="20260608_rewarddiag_craft_close_loop_grpo_auxrisk.yaml",
                    tag="rewarddiag_craft_grpo_aux",
                    max_updates=max_updates,
                    variant="craft_grpo_aux",
                ),
                "rewarddiag_craft_grpo_aux",
            )
        )
    else:
        experiments.append(
            (
                _make_pipeline_config(
                    base,
                    name="20260608_rewarddiag_sac_dense_progress_safety.yaml",
                    tag="rewarddiag_sac_dense",
                    max_updates=max_updates,
                    variant="sac_dense",
                ),
                "rewarddiag_sac_dense",
            )
        )
    return experiments


def _run_pipeline(config_path: Path, *, tag: str) -> Path:
    cmd = [
        str(PIPELINE),
        "--reinforcepp-config",
        str(config_path),
        "--reinforcepp-algo-tag",
        tag,
        "--max-scenes",
        "24",
        "--repeat-evals",
        "1",
        "--retry-count",
        "1",
    ]
    return _run_command_until_success(cmd, cwd=REPO_ROOT, log_prefix=f"pipeline_{tag}", retry=1)


def _find_eval_summaries(tag: str) -> list[str]:
    root = Path("/root/clone/HUGSIM-ORI/outputs/evaluate-auto")
    if not root.exists():
        return []
    return [
        str(path)
        for path in sorted(root.glob(f"*{tag}*/*/summary/summary.json"), key=lambda p: p.stat().st_mtime)
    ]


def _load_eval_summary(path: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_report(data: dict[str, Any]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "report.json").write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# 20260608 reward diagnostic overnight report",
        "",
        f"updated_at: {_now()}",
        "",
        "## Diagnostics",
    ]
    for key, item in data.get("diagnostics", {}).items():
        b = item.get("buffer", {})
        m = item.get("log", {}).get("last_update_metrics", {})
        lines.append(
            f"- {key}: samples={b.get('samples')} reward_mean={b.get('reward_mean')} "
            f"reward_std={b.get('reward_std')} cost_mean={b.get('cost_reward_mean')} "
            f"done_rate={b.get('done_rate')} loss_pi={m.get('loss_pi')} "
            f"clip_frac={m.get('clip_frac')} approx_kl={m.get('approx_kl')}"
        )
    lines.append("")
    lines.append("## Experiments")
    for exp in data.get("experiments", []):
        lines.append(
            f"- {exp.get('tag')}: config={exp.get('config')} log={exp.get('log')} "
            f"eval_summaries={exp.get('eval_summaries')}"
        )
    lines.append("")
    lines.append("## Conclusion")
    lines.append(data.get("conclusion", "Pipeline still running or no conclusion yet."))
    (OUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    base = _load_yaml(BASE_CONFIG)
    report: dict[str, Any] = {
        "started_at": _now(),
        "base_config": str(BASE_CONFIG),
        "diagnostics": {},
        "experiments": [],
    }
    _write_report(report)

    diag_specs = [
        ("diag_collision100", 100.0),
        ("diag_collision1000", 1000.0),
    ]
    for tag, weight in diag_specs:
        cfg_path = _diagnostic_config(base, tag=tag, collision_weight=weight)
        log_path = _run_diagnostic(cfg_path, tag=tag)
        report["diagnostics"][tag] = {
            "config": str(cfg_path),
            "log_path": str(log_path),
            "log": _parse_metrics_from_log(log_path),
            "buffer": _analyze_buffer(cfg_path),
        }
        _write_report(report)

    experiments = _generate_experiments(base, report["diagnostics"])
    report["generated_experiments"] = [{"config": str(path), "tag": tag} for path, tag in experiments]
    _write_report(report)

    for config_path, tag in experiments:
        exp: dict[str, Any] = {"config": str(config_path), "tag": tag, "started_at": _now()}
        report["experiments"].append(exp)
        _write_report(report)
        log_path = _run_pipeline(config_path, tag=tag)
        summaries = _find_eval_summaries(tag)
        exp.update(
            {
                "finished_at": _now(),
                "log": str(log_path),
                "eval_summaries": summaries,
                "last_eval_summary": _load_eval_summary(summaries[-1]) if summaries else {},
            }
        )
        _write_report(report)

    report["finished_at"] = _now()
    high = report["diagnostics"].get("diag_collision1000", {}).get("buffer", {})
    if float(high.get("cost_reward_mean", 0.0) or 0.0) == 0.0:
        conclusion = (
            "collision-only reward did not produce cost signal in diagnostic shards; "
            "prefer dense progress/safety or CRAFT/GRPO variants over further collision-weight sweeps."
        )
    else:
        conclusion = (
            "collision reward is present in diagnostic shards; compare eval summaries to decide whether "
            "dense step_path, CRAFT+GRPO, or the auxiliary/SAC branch improves closed-loop metrics."
        )
    report["conclusion"] = conclusion
    _write_report(report)
    print(f"[{_now()}] complete report={OUT_ROOT / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
