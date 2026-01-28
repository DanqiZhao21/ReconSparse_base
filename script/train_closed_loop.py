import os
import sys
import time
import random
import functools
from typing import Any, Dict, Optional
from contextlib import nullcontext

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import imageio
# Simple stage logger
def stage(msg: str) -> None:
    # DDP-friendly logger: only rank0 prints by default.
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        log_all = os.environ.get("LOG_ALL_RANKS", "0").strip() in {"1", "true", "True"}
    except Exception:
        world_size, rank, log_all = 1, 0, False
    if world_size > 1 and rank != 0 and not log_all:
        return
    prefix = f"[{time.strftime('%H:%M:%S')}]"
    if world_size > 1:
        prefix = f"{prefix} [rank {rank}]"
    print(f"{prefix} {msg}", flush=True)
# Optional: Weights & Biases logging
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False

# Ensure project root is on sys.path before importing internal packages
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconsimulator.envs.rl_wrapper import RLReconEnv
from reconsimulator.envs.subproc_vec_env import SubprocVecEnv, SerialVecEnv, make_scene_sampling_env
from rl.policy_diffusiondrivev2 import DiffusionDriveV2Policy
from rl.ppo import _obs_to_tensor as obs_to_tensor
from rl.ppo_ddv2_core import compute_gae, normalize_advantages, ddv2_ppo_update
from reconsimulator.envs import nus_config as nus_cfg


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "ppo_closed_loop.yaml"))

    # -------------------- DDP setup (torchrun) --------------------
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_enabled = world_size > 1

    ddp_cfg = (cfg.get("train", {}).get("ddp", {}) or {})
    grad_accum_steps = int(ddp_cfg.get("grad_accum_steps", 1))
    grad_accum_steps = max(1, grad_accum_steps)
    ddp_backend = str(ddp_cfg.get("backend", "nccl"))
    # NOTE: Gathering rollout/replay across ranks can be very memory-heavy for DDV2 PPO.
    # Prefer gather_rollout=false and rely on DDP gradient aggregation.
    gather_rollout = bool(ddp_cfg.get("gather_rollout", False))
    gather_ddv2_replay = bool(ddp_cfg.get("gather_ddv2_replay", False))
    ddp_seed = int(ddp_cfg.get("seed", 0))

    # -------------------- Reproducibility (seed) --------------------
    # Use ddp.seed as the global base seed unless overridden by train.seed.
    try:
        base_seed = int((cfg.get("train", {}) or {}).get("seed", ddp_seed))
    except Exception:
        base_seed = int(ddp_seed)
    per_rank_seed = int(base_seed) + int(rank)
    try:
        random.seed(per_rank_seed)
    except Exception:
        pass
    try:
        np.random.seed(per_rank_seed)
    except Exception:
        pass
    try:
        torch.manual_seed(per_rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(per_rank_seed)
    except Exception:
        pass
    # Optional: enforce deterministic behavior (may reduce performance).
    try:
        deterministic = bool((cfg.get("train", {}) or {}).get("deterministic", False))
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        deterministic = False
    # Local RNG for all sampling in this script (avoid accidental global RNG drift).
    rng = np.random.default_rng(per_rank_seed)
    try:
        stage(f"Seed: base={base_seed} per_rank={per_rank_seed} deterministic={bool(deterministic)}")
    except Exception:
        pass

    # -------------------- Replay storage/compute dtype (memory optimization) --------------------
    # Storing camera_feature as fp16 roughly halves replay RAM.
    # Default behavior keeps compute in fp32 for stability; set compute_dtype.camera_feature=fp16 to run fp16 end-to-end.
    replay_cfg = (cfg.get("train", {}).get("replay", {}) or {})
    replay_storage_dtype_cfg = (replay_cfg.get("storage_dtype", {}) or {})
    replay_compute_dtype_cfg = (replay_cfg.get("compute_dtype", {}) or {})

    def _parse_torch_dtype(x: Optional[str]) -> Optional[torch.dtype]:
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in {"fp16", "float16", "half"}:
            return torch.float16
        if s in {"fp32", "float32", "float"}:
            return torch.float32
        if s in {"bf16", "bfloat16"}:
            return torch.bfloat16
        raise ValueError(f"Unsupported dtype string: {x}. Supported: fp16/fp32/bf16")

    replay_storage_camera_dtype = _parse_torch_dtype(replay_storage_dtype_cfg.get("camera_feature", "fp16"))
    replay_storage_chain_dtype = _parse_torch_dtype(replay_storage_dtype_cfg.get("diffusion_chain", "fp32"))
    replay_compute_camera_dtype = _parse_torch_dtype(replay_compute_dtype_cfg.get("camera_feature", "fp32"))
    replay_compute_chain_dtype = _parse_torch_dtype(replay_compute_dtype_cfg.get("diffusion_chain", "fp32"))

    def _cast_tensor_if_needed(t: torch.Tensor, target: Optional[torch.dtype]) -> torch.Tensor:
        if target is None:
            return t
        if (not torch.is_tensor(t)) or (not t.is_floating_point()):
            return t
        if t.dtype == target:
            return t
        return t.to(dtype=target)

    def _cast_replay_for_storage(replay_obj: Any) -> Any:
        if not isinstance(replay_obj, dict):
            return replay_obj
        out: Dict[str, Any] = dict(replay_obj)
        if "camera_feature" in out and torch.is_tensor(out["camera_feature"]):
            out["camera_feature"] = _cast_tensor_if_needed(out["camera_feature"], replay_storage_camera_dtype)
        if "diffusion_chain" in out and torch.is_tensor(out["diffusion_chain"]):
            out["diffusion_chain"] = _cast_tensor_if_needed(out["diffusion_chain"], replay_storage_chain_dtype)
        return out

    # If user requests gathering rollouts, we must also gather DDV2 replay for ddv2_rl_ppo.
    if gather_rollout:
        gather_ddv2_replay = True

    if ddp_enabled:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=ddp_backend, init_method="env://")

    try:
        stage(
            "DDP status: "
            f"enabled={ddp_enabled} backend={ddp_backend} dist_initialized={bool(dist.is_initialized())} "
            f"world_size={world_size} rank={rank} local_rank={local_rank} "
            f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()} "
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}"
        )
    except Exception:
        pass

    def _destroy_dist() -> None:
        if ddp_enabled and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def _dist_all_reduce_mean(x: float) -> float:
        if not (ddp_enabled and dist.is_initialized()):
            return float(x)
        t = torch.tensor([float(x)], device=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu"))
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float((t / float(world_size)).item())

    def _dist_all_reduce_sum(x: float) -> float:
        if not (ddp_enabled and dist.is_initialized()):
            return float(x)
        t = torch.tensor([float(x)], device=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu"))
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def _dist_all_reduce_min_int(x: int) -> int:
        if not (ddp_enabled and dist.is_initialized()):
            return int(x)
        t = torch.tensor([int(x)], device=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu"), dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t.item())

    def _dist_all_gather_object(obj: Any) -> list[Any]:
        if not (ddp_enabled and dist.is_initialized()):
            return [obj]
        out: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(out, obj)
        return out
    # Optional per-process suffix for outputs and wandb run name
    run_suffix = os.environ.get("RUN_SUFFIX", "").strip()
    # In DDP, prefer a single shared output directory and a single wandb run.
    # If per-rank outputs are desired (debug), set RUN_SUFFIX explicitly.
    try:
        base_out = str(cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop"))
        if run_suffix:
            # Ensure nested directory per process to avoid collisions
            if "train" not in cfg:
                cfg["train"] = {}
            cfg["train"]["out_dir"] = os.path.join(base_out, run_suffix)
        # Append suffix to wandb run name if enabled
        wb_cfg = cfg.get("train", {}).get("wandb", {}) or {}
        rn = str(wb_cfg.get("run_name", f"ppo_closed_loop_{time.strftime('%Y%m%d-%H%M%S')}"))
        if run_suffix:
            wb_cfg["run_name"] = f"{rn}-{run_suffix}"
            if "train" not in cfg:
                cfg["train"] = {}
            cfg["train"]["wandb"] = wb_cfg
    except Exception:
        pass

    env_cfg = cfg.get("env", {})
    reward_cfg = env_cfg.get("reward", {})
    # Each DDP process binds to one GPU.
    cuda = int(local_rank) if (ddp_enabled and torch.cuda.is_available()) else int(env_cfg.get("cuda", 0))
    scene = int(env_cfg.get("scene", 0))
    max_steps = int(env_cfg.get("max_steps", 185))

    # Speed knobs for the environment itself.
    # NOTE: this env does GPU rendering per step; lowering resolution and/or skipping frames can drastically speed up rollout.
    env_render_w = env_cfg.get("render_w", None)
    env_render_h = env_cfg.get("render_h", None)
    env_step_frames = env_cfg.get("step_frames", None)
    try:
        env_render_w = int(env_render_w) if env_render_w is not None else None
    except Exception:
        env_render_w = None
    try:
        env_render_h = int(env_render_h) if env_render_h is not None else None
    except Exception:
        env_render_h = None
    try:
        env_step_frames = int(env_step_frames) if env_step_frames is not None else None
    except Exception:
        env_step_frames = None
    if env_step_frames is not None and env_step_frames <= 0:
        env_step_frames = None

    # Optional: start each episode from a non-zero frame within the scene.
    # Useful when a scene contains many frames (e.g., 185) but max_steps is smaller.
    start_cfg = (env_cfg.get("start_frame", {}) or {})
    start_mode = str(start_cfg.get("mode", "zero")).strip().lower()  # zero | random | sequential
    allow_short_tail = bool(start_cfg.get("allow_short_tail", False))
    start_min = int(start_cfg.get("min", 0))
    start_max_cfg = start_cfg.get("max", None)
    start_stride_cfg = start_cfg.get("stride", None)#NOTE
    # ---- Discover available scene ids (optional) ----
    use_all_scenes = bool(env_cfg.get("use_all_scenes", True))
    scene_sampling = str(env_cfg.get("scene_sampling", "random")).lower()  # random | sequential
    require_ckpt = bool(env_cfg.get("require_ckpt", True))

    # -------------------- Slow-op timing (helps diagnose "stuck" I/O) --------------------
    timing_cfg = (cfg.get("train", {}) or {}).get("timing", {}) or {}
    timing_enable = bool(timing_cfg.get("enable", True))
    slow_reset_s = float(timing_cfg.get("slow_reset_s", 8.0))
    slow_step_s = float(timing_cfg.get("slow_step_s", 2.0))

    def _discover_scene_ids(base_dir: str) -> list[int]:
        ids: list[int] = []
        missing_ckpt: int = 0
        try:
            for name in os.listdir(base_dir):
                if name.isdigit():
                    name3 = f"{int(name):03d}"
                    cam0 = os.path.join(base_dir, name3, "cam2ego", "0.txt")
                    ego0 = os.path.join(base_dir, name3, "ego_pose", "000.txt")
                    ckpt = os.path.join(base_dir, name3, "3DGS_without_prior", "checkpoint_final.pth")
                    if os.path.exists(cam0) and os.path.exists(ego0):
                        if (not require_ckpt) or os.path.exists(ckpt):
                            ids.append(int(name))
                        else:
                            missing_ckpt += 1
        except Exception:
            pass
        ids.sort()
        try:
            stage(f"Scene discovery @ {base_dir}: usable={len(ids)} require_ckpt={require_ckpt}")
            if require_ckpt:
                stage(f"Scenes missing ckpt skipped: {missing_ckpt}")
        except Exception:
            pass
        return ids

    _scene_ids: list[int] = [scene]
    if use_all_scenes:
        _scene_ids = _discover_scene_ids(nus_cfg.BASE_DATA_DIR) or [scene]
    _seq_idx: int = 0

    def _next_scene_id() -> int:
        nonlocal _seq_idx
        if len(_scene_ids) == 0:
            return scene
        if scene_sampling.startswith("seq"):
            sid = _scene_ids[_seq_idx % len(_scene_ids)]
            _seq_idx += 1
            return int(sid)
        else:
            # Deterministic given fixed seed + same control flow.
            return int(rng.choice(_scene_ids))

    _reset_counter: int = 0
    _scene_start_cursor: dict[int, int] = {}

    def _sample_start_frame(env_local: RLReconEnv, *, scene_id: int) -> int:
        if start_mode.startswith("zero"):
            return 0
        if (not start_mode.startswith("rand")) and (not start_mode.startswith("seq")):
            return 0

        try:
            final_frame = int(getattr(env_local.env, "final_frame", 0))
        except Exception:
            final_frame = 0
        try:
            step_frames = int(getattr(env_local.env, "step_frames", 1))
        except Exception:
            step_frames = 1
        if final_frame <= 1:
            return 0

        if allow_short_tail:
            max_start = final_frame - 1
        else:
            max_start = (final_frame - 1) - (max_steps * step_frames)
        max_start = max(0, int(max_start))

        lo = max(0, int(start_min))
        hi = max_start
        if start_max_cfg is not None:
            try:
                hi = min(int(hi), int(start_max_cfg))
            except Exception:
                pass
        hi = max(lo, int(hi))

        if start_mode.startswith("seq"):
            # Systematically slide a window through the scene.
            stride = None
            try:
                if start_stride_cfg is not None:
                    stride = int(start_stride_cfg)
            except Exception:
                stride = None
            if stride is None or stride <= 0:
                stride = max(1, int(max_steps * step_frames))
            cur = int(_scene_start_cursor.get(int(scene_id), lo))
            if cur < lo or cur > hi:
                cur = lo
            sf = cur
            nxt = cur + int(stride)
            if nxt > hi:
                nxt = lo
            _scene_start_cursor[int(scene_id)] = int(nxt)
        else:
            # Deterministic-ish per rank; varies across resets.
            seed = int(ddp_seed) + int(rank) * 100003 + int(scene_id) * 97 + int(_reset_counter) * 1009
            rng = np.random.RandomState(seed)
            sf = int(rng.randint(lo, hi + 1))
        if step_frames > 1:
            sf = (sf // step_frames) * step_frames
        return int(sf)

    # Track skipped scenes due to missing checkpoints
    scene_skips: int = 0

    # Helper: create env with a valid scene, skipping missing ones
    def _safe_create_env() -> tuple[RLReconEnv, Dict[str, np.ndarray], Dict[str, Any], int]:
        nonlocal scene_skips
        nonlocal _reset_counter
        max_attempts = max(1, len(_scene_ids))
        attempts = 0
        while attempts < max_attempts:
            sid = _next_scene_id()
            stage(f"Init scene candidate: {sid}")
            try:
                t0 = time.perf_counter()
                env_local = RLReconEnv(
                    cuda=cuda,
                    scene=sid,
                    reward_cfg=reward_cfg,
                    debug=debug,
                    render_w=env_render_w,
                    render_h=env_render_h,
                )
                t1 = time.perf_counter()
                sf = _sample_start_frame(env_local, scene_id=sid)
                _reset_counter += 1
                obs_local, info_local = env_local.reset(scene=sid, start_frame=sf, step_frames=env_step_frames)
                t2 = time.perf_counter()
                if timing_enable and (t2 - t0) > slow_reset_s:
                    stage(f"[slow] init+reset scene={sid} start_frame={sf} took {(t2 - t0):.2f}s (init={(t1 - t0):.2f}s reset={(t2 - t1):.2f}s)")
                stage(f"Init scene selected: {sid}")
                return env_local, obs_local, info_local, sid
            except FileNotFoundError:
                scene_skips += 1
                stage(f"[skip] Scene {sid} missing checkpoint; trying next...")
                attempts += 1
            except Exception as e:
                scene_skips += 1
                stage(f"[skip] Scene {sid} failed ({e}); trying next...")
                attempts += 1
        raise RuntimeError("No valid scenes found to initialize environment.")

    # Helper: reset env to next valid scene
    def _safe_reset_env() -> tuple[Dict[str, np.ndarray], Dict[str, Any], int]:
        nonlocal scene_skips
        nonlocal _reset_counter
        max_attempts = max(1, len(_scene_ids))
        attempts = 0
        while attempts < max_attempts:
            sid = _next_scene_id()
            try:
                t0 = time.perf_counter()
                sf = _sample_start_frame(env, scene_id=sid)
                _reset_counter += 1
                obs_local, info_local = env.reset(scene=sid, start_frame=sf, step_frames=env_step_frames)
                t1 = time.perf_counter()
                if timing_enable and (t1 - t0) > slow_reset_s:
                    stage(f"[slow] reset scene={sid} start_frame={sf} took {(t1 - t0):.2f}s")
                stage(f"Switched to scene {sid}")
                return obs_local, info_local, sid
            except FileNotFoundError:
                scene_skips += 1
                stage(f"[skip] Scene {sid} missing checkpoint; trying next...")
                attempts += 1
            except Exception as e:
                scene_skips += 1
                stage(f"[skip] Scene {sid} failed ({e}); trying next...")
                attempts += 1
        raise RuntimeError("No valid scenes found to reset environment.")
    debug = bool(env_cfg.get("debug", False))

#NOTE  # Initialize env with a valid scene
    env, obs, info, init_scene = _safe_create_env()
    cur_scene: int = int(init_scene)
    try:
        stage(f"Discovered {len(_scene_ids)} usable scenes (require_ckpt={require_ckpt})")
    except Exception:
        pass

    # Anchor sizes (from env attributes)
    x_anchor = getattr(env.env, "x_anchor", 61)
    y_anchor = getattr(env.env, "y_anchor", 61)
    agent_cfg = cfg.get("agent", {})
    ckpt_path = agent_cfg.get("ckpt", "/root/clone/ReconDreamer-RL/DiffusionDriveV2/ckpt/diffusiondrivev2_rl.ckpt")
    use_ddv2 = bool(agent_cfg.get("use_ddv2", True))
    ddv2 = None
    if use_ddv2:
        ddv2 = DiffusionDriveV2Policy(
            x_anchor=x_anchor,
            y_anchor=y_anchor,
            ckpt_path=ckpt_path,
            device=f"cuda:{cuda}" if torch.cuda.is_available() else "cpu",
            rl_lr=float(cfg.get("train", {}).get("ddv2_lr", 1e-5)),
            reinforce_baseline_beta=float(cfg.get("train", {}).get("ddv2_baseline_beta", 0.98)),
        )

        if ddp_enabled and torch.cuda.is_available():
            # DDP wrap DDV2 planner for multi-GPU fine-tuning
            ddv2.wrap_ddp(device_id=local_rank, process_group=dist.group.WORLD)

    train_cfg = cfg.get("train", {})
    algo = str(train_cfg.get("algo", "ddv2_rl_ppo"))
    lr = float(train_cfg.get("lr", 2e-4))
    lr_value = float(train_cfg.get("lr_value", lr))
    ppo_epochs = int(train_cfg.get("epochs", 1))
    horizon = int(train_cfg.get("horizon", 64))
    clip_eps = float(train_cfg.get("clip_eps", 0.2))
    total_updates = int(train_cfg.get("updates", 50))
    minibatch_size = int(train_cfg.get("minibatch_size", 64))
    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))

    guidance_weight = float(train_cfg.get("guidance_weight", 0.0 if not use_ddv2 else 1.0))
    guidance_sigma = float(train_cfg.get("guidance_sigma", 4.0))

    if algo not in {"ddv2_rl_reinforce", "ddv2_rl_ppo"}:
        _destroy_dist()
        raise RuntimeError(
            f"Unsupported train.algo={algo}. This training script only supports ddv2 fine-tuning: "
            "{ddv2_rl_reinforce, ddv2_rl_ppo}."
        )

    # max_steps is parsed above (used for start_frame sampling)

    reward_mode = str((reward_cfg or {}).get("mode", "four_component")).lower()
    episode_reward_mode = reward_mode.startswith("episode_")

    # ---- Training loop ----
    # Guidance is handled inside DiffusionDriveV2Policy.step() when use_ddv2=true.

    # Only save video for the very first episode by default (full training video is huge).
    ep_reward = 0.0
    max_video_frames = int(train_cfg.get("max_video_frames", horizon))
    video_frames_written = 0
    # ---- wandb init (optional via config) ----
    wb_cfg = cfg.get("train", {}).get("wandb", {})
    wb_enabled = bool(wb_cfg.get("enabled", False)) and _WANDB_AVAILABLE
    if wb_enabled and (not ddp_enabled or rank == 0):
        project = str(wb_cfg.get("project", "ReconDreamerRL"))
        run_name = str(wb_cfg.get("run_name", f"ppo_closed_loop_{time.strftime('%Y%m%d-%H%M%S')}"))
        wandb.init(project=project, name=run_name, config=cfg)
        wandb.define_metric("update")
        wandb.define_metric("global_step")
        wandb.define_metric("reward_sum", summary="mean")
        wandb.define_metric("loss_pi", summary="last")
        wandb.define_metric("loss_v", summary="last")
        wandb.define_metric("approx_kl", summary="last")
        wandb.define_metric("ddv2_param_delta", summary="last")
        # custom times & component summaries
        wandb.define_metric("collect_time_s", summary="last")
        wandb.define_metric("opt_time_s", summary="last")
        wandb.define_metric("update_time_s", summary="last")
        wandb.define_metric("rpd_mean", summary="last")
        wandb.define_metric("rhd_mean", summary="last")
        wandb.define_metric("rsc_rate", summary="last")
        wandb.define_metric("rdc_rate", summary="last")
        wandb.define_metric("jerk_pen_mean", summary="last")
        wandb.define_metric("yaw_jerk_pen_mean", summary="last")
    
#ADD START
    # ---- Video saving config ----
    train_cfg = cfg.get("train", {})
    save_video = bool(train_cfg.get("save_video", False))
    video_path = str(train_cfg.get("video_path", os.path.join("outputs/ppo_closed_loop", "episode.mp4")))
    fps = int(train_cfg.get("fps", 2))
    draw_traj_overlay = bool(train_cfg.get("draw_traj_overlay", False))
    writer = None
    final_video_path = video_path

    exp_hist: list[tuple[float, float]] = []
    act_hist: list[tuple[float, float]] = []

    def _grid_frame(observation: Dict[str, np.ndarray], info: Dict[str, Any] | None = None) -> np.ndarray:
        """Stack 6 views to 2x3 grid (H*2 x W*3 x 3), optionally draw trajectory overlay bottom-left."""
        keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
        imgs = [observation[k] for k in keys]
        h, w = imgs[0].shape[:2]
        row1 = np.concatenate(imgs[:3], axis=1)
        row2 = np.concatenate(imgs[3:6], axis=1)
        grid = np.concatenate([row1, row2], axis=0)

        # Append positions to history from info
        if info is not None:
            exp_pos = info.get("exp_pos", None)
            act_pos = info.get("act_pos", None)
            if exp_pos is not None and act_pos is not None:
                try:
                    exp_hist.append((float(exp_pos[0]), float(exp_pos[2])))
                    act_hist.append((float(act_pos[0]), float(act_pos[2])))
                except Exception:
                    pass

        if draw_traj_overlay:
            try:
                gh, gw = grid.shape[:2]
                box_w, box_h = 320, 120
                margin = 10
                x0, y0 = margin, gh - box_h - margin
                # Blend gray transparent rectangle
                roi_bg = grid[y0:y0+box_h, x0:x0+box_w].copy()
                overlay = roi_bg.copy()
                import cv2
                cv2.rectangle(overlay, (0, 0), (box_w - 1, box_h - 1), (128, 128, 128), thickness=-1)
                blended = cv2.addWeighted(overlay, 0.4, roi_bg, 0.6, 0)
                grid[y0:y0+box_h, x0:x0+box_w] = blended

                # Compose text lines from info
                exp_pos = info.get("exp_pos") if info else None
                act_pos = info.get("act_pos") if info else None
                exp_yaw_deg = info.get("exp_yaw_deg") if info else None
                act_yaw_deg = info.get("act_yaw_deg") if info else None
                xz_err_m = info.get("xz_err_m") if info else None
                yaw_err_deg = info.get("yaw_err_deg") if info else None

                def fmt_pose(tag, pos, yaw):
                    if pos is None or yaw is None:
                        return f"{tag}: (x=?, y=?) yaw=?"
                    return f"{tag}: x={pos[0]:.3f}, y={pos[1]:.3f}, yaw={float(yaw):.2f}deg"

                line1 = fmt_pose("EXP", exp_pos, exp_yaw_deg)
                line2 = fmt_pose("ACT", act_pos, act_yaw_deg)
                line3 = None
                if xz_err_m is not None and yaw_err_deg is not None:
                    line3 = f"err: xz={float(xz_err_m):.3f}m, yaw={float(yaw_err_deg):.2f}deg"

                # Draw text
                base_x = x0 + 8
                base_y = y0 + 22
                cv2.putText(grid, line1, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(grid, line2, (base_x, base_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                if line3:
                    cv2.putText(grid, line3, (base_x, base_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass
        return grid

    if save_video and (not ddp_enabled or rank == 0):
        # Append timestamp to avoid overwrite, e.g., episode_20260104-153045.mp4
        ts = time.strftime("%Y%m%d-%H%M%S")
        base_dir = os.path.dirname(video_path)
        base_name = os.path.basename(video_path)
        name, ext = os.path.splitext(base_name)
        final_video_path = os.path.join(base_dir, f"{name}_{ts}{ext}")
        os.makedirs(os.path.dirname(final_video_path), exist_ok=True)
        # Use macro_block_size=1 to avoid automatic resizing warnings (may reduce codec compatibility)
        writer = imageio.get_writer(final_video_path, mode="I", fps=fps, macro_block_size=1)
        stage(f"Video writer opened: {final_video_path} (fps={fps})")
        writer.append_data(_grid_frame(obs, None))
#ADD END DDV2 REINFORCE training loop
# #NOTE DDV2 RL training loop
    if algo == "ddv2_rl_reinforce":
        if ddv2 is None:
            raise RuntimeError("train.algo=ddv2_rl_reinforce requires agent.use_ddv2=true")

        agent = ddv2

        global_step = 0
        for upd in range(total_updates):
            stage(f"💜[reinforce] Update {upd+1}/{total_updates} start")
            ep_reward = 0.0
            steps_in_episode = 0
            losses = []
            grad_norms = []

            # # Parameter-delta sanity check (must move if gradients are flowing)
            # p_before = None
            # try:
            #     p_before = next(p for p in agent._agent.parameters() if getattr(p, "requires_grad", False)).detach().clone().cpu()
            # except Exception:
            #     p_before = None

            # One "update" here is horizon interaction steps.
            # If episode_reward_mode, we do episode-level REINFORCE: one update per episode
            # using summed logp and a single scalar episode reward.
            ep_logps: list[torch.Tensor] = []
            # aggregate per-step four components for logging
            comp_sum = {"rpd":0.0, "rhd":0.0, "rsc":0.0, "rdc":0.0, "jerk_pen":0.0, "yaw_jerk_pen":0.0}
            comp_steps = 0
            static_col_steps = 0
            dynamic_col_steps = 0
            t0_collect = time.perf_counter()
            # Each rank collects its own samples; total effective samples per update stays ~horizon.
            horizon_local = int((horizon + world_size - 1) // world_size) if ddp_enabled else int(horizon)
            stage(f"💜[reinforce] Collecting horizon_local={horizon_local} (global={horizon})")
            # Accumulate gradients and only step every grad_accum_steps.
            ddp_model = getattr(agent._agent, "_transfuser_model", None)
            opt = agent._ddv2_optimizer
            if opt is None:
                raise RuntimeError("DDV2 optimizer not initialized")
            opt.zero_grad(set_to_none=True)
            accum_i = 0
            for t in range(horizon_local):
                action, logp = agent.step_ddv2rl(obs, eta=1.0)
                t_step0 = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                t_step1 = time.perf_counter()
                if timing_enable and (t_step1 - t_step0) > slow_step_s:
                    try:
                        scene_id = getattr(getattr(env, "env", None), "scene", None)
                        now_frame = getattr(getattr(env, "env", None), "now_frame", None)
                    except Exception:
                        scene_id, now_frame = None, None
                    stage(f"[slow] step scene={scene_id} frame={now_frame} took {(t_step1 - t_step0):.2f}s")
                done = bool(terminated or truncated)

                # aggregate components if provided
                if isinstance(info, dict):
                    comp_sum["rpd"] += float(info.get("rpd", 0.0))
                    comp_sum["rhd"] += float(info.get("rhd", 0.0))
                    comp_sum["rsc"] += float(info.get("rsc", 0.0))
                    comp_sum["rdc"] += float(info.get("rdc", 0.0))
                    comp_sum["jerk_pen"] += float(info.get("jerk_pen", 0.0))
                    comp_sum["yaw_jerk_pen"] += float(info.get("yaw_jerk_pen", 0.0))
                    static_col_steps += 1 if bool(info.get("static_collision", False)) else 0
                    dynamic_col_steps += 1 if bool(info.get("dynamic_collision", False)) else 0
                    comp_steps += 1

                # Keep local reward for policy-gradient, but update baseline using global mean reward
                # to keep baseline consistent across ranks.
                r_local = float(reward)
                r_mean = float(_dist_all_reduce_mean(r_local))

                if episode_reward_mode:
                    ep_logps.append(logp)
                else:
                    # Inline REINFORCE backward with grad accumulation + no_sync
                    agent._reward_baseline = agent._baseline_beta * agent._reward_baseline + (1.0 - agent._baseline_beta) * r_mean
                    adv = r_local - agent._reward_baseline
                    loss = -(float(adv) * logp) / float(grad_accum_steps)
                    sync_now = ((accum_i + 1) % grad_accum_steps) == 0
                    cm = nullcontext()
                    if ddp_enabled and ddp_model is not None and hasattr(ddp_model, "no_sync") and not sync_now:
                        cm = ddp_model.no_sync()
                    with cm:
                        loss.backward()
                    losses.append(float(loss.detach().cpu().item()))
                    accum_i += 1
                    if sync_now:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                # grad_norms.append(float(m.get("grad_norm", 0.0)))

                # In episode reward mode, per-step reward is 0; log episode_reward on termination.
                if not episode_reward_mode:
                    ep_reward += float(reward)
                steps_in_episode += 1
                global_step += 1

                if save_video and writer is not None and video_frames_written < max_video_frames:
                    writer.append_data(_grid_frame(obs, info))
                    video_frames_written += 1

                if steps_in_episode >= max_steps:
                    done = True
                    if episode_reward_mode:
                        ep_r, ep_m = env.finalize_episode_reward(done_reason="timeout")
                        info = dict(info or {})
                        info["episode_reward"] = float(ep_r)
                        info["episode_metrics"] = ep_m
                        info["episode_len"] = int(ep_m.get("episode_len", 0))
                        info["done_reason"] = str(ep_m.get("done_reason", "timeout"))

                if done:
                    dr = None
                    try:
                        dr = (info or {}).get("done_reason", None)
                    except Exception:
                        dr = None
                    stage(f"💜[reinforce] Episode done: steps={steps_in_episode} reward={(info or {}).get('episode_reward', ep_reward):.4f} reason={str(dr) if dr is not None else 'n/a'}")
                    if episode_reward_mode:
                        ep_reward_local = float(info.get("episode_reward", 0.0))
                        ep_reward_mean = float(_dist_all_reduce_mean(float(ep_reward_local)))
                        ep_reward += ep_reward_local
                        if len(ep_logps) > 0:
                            ep_logp_sum = torch.stack(ep_logps, dim=0).sum()
                            agent._reward_baseline = agent._baseline_beta * agent._reward_baseline + (1.0 - agent._baseline_beta) * ep_reward_mean
                            adv = ep_reward_local - agent._reward_baseline
                            loss = -(float(adv) * ep_logp_sum) / float(grad_accum_steps)
                            sync_now = ((accum_i + 1) % grad_accum_steps) == 0
                            cm = nullcontext()
                            if ddp_enabled and ddp_model is not None and hasattr(ddp_model, "no_sync") and not sync_now:
                                cm = ddp_model.no_sync()
                            with cm:
                                loss.backward()
                            losses.append(float(loss.detach().cpu().item()))
                            accum_i += 1
                            if sync_now:
                                opt.step()
                                opt.zero_grad(set_to_none=True)
                        ep_logps = []
                    # Switch to next scene for next episode
                    obs, info, _ = _safe_reset_env()
                    steps_in_episode = 0
            collect_time = time.perf_counter()-t0_collect
            stage(f"💜[reinforce] Collection finished in {collect_time:.2f}s")

            # Flush leftover accumulated grads
            if (accum_i % grad_accum_steps) != 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
            os.makedirs(out_dir, exist_ok=True)

            # param_delta_max = 0.0
            # if p_before is not None:
            #     try:
            #         p_after = next(p for p in agent._agent.parameters() if getattr(p, "requires_grad", False)).detach().cpu()
            #         param_delta_max = float((p_after - p_before).abs().max().item())
            #     except Exception:
            #         param_delta_max = 0.0

            ep_reward_log = float(_dist_all_reduce_mean(float(ep_reward)))
            loss_sum_local = float(np.sum(losses)) if len(losses) else 0.0
            loss_cnt_local = float(len(losses))
            loss_sum = float(_dist_all_reduce_sum(loss_sum_local))
            loss_cnt = float(_dist_all_reduce_sum(loss_cnt_local))
            loss_mean = float(loss_sum / max(1.0, loss_cnt))

            if (not ddp_enabled) or rank == 0:
                with open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8") as f:
                    f.write(
                        f"{time.time():.0f}\tupdate={upd}\tglobal_step={global_step}"
                        f"\trew_sum={ep_reward_log:.4f}\t"
                        f"loss_reinforce={loss_mean:.6f}\t"
                    )
                print(
                    f"[ddv2-rl update {upd}/{total_updates}] steps={global_step} "
                    f"rew_sum={ep_reward_log:.4f} loss={loss_mean:.4f} "
                )
            if wb_enabled:
                comp_mean = (lambda x: (x/comp_steps) if comp_steps>0 else 0.0)
                wandb.log({
                    "update": upd,
                    "global_step": global_step,
                    "reward_sum": float(ep_reward_log),
                    "loss_reinforce": float(loss_mean),
                    "collect_time_s": float(collect_time),
                    "rpd_mean": float(comp_mean(comp_sum["rpd"])),
                    "rhd_mean": float(comp_mean(comp_sum["rhd"])),
                    "rsc_rate": float((static_col_steps/comp_steps) if comp_steps>0 else 0.0),
                    "rdc_rate": float((dynamic_col_steps/comp_steps) if comp_steps>0 else 0.0),
                    "jerk_pen_mean": float(comp_mean(comp_sum["jerk_pen"])),
                    "yaw_jerk_pen_mean": float(comp_mean(comp_sum["yaw_jerk_pen"])),
                    "scene_skips": int(scene_skips),
                })

            # Save updated DDV2 weights (optional)
            save_cfg = cfg.get("train", {}).get("save", {})
            save_every = int(save_cfg.get("every", 0))
            out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
            if (not ddp_enabled or rank == 0) and save_every > 0 and ((upd + 1) % save_every == 0):
                os.makedirs(out_dir, exist_ok=True)
                try:
                    m = agent._agent._transfuser_model
                    sd = (m.module.state_dict() if hasattr(m, "module") else m.state_dict())
                    sd_pref = {f"agent.{k}": v for k, v in sd.items()}
                    ckpt_path = os.path.join(out_dir, f"ddv2_reinforce_{upd+1}.ckpt")
                    torch.save({"state_dict": sd_pref}, ckpt_path)
                    print(f"Saved DDV2 ckpt: {ckpt_path}")
                    upload_to_wandb = bool(save_cfg.get("upload_to_wandb", False))
                    if wb_enabled and upload_to_wandb:
                        try:
                            wandb.save(ckpt_path)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[WARN] Failed to save DDV2 ckpt: {e}")

        if writer is not None:
            writer.close()
            stage(f"Video saved: {final_video_path}")
        # Final checkpoint save to local & wandb
        try:
            out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
            os.makedirs(out_dir, exist_ok=True)
            m = agent._agent._transfuser_model
            sd = (m.module.state_dict() if hasattr(m, "module") else m.state_dict())
            sd_pref = {f"agent.{k}": v for k, v in sd.items()}
            final_ckpt = os.path.join(out_dir, "ddv2_reinforce_final.ckpt")
            if (not ddp_enabled) or rank == 0:
                torch.save({"state_dict": sd_pref}, final_ckpt)
                print(f"Saved final DDV2 ckpt: {final_ckpt}")
            if wb_enabled and (not ddp_enabled or rank == 0):
                try:
                    upload_final = bool(cfg.get("train", {}).get("save", {}).get("upload_final_to_wandb", True))
                    if upload_final:
                        wandb.save(final_ckpt)
                    art = wandb.Artifact("ddv2_reinforce_ckpt", type="model")
                    art.add_file(final_ckpt)
                    wandb.log_artifact(art)
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Failed final DDV2 ckpt save: {e}")
        stage("ddv2_rl_REINFORCE Training finished.")
        _destroy_dist()
        return
#ADD ddv2 PPO traj_head training loop
    '''
    用 PPO 在闭环环境中微调 DiffusionDriveV2 的 trajectory head
        policy：DDV2 的 diffusion trajectory head
        value：一个额外的小 CNN critic
        rollout：按 episode 收集（而不是固定 horizon）
        log-prob：一整条 trajectory 的 log-prob 之和
        update：只更新 DDV2 的 trajectory head + value net
    '''
    if algo == "ddv2_rl_ppo":
        if ddv2 is None:
            raise RuntimeError("train.algo=ddv2_rl_ppo requires agent.use_ddv2=true")

        agent = ddv2

        ddv2_eta = float(train_cfg.get("ddv2_eta", 1.0))#diffusion 噪声强度
        ddv2_mode_idx = int(train_cfg.get("ddv2_mode_idx", 0))#多模态轨迹的模式索引,实际设置为-1 ，表示采样而不是每次抽取一样的
        ddv2_mode_select = str(train_cfg.get("ddv2_mode_select", "sample")).strip().lower()#怎样选择每一步的Mode 有sample和greedy
        max_grad_norm = float(train_cfg.get("ddv2_max_grad_norm", 0.5))#梯度裁剪阈值

        batch_episodes = int(train_cfg.get("batch_episodes", 2))#每次 update 用多少完整 episode。4096条轨迹*60步
        vf_coef = float(train_cfg.get("vf_coef", 0.5))#这个是valueNet的损失系数 value loss 权重

        # -------------------- Vectorized env (per-rank multi-env) --------------------
        # Note: RLReconEnv is Gymnasium-style but not a gym.Env, so we use a small
        # custom SubprocVecEnv to parallelize env stepping across multiple processes.
        vec_cfg = (train_cfg.get("vec_env", {}) or {})
        num_envs_cfg = int(vec_cfg.get("num_envs", train_cfg.get("num_envs", 1)))
        num_envs_cfg = max(1, int(num_envs_cfg))
        vec_backend = str(vec_cfg.get("backend", "serial")).strip().lower()  # serial | subproc
        vec_start_method = str(vec_cfg.get("start_method", "spawn")).strip().lower()
        # For episode-centric collection, having num_envs >= batch_episodes_local works best.
        vec_env: Any | None = None

        # A tiny critic for GAE baseline (separate from DDV2 planner).给 PPO 提供 GAE baseline
        class _ValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(18, 32, kernel_size=8, stride=4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(inplace=True),
                )
                with torch.no_grad():
                    dummy = torch.zeros(1, 18, 64, 64)
                    n_flat = int(self.conv(dummy).view(1, -1).shape[1])
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_flat, 512),
                    nn.ReLU(inplace=True),
                )
                self.v = nn.Linear(512, 1)

            def forward(self, obs_t: torch.Tensor) -> torch.Tensor:
                h = self.fc(self.conv(obs_t))
                return self.v(h).squeeze(-1)
#NOTE 初始化优化器和 rollout buffer
        model_device = next(agent._agent.parameters()).device if hasattr(agent._agent, "parameters") else torch.device("cpu")
        value_net = _ValueNet().to(model_device)
        if ddp_enabled and torch.cuda.is_available():
            value_net = nn.parallel.DistributedDataParallel(value_net, device_ids=[local_rank], output_device=local_rank)
        value_lr = float(train_cfg.get("lr_value", 1e-4))
        value_optim = torch.optim.Adam(value_net.parameters(), lr=value_lr)

        rollout_obs = []#critic 输入
        rollout_replay = []#重算 logp 的 diffusion 中间态
        rollout_logp_old = []#trajectory logp
        rollout_val = []#V(s) 价值估计
        rollout_rew = []#reward
        rollout_done = []#episode 结束标志

#NOTE 主训练循环（update）
        global_step = 0
        for upd in range(total_updates):#50总共更新的次数 
            stage(f"💜[ppo-ddv2] Update {upd+1}/{total_updates} start")
            # Explicitly announce rollout mode to avoid confusion.
            stage(
                f"💜[ppo-ddv2] Rollout mode={'vec' if int(num_envs_cfg) > 1 else 'single'} "
                f"(train.vec_env.num_envs={int(num_envs_cfg)}, backend={vec_backend}, start_method={vec_start_method})"
            )
            rollout_obs.clear()
            rollout_replay.clear()
            rollout_logp_old.clear()
            rollout_val.clear()
            rollout_rew.clear()
            rollout_done.clear()

            # NOTE: reward sums printed during collection.
            # `ep_reward` is the cumulative return across episodes collected in this update.
            ep_reward = 0.0
            episodes_collected = 0
            steps_in_episode = 0
            ep_start_idx = 0 #记录每个 episode 在 rollout buffer 中的起始索引，用于 batch reward 替换

            # Collect full episodes (episode-centric closed-loop)
#NOTE 收集 batch episodes（rollout）
            '''
            buffer 是时间维度堆叠，不是按 episode划分的。
            eg：每次 update 会收集两条完整的 episode,每条 episode 可能有很多步，例如 50 步,所以最终 rollout_replay 里可能有 2 * 50 = 100 条条目
            '''
            t0_collect = time.perf_counter()
            # Split episodes across ranks to keep total episodes ~= batch_episodes.
            if ddp_enabled:
                base = int(batch_episodes) // int(world_size)
                rem = int(batch_episodes) % int(world_size)
                batch_episodes_local = base + (1 if rank < rem else 0)
            else:
                batch_episodes_local = int(batch_episodes)

            stage(f"💜[ppo-ddv2] Collecting episodes_local={batch_episodes_local} (global={batch_episodes})...")
            comp_sum = {"rpd":0.0, "rhd":0.0, "rsc":0.0, "rdc":0.0, "jerk_pen":0.0, "yaw_jerk_pen":0.0}
            comp_steps = 0
            static_col_steps = 0
            dynamic_col_steps = 0
            # Lazily create vec_env only if needed.
            if vec_env is None and num_envs_cfg > 1:
                # Each worker self-samples scenes/start_frame; we pass the discovered scene list.
                # IMPORTANT: GPU-heavy envs frequently deadlock under subprocess spawn.
                # Default backend is serial (single process). Use subproc only if you know it works.
                n_eff = int(min(int(num_envs_cfg), max(1, int(batch_episodes_local))))
                env_fns = [
                    functools.partial(
                        make_scene_sampling_env,
                        cuda=int(cuda),
                        reward_cfg=reward_cfg,
                        debug=debug,
                        scene_ids=list(_scene_ids),
                        scene_sampling=str(scene_sampling),
                        ddp_seed=int(ddp_seed),
                        rank=int(rank),
                        worker_id=int(i),
                        start_mode=str(start_mode),
                        allow_short_tail=bool(allow_short_tail),
                        start_min=int(start_min),
                        start_max=(int(start_max_cfg) if start_max_cfg is not None else None),
                        start_stride=(int(start_stride_cfg) if start_stride_cfg is not None else None),
                        max_steps=int(max_steps),
                        render_w=env_render_w,
                        render_h=env_render_h,
                        step_frames=env_step_frames,
                    )
                    for i in range(int(n_eff))
                ]
                if vec_backend in {"subproc", "subprocess", "mp", "multiproc"}:
                    if torch.cuda.is_available():
                        stage("[WARN] vec_env.backend=subproc with CUDA env may hang; prefer backend=serial")
                    vec_env = SubprocVecEnv(env_fns, start_method=str(vec_start_method))
                    stage(f"💜[ppo-ddv2] SubprocVecEnv ready: num_envs={vec_env.num_envs} start_method={vec_start_method}")
                else:
                    vec_env = SerialVecEnv(env_fns)
                    stage(f"💜[ppo-ddv2] SerialVecEnv ready: num_envs={vec_env.num_envs}")

            if vec_env is None:
                # -------------------- Single-env collection (original) --------------------
                while episodes_collected < batch_episodes_local:
                    # Observation tensor for critic
                    obs_t = obs_to_tensor(obs, model_device)  # (1,18,64,64)
                    value_net.eval()
                    with torch.inference_mode():
                        v = value_net(obs_t).squeeze(0)

                    action, logp, replay = agent.sample_ddv2rl_with_replay(
                        obs,
                        eta=ddv2_eta,
                        mode_idx=ddv2_mode_idx,
                        mode_select=ddv2_mode_select,
                    )
                    rollout_obs.append(obs_t.detach().cpu())
                    rollout_val.append(v.detach().cpu())
                    # 将 replay 字典中所有张量字段转为 CPU 并分离梯度
                    if isinstance(replay, dict):
                        replay_cpu = {
                            k: (v.detach().cpu() if torch.is_tensor(v) else v)
                            for k, v in replay.items()
                        }
                        replay_cpu = _cast_replay_for_storage(replay_cpu)
                    else:
                        # 兼容旧返回格式：若不是字典，则尝试按张量处理
                        replay_cpu = replay.detach().cpu() if torch.is_tensor(replay) else replay
                    rollout_replay.append(replay_cpu)
                    rollout_logp_old.append(logp.detach().cpu())#先将第一初始化的存进去
                    #NOTE 环境交互一步
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    rollout_rew.append(float(reward))
                    rollout_done.append(1.0 if done else 0.0)
                    # aggregate components for logging
                    if isinstance(info, dict):
                        comp_sum["rpd"] += float(info.get("rpd", 0.0))
                        comp_sum["rhd"] += float(info.get("rhd", 0.0))
                        comp_sum["rsc"] += float(info.get("rsc", 0.0))
                        comp_sum["rdc"] += float(info.get("rdc", 0.0))
                        comp_sum["jerk_pen"] += float(info.get("jerk_pen", 0.0))
                        comp_sum["yaw_jerk_pen"] += float(info.get("yaw_jerk_pen", 0.0))
                        static_col_steps += 1 if bool(info.get("static_collision", False)) else 0
                        dynamic_col_steps += 1 if bool(info.get("dynamic_collision", False)) else 0
                        comp_steps += 1

                    steps_in_episode += 1
                    global_step += 1#训练的总步数

                    if save_video and writer is not None and video_frames_written < max_video_frames:
                        writer.append_data(_grid_frame(obs, info))
                        video_frames_written += 1

                    if steps_in_episode >= max_steps:
                        done = True
                        rollout_done[-1] = 1.0
                        if episode_reward_mode:
                            ep_r, ep_m = env.finalize_episode_reward(done_reason="timeout")
                            info = dict(info or {})
                            info["episode_reward"] = float(ep_r)
                            info["episode_metrics"] = ep_m
                            info["episode_len"] = int(ep_m.get("episode_len", 0))
                            info["done_reason"] = str(ep_m.get("done_reason", "timeout"))

                    if done:
                        dr = None
                        try:
                            dr = (info or {}).get("done_reason", None)
                        except Exception:
                            dr = None

                        #NOTE Compute *this episode* return for logging.
                        ep_return = None
                        if episode_reward_mode and isinstance(info, dict) and ("episode_reward" in info):
                            try:
                                ep_return = float(info.get("episode_reward", 0.0))
                            except Exception:
                                ep_return = None
                        if ep_return is None:
                            try:
                                ep_return = float(np.sum(rollout_rew[int(ep_start_idx) :]))
                            except Exception:
                                ep_return = 0.0

                        stage(
                            f"💜[ppo-ddv2] Episode {episodes_collected+1} done(env=0): steps={steps_in_episode} "
                            f"ep_return={float(ep_return):.4f} upd_return_sum(before_add)={ep_reward:.4f} "
                            f"reason={str(dr) if dr is not None else 'n/a'}"
                        )

                        #NOTE r0 = r1 = r2 = ... = rT = R_episode
                        if episode_reward_mode and isinstance(info, dict) and ("episode_reward" in info):
                            ep_r = float(info.get("episode_reward", 0.0))
                            ep_reward += ep_r
                            for j in range(int(ep_start_idx), len(rollout_rew)):
                                rollout_rew[j] = ep_r
                            ep_start_idx = len(rollout_rew)
                        else:
                            # fallback: sum raw rewards G=r0 + r1 + ... + rT
                            ep_reward += float(np.sum(rollout_rew[int(ep_start_idx) :]))
                            ep_start_idx = len(rollout_rew)

                        episodes_collected += 1
                        obs, info, cur_scene = _safe_reset_env()
                        steps_in_episode = 0
            else:
                # -------------------- Multi-env collection (per-rank subprocess envs) --------------------
                n_envs = int(vec_env.num_envs)
                # Periodic progress logging (CLI)
                try:
                    # Default off to avoid noisy logs; episode-done logs are always printed.
                    vec_log_enable = bool(vec_cfg.get("log_progress", False))
                except Exception:
                    vec_log_enable = False
                try:
                    vec_log_interval_s = float(vec_cfg.get("log_interval_s", 5.0))
                except Exception:
                    vec_log_interval_s = 5.0
                vec_log_interval_s = max(0.2, float(vec_log_interval_s))
                try:
                    vec_log_max_envs = int(vec_cfg.get("log_max_envs", 4))
                except Exception:
                    vec_log_max_envs = 4
                vec_log_max_envs = max(1, int(vec_log_max_envs))

                # Per-env episode buffers (kept until that env finishes an episode)
                ep_obs: list[list[torch.Tensor]] = [[] for _ in range(n_envs)]
                ep_val: list[list[torch.Tensor]] = [[] for _ in range(n_envs)]
                ep_replay: list[list[Any]] = [[] for _ in range(n_envs)]
                ep_logp: list[list[torch.Tensor]] = [[] for _ in range(n_envs)]
                ep_rew: list[list[float]] = [[] for _ in range(n_envs)]
                ep_done: list[list[float]] = [[] for _ in range(n_envs)]
                ep_steps: list[int] = [0 for _ in range(n_envs)]

                # Per-env timing accumulators (seconds). Approximate attribution across batch.
                ep_t_obs: list[float] = [0.0 for _ in range(n_envs)]
                ep_t_value: list[float] = [0.0 for _ in range(n_envs)]
                ep_t_policy: list[float] = [0.0 for _ in range(n_envs)]
                ep_t_env_step: list[float] = [0.0 for _ in range(n_envs)]

                # Reset all envs at update start to keep episode-level reward modes consistent.
                obs_list, info_list = vec_env.reset()
                active0 = int(min(int(n_envs), max(1, int(batch_episodes_local))))
                running: list[bool] = [bool(i < active0) for i in range(n_envs)]

                # Clear episode start index semantics (we now append per-episode blocks).
                ep_start_idx = 0
                steps_in_episode = 0

                last_prog_t = time.perf_counter()
                last_prog_step = int(global_step)

                while episodes_collected < batch_episodes_local:
                    actions: list[Any] = [None for _ in range(n_envs)]

                    # Compute actions only for active envs; inactive envs idle (no stepping).
                    active_idx = [i for i in range(n_envs) if bool(running[i])]
                    if len(active_idx) > 0:
                        obs_active = [obs_list[i] for i in active_idx]

                        t_obs0 = time.perf_counter()

                        # Critic value (batched)
                        obs_t_list = [obs_to_tensor(o, model_device) for o in obs_active]
                        obs_t_batch = torch.cat(obs_t_list, dim=0)
                        value_net.eval()
                        t_val0 = time.perf_counter()
                        with torch.inference_mode():
                            v_batch = value_net(obs_t_batch).detach()

                        t_pol0 = time.perf_counter()

                        # Policy (batched if available)
                        if hasattr(agent, "sample_ddv2rl_with_replay_batch"):
                            a_list, lp_list, rp_list = agent.sample_ddv2rl_with_replay_batch(
                                obs_active,
                                eta=ddv2_eta,
                                mode_idx=ddv2_mode_idx,
                                mode_select=ddv2_mode_select,
                            )
                        else:
                            a_list, lp_list, rp_list = [], [], []
                            for o in obs_active:
                                a, lp, rp = agent.sample_ddv2rl_with_replay(
                                    o,
                                    eta=ddv2_eta,
                                    mode_idx=ddv2_mode_idx,
                                    mode_select=ddv2_mode_select,
                                )
                                a_list.append(a)
                                lp_list.append(lp)
                                rp_list.append(rp)

                        t_pol1 = time.perf_counter()

                        # Attribute timing roughly evenly to each active env.
                        bsz = max(1, int(len(active_idx)))
                        dt_obs = float(t_val0 - t_obs0)
                        dt_val = float(t_pol0 - t_val0)
                        dt_pol = float(t_pol1 - t_pol0)
                        for env_i in active_idx:
                            ep_t_obs[env_i] += dt_obs / float(bsz)
                            ep_t_value[env_i] += dt_val / float(bsz)
                            ep_t_policy[env_i] += dt_pol / float(bsz)

                        for j, env_i in enumerate(active_idx):
                            actions[env_i] = a_list[j]
                            obs_t = obs_t_list[j]
                            v = v_batch[j]
                            logp = lp_list[j]
                            replay = rp_list[j]

                            ep_obs[env_i].append(obs_t.detach().cpu())
                            ep_val[env_i].append(v.detach().cpu())
                            ep_logp[env_i].append((logp.detach().cpu() if torch.is_tensor(logp) else torch.as_tensor(logp).cpu()))

                            if isinstance(replay, dict):
                                replay_cpu = {
                                    k: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                                    for k, vv in replay.items()
                                }
                                replay_cpu = _cast_replay_for_storage(replay_cpu)
                            else:
                                replay_cpu = replay.detach().cpu() if torch.is_tensor(replay) else replay
                            ep_replay[env_i].append(replay_cpu)

                    t_step0 = time.perf_counter()
                    obs_next_list, reward_list, term_list, trunc_list, info_next_list = vec_env.step(actions)
                    t_step1 = time.perf_counter()
                    if len(active_idx) > 0:
                        bsz = max(1, int(len(active_idx)))
                        dt_step = float(t_step1 - t_step0)
                        for env_i in active_idx:
                            ep_t_env_step[env_i] += dt_step / float(bsz)

                    for i in range(n_envs):
                        if not running[i]:
                            # idling env; keep cached obs
                            obs_list[i] = obs_next_list[i]
                            info_list[i] = info_next_list[i]
                            continue

                        obs_list[i] = obs_next_list[i]
                        info_list[i] = info_next_list[i]
                        info_i = info_next_list[i]

                        done = bool(term_list[i] or trunc_list[i])
                        ep_rew[i].append(float(reward_list[i]))
                        ep_done[i].append(1.0 if done else 0.0)

                        # aggregate components for logging
                        if isinstance(info_i, dict):
                            comp_sum["rpd"] += float(info_i.get("rpd", 0.0))
                            comp_sum["rhd"] += float(info_i.get("rhd", 0.0))
                            comp_sum["rsc"] += float(info_i.get("rsc", 0.0))
                            comp_sum["rdc"] += float(info_i.get("rdc", 0.0))
                            comp_sum["jerk_pen"] += float(info_i.get("jerk_pen", 0.0))
                            comp_sum["yaw_jerk_pen"] += float(info_i.get("yaw_jerk_pen", 0.0))
                            static_col_steps += 1 if bool(info_i.get("static_collision", False)) else 0
                            dynamic_col_steps += 1 if bool(info_i.get("dynamic_collision", False)) else 0
                            comp_steps += 1

                        ep_steps[i] += 1
                        global_step += 1

                        # Video: only write env0 to avoid huge overhead
                        if (
                            i == 0
                            and save_video
                            and writer is not None
                            and video_frames_written < max_video_frames
                        ):
                            try:
                                writer.append_data(_grid_frame(obs_list[i], info_i))
                                video_frames_written += 1
                            except Exception:
                                pass

                        # Forced timeout
                        if ep_steps[i] >= max_steps:
                            done = True
                            ep_done[i][-1] = 1.0
                            if episode_reward_mode:
                                try:
                                    ep_r, ep_m = vec_env.call_one(i, "finalize_episode_reward", done_reason="timeout")
                                    info_i = dict(info_i or {})
                                    info_i["episode_reward"] = float(ep_r)
                                    info_i["episode_metrics"] = ep_m
                                    info_i["episode_len"] = int(ep_m.get("episode_len", 0))
                                    info_i["done_reason"] = str(ep_m.get("done_reason", "timeout"))
                                    info_next_list[i] = info_i
                                    info_list[i] = info_i
                                except Exception:
                                    pass

                        if done:
                            dr = None
                            try:
                                dr = (info_i or {}).get("done_reason", None)
                            except Exception:
                                dr = None

                            # Compute this episode return for logging.
                            ep_return = None
                            if episode_reward_mode and isinstance(info_i, dict) and ("episode_reward" in info_i):
                                try:
                                    ep_return = float(info_i.get("episode_reward", 0.0))
                                except Exception:
                                    ep_return = None
                            if ep_return is None:
                                try:
                                    ep_return = float(np.sum(ep_rew[i]))
                                except Exception:
                                    ep_return = 0.0

                            stage(
                                f"💜[ppo-ddv2] Episode {episodes_collected+1} done(env={i}): steps={ep_steps[i]} "
                                f"ep_return={float(ep_return):.4f} upd_return_sum(before_add)={ep_reward:.4f} "
                                f"reason={str(dr) if dr is not None else 'n/a'} "
                                f"scene={info_i.get('scene', None) if isinstance(info_i, dict) else None} "
                                f"frame={info_i.get('now_frame', None) if isinstance(info_i, dict) else None} "
                                f"ms/step(step={1000.0*ep_t_env_step[i]/max(1,int(ep_steps[i])):.1f} "
                                f"pol={1000.0*ep_t_policy[i]/max(1,int(ep_steps[i])):.1f} "
                                f"v={1000.0*ep_t_value[i]/max(1,int(ep_steps[i])):.1f} "
                                f"obs={1000.0*ep_t_obs[i]/max(1,int(ep_steps[i])):.1f})"
                            )

                            if episode_reward_mode and isinstance(info_i, dict) and ("episode_reward" in info_i):
                                ep_r = float(info_i.get("episode_reward", 0.0))
                                ep_reward += ep_r
                                ep_rew[i] = [ep_r for _ in range(len(ep_rew[i]))]
                            else:
                                ep_reward += float(np.sum(ep_rew[i]))

                            # Flush per-episode block into global rollout buffer
                            rollout_obs.extend(ep_obs[i])
                            rollout_val.extend(ep_val[i])
                            rollout_replay.extend(ep_replay[i])
                            rollout_logp_old.extend(ep_logp[i])
                            rollout_rew.extend(ep_rew[i])
                            rollout_done.extend(ep_done[i])

                            # Reset per-env episode buffers
                            ep_obs[i].clear()
                            ep_val[i].clear()
                            ep_replay[i].clear()
                            ep_logp[i].clear()
                            ep_rew[i].clear()
                            ep_done[i].clear()
                            ep_steps[i] = 0

                            ep_t_obs[i] = 0.0
                            ep_t_value[i] = 0.0
                            ep_t_policy[i] = 0.0
                            ep_t_env_step[i] = 0.0

                            episodes_collected += 1
                            running[i] = False

                    # Decide whether to start new episodes on paused envs
                    need = int(batch_episodes_local) - int(episodes_collected)
                    if need <= 0:
                        break
                    n_running = int(sum(1 for x in running if x))
                    if n_running < need:
                        # Reactivate some paused envs
                        for i in range(n_envs):
                            if need <= 0:
                                break
                            if running[i]:
                                continue
                            try:
                                obs_i, info_i = vec_env.reset_one(i)
                                obs_list[i] = obs_i
                                info_list[i] = info_i
                            except Exception:
                                # If reset fails, keep it paused; another env may fill the gap.
                                continue
                            running[i] = True
                            n_running += 1
                            need = int(batch_episodes_local) - int(episodes_collected)
                            if n_running >= need:
                                break

                    # Periodic CLI progress summary
                    if vec_log_enable:
                        now_t = time.perf_counter()
                        if (now_t - last_prog_t) >= float(vec_log_interval_s):
                            d_steps = int(global_step) - int(last_prog_step)
                            dt_s = max(1e-6, float(now_t - last_prog_t))
                            sps = float(d_steps) / dt_s
                            try:
                                env_lines = []
                                for ei in range(min(int(n_envs), int(vec_log_max_envs))):
                                    ii = info_list[ei]
                                    sc = None
                                    fr = None
                                    if isinstance(ii, dict):
                                        sc = ii.get("scene", None)
                                        fr = ii.get("now_frame", None)
                                    env_lines.append(
                                        f"e{ei}:run={1 if running[ei] else 0} step={ep_steps[ei]} scene={sc} frame={fr}"
                                    )
                                stage(
                                    f"💜[ppo-ddv2][progress] episodes={episodes_collected}/{batch_episodes_local} "
                                    f"active={int(sum(1 for x in running if x))}/{n_envs} "
                                    f"steps/s={sps:.1f} | " + " | ".join(env_lines)
                                )
                            except Exception:
                                stage(
                                    f"💜[ppo-ddv2][progress] episodes={episodes_collected}/{batch_episodes_local} "
                                    f"active={int(sum(1 for x in running if x))}/{n_envs} steps/s={sps:.1f}"
                                )
                            last_prog_t = now_t
                            last_prog_step = int(global_step)
            collect_time = time.perf_counter()-t0_collect
            stage(f"💜[ppo-ddv2] Collection finished: steps={len(rollout_rew)} episodes={episodes_collected} in {collect_time:.2f}s")

            # -------------------- Build local PPO-style tensors --------------------
            rewards = torch.tensor(rollout_rew, dtype=torch.float32, device=model_device)
            dones = torch.tensor(rollout_done, dtype=torch.float32, device=model_device)
            values = torch.stack(rollout_val).to(device=model_device, dtype=torch.float32)#实际走出的轨迹

            # Closed-loop collects full episodes; the last transition is terminal so bootstrap=0.
            last_value = torch.tensor(0.0, device=model_device, dtype=values.dtype)
            adv, ret = compute_gae(
                rewards=rewards,
                dones=dones,
                values=values,
                last_value=last_value,
                gamma=float(gamma),
                gae_lambda=float(gae_lambda),
            )
            # Normalize advantages later (after optional gather) to match global semantics.

            old_logp = torch.stack(rollout_logp_old).to(device=model_device, dtype=torch.float32)
            obs_batch = torch.cat(rollout_obs, dim=0).to(device=model_device, dtype=torch.float32)

            # If not gathering rollouts, make per-rank batch sizes equal to avoid different
            # numbers of backward sync steps across ranks.
            if ddp_enabled and (not gather_rollout):
                n_local = int(obs_batch.shape[0])
                n_min = int(_dist_all_reduce_min_int(n_local))
                if n_min <= 0:
                    raise RuntimeError("ddv2_rl_ppo got empty rollout on at least one rank")
                if n_local != n_min:
                    obs_batch = obs_batch[:n_min]
                    old_logp = old_logp[:n_min]
                    adv = adv[:n_min]
                    ret = ret[:n_min]
                    rollout_replay = rollout_replay[:n_min]

            # Gather batches across ranks (optionally including diffusion replay)
            local_pack = {
                "obs": obs_batch.detach().cpu(),
                "old_logp": old_logp.detach().cpu(),
                "adv": adv.detach().cpu(),
                "ret": ret.detach().cpu(),
                "replay": rollout_replay if gather_ddv2_replay else None,
            }
            packs = _dist_all_gather_object(local_pack) if (ddp_enabled and gather_rollout) else [local_pack]

            obs_batch = torch.cat([p["obs"] for p in packs if p["obs"] is not None], dim=0).to(device=model_device)
            old_logp = torch.cat([p["old_logp"] for p in packs if p["old_logp"] is not None], dim=0).to(device=model_device)
            adv = torch.cat([p["adv"] for p in packs if p["adv"] is not None], dim=0).to(device=model_device)
            ret = torch.cat([p["ret"] for p in packs if p["ret"] is not None], dim=0).to(device=model_device)

            if gather_ddv2_replay and (ddp_enabled and gather_rollout):
                rollout_replay = []
                for p in packs:
                    if p.get("replay") is not None:
                        rollout_replay.extend(list(p["replay"]))

            # Global (cross-rank) advantage normalization without gathering the full buffer.
            # When gather_rollout=true, all ranks have identical adv and this reduces to the same result.
            adv = normalize_advantages(adv, ddp_enabled=ddp_enabled, dist_module=dist, device=model_device)
                    
            #已经完成 batch_episodes 个 episode 采样
#NOTE 计算 advantage 和 return（GAE）
            # PPO update on DDV2 parameters
            clip_eps_ddv2 = float(train_cfg.get("clip_eps", 0.2))
            ddv2_ppo_epochs = int(train_cfg.get("epochs", 1))
            ddv2_minibatch_size = int(train_cfg.get("minibatch_size", 16))#mini-batch 是从整个 rollout buffer（按时间顺序存储的 episode 数据）里随机抽取索引,跨 episode 的,mini-batch 16 个时间步
            n = int(obs_batch.shape[0])
            idxs = np.arange(n)

            last_loss_pi = 0.0
            last_loss_v = 0.0
            last_approx_kl = 0.0
#NOTE PPO 更新循环
            # Track DDV2 param delta within update (trajectory head only)
            ddv2_params_before = None
            try:
                m = agent._agent._transfuser_model
                core = m.module if hasattr(m, "module") else m
                ddv2_params_before = torch.cat([p.detach().cpu().flatten() for p in core._trajectory_head.parameters()])
            except Exception:
                ddv2_params_before = None

            t0_opt = time.perf_counter()
            stage(f"💜[ppo-ddv2] Optimizing: samples={n}, epochs={ddv2_ppo_epochs}, minibatch={ddv2_minibatch_size}")
            if gather_rollout and (not gather_ddv2_replay):
                raise RuntimeError("ddv2_rl_ppo requires train.ddp.gather_ddv2_replay=true when train.ddp.gather_rollout=true")

            res = ddv2_ppo_update(
                agent=agent,
                value_net=value_net,
                value_optim=value_optim,
                obs_batch=obs_batch,
                old_logp=old_logp,
                adv=adv,
                ret=ret,
                replay=rollout_replay,
                device=model_device,
                ddv2_eta=float(ddv2_eta),
                ddv2_mode_idx_default=int(ddv2_mode_idx),
                clip_eps=float(clip_eps_ddv2),
                vf_coef=float(vf_coef),
                ppo_epochs=int(ddv2_ppo_epochs),
                minibatch_size=int(ddv2_minibatch_size),
                max_grad_norm=float(max_grad_norm),
                grad_accum_steps=int(grad_accum_steps),
                ddp_enabled=bool(ddp_enabled),
                world_size=int(world_size),
                rank=int(rank),
                ddp_seed=int(ddp_seed),
                update_seed=int(upd),
                replay_compute_camera_dtype=replay_compute_camera_dtype,
                replay_compute_chain_dtype=replay_compute_chain_dtype,
                # Preserve old behavior: only use DistributedSampler when we gathered a global buffer.
                use_distributed_sampler=bool(ddp_enabled and gather_rollout),
            )

            last_loss_pi = float(res.loss_pi)
            last_loss_v = float(res.loss_v)
            last_approx_kl = float(res.approx_kl)
            last_ratio_mean = float(res.ratio_mean)
            last_adv_mean = float(res.adv_mean)

            opt_time = time.perf_counter()-t0_opt
            stage(f"💜[ppo-ddv2] Optimization finished in {opt_time:.2f}s")

            ep_reward_log = float(_dist_all_reduce_mean(float(ep_reward)))
            last_loss_pi_log = float(_dist_all_reduce_mean(float(last_loss_pi)))
            last_loss_v_log = float(_dist_all_reduce_mean(float(last_loss_v)))
            last_approx_kl_log = float(_dist_all_reduce_mean(float(last_approx_kl)))

            out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
            os.makedirs(out_dir, exist_ok=True)
            ddv2_param_delta = 0.0
            if ddv2_params_before is not None:
                try:
                    m = agent._agent._transfuser_model
                    core = m.module if hasattr(m, "module") else m
                    ddv2_params_after = torch.cat([p.detach().cpu().flatten() for p in core._trajectory_head.parameters()])
                    ddv2_param_delta = float((ddv2_params_after - ddv2_params_before).abs().max().item())
                except Exception:
                    ddv2_param_delta = 0.0
            if (not ddp_enabled) or rank == 0:
                with open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8") as f:
                    f.write(
                        f"{time.time():.0f}\tupdate={upd}\tglobal_step={global_step}"
                        f"\trew_sum={ep_reward_log:.4f}\t"
                        f"loss_pi={last_loss_pi_log:.6f}\t"
                        f"loss_v={last_loss_v_log:.6f}\t"
                        f"approx_kl={last_approx_kl_log:.6f}\t"
                        f"ddv2_param_delta={ddv2_param_delta:.6e}\n"
                    )

                print(
                    f"[ddv2-ppo update {upd}/{total_updates}] steps={global_step} "
                    f"episodes={batch_episodes} rew_sum={ep_reward_log:.4f} "
                    f"loss_pi={last_loss_pi_log:.4f} loss_v={last_loss_v_log:.4f} kl={last_approx_kl_log:.4f} "
                    f"ratio={last_ratio_mean:.4f} adv={last_adv_mean:.4f} "
                    f"dP={ddv2_param_delta:.2e}"
                )

            if wb_enabled:
                comp_mean = (lambda x: (x/comp_steps) if comp_steps>0 else 0.0)
                wandb.log({
                    "update": upd,
                    "global_step": global_step,
                    "reward_sum": float(ep_reward_log),
                    "loss_pi": float(last_loss_pi_log),
                    "loss_v": float(last_loss_v_log),
                    "approx_kl": float(last_approx_kl_log),
                    "ratio_mean": float(last_ratio_mean),
                    "adv_mean": float(last_adv_mean),
                    "ddv2_param_delta": float(ddv2_param_delta),
                    "collect_time_s": float(collect_time),
                    "opt_time_s": float(opt_time),
                    "rpd_mean": float(comp_mean(comp_sum["rpd"])),
                    "rhd_mean": float(comp_mean(comp_sum["rhd"])),
                    "rsc_rate": float((static_col_steps/comp_steps) if comp_steps>0 else 0.0),
                    "rdc_rate": float((dynamic_col_steps/comp_steps) if comp_steps>0 else 0.0),
                    "jerk_pen_mean": float(comp_mean(comp_sum["jerk_pen"])),
                    "yaw_jerk_pen_mean": float(comp_mean(comp_sum["yaw_jerk_pen"])),
                    "scene_skips": int(scene_skips),
                })

            # Save updated DDV2 weights (optional)
            save_cfg = cfg.get("train", {}).get("save", {})
            save_every = int(save_cfg.get("every", 0))
            if (not ddp_enabled or rank == 0) and save_every > 0 and ((upd + 1) % save_every == 0):
                try:
                    m = agent._agent._transfuser_model
                    sd = (m.module.state_dict() if hasattr(m, "module") else m.state_dict())
                    sd_pref = {f"agent.{k}": v for k, v in sd.items()}
                    ckpt_path = os.path.join(out_dir, f"ddv2_ppo_{upd+1}.ckpt")
                    torch.save({"state_dict": sd_pref}, ckpt_path)
                    print(f"Saved DDV2 ckpt: {ckpt_path}")
                    upload_to_wandb = bool(save_cfg.get("upload_to_wandb", False))
                    if wb_enabled and upload_to_wandb:
                        try:
                            wandb.save(ckpt_path)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[WARN] Failed to save DDV2 ckpt: {e}")

        if writer is not None:
            writer.close()
            stage(f"💜Video saved: {final_video_path}")
        # Final checkpoint save to local & wandb
        try:
            out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
            os.makedirs(out_dir, exist_ok=True)
            m = agent._agent._transfuser_model
            sd = (m.module.state_dict() if hasattr(m, "module") else m.state_dict())
            sd_pref = {f"agent.{k}": v for k, v in sd.items()}
            final_ckpt = os.path.join(out_dir, "ddv2_ppo_final.ckpt")
            if (not ddp_enabled) or rank == 0:
                torch.save({"state_dict": sd_pref}, final_ckpt)
                print(f"Saved final DDV2 ckpt: {final_ckpt}")
            if wb_enabled and (not ddp_enabled or rank == 0):
                try:
                    upload_final = bool(cfg.get("train", {}).get("save", {}).get("upload_final_to_wandb", True))
                    if upload_final:
                        wandb.save(final_ckpt)
                    art = wandb.Artifact("ddv2_ppo_ckpt", type="model")
                    art.add_file(final_ckpt)
                    wandb.log_artifact(art)
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Failed final DDV2 ckpt save: {e}")
        stage("Training PPO_ddv2 finished.")
        try:
            if vec_env is not None:
                vec_env.close()
        except Exception:
            pass
        _destroy_dist()
        return
    # Should never reach here due to returns above.
    _destroy_dist()
    stage("Training finished.")


if __name__ == "__main__":
    main()
