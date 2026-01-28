import argparse
import os
import sys
from typing import Any, Dict, Tuple, Optional

import yaml
import torch
import numpy as np

# Ensure repo root on path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rl.policy_diffusiondrivev2 import DiffusionDriveV2Policy


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _summarize_tensors(obj: Any, *, prefix: str = "") -> Tuple[int, int, list[str]]:
    """Return (total_bytes, tensor_count, lines). Handles nested dict/list/tuple."""
    total = 0
    count = 0
    lines: list[str] = []

    if torch.is_tensor(obj):
        b = _tensor_nbytes(obj)
        total += b
        count += 1
        lines.append(
            f"{prefix}Tensor shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device} nbytes={_human_bytes(b)}"
        )
        return total, count, lines

    if isinstance(obj, dict):
        for k, v in obj.items():
            b, c, l = _summarize_tensors(v, prefix=f"{prefix}{k}." if prefix else f"{k}.")
            total += b
            count += c
            lines.extend(l)
        return total, count, lines

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            b, c, l = _summarize_tensors(v, prefix=f"{prefix}[{i}].")
            total += b
            count += c
            lines.extend(l)
        return total, count, lines

    return 0, 0, []


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect replay tensor sizes: camera_feature & diffusion_chain")
    ap.add_argument(
        "--config",
        type=str,
        default=os.path.join(_REPO_ROOT, "script", "configs", "ppo_closed_loop.yaml"),
        help="Path to config yaml (default: script/configs/ppo_closed_loop.yaml)",
    )
    ap.add_argument("--scene", type=int, default=None, help="Override env.scene")
    ap.add_argument("--cuda", type=int, default=None, help="Override env.cuda")
    ap.add_argument("--start-frame", type=int, default=None, help="Start frame for env.reset")
    ap.add_argument("--eta", type=float, default=None, help="Override train.ddv2_eta")
    ap.add_argument("--mode-idx", type=int, default=None, help="Override train.ddv2_mode_idx")
    ap.add_argument("--samples", type=int, default=1, help="How many steps to sample")
    ap.add_argument("--no-step", action="store_true", help="Only reset+sample once (do not env.step in a loop)")

    ap.add_argument(
        "--cast-camera-dtype",
        type=str,
        default="none",
        choices=["none", "fp16", "fp32", "bf16"],
        help="Optionally cast replay['camera_feature'] before size accounting (simulates storage dtype)",
    )

    # Avoid renderer/nvdiffrast JIT build by bypassing RLReconEnv.
    ap.add_argument(
        "--synthetic-obs",
        action="store_true",
        help="Use random uint8 camera images as observation (bypass RL env reset/render)",
    )
    ap.add_argument(
        "--obs-h",
        type=int,
        default=450,
        help="Synthetic observation image height (default 450)",
    )
    ap.add_argument(
        "--obs-w",
        type=int,
        default=800,
        help="Synthetic observation image width (default 800)",
    )
    ap.add_argument(
        "--direct-camera-feature",
        action="store_true",
        help="Bypass _build_camera_feature/cv2: directly generate camera_feature (1,3,256,1024) and run DDV2 forward",
    )

    args = ap.parse_args()

    def _parse_dtype(s: str) -> Optional[torch.dtype]:
        ss = str(s).strip().lower()
        if ss in {"none", ""}:
            return None
        if ss in {"fp16", "float16", "half"}:
            return torch.float16
        if ss in {"fp32", "float32", "float"}:
            return torch.float32
        if ss in {"bf16", "bfloat16"}:
            return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {s}")

    cast_cam_dtype = _parse_dtype(args.cast_camera_dtype)

    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("env", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    agent_cfg = cfg.get("agent", {}) or {}

    scene = int(args.scene if args.scene is not None else env_cfg.get("scene", 0))
    cuda = int(args.cuda if args.cuda is not None else env_cfg.get("cuda", 0))
    start_frame = args.start_frame

    ckpt_path = str(agent_cfg.get("ckpt"))
    if not ckpt_path:
        raise RuntimeError("agent.ckpt is empty")

    eta = float(args.eta if args.eta is not None else train_cfg.get("ddv2_eta", 1.0))
    mode_idx = int(args.mode_idx if args.mode_idx is not None else train_cfg.get("ddv2_mode_idx", 0))

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    def _to_cpu(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_cpu(v) for v in obj)
        return obj

    def _make_synth_obs() -> Dict[str, np.ndarray]:
        h = int(args.obs_h)
        w = int(args.obs_w)
        rng = np.random.RandomState(0)
        def img() -> np.ndarray:
            return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        # Provide all keys RL env would provide.
        return {
            "front": img(),
            "front_left": img(),
            "front_right": img(),
            "back_left": img(),
            "back_right": img(),
            "back": img(),
        }

    # For sizing, anchors don't matter much; keep defaults.
    x_anchor = 61
    y_anchor = 61

    policy = DiffusionDriveV2Policy(
        x_anchor=x_anchor,
        y_anchor=y_anchor,
        ckpt_path=ckpt_path,
        device=str(device),
        rl_lr=float(train_cfg.get("ddv2_lr", 1e-5)),
        reinforce_baseline_beta=float(train_cfg.get("ddv2_baseline_beta", 0.98)),
    )

    print("==== Inspect replay sizes ====")
    print(f"scene={scene} cuda={cuda} device={device} start_frame={start_frame}")
    print(f"ddv2_eta={eta} ddv2_mode_idx={mode_idx}")
    print(f"cast_camera_dtype={cast_cam_dtype if cast_cam_dtype is not None else 'none'}")
    if args.synthetic_obs:
        print(f"obs=synthetic uint8 ({int(args.obs_h)}x{int(args.obs_w)})")
    if args.direct_camera_feature:
        print("mode=direct-camera-feature")

    total_cam = 0
    total_chain = 0

    # Observation source
    obs: Optional[Dict[str, np.ndarray]] = None
    if args.synthetic_obs:
        obs = _make_synth_obs()

    for i in range(int(args.samples)):
        with torch.no_grad():
            if args.direct_camera_feature:
                # Generate camera_feature directly and run DDV2 forward.
                cam_gpu = torch.rand((1, 3, 256, 1024), dtype=torch.float32, device=device)
                features = {
                    "camera_feature": cam_gpu,
                    "lidar_feature": torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=device),
                    "status_feature": torch.zeros((1, 8), dtype=torch.float32, device=device),
                }
                model = policy._agent._transfuser_model
                pred = model(
                    features,
                    targets=None,
                    eta=float(eta),
                    metric_cache=None,
                    cal_pdm=False,
                    token=None,
                )
                diffusion_chain = pred.get("all_diffusion_output", None)
                cam = cam_gpu.detach().cpu()
                if cast_cam_dtype is not None and cam.is_floating_point():
                    cam = cam.to(dtype=cast_cam_dtype)
                if diffusion_chain is None:
                    raise KeyError(
                        "DDV2 forward output missing 'all_diffusion_output'. "
                        "If the model API changed, update tools/inspect_replay_sizes.py accordingly."
                    )
                chain = _to_cpu(diffusion_chain)
                action = (0.0, 0.0, 0.0, 2)
                logp = torch.tensor(0.0)
            else:
                if obs is None:
                    # Lazy-import RL env only if needed (renderer may trigger nvdiffrast build).
                    from reconsimulator.envs.rl_wrapper import RLReconEnv
                    env = RLReconEnv(cuda=cuda, scene=scene, reward_cfg=env_cfg.get("reward", {}), debug=bool(env_cfg.get("debug", False)))
                    obs, info = env.reset(scene=scene, start_frame=start_frame)
                    try:
                        x_anchor = getattr(env.env, "x_anchor", 61)
                        y_anchor = getattr(env.env, "y_anchor", 61)
                    except Exception:
                        x_anchor = 61
                        y_anchor = 61
                action, logp, replay = policy.sample_ddv2rl_with_replay(obs, eta=eta, mode_idx=mode_idx)
                cam = replay.get("camera_feature", None) if isinstance(replay, dict) else None
                chain = replay.get("diffusion_chain", None) if isinstance(replay, dict) else None

                if cast_cam_dtype is not None and torch.is_tensor(cam) and cam.is_floating_point():
                    cam = cam.to(dtype=cast_cam_dtype)
                    replay = dict(replay)
                    replay["camera_feature"] = cam

        print(f"\n-- sample {i} --")
        print(f"action={action} logp={float(logp.detach().cpu().item()) if torch.is_tensor(logp) else logp}")

        if cam is None:
            print("camera_feature: <missing>")
        else:
            b_cam, c_cam, lines_cam = _summarize_tensors(cam)
            print(f"camera_feature: tensors={c_cam} total={_human_bytes(b_cam)}")
            for ln in lines_cam:
                print("  ", ln)
            total_cam += b_cam

        if chain is None:
            print("diffusion_chain: <missing>")
        else:
            b_chain, c_chain, lines_chain = _summarize_tensors(chain)
            print(f"diffusion_chain: tensors={c_chain} total={_human_bytes(b_chain)}")
            # diffusion_chain can be huge; only print first 40 tensor lines.
            for ln in lines_chain[:40]:
                print("  ", ln)
            if len(lines_chain) > 40:
                print(f"  ... ({len(lines_chain)-40} more tensor entries)")
            total_chain += b_chain

        if args.no_step:
            continue

        # step env to move forward (only when using real env)
        if (not args.synthetic_obs) and (not args.direct_camera_feature):
            try:
                obs, reward, terminated, truncated, info = env.step(action)  # type: ignore[name-defined]
                if bool(terminated or truncated):
                    obs, info = env.reset(scene=scene, start_frame=start_frame)  # type: ignore[name-defined]
            except Exception as e:
                print(f"[WARN] env.step/reset failed: {e}")
                break

    if int(args.samples) > 0:
        print("\n==== Summary ====")
        print(f"avg camera_feature per step:  {_human_bytes(int(total_cam / int(args.samples)))}")
        print(f"avg diffusion_chain per step: {_human_bytes(int(total_chain / int(args.samples)))}")
        print(f"avg replay total per step:    {_human_bytes(int((total_cam + total_chain) / int(args.samples)))}")


if __name__ == "__main__":
    main()
