import os
import sys
import time
from typing import Any, Dict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
# Simple stage logger
def stage(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
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
from rl.ppo import PPOAgent
from rl.policy_diffusiondrivev2 import DiffusionDriveV2Policy
from rl.ppo import PPOBatch
from rl.ppo import _obs_to_tensor as obs_to_tensor
from reconsimulator.envs import nus_config as nus_cfg


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "ppo_closed_loop.yaml"))
    # Optional per-process suffix for outputs and wandb run name
    run_suffix = os.environ.get("RUN_SUFFIX", "").strip()
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
    cuda = int(env_cfg.get("cuda", 0))
    scene = int(env_cfg.get("scene", 0))
    # ---- Discover available scene ids (optional) ----
    use_all_scenes = bool(env_cfg.get("use_all_scenes", True))
    scene_sampling = str(env_cfg.get("scene_sampling", "random")).lower()  # random | sequential
    require_ckpt = bool(env_cfg.get("require_ckpt", True))

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
            return int(np.random.choice(_scene_ids))

    # Track skipped scenes due to missing checkpoints
    scene_skips: int = 0

    # Helper: create env with a valid scene, skipping missing ones
    def _safe_create_env() -> tuple[RLReconEnv, Dict[str, np.ndarray], Dict[str, Any], int]:
        nonlocal scene_skips
        max_attempts = max(1, len(_scene_ids))
        attempts = 0
        while attempts < max_attempts:
            sid = _next_scene_id()
            stage(f"Init scene candidate: {sid}")
            try:
                env_local = RLReconEnv(cuda=cuda, scene=sid, reward_cfg=reward_cfg, debug=debug)
                obs_local, info_local = env_local.reset(scene=sid)
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
        max_attempts = max(1, len(_scene_ids))
        attempts = 0
        while attempts < max_attempts:
            sid = _next_scene_id()
            try:
                obs_local, info_local = env.reset(scene=sid)
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
            device=f"cuda:{cuda}",
            rl_lr=float(cfg.get("train", {}).get("ddv2_lr", 1e-5)),
            reinforce_baseline_beta=float(cfg.get("train", {}).get("ddv2_baseline_beta", 0.98)),
        )

    train_cfg = cfg.get("train", {})
    algo = str(train_cfg.get("algo", "ppo"))
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
        ppo = PPOAgent(
            x_anchor=x_anchor,
            y_anchor=y_anchor,
            device=f"cuda:{cuda}" if torch.cuda.is_available() else "cpu",
            lr=lr,
            lr_value=lr_value,
            clip_eps=clip_eps,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            guidance_weight=guidance_weight,
            guidance_sigma=guidance_sigma,
        )

        if ddv2 is not None:
            ddv2.set_rl_agent(ppo)
            agent = ddv2
        else:
            agent = ppo

    max_steps = int(env_cfg.get("max_steps", 200))

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
    if wb_enabled:
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
    fps = int(train_cfg.get("fps", 10))
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

    if save_video:
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
            stage(f"💜[reinforce] Collecting horizon={horizon} steps")
            for t in range(horizon):
                action, logp = agent.step_ddv2rl(obs, eta=1.0)
                obs, reward, terminated, truncated, info = env.step(action)
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

                if episode_reward_mode:
                    ep_logps.append(logp)
                else:
                    m = agent.reinforce_update(logp, float(reward))
                    losses.append(float(m.get("loss_reinforce", 0.0)))
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
                        ep_reward_scalar = float(info.get("episode_reward", 0.0))
                        ep_reward += ep_reward_scalar
                        if len(ep_logps) > 0:
                            ep_logp_sum = torch.stack(ep_logps, dim=0).sum()
                            m = agent.reinforce_update(ep_logp_sum, ep_reward_scalar)
                            losses.append(float(m.get("loss_reinforce", 0.0)))
                        ep_logps = []
                    # Switch to next scene for next episode
                    obs, info, _ = _safe_reset_env()
                    steps_in_episode = 0
            collect_time = time.perf_counter()-t0_collect
            stage(f"💜[reinforce] Collection finished in {collect_time:.2f}s")

            out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
            os.makedirs(out_dir, exist_ok=True)

            # param_delta_max = 0.0
            # if p_before is not None:
            #     try:
            #         p_after = next(p for p in agent._agent.parameters() if getattr(p, "requires_grad", False)).detach().cpu()
            #         param_delta_max = float((p_after - p_before).abs().max().item())
            #     except Exception:
            #         param_delta_max = 0.0

            with open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8") as f:
                f.write(
                    f"{time.time():.0f}\tupdate={upd}\tglobal_step={global_step}"
                    f"\trew_sum={ep_reward:.4f}\t"
                    f"loss_reinforce={float(np.mean(losses)):.6f}\t"
                    # f"grad_norm={float(np.mean(grad_norms)):.6f}\t"
                    # f"param_delta_max={param_delta_max:.6e}\n"
                )
            print(
                f"[ddv2-rl update {upd}/{total_updates}] steps={global_step} "
                f"rew_sum={ep_reward:.4f} loss={float(np.mean(losses)):.4f} "
                # f"g={float(np.mean(grad_norms)):.2e} dP={param_delta_max:.2e}"
            )
            if wb_enabled:
                comp_mean = (lambda x: (x/comp_steps) if comp_steps>0 else 0.0)
                wandb.log({
                    "update": upd,
                    "global_step": global_step,
                    "reward_sum": float(ep_reward),
                    "loss_reinforce": float(np.mean(losses) if len(losses) else 0.0),
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
            if save_every > 0 and ((upd + 1) % save_every == 0):
                os.makedirs(out_dir, exist_ok=True)
                try:
                    sd = agent._agent._transfuser_model.state_dict()
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
            sd = agent._agent._transfuser_model.state_dict()
            sd_pref = {f"agent.{k}": v for k, v in sd.items()}
            final_ckpt = os.path.join(out_dir, "ddv2_reinforce_final.ckpt")
            torch.save({"state_dict": sd_pref}, final_ckpt)
            print(f"Saved final DDV2 ckpt: {final_ckpt}")
            if wb_enabled:
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
        ddv2_mode_idx = int(train_cfg.get("ddv2_mode_idx", 0))#多模态轨迹的模式索引
        max_grad_norm = float(train_cfg.get("ddv2_max_grad_norm", 0.5))#梯度裁剪阈值

        batch_episodes = int(train_cfg.get("batch_episodes", 2))#每次 update 用多少完整 episode
        vf_coef = float(train_cfg.get("vf_coef", 0.5))#这个是valueNet的损失系数 value loss 权重

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
            rollout_obs.clear()
            rollout_replay.clear()
            rollout_logp_old.clear()
            rollout_val.clear()
            rollout_rew.clear()
            rollout_done.clear()

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
            stage(f"💜[ppo-ddv2] Collecting {batch_episodes} episodes...")
            comp_sum = {"rpd":0.0, "rhd":0.0, "rsc":0.0, "rdc":0.0, "jerk_pen":0.0, "yaw_jerk_pen":0.0}
            comp_steps = 0
            static_col_steps = 0
            dynamic_col_steps = 0
            while episodes_collected < batch_episodes:#cfg=2
                # Observation tensor for critic
                obs_t = obs_to_tensor(obs, model_device)  # (1,18,64,64)
                value_net.eval()
                with torch.no_grad():
                    v = value_net(obs_t).squeeze(0)
                #new_logp_vec、logp 对应的是 整条轨迹的 log probability 和，而不是单步
                #ddv2_mode_idx=-1 ddv2_eta=1.0
                #NOTE 采样动作，记录数据(该环境下原始model生成的一些列轨迹以及对应的去噪过程)
                '''
                存的东西：
                1)cacera feature 原始 observation 经 encoder 前的视觉输入;存 已经 preprocess 好的 feature
                2)diffusion 每一步的 latent / noise / x_t;采样路径的随机性来源
                
                
                '''
                #这个obs都是图片
                action, logp, replay = agent.sample_ddv2rl_with_replay(obs, eta=ddv2_eta, mode_idx=ddv2_mode_idx)
                rollout_obs.append(obs_t.detach().cpu())
                rollout_val.append(v.detach().cpu())
                # 将 replay 字典中所有张量字段转为 CPU 并分离梯度
                if isinstance(replay, dict):
                    replay_cpu = {
                        k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in replay.items()
                    }
                else:
                    # 兼容旧返回格式：若不是字典，则尝试按张量处理
                    replay_cpu = replay.detach().cpu() if torch.is_tensor(replay) else replay
                rollout_replay.append(replay_cpu)
                rollout_logp_old.append(logp.detach().cpu())
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
                global_step += 1#8*2*50=800训练的总步数

                if save_video and writer is not None and video_frames_written < max_video_frames:
                    writer.append_data(_grid_frame(obs, info))
                    video_frames_written += 1
                #NOTE episode 完成或超时处理;
                # 使用每一步的reward加和(advantage 逐步累积，反映每一步的即时贡献)
                # 还是最终finalize_episode_reward(env.finalize_episode_reward() 返回总 reward)
                # 如果环境有 finalize_episode_reward 并且 episode_reward_mode=True，
                # 那么整条 episode 的每一步 reward 都被替换成 total reward。
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
                    stage(f"💜[ppo-ddv2] Episode {episodes_collected+1} done: steps={steps_in_episode} r_sum={ep_reward:.4f} reason={str(dr) if dr is not None else 'n/a'}")
                    #NOTE r0 = r1 = r2 = ... = rT = R_episode
                    if episode_reward_mode and "episode_reward" in info:
                        ep_r = float(info.get("episode_reward", 0.0))
                        ep_reward += ep_r
                        for j in range(int(ep_start_idx), len(rollout_rew)):
                            rollout_rew[j] = ep_r
                        ep_start_idx = len(rollout_rew)
                        #episode 1: t=0..49  保证 reward 替换只作用于本 episode，不会影响后续 episode
                        # episode 2: t=50..99
                        
                    #NOTE标准的RL reward 
                    else:
                        # fallback: sum raw rewards G=r0 + r1 + ... + rT(gamma=1此时)
                        ep_reward += float(np.sum(rollout_rew[int(ep_start_idx) :]))
                        ep_start_idx = len(rollout_rew)

                    episodes_collected += 1
                    obs, info, _ = _safe_reset_env()
                    steps_in_episode = 0
            collect_time = time.perf_counter()-t0_collect
            stage(f"💜[ppo-ddv2] Collection finished: steps={len(rollout_rew)} episodes={episodes_collected} in {collect_time:.2f}s")
                    
            #已经完成 batch_episodes 个 episode 采样
#NOTE 计算 advantage 和 return（GAE）
            # Prepare tensors
            rewards = torch.tensor(rollout_rew, dtype=torch.float32, device=model_device)
            dones = torch.tensor(rollout_done, dtype=torch.float32, device=model_device)
            values = torch.stack(rollout_val).to(device=model_device, dtype=torch.float32)

            adv = torch.zeros_like(rewards)
            last_gae = torch.tensor(0.0, device=model_device)
            for t in reversed(range(len(rewards))):
                mask = 1.0 - dones[t]
                v_next = torch.tensor(0.0, device=model_device) if t == len(rewards) - 1 else values[t + 1]
                delta = rewards[t] + gamma * v_next * mask - values[t]
                last_gae = delta + gamma * gae_lambda * mask * last_gae
                adv[t] = last_gae
            ret = adv + values
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            old_logp = torch.stack(rollout_logp_old).to(device=model_device, dtype=torch.float32)
            obs_batch = torch.cat(rollout_obs, dim=0).to(device=model_device, dtype=torch.float32)

            # PPO update on DDV2 parameters
            clip_eps_ddv2 = float(train_cfg.get("clip_eps", 0.2))
            ddv2_ppo_epochs = int(train_cfg.get("epochs", 1))
            ddv2_minibatch_size = int(train_cfg.get("minibatch_size", 16))#mini-batch 是从整个 rollout buffer（按时间顺序存储的 episode 数据）里随机抽取索引,跨 episode 的,mini-batch 16 个时间步
            n = len(rollout_rew)
            idxs = np.arange(n)

            last_loss_pi = 0.0
            last_loss_v = 0.0
            last_approx_kl = 0.0
#NOTE PPO 更新循环
            # Track DDV2 param delta within update (trajectory head only)
            ddv2_params_before = None
            try:
                ddv2_params_before = torch.cat([
                    p.detach().cpu().flatten() for p in agent._agent._transfuser_model._trajectory_head.parameters()
                ])
            except Exception:
                ddv2_params_before = None

            t0_opt = time.perf_counter()
            stage(f"💜[ppo-ddv2] Optimizing: samples={n}, epochs={ddv2_ppo_epochs}, minibatch={ddv2_minibatch_size}")
            for _ in range(ddv2_ppo_epochs):
                np.random.shuffle(idxs)#PPO 是 小批量随机梯度下降mini-batch SGD
                for start in range(0, n, ddv2_minibatch_size):
                    mb_idx = idxs[start : start + ddv2_minibatch_size]
                    mb_idx_t = torch.tensor(mb_idx, dtype=torch.long)

                    # Batch replays
                    cam = torch.cat([rollout_replay[i]["camera_feature"] for i in mb_idx], dim=0)
                    chain = torch.cat([rollout_replay[i]["diffusion_chain"] for i in mb_idx], dim=0)
                    # 每条样本各自的模式索引
                    mb_mode_idx = torch.as_tensor(
                        [int(rollout_replay[i].get("mode_idx", ddv2_mode_idx)) for i in mb_idx],
                        dtype=torch.long,
                        device=model_device,
                    )

                    # Recompute logp under current params (vectorized over batch)
                    features = {
                        "camera_feature": cam.to(model_device),
                        "lidar_feature": torch.zeros((cam.shape[0], 1, 256, 256), dtype=torch.float32, device=model_device),
                        "status_feature": torch.zeros((cam.shape[0], 8), dtype=torch.float32, device=model_device),
                    }
                    #重新计算 log probability（在当前参数下）
                    all_logps = agent._agent._transfuser_model.compute_log_probs_from_diffusion_chain(
                        features,
                        chain.to(model_device),
                        eta=float(ddv2_eta),
                    )
                    # 逐样本按其各自的 mode_idx 选择对应模式的 logp
                    # mb_mode_idx 指明每条样本在 diffusion chain 中实际使用的模式
                    bsz = cam.shape[0]
                    sel = all_logps[torch.arange(bsz, device=model_device), mb_mode_idx, :]#PPO loss 只关心实际选择的那个模式
                    new_logp_vec = sel.sum(dim=-1).to(dtype=torch.float32)#把轨迹序列上每一步的 log probability 累加，得到整条轨迹的 log probability

                    mb_idx_t = mb_idx_t.to(device=model_device)
                    old_logp_mb = old_logp[mb_idx_t]
                    adv_mb = adv[mb_idx_t]#GAE_mini_batch
                    ret_mb = ret[mb_idx_t]#return_mini_batch

                    v_pred = value_net(obs_batch[mb_idx_t])
                    loss_v = F.mse_loss(v_pred, ret_mb)

                    ratio = torch.exp(new_logp_vec - old_logp_mb)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps_ddv2, 1.0 + clip_eps_ddv2) * adv_mb
                    loss_pi = -(torch.min(surr1, surr2)).mean()

                    loss = loss_pi + vf_coef * loss_v

                    approx_kl = (old_logp_mb - new_logp_vec).mean().detach()

                    agent._ddv2_optimizer.zero_grad(set_to_none=True)
                    value_optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in agent._ddv2_optimizer.param_groups[0]["params"] if p.grad is not None],
                        max_grad_norm,
                    )
                    agent._ddv2_optimizer.step()
                    value_optim.step()

                    last_loss_pi = float(loss_pi.detach().cpu().item())
                    last_loss_v = float(loss_v.detach().cpu().item())
                    last_approx_kl = float(approx_kl.cpu().item())
                    last_ratio_mean = float(ratio.detach().cpu().mean().item())
                    last_adv_mean = float(adv_mb.detach().cpu().mean().item())
                    opt_time = time.perf_counter()-t0_opt
                    stage(f"💜[ppo-ddv2] Optimization finished in {opt_time:.2f}s")

            out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
            os.makedirs(out_dir, exist_ok=True)
            ddv2_param_delta = 0.0
            if ddv2_params_before is not None:
                try:
                    ddv2_params_after = torch.cat([
                        p.detach().cpu().flatten() for p in agent._agent._transfuser_model._trajectory_head.parameters()
                    ])
                    ddv2_param_delta = float((ddv2_params_after - ddv2_params_before).abs().max().item())
                except Exception:
                    ddv2_param_delta = 0.0
            with open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8") as f:
                f.write(
                    f"{time.time():.0f}\tupdate={upd}\tglobal_step={global_step}"
                    f"\trew_sum={ep_reward:.4f}\t"
                    f"loss_pi={last_loss_pi:.6f}\t"
                    f"loss_v={last_loss_v:.6f}\t"
                    f"approx_kl={last_approx_kl:.6f}\t"
                    f"ddv2_param_delta={ddv2_param_delta:.6e}\n"
                )

            print(
                f"[ddv2-ppo update {upd}/{total_updates}] steps={global_step} "
                f"episodes={batch_episodes} rew_sum={ep_reward:.4f} "
                f"loss_pi={last_loss_pi:.4f} loss_v={last_loss_v:.4f} kl={last_approx_kl:.4f} "
                f"ratio={last_ratio_mean:.4f} adv={last_adv_mean:.4f} "
                f"dP={ddv2_param_delta:.2e}"
            )

            if wb_enabled:
                comp_mean = (lambda x: (x/comp_steps) if comp_steps>0 else 0.0)
                wandb.log({
                    "update": upd,
                    "global_step": global_step,
                    "reward_sum": float(ep_reward),
                    "loss_pi": float(last_loss_pi),
                    "loss_v": float(last_loss_v),
                    "approx_kl": float(last_approx_kl),
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
            if save_every > 0 and ((upd + 1) % save_every == 0):
                try:
                    sd = agent._agent._transfuser_model.state_dict()
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
            sd = agent._agent._transfuser_model.state_dict()
            sd_pref = {f"agent.{k}": v for k, v in sd.items()}
            final_ckpt = os.path.join(out_dir, "ddv2_ppo_final.ckpt")
            torch.save({"state_dict": sd_pref}, final_ckpt)
            print(f"Saved final DDV2 ckpt: {final_ckpt}")
            if wb_enabled:
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
        return

    # PPO rollout buffer (stores already-preprocessed obs tensors)
    rollout_obs = []
    rollout_ax = []
    rollout_ay = []
    rollout_logp = []
    rollout_val = []
    rollout_rew = []
    rollout_done = []
    ep_start_idx = 0

    def _flush_and_update(next_obs: Dict[str, np.ndarray]) -> Dict[str, float]:
        # Compute GAE and returns
        with torch.no_grad():
            next_v = torch.tensor(agent.value(next_obs), dtype=torch.float32)

        values = torch.stack(rollout_val)  # (T,)
        rewards = torch.tensor(rollout_rew, dtype=torch.float32)
        dones = torch.tensor(rollout_done, dtype=torch.float32)

        adv = torch.zeros_like(rewards)
        last_gae = torch.tensor(0.0)
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            v_next = next_v if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * v_next * mask - values[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            adv[t] = last_gae
        ret = adv + values

        batch = PPOBatch(
            obs=torch.cat(rollout_obs, dim=0).detach().cpu(),
            act_x=torch.tensor(rollout_ax, dtype=torch.int64),
            act_y=torch.tensor(rollout_ay, dtype=torch.int64),
            logp=torch.stack(rollout_logp).detach().cpu(),
            adv=adv.detach().cpu(),
            ret=ret.detach().cpu(),
        )
        return agent.update(batch)

    # Training updates
    global_step = 0
    for upd in range(total_updates):
        stage(f"[ppo] Update {upd+1}/{total_updates} start")
        rollout_obs.clear()
        rollout_ax.clear()
        rollout_ay.clear()
        rollout_logp.clear()
        rollout_val.clear()
        rollout_rew.clear()
        rollout_done.clear()

        ep_start_idx = 0

        ep_reward = 0.0
        steps_in_episode = 0

        # Collect horizon steps (may span multiple episodes)
        t0_collect = time.perf_counter()
        stage(f"[ppo] Collecting rollout: target horizon={horizon}")
        comp_sum = {"rpd":0.0, "rhd":0.0, "rsc":0.0, "rdc":0.0, "jerk_pen":0.0, "yaw_jerk_pen":0.0}
        comp_steps = 0
        static_col_steps = 0
        dynamic_col_steps = 0
        while len(rollout_rew) < horizon:
            action, logp, v, ent = agent.step(obs, sample=True)

            # Save current obs (as tensor) for training
            rollout_obs.append(obs_to_tensor(obs, agent.device).detach().cpu())
            rollout_ax.append(int(action[0]))
            rollout_ay.append(int(action[1]))
            rollout_logp.append(logp.detach().cpu())
            rollout_val.append(v.detach().cpu())

            obs, reward, terminated, truncated, info = env.step(action)
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

            rollout_rew.append(float(reward))
            rollout_done.append(1.0 if done else 0.0)

            ep_reward += float(reward)
            steps_in_episode += 1
            if steps_in_episode == 1:
                stage("[ppo] Episode started")
            if steps_in_episode % 50 == 0:
                stage(f"[ppo] Episode progress: {steps_in_episode} steps")
            global_step += 1

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
                stage(f"[ppo] Episode done: steps={steps_in_episode} r_sum={ep_reward:.4f} reason={str(dr) if dr is not None else 'n/a'}")
                if episode_reward_mode and "episode_reward" in info:
                    ep_r = float(info.get("episode_reward", 0.0))
                    start = int(ep_start_idx)
                    for j in range(start, len(rollout_rew)):
                        rollout_rew[j] = ep_r
                    ep_start_idx = len(rollout_rew)
                # reset episode
                obs, info, _ = _safe_reset_env()
                steps_in_episode = 0
        collect_time = time.perf_counter()-t0_collect
        stage(f"[ppo] Collection finished: steps={len(rollout_rew)} in {collect_time:.2f}s")

        assert_update = bool(train_cfg.get("assert_update", True))
        w_before = None
        if assert_update:
            try:
                # Works whether agent is PPOAgent or DiffusionDriveV2Policy (attached PPO).
                model = agent._rl.model if hasattr(agent, "_rl") and agent._rl is not None else agent.model
                w_before = model.pi_x.weight.detach().clone().cpu()
            except Exception:
                w_before = None

        stage("[ppo] Computing GAE/returns and updating...")
        t0_update = time.perf_counter()
        metrics = _flush_and_update(obs)
        update_time = time.perf_counter()-t0_update
        stage(f"[ppo] Update finished in {update_time:.2f}s")

        if assert_update and w_before is not None:
            try:
                model = agent._rl.model if hasattr(agent, "_rl") and agent._rl is not None else agent.model
                w_after = model.pi_x.weight.detach().cpu()
                delta = float((w_after - w_before).abs().max().item())
                metrics["param_delta_max"] = delta
            except Exception:
                pass

        out_dir = cfg.get("train", {}).get("out_dir", "outputs/ppo_closed_loop")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8") as f:
            f.write(
                f"{time.time():.0f}\tupdate={upd}\tglobal_step={global_step}"
                f"\trew_sum={ep_reward:.4f}\t"
                f"loss_pi={metrics.get('loss_pi', 0.0):.6f}\t"
                f"loss_v={metrics.get('loss_v', 0.0):.6f}\t"
                f"entropy={metrics.get('entropy', 0.0):.6f}\t"
                f"approx_kl={metrics.get('approx_kl', 0.0):.6f}\n"
            )

        print(
            f"[update {upd}/{total_updates}] steps={global_step} "
            f"loss_pi={metrics.get('loss_pi', 0.0):.4f} "
            f"loss_v={metrics.get('loss_v', 0.0):.4f} "
            f"ent={metrics.get('entropy', 0.0):.4f} "
            f"kl={metrics.get('approx_kl', 0.0):.4f} "
            f"dW={metrics.get('param_delta_max', 0.0):.2e}"
        )
    
        if wb_enabled:
            comp_mean = (lambda x: (x/comp_steps) if comp_steps>0 else 0.0)
            wandb.log({
                "update": upd,
                "global_step": global_step,
                "reward_sum": float(ep_reward),
                "loss_pi": float(metrics.get('loss_pi', 0.0)),
                "loss_v": float(metrics.get('loss_v', 0.0)),
                "approx_kl": float(metrics.get('approx_kl', 0.0)),
                "collect_time_s": float(collect_time),
                "update_time_s": float(update_time),
                "rpd_mean": float(comp_mean(comp_sum["rpd"])),
                "rhd_mean": float(comp_mean(comp_sum["rhd"])),
                "rsc_rate": float((static_col_steps/comp_steps) if comp_steps>0 else 0.0),
                "rdc_rate": float((dynamic_col_steps/comp_steps) if comp_steps>0 else 0.0),
                "jerk_pen_mean": float(comp_mean(comp_sum["jerk_pen"])),
                "yaw_jerk_pen_mean": float(comp_mean(comp_sum["yaw_jerk_pen"])),
                "scene_skips": int(scene_skips),
            })
    if writer is not None:
        writer.close()
        stage(f"Video saved: {final_video_path}")

    stage("Training finished.")


if __name__ == "__main__":
    main()
