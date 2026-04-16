from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))

#从轨迹点 (x, y) 计算每一帧的朝向 yaw
#TODO: 第一个点prev只是一个占位；
def _path_yaw_from_xy(points_xy: np.ndarray) -> np.ndarray:
    if int(points_xy.shape[0]) <= 0:
        return np.zeros((0,), dtype=np.float32)
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), points_xy[:-1]], axis=0)
    delta = points_xy - prev
    yaw = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)
    if int(yaw.shape[0]) > 1:
        yaw[0] = yaw[1]
    return yaw

#计算轨迹的累计路径长度（s）;;把一串轨迹点，变成“沿路径的累计距离（arclength）”
def _polyline_arclength(points_xy: np.ndarray) -> np.ndarray:
    if int(points_xy.shape[0]) <= 0:
        return np.zeros((0,), dtype=np.float32)  #np.zeros(shape, dtype=...)   array([], dtype=float32) ; NumPy 的 shape 必须是“元组（tuple）”，而 (0) 不是元组
    if int(points_xy.shape[0]) == 1:
        return np.zeros((1,), dtype=np.float32)
    seg = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg, dtype=np.float32)], axis=0)

#_project_progress(cand_xy[-1], gt_xy, gt_s)
def _project_progress(point_xy: np.ndarray, path_xy: np.ndarray, path_s: np.ndarray) -> float:
    if int(path_xy.shape[0]) <= 1:
        return 0.0
    best_dist = float("inf")
    best_s = 0.0
    for idx in range(int(path_xy.shape[0]) - 1):
        p0 = path_xy[idx]
        p1 = path_xy[idx + 1]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-12:
            continue
        alpha = float(np.dot(point_xy - p0, seg) / seg_len_sq)
        alpha = max(0.0, min(1.0, alpha))
        proj = p0 + alpha * seg
        dist = float(np.linalg.norm(point_xy - proj))
        if dist < best_dist:
            best_dist = dist
            best_s = float(path_s[idx] + alpha * math.sqrt(seg_len_sq))
    return best_s


class NuScenesTokenScorer:
    def __init__(self, *, token2vad_path: str | Path) -> None:
        self.token2vad_path = Path(token2vad_path)
        self._token2vad: dict[str, dict[str, Any]] | None = None

    def _ensure_loaded(self) -> dict[str, dict[str, Any]]:
        if self._token2vad is None:
            with self.token2vad_path.open("rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, dict):
                raise RuntimeError(f"token2vad file must contain a dict, got {type(loaded)!r}")
            self._token2vad = loaded
        return self._token2vad

    @staticmethod
    def _gt_to_env_xy(gt_ego_fut_trajs: np.ndarray) -> np.ndarray:
        gt = np.asarray(gt_ego_fut_trajs, dtype=np.float32)
        if gt.ndim != 2 or gt.shape[1] < 2:
            raise RuntimeError(f"Expected gt_ego_fut_trajs with shape (T, 2+), got {gt.shape}")
        # token2vad stores local future as (lateral, forward); simulator/policy uses (forward, left).
        out = np.zeros((gt.shape[0], 2), dtype=np.float32)
        out[:, 0] = gt[:, 1]
        out[:, 1] = gt[:, 0]
        return out

    def _lookup_gt(self, sample_token: str) -> np.ndarray:
        token2vad = self._ensure_loaded()
        row = token2vad.get(str(sample_token), None)
        if not isinstance(row, dict):
            raise RuntimeError(f"sample_token={sample_token!r} not found in token2vad index")
        gt = row.get("gt_ego_fut_trajs", None)
        if gt is None:
            raise RuntimeError(f"sample_token={sample_token!r} missing gt_ego_fut_trajs")
        return self._gt_to_env_xy(np.asarray(gt, dtype=np.float32))

    def _score_batch(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if traj_xyyaw.ndim != 4 or traj_xyyaw.shape[-1] < 2:
            raise RuntimeError(
                "NuScenesTokenScorer expects traj_xyyaw with shape (batch, candidates, horizon, 3); "
                f"got {tuple(traj_xyyaw.shape)}"
            )
        if len(replays) != int(traj_xyyaw.shape[0]):
            raise RuntimeError(f"Replay batch length mismatch: replays={len(replays)} traj_batch={int(traj_xyyaw.shape[0])}")

        traj_np = traj_xyyaw.detach().cpu().numpy().astype(np.float32)
        scores = np.zeros((traj_np.shape[0], traj_np.shape[1]), dtype=np.float32)
        details: list[dict[str, Any]] = []
        for batch_idx, replay in enumerate(replays):
            sample_token = replay.get("sample_token", None)
            if sample_token is None:
                raise RuntimeError("NuScenesTokenScorer requires replay['sample_token']")
            gt_xy = self._lookup_gt(str(sample_token))
            gt_yaw = _path_yaw_from_xy(gt_xy)
            gt_s = _polyline_arclength(gt_xy)#给 GT 轨迹加一个“里程表”
            gt_total_len = float(max(1e-6, gt_s[-1] if int(gt_s.shape[0]) > 0 else 1.0))

            horizon = min(int(gt_xy.shape[0]), int(traj_np.shape[2]))
            gt_xy_cmp = gt_xy[:horizon]
            gt_yaw_cmp = gt_yaw[:horizon]
            sample_detail: dict[str, Any] = {
                "batch_index": int(batch_idx),
                "sample_token": str(sample_token),
                "gt_xy": gt_xy_cmp.copy(),
                "gt_yaw": gt_yaw_cmp.copy(),
                "candidates": [],
            }
            for cand_idx in range(int(traj_np.shape[1])):
                cand = traj_np[batch_idx, cand_idx, :horizon, :]
                cand_xy = cand[:, :2]
                cand_yaw = cand[:, 2] if cand.shape[1] >= 3 else _path_yaw_from_xy(cand_xy)

                pos_err = np.linalg.norm(cand_xy - gt_xy_cmp, axis=1)
                first_err = float(pos_err[0]) if pos_err.size else 0.0
                final_err = float(pos_err[-1]) if pos_err.size else 0.0
                mean_err = float(pos_err.mean()) if pos_err.size else 0.0
                yaw_err = np.abs(_wrap_angle(cand_yaw - gt_yaw_cmp))
                mean_yaw_err = float(yaw_err.mean()) if yaw_err.size else 0.0

                cand_progress = _project_progress(cand_xy[-1], gt_xy, gt_s) if cand_xy.shape[0] > 0 else 0.0
                progress_ratio = float(np.clip(cand_progress / gt_total_len, 0.0, 1.5))

                cand_prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), cand_xy[:-1]], axis=0)
                cand_vel = cand_xy - cand_prev
                cand_acc = cand_vel[1:] - cand_vel[:-1] if cand_vel.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
                smooth_pen = float(np.linalg.norm(cand_acc, axis=1).mean()) if cand_acc.size else 0.0

                score_terms = {
                    "progress_reward": 2.0 * progress_ratio,
                    "mean_error_penalty": -0.35 * mean_err,
                    "final_error_penalty": -0.50 * final_err,
                    "first_error_penalty": -0.35 * first_err,
                    "yaw_error_penalty": -0.20 * mean_yaw_err,
                    "smoothness_penalty": -0.05 * smooth_pen,
                }
                score = float(sum(score_terms.values()))
                scores[batch_idx, cand_idx] = np.float32(score)
                sample_detail["candidates"].append(
                    {
                        "candidate_index": int(cand_idx),
                        "traj_xyyaw": cand.copy(),
                        "score": score,
                        "progress_ratio": progress_ratio,
                        "mean_error_m": mean_err,
                        "final_error_m": final_err,
                        "first_error_m": first_err,
                        "mean_yaw_error_rad": mean_yaw_err,
                        "smoothness_penalty_raw": smooth_pen,
                        "score_terms": score_terms,
                    }
                )
            details.append(sample_detail)
        return scores, details

    def score_with_details(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        scores, details = self._score_batch(replays, traj_xyyaw)
        return torch.from_numpy(scores), details

    def score(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        scores, _ = self._score_batch(replays, traj_xyyaw)
        return torch.from_numpy(scores)

    def dump_debug_artifacts(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        *,
        out_dir: str | Path,
        step_tag: str,
        top_k: int = 4,
    ) -> list[dict[str, str]]:
        import matplotlib.pyplot as plt

        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        _, details = self.score_with_details(replays, traj_xyyaw)
        artifacts: list[dict[str, str]] = []
        for sample_detail in details:
            sample_token = str(sample_detail["sample_token"])
            token_slug = sample_token.replace("/", "_")[:24]
            prefix = f"{step_tag}_b{int(sample_detail['batch_index']):03d}_{token_slug}"
            json_path = out_root / f"{prefix}.json"
            png_path = out_root / f"{prefix}.png"

            ranked = sorted(sample_detail["candidates"], key=lambda item: float(item["score"]), reverse=True)
            kept = ranked[: max(1, int(top_k))]
            payload = {
                "sample_token": sample_token,
                "step_tag": str(step_tag),
                "batch_index": int(sample_detail["batch_index"]),
                "gt_xy": np.asarray(sample_detail["gt_xy"], dtype=np.float32).tolist(),
                "gt_yaw": np.asarray(sample_detail["gt_yaw"], dtype=np.float32).tolist(),
                "candidates": [
                    {
                        "candidate_index": int(item["candidate_index"]),
                        "score": float(item["score"]),
                        "progress_ratio": float(item["progress_ratio"]),
                        "mean_error_m": float(item["mean_error_m"]),
                        "final_error_m": float(item["final_error_m"]),
                        "first_error_m": float(item["first_error_m"]),
                        "mean_yaw_error_rad": float(item["mean_yaw_error_rad"]),
                        "smoothness_penalty_raw": float(item["smoothness_penalty_raw"]),
                        "score_terms": {key: float(val) for key, val in item["score_terms"].items()},
                        "traj_xyyaw": np.asarray(item["traj_xyyaw"], dtype=np.float32).tolist(),
                    }
                    for item in kept
                ],
            }
            json_path.write_text(json.dumps(payload, indent=2))

            fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=160)
            gt_xy = np.asarray(sample_detail["gt_xy"], dtype=np.float32)
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=2.5, label="gt_path")
            ax.scatter([0.0], [0.0], color="red", marker="x", s=60, label="ego_origin")
            cmap = plt.get_cmap("viridis", max(2, len(kept)))
            for rank, item in enumerate(kept):
                cand_xyyaw = np.asarray(item["traj_xyyaw"], dtype=np.float32)
                cand_xy = cand_xyyaw[:, :2]
                label = f"rank{rank + 1} idx{int(item['candidate_index'])} score={float(item['score']):.3f}"
                ax.plot(cand_xy[:, 0], cand_xy[:, 1], color=cmap(rank), linewidth=1.8, alpha=0.95, label=label)
                ax.scatter([cand_xy[-1, 0]], [cand_xy[-1, 1]], color=cmap(rank), s=18)
            ax.set_title(f"NuScenes GRPO debug\n{sample_token}")
            ax.set_xlabel("forward x (m)")
            ax.set_ylabel("left y (m)")
            ax.axis("equal")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)

            artifacts.append(
                {
                    "sample_token": sample_token,
                    "json_path": str(json_path),
                    "png_path": str(png_path),
                }
            )
        return artifacts


__all__ = [
    "NuScenesTokenScorer",
]
