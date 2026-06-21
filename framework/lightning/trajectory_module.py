from __future__ import annotations

import math
import os
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F

from framework.algorithms.trajectory_policy_core import (
    agent_logp_from_replay_batch,
    compute_grpo_objective,
    compute_ppo_metrics,
    compute_ppo_objective,
    compute_reinforce_metrics,
    compute_reinforce_objective,
    compute_risk_decel_auxiliary_objective,
    compute_sac_objective,
    score_counterfactual_trajectories,
)
from framework.lightning.config import ActorLearnerLightningConfig
from framework.lightning_compat import L
from framework.replay_schema import get_front_obstacle_aux, get_policy_model_inputs
from framework.runner.logging import _exception_is_cuda_oom, log_cuda_memory_snapshot


def _trainable_parameters(module: Any) -> list[torch.nn.Parameter]:
    if module is None or not hasattr(module, "parameters"):
        return []
    return [param for param in module.parameters() if getattr(param, "requires_grad", False)]


def _use_agent_value_features(agent: Any, value_net: Any, batch: Dict[str, Any]) -> bool:
    if value_net is None:
        return False
    value_module = getattr(value_net, "module", value_net)
    if bool(getattr(value_module, "expects_value_features", False)):
        return True
    if "obs" in batch:
        return False
    feature_fn = getattr(agent, "value_features_from_replay_batch", None)
    return callable(feature_fn)


def _maybe_compute_distillation_metrics(
    agent: Any,
    replay: list[Dict[str, Any]],
    *,
    device: torch.device,
    temperature: float,
    forward_kl_coef: float,
    reverse_kl_coef: float,
) -> Dict[str, torch.Tensor] | None:
    if float(forward_kl_coef) <= 0.0 and float(reverse_kl_coef) <= 0.0:
        return None

    student_fn = getattr(agent, "distill_student_log_probs_from_replay_batch", None)
    teacher_fn = getattr(agent, "distill_teacher_log_probs_from_replay_batch", None)
    if not callable(student_fn) or not callable(teacher_fn):
        return None

    student_log_probs = student_fn(replay, temperature=float(temperature))
    teacher_log_probs = teacher_fn(replay, temperature=float(temperature))
    if not torch.is_tensor(student_log_probs) or not torch.is_tensor(teacher_log_probs):
        raise TypeError("distillation hooks must return tensors")

    student_log_probs = student_log_probs.to(device=device, dtype=torch.float32)
    teacher_log_probs = teacher_log_probs.to(device=device, dtype=torch.float32).detach()
    if student_log_probs.shape != teacher_log_probs.shape:
        raise RuntimeError(
            "student/teacher distillation log-prob shapes must match: "
            f"student={tuple(student_log_probs.shape)} teacher={tuple(teacher_log_probs.shape)}"
        )
    if student_log_probs.ndim != 2:
        raise RuntimeError(
            "distillation hooks must return batched mode log-probs with shape (batch, num_modes); "
            f"got {tuple(student_log_probs.shape)}"
        )

    teacher_probs = teacher_log_probs.exp()
    student_probs = student_log_probs.exp()
    forward_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    reverse_kl = F.kl_div(teacher_log_probs, student_probs, reduction="batchmean")
    loss_forward_kl = torch.as_tensor(float(forward_kl_coef), device=device, dtype=torch.float32) * forward_kl
    loss_reverse_kl = torch.as_tensor(float(reverse_kl_coef), device=device, dtype=torch.float32) * reverse_kl
    return {
        "forward_kl": forward_kl,
        "reverse_kl": reverse_kl,
        "loss_forward_kl": loss_forward_kl,
        "loss_reverse_kl": loss_reverse_kl,
    }


class TrajectoryLightningModule(L.LightningModule):
    def __init__(
        self,
        *,
        agent: Any,
        learner_config: ActorLearnerLightningConfig,
        value_net: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.learner_config = learner_config
        self.policy_module = getattr(agent, "trainable_module", None)
        self.value_net = value_net
        self.latest_metrics: Dict[str, float] = {}
        self._grpo_debug_dump_count = 0
        self._reset_update_metric_aggregates()

    def _maybe_report_slow_step_part(self, *, name: str, seconds: float, batch_idx: int) -> None:
        threshold_s = 2.0
        if float(seconds) < float(threshold_s):
            return
        stage_fn = getattr(self, "stage_fn", None)
        if not callable(stage_fn):
            return
        update_fn = getattr(self, "_update_index", None)
        update_idx = "unknown"
        if callable(update_fn):
            try:
                update_idx = str(int(update_fn()))
            except Exception:
                update_idx = "unknown"
        stage_fn(
            f"[learner] slow_step update={update_idx} batch_idx={int(batch_idx)} "
            f"part={name} took={float(seconds):.2f}s"
        )

    def _maybe_apply_grpo_loss(
        self,
        *,
        replay: list[Dict[str, Any]],
        device: torch.device,
        batch_idx: int,
        loss: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
        timing_parts: Dict[str, float] | None = None,
        candidates: Dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        grpo_coef = float(self.learner_config.grpo_coef)
        grpo_enabled = bool(getattr(self.learner_config, "grpo_enabled", False)) or (grpo_coef > 0.0)
        aux_coef = float(getattr(self.learner_config, "aux_risk_decel_coef", 0.0))
        aux_requested = bool(getattr(self.learner_config, "aux_risk_decel_enabled", False)) and aux_coef > 0.0
        debug_requested = bool(getattr(self.learner_config, "grpo_debug_visualize", False))
        loss_requested = bool(grpo_enabled and grpo_coef > 0.0)
        if not loss_requested and not debug_requested and not aux_requested:
            return loss, metrics
        objective_key = str(getattr(self.learner_config, "grpo_objective", "grpo")).strip().lower()
        if objective_key != "grpo":
            raise ValueError(f"learner_config.grpo_objective must be 'grpo', got {objective_key!r}")

        if candidates is None:
            fused_fn = getattr(self.agent, "replay_policy_outputs_from_replay_batch", None)
            if callable(fused_fn):
                t0 = time.perf_counter()
                outputs = fused_fn(
                    replay,
                    eta=float(self.learner_config.eta),
                    num_candidates=int(self.learner_config.grpo_num_candidates),
                    candidate_select=str(self.learner_config.grpo_candidate_select),
                )
                if timing_parts is not None:
                    timing_parts["fused_policy_s"] = timing_parts.get("fused_policy_s", 0.0) + float(time.perf_counter() - t0)
                candidates = outputs.get("counterfactual", None) if isinstance(outputs, dict) else None
            if candidates is None:
                raise RuntimeError(
                    "GRPO requires actor-stored counterfactual candidates in replay. "
                    "Expected replay['grpo']['candidates'] with mode_indices, "
                    "old_log_probs, and traj_xyyaw."
                )
        else:
            if timing_parts is not None:
                timing_parts.setdefault("grpo_sample_s", 0.0)
        candidate_log_probs = candidates["log_probs"].to(device=device, dtype=torch.float32)
        old_candidate_log_probs = candidates.get("old_log_probs", None)
        if old_candidate_log_probs is None:
            raise RuntimeError("GRPO requires candidates['old_log_probs']")
        t0 = time.perf_counter()
        candidate_scores = score_counterfactual_trajectories(
            self.agent,
            replay,
            candidates["traj_xyyaw"],
            device=device,
        )
        score_s = float(time.perf_counter() - t0)
        if timing_parts is not None:
            timing_parts["grpo_score_s"] = timing_parts.get("grpo_score_s", 0.0) + score_s
        self._maybe_report_slow_step_part(name="grpo_score", seconds=score_s, batch_idx=batch_idx)
        t0 = time.perf_counter()
        self._maybe_dump_grpo_debug(
            replay=replay,
            traj_xyyaw=candidates["traj_xyyaw"],
            candidate_scores=candidate_scores,
            batch_idx=batch_idx,
        )
        debug_s = float(time.perf_counter() - t0)
        if timing_parts is not None:
            timing_parts["grpo_debug_s"] = timing_parts.get("grpo_debug_s", 0.0) + debug_s
        self._maybe_report_slow_step_part(name="grpo_debug", seconds=debug_s, batch_idx=batch_idx)
        out_loss = loss
        out_metrics = dict(metrics)
        if loss_requested:
            t0 = time.perf_counter()
            grpo_loss = compute_grpo_objective(
                candidate_log_probs=candidate_log_probs,
                old_candidate_log_probs=old_candidate_log_probs,
                candidate_scores=candidate_scores,
                candidate_score_logits=candidates.get("score_logits", None),
                score_norm_eps=float(self.learner_config.grpo_norm_eps),
                use_rank_adv=bool(self.learner_config.grpo_use_rank_adv),
                score_clip=self.learner_config.grpo_score_clip,
                objective=objective_key,
                temperature=float(getattr(self.learner_config, "grpo_temperature", 1.0)),
                clip_eps=float(self.learner_config.clip_eps),
            )
            objective_s = float(time.perf_counter() - t0)
            if timing_parts is not None:
                timing_parts["grpo_objective_s"] = timing_parts.get("grpo_objective_s", 0.0) + objective_s
            self._maybe_report_slow_step_part(name="grpo_objective", seconds=objective_s, batch_idx=batch_idx)
            out_loss = out_loss + grpo_coef * grpo_loss.loss
            out_metrics.update(
                {
                    "grpo_loss": grpo_loss.loss.detach(),
                    "grpo_score_mean": grpo_loss.score_mean.detach(),
                    "grpo_score_std": grpo_loss.score_std.detach(),
                    "grpo_score_min": grpo_loss.score_min.detach(),
                    "grpo_score_max": grpo_loss.score_max.detach(),
                    "grpo_approx_kl": grpo_loss.approx_kl.detach(),
                    "grpo_clip_frac": grpo_loss.clip_frac.detach(),
                    "grpo_ratio_mean": grpo_loss.ratio_mean.detach(),
                }
            )

        if aux_requested:
            score_logits = candidates.get("score_logits", None)
            if score_logits is not None:
                t0 = time.perf_counter()
                aux_loss = compute_risk_decel_auxiliary_objective(
                    candidate_score_logits=score_logits,
                    candidate_traj_xyyaw=candidates["traj_xyyaw"],
                    high_risk_mask=self._risk_decel_high_risk_mask_from_replay(replay, device=device),
                    ego_speed_mps=self._risk_decel_ego_speed_from_replay(replay, device=device),
                    dt_s=float(getattr(self.learner_config, "aux_risk_decel_dt_s", 0.5)),
                    speed_margin_mps=float(getattr(self.learner_config, "aux_risk_decel_speed_margin_mps", 0.1)),
                    eps=float(getattr(self.learner_config, "aux_risk_decel_eps", 1.0e-6)),
                )
                aux_s = float(time.perf_counter() - t0)
                if timing_parts is not None:
                    timing_parts["aux_risk_decel_s"] = timing_parts.get("aux_risk_decel_s", 0.0) + aux_s
                self._maybe_report_slow_step_part(name="aux_risk_decel", seconds=aux_s, batch_idx=batch_idx)
                out_loss = out_loss + aux_coef * aux_loss.loss
                out_metrics.update(
                    {
                        "aux_risk_decel_loss": aux_loss.loss.detach(),
                        "aux_risk_decel_active_count": aux_loss.active_count.detach(),
                        "aux_risk_decel_decel_prob": aux_loss.decel_prob_mean.detach(),
                        "aux_risk_decel_accel_prob": aux_loss.accel_prob_mean.detach(),
                    }
                )
        return out_loss, out_metrics

    def _maybe_fused_replay_policy_outputs(
        self,
        replay: list[Dict[str, Any]],
        *,
        device: torch.device,
        timing_parts: Dict[str, float],
    ) -> Dict[str, Any] | None:
        grpo_coef = float(self.learner_config.grpo_coef)
        loss_requested = (bool(getattr(self.learner_config, "grpo_enabled", False)) or grpo_coef > 0.0) and grpo_coef > 0.0
        aux_requested = (
            bool(getattr(self.learner_config, "aux_risk_decel_enabled", False))
            and float(getattr(self.learner_config, "aux_risk_decel_coef", 0.0)) > 0.0
        )
        debug_requested = bool(getattr(self.learner_config, "grpo_debug_visualize", False))
        if not loss_requested and not debug_requested and not aux_requested:
            return None

        fused_fn = getattr(self.agent, "replay_policy_outputs_from_replay_batch", None)
        if not callable(fused_fn):
            return None

        t0 = time.perf_counter()
        outputs = fused_fn(
            replay,
            eta=float(self.learner_config.eta),
            num_candidates=int(self.learner_config.grpo_num_candidates),
            candidate_select=str(self.learner_config.grpo_candidate_select),
        )
        timing_parts["fused_policy_s"] = float(time.perf_counter() - t0)
        new_logp = outputs.get("new_logp", None)
        if torch.is_tensor(new_logp):
            outputs["new_logp"] = new_logp.to(device=device, dtype=torch.float32).view(-1)
        return outputs

    def _risk_decel_high_risk_mask_from_replay(
        self,
        replay: list[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        high_risk: list[bool] = []
        max_gap = float(getattr(self.learner_config, "aux_risk_decel_high_risk_gap_m", 8.0))
        max_ttc = float(getattr(self.learner_config, "aux_risk_decel_high_risk_ttc_s", 3.0))
        max_lateral = float(getattr(self.learner_config, "aux_risk_decel_lateral_m", 2.5))
        for rep in replay:
            if not isinstance(rep, dict):
                high_risk.append(False)
                continue
            front = get_front_obstacle_aux(rep)
            if front is None or not bool(front.get("available", False)):
                high_risk.append(False)
                continue
            try:
                gap_m = float(front.get("gap_m", float("inf")))
                lateral_m = abs(float(front.get("lateral_m", float("inf"))))
                ttc_s = float(front.get("ttc_s", float("inf")))
            except Exception:
                high_risk.append(False)
                continue
            in_corridor = lateral_m <= max_lateral
            risky_gap = gap_m <= max_gap
            risky_ttc = ttc_s <= max_ttc
            high_risk.append(bool(in_corridor and (risky_gap or risky_ttc)))
        return torch.as_tensor(high_risk, device=device, dtype=torch.bool)

    @staticmethod
    def _risk_decel_ego_speed_from_replay(
        replay: list[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        speeds: list[float] = []
        for rep in replay:
            speed = None
            if isinstance(rep, dict):
                raw_speed = rep.get("ego_speed_mps", None)
                if raw_speed is not None:
                    try:
                        speed = float(raw_speed)
                    except Exception:
                        speed = None
                if speed is None:
                    try:
                        status = get_policy_model_inputs(rep).get("status_feature", None)
                    except RuntimeError:
                        status = None
                    if torch.is_tensor(status):
                        status_t = status.detach().to(dtype=torch.float32).view(-1)
                        if int(status_t.numel()) >= 6:
                            speed = float(torch.linalg.norm(status_t[4:6]).item())
            speeds.append(0.0 if speed is None else float(speed))
        return torch.as_tensor(speeds, device=device, dtype=torch.float32)

    def _compute_grpo_only_loss(
        self,
        *,
        replay: list[Dict[str, Any]],
        device: torch.device,
        batch_idx: int,
        timing_parts: Dict[str, float],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        grpo_coef = float(self.learner_config.grpo_coef)
        if not bool(getattr(self.learner_config, "grpo_enabled", False)) or grpo_coef <= 0.0:
            raise RuntimeError("train.algo=grpo_only requires train.grpo.enable=true and train.grpo.coef > 0")

        zero = torch.zeros((), device=device, dtype=torch.float32)
        fused_candidates = None
        fused_fn = getattr(self.agent, "replay_policy_outputs_from_replay_batch", None)
        if callable(fused_fn):
            t0 = time.perf_counter()
            fused_outputs = fused_fn(
                replay,
                eta=float(self.learner_config.eta),
                num_candidates=int(self.learner_config.grpo_num_candidates),
                candidate_select=str(self.learner_config.grpo_candidate_select),
            )
            timing_parts["fused_policy_s"] = float(time.perf_counter() - t0)
            fused_candidates = fused_outputs.get("counterfactual", None) if isinstance(fused_outputs, dict) else None
        loss, metrics = self._maybe_apply_grpo_loss(
            replay=replay,
            device=device,
            batch_idx=batch_idx,
            loss=zero,
            metrics={},
            timing_parts=timing_parts,
            candidates=fused_candidates,
        )
        if "grpo_loss" not in metrics:
            raise RuntimeError("train.algo=grpo_only did not produce a GRPO loss")
        return loss, metrics

    def _maybe_dump_grpo_debug(
        self,
        *,
        replay: list[Dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        candidate_scores: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if not bool(self.learner_config.grpo_debug_visualize):
            return
        out_dir = self.learner_config.grpo_debug_dir
        if out_dir is None or str(out_dir).strip() == "":
            return
        max_batches = int(self.learner_config.grpo_debug_max_batches)
        if max_batches > 0 and self._grpo_debug_dump_count >= max_batches:
            return
        dump_fn = getattr(self.agent, "dump_counterfactual_debug_from_replay_batch", None)
        if not callable(dump_fn):
            return

        step_tag = f"step{int(getattr(self, 'global_step', 0)):06d}_batch{int(batch_idx):04d}"
        os.makedirs(str(out_dir), exist_ok=True)
        dump_fn(
            replay,
            traj_xyyaw.detach(),
            candidate_scores.detach(),
            out_dir=str(out_dir),
            step_tag=step_tag,
            top_k=max(1, int(self.learner_config.grpo_debug_top_k)),
        )
        self._grpo_debug_dump_count += 1

    def _reset_update_metric_aggregates(self) -> None:
        self._update_metric_weight = 0.0
        self._update_metric_steps = 0
        self._update_metric_sums: Dict[str, float] = {}
        self._update_metric_max: Dict[str, float] = {}
        self._update_timing_sums: Dict[str, float] = {}
        self._update_timing_max: Dict[str, float] = {}

    def _record_update_metrics(self, metrics: Dict[str, torch.Tensor], *, batch_size: int) -> None:
        weight = float(max(1, int(batch_size)))
        self._update_metric_weight += weight
        self._update_metric_steps += 1
        for key, value in metrics.items():
            scalar = float(value.detach().cpu().item())
            self._update_metric_sums[key] = self._update_metric_sums.get(key, 0.0) + (scalar * weight)
            prev_max = self._update_metric_max.get(key, scalar)
            self._update_metric_max[key] = scalar if scalar > prev_max else prev_max

    def aggregated_update_metrics(self) -> Dict[str, float]:
        if self._update_metric_weight <= 0.0:
            return dict(self.latest_metrics)

        out = {
            key: total / float(self._update_metric_weight)
            for key, total in self._update_metric_sums.items()
        }
        if "approx_kl" in self._update_metric_max:
            out["approx_kl_max"] = float(self._update_metric_max["approx_kl"])
        out["num_minibatches"] = float(self._update_metric_steps)
        return out

    def _record_update_timing(self, timing_parts: Dict[str, float]) -> None:
        for key, value in timing_parts.items():
            scalar = float(value)
            self._update_timing_sums[key] = self._update_timing_sums.get(key, 0.0) + scalar
            prev_max = self._update_timing_max.get(key, scalar)
            self._update_timing_max[key] = scalar if scalar > prev_max else prev_max

    def aggregated_update_timing(self) -> Dict[str, float]:
        out = dict(self._update_timing_sums)
        if self._update_metric_steps > 0:
            denom = float(self._update_metric_steps)
            for key, value in self._update_timing_sums.items():
                out[f"{key}_avg"] = float(value / denom)
        for key, value in self._update_timing_max.items():
            out[f"{key}_max"] = float(value)
        out["timed_minibatches"] = float(self._update_metric_steps)
        return out

    @staticmethod
    def _build_lr_lambda(
        *,
        kind: str,
        warmup_updates: int,
        total_updates: int,
        min_lr_scale: float,
    ) -> Any:
        scheduler_kind = str(kind).strip().lower()
        warmup = max(0, int(warmup_updates))
        total = max(1, int(total_updates))
        min_scale = max(0.0, min(1.0, float(min_lr_scale)))

        if scheduler_kind not in {"linear_warmup_cosine_decay", "warmup_cosine", "cosine"}:
            raise ValueError(f"Unsupported lr scheduler kind: {kind}")

        def _lr_lambda(step: int) -> float:
            step_i = max(0, int(step))
            if warmup > 0 and step_i < warmup:
                return float((step_i + 1) / warmup)
            if total <= warmup:
                return float(min_scale)
            progress = min(1.0, max(0.0, float(step_i - warmup + 1) / float(total - warmup)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_scale + (1.0 - min_scale) * cosine)

        return _lr_lambda

    def configure_optimizers(self) -> Any:
        policy_params = _trainable_parameters(self.policy_module or self.agent)
        if len(policy_params) == 0:
            raise RuntimeError("No trainable policy parameters found for learner optimizer setup")

        param_groups = [
            {
                "params": policy_params,
                "lr": float(self.learner_config.optimizer_config.policy_lr),
                "weight_decay": float(self.learner_config.optimizer_config.weight_decay),
            }
        ]
        if self.learner_config.algo_kind.startswith("ppo"):
            if self.value_net is None:
                raise RuntimeError("PPO Lightning module requires value_net")
            value_params = _trainable_parameters(self.value_net)
            if len(value_params) == 0:
                raise RuntimeError("No trainable value parameters found for PPO optimizer setup")
            param_groups.append(
                {
                    "params": value_params,
                    "lr": float(
                        self.learner_config.optimizer_config.value_lr
                        if self.learner_config.optimizer_config.value_lr is not None
                        else self.learner_config.optimizer_config.policy_lr
                    ),
                    "weight_decay": 0.0,
                }
            )

        optimizer = torch.optim.Adam(param_groups)
        if not bool(self.learner_config.lr_scheduler_enabled):
            return optimizer

        lr_lambda = self._build_lr_lambda(
            kind=str(self.learner_config.lr_scheduler_kind),
            warmup_updates=int(self.learner_config.lr_warmup_updates),
            total_updates=int(self.learner_config.lr_total_updates or self.learner_config.max_updates),
            min_lr_scale=float(self.learner_config.lr_min_scale),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        try:
            step_t0 = time.perf_counter()
            timing_parts: Dict[str, float] = {}
            device = self.device
            replay = list(batch["replay"])
            adv = batch["adv"].to(device=device, dtype=torch.float32).view(-1)
            ret = batch["ret"].to(device=device, dtype=torch.float32).view(-1)
            old_logp = batch.get("old_logp", None)
            if torch.is_tensor(old_logp):
                old_logp = old_logp.to(device=device, dtype=torch.float32).view(-1)

            if self.learner_config.algo_kind == "grpo_only":
                loss, metrics = self._compute_grpo_only_loss(
                    replay=replay,
                    device=device,
                    batch_idx=batch_idx,
                    timing_parts=timing_parts,
                )
            else:
                fused_policy_outputs = self._maybe_fused_replay_policy_outputs(
                    replay,
                    device=device,
                    timing_parts=timing_parts,
                )
                fused_candidates = None
                if fused_policy_outputs is not None:
                    new_logp = fused_policy_outputs["new_logp"]
                    fused_candidates = fused_policy_outputs.get("counterfactual", None)
                    timing_parts["new_logp_s"] = 0.0
                    timing_parts["grpo_sample_s"] = 0.0
                else:
                    t0 = time.perf_counter()
                    new_logp = agent_logp_from_replay_batch(
                        self.agent,
                        replay,
                        device=device,
                        eta=float(self.learner_config.eta),
                    )
                    timing_parts["new_logp_s"] = float(time.perf_counter() - t0)

                if self.learner_config.algo_kind.startswith("ppo"):
                    if self.value_net is None:
                        raise RuntimeError("PPO Lightning module requires value_net")
                    t0 = time.perf_counter()
                    if _use_agent_value_features(self.agent, self.value_net, batch):
                        value_input = self.agent.value_features_from_replay_batch(replay).to(device=device, dtype=torch.float32)
                    else:
                        value_input = batch["obs"].to(device=device, dtype=torch.float32)
                    old_value = batch.get("old_value", None)
                    if torch.is_tensor(old_value):
                        old_value = old_value.to(device=device, dtype=torch.float32).view(-1)
                    value_pred = self.value_net(value_input).view(-1)
                    timing_parts["value_s"] = float(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    ppo_loss = compute_ppo_objective(
                        new_logp=new_logp,
                        old_logp=old_logp,
                        adv=adv,
                        ret=ret,
                        value_pred=value_pred,
                        old_value=old_value,
                        clip_eps=float(self.learner_config.clip_eps),
                        vf_coef=float(self.learner_config.vf_coef),
                        value_clip_eps=float(self.learner_config.value_clip_eps),
                        kl_coef=float(self.learner_config.kl_coef),
                        dual_clip=self.learner_config.dual_clip,
                    )
                    timing_parts["objective_s"] = float(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    metrics = compute_ppo_metrics(
                        new_logp=new_logp,
                        old_logp=old_logp,
                        adv=adv,
                        ret=ret,
                        value_pred=value_pred,
                        loss=ppo_loss,
                    )
                    timing_parts["metrics_compute_s"] = float(time.perf_counter() - t0)
                    loss = ppo_loss.loss * float(getattr(self.learner_config, "closed_loop_loss_coef", 1.0))
                    metrics["closed_loop_loss_coef"] = torch.as_tensor(
                        float(getattr(self.learner_config, "closed_loop_loss_coef", 1.0)),
                        device=device,
                        dtype=torch.float32,
                    )
                elif self.learner_config.algo_kind == "sac":
                    sac_old_logp = old_logp if torch.is_tensor(old_logp) and int(old_logp.numel()) == int(adv.numel()) else None
                    t0 = time.perf_counter()
                    sac_loss = compute_sac_objective(
                        new_logp=new_logp,
                        old_logp=sac_old_logp,
                        adv=adv,
                        entropy_coef=float(getattr(self.learner_config, "sac_entropy_coef", 0.0)),
                        kl_coef=float(self.learner_config.kl_coef),
                        clip_eps=float(self.learner_config.clip_eps),
                    )
                    timing_parts["objective_s"] = float(time.perf_counter() - t0)
                    metrics = {
                        "loss_pi": sac_loss.loss_pi.detach(),
                        "sac_pg_loss": sac_loss.loss_pg.detach(),
                        "sac_entropy_loss": sac_loss.loss_entropy.detach(),
                        "sac_logp_mean": sac_loss.logp_mean.detach(),
                        "sac_entropy_coef": sac_loss.entropy_coef.detach(),
                        "approx_kl": sac_loss.approx_kl.detach(),
                        "clip_frac": sac_loss.clip_frac.detach(),
                        "ratio_mean": sac_loss.ratio_mean.detach(),
                        "adv_mean": sac_loss.adv_mean.detach(),
                    }
                    timing_parts["metrics_compute_s"] = 0.0
                    loss = sac_loss.loss * float(getattr(self.learner_config, "closed_loop_loss_coef", 1.0))
                    metrics["closed_loop_loss_coef"] = torch.as_tensor(
                        float(getattr(self.learner_config, "closed_loop_loss_coef", 1.0)),
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    reinforce_old_logp = old_logp if self.learner_config.algo_kind in {"reinforcepp", "reinforce_kl"} else None
                    t0 = time.perf_counter()
                    r_loss = compute_reinforce_objective(
                        new_logp=new_logp,
                        old_logp=reinforce_old_logp,
                        adv=adv,
                        clip_eps=float(self.learner_config.clip_eps),
                        kl_coef=float(self.learner_config.kl_coef),
                    )
                    timing_parts["objective_s"] = float(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    metrics = compute_reinforce_metrics(
                        new_logp=new_logp,
                        old_logp=reinforce_old_logp,
                        adv=adv,
                        loss=r_loss,
                    )
                    timing_parts["metrics_compute_s"] = float(time.perf_counter() - t0)
                    loss = r_loss.loss * float(getattr(self.learner_config, "closed_loop_loss_coef", 1.0))
                    metrics["closed_loop_loss_coef"] = torch.as_tensor(
                        float(getattr(self.learner_config, "closed_loop_loss_coef", 1.0)),
                        device=device,
                        dtype=torch.float32,
                    )

                loss, metrics = self._maybe_apply_grpo_loss(
                    replay=replay,
                    device=device,
                    batch_idx=batch_idx,
                    loss=loss,
                    metrics=metrics,
                    timing_parts=timing_parts,
                    candidates=fused_candidates,
                )

            t0 = time.perf_counter()
            distill_metrics = _maybe_compute_distillation_metrics(
                self.agent,
                replay,
                device=device,
                temperature=float(self.learner_config.distill_temperature),
                forward_kl_coef=float(self.learner_config.forward_kl_coef),
                reverse_kl_coef=float(self.learner_config.reverse_kl_coef),
            )
            timing_parts["distill_s"] = float(time.perf_counter() - t0)
            if distill_metrics is not None:
                loss = loss + distill_metrics["loss_forward_kl"] + distill_metrics["loss_reverse_kl"]
                metrics = {**metrics, **distill_metrics}

            t0 = time.perf_counter()
            self.latest_metrics = {key: float(val.detach().cpu().item()) for key, val in metrics.items()}
            self._record_update_metrics(metrics, batch_size=int(adv.shape[0]))
            for key, value in metrics.items():
                self.log(
                    f"train/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(key == "loss_pi"),
                    logger=False,
                    batch_size=int(adv.shape[0]),
                )
            timing_parts["metrics_log_s"] = float(time.perf_counter() - t0)
            timing_parts["training_step_total_s"] = float(time.perf_counter() - step_t0)
            self._record_update_timing(timing_parts)
            return loss
        except Exception as exc:
            if _exception_is_cuda_oom(exc):
                stage_fn = getattr(self, "stage_fn", None)
                update_fn = getattr(self, "_update_index", None)
                update_idx = "unknown"
                if callable(update_fn):
                    try:
                        update_idx = str(int(update_fn()))
                    except Exception:
                        update_idx = "unknown"
                writer = stage_fn if callable(stage_fn) else None
                if callable(writer):
                    writer(
                        f"[learner] CUDA OOM inside training_step "
                        f"update={update_idx} batch_idx={batch_idx}"
                    )
                    log_cuda_memory_snapshot(
                        label=f"training_step_oom update={update_idx} batch_idx={batch_idx}",
                        log_fn=writer,
                    )
            raise
