from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from framework.algorithms.trajectory_policy_core import (
    agent_logp_from_replay_batch,
    compute_ppo_metrics,
    compute_ppo_objective,
    compute_reinforce_metrics,
    compute_reinforce_objective,
)
from framework.lightning.config import ActorLearnerLightningConfig
from framework.lightning_compat import L
from framework.runner.logging import _exception_is_cuda_oom, log_cuda_memory_snapshot

try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore


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
        self._reset_update_metric_aggregates()

    def _maybe_log_train_seen_samples(
        self,
        *,
        metrics: Dict[str, torch.Tensor],
        adv: torch.Tensor,
        ret: torch.Tensor,
        batch_size: int,
    ) -> None:
        if not bool(getattr(self, "wandb_enabled", False)) or wandb is None:
            return

        seen_step = int(getattr(self, "global_train_seen_sample_step", 0)) + int(max(1, int(batch_size)))
        setattr(self, "global_train_seen_sample_step", int(seen_step))

        payload: Dict[str, float | int] = {
            "global_train_seen_sample_step": int(seen_step),
            "ret_mean": float(ret.detach().mean().item()) if int(ret.numel()) > 0 else 0.0,
            "ret_std": float(ret.detach().std(unbiased=False).item()) if int(ret.numel()) > 0 else 0.0,
            "adv_std": float(adv.detach().std(unbiased=False).item()) if int(adv.numel()) > 0 else 0.0,
            "seen_batch_size": int(batch_size),
        }
        update_fn = getattr(self, "_update_index", None)
        if callable(update_fn):
            try:
                payload["update"] = int(update_fn())
            except Exception:
                pass
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            try:
                payload["global_step"] = int(getattr(trainer, "global_step"))
            except Exception:
                pass

        for key, value in metrics.items():
            try:
                payload[f"train_seen_samples/{key}"] = float(value.detach().cpu().item())
            except Exception:
                continue
        payload["train_seen_samples/ret_mean"] = float(payload["ret_mean"])
        payload["train_seen_samples/ret_std"] = float(payload["ret_std"])
        payload["train_seen_samples/adv_std"] = float(payload["adv_std"])
        payload["train_seen_samples/seen_batch_size"] = float(payload["seen_batch_size"])
        try:
            wandb.log(payload)
        except Exception:
            pass

    def _reset_update_metric_aggregates(self) -> None:
        self._update_metric_weight = 0.0
        self._update_metric_steps = 0
        self._update_metric_sums: Dict[str, float] = {}
        self._update_metric_max: Dict[str, float] = {}

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
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

        return torch.optim.Adam(param_groups)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        try:
            device = self.device
            replay = list(batch["replay"])
            adv = batch["adv"].to(device=device, dtype=torch.float32).view(-1)
            ret = batch["ret"].to(device=device, dtype=torch.float32).view(-1)
            old_logp = batch.get("old_logp", None)
            if torch.is_tensor(old_logp):
                old_logp = old_logp.to(device=device, dtype=torch.float32).view(-1)

            new_logp = agent_logp_from_replay_batch(
                self.agent,
                replay,
                device=device,
                eta=float(self.learner_config.eta),
            )

            if self.learner_config.algo_kind.startswith("ppo"):
                if self.value_net is None:
                    raise RuntimeError("PPO Lightning module requires value_net")
                if _use_agent_value_features(self.agent, self.value_net, batch):
                    value_input = self.agent.value_features_from_replay_batch(replay).to(device=device, dtype=torch.float32)
                else:
                    value_input = batch["obs"].to(device=device, dtype=torch.float32)
                old_value = batch.get("old_value", None)
                if torch.is_tensor(old_value):
                    old_value = old_value.to(device=device, dtype=torch.float32).view(-1)
                value_pred = self.value_net(value_input).view(-1)
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
                metrics = compute_ppo_metrics(
                    new_logp=new_logp,
                    old_logp=old_logp,
                    adv=adv,
                    ret=ret,
                    value_pred=value_pred,
                    loss=ppo_loss,
                )
                loss = ppo_loss.loss
            else:
                reinforce_old_logp = old_logp if self.learner_config.algo_kind in {"reinforcepp", "reinforce_kl"} else None
                r_loss = compute_reinforce_objective(
                    new_logp=new_logp,
                    old_logp=reinforce_old_logp,
                    adv=adv,
                    clip_eps=float(self.learner_config.clip_eps),
                    kl_coef=float(self.learner_config.kl_coef),
                )
                metrics = compute_reinforce_metrics(
                    new_logp=new_logp,
                    old_logp=reinforce_old_logp,
                    adv=adv,
                    loss=r_loss,
                )
                loss = r_loss.loss

            distill_metrics = _maybe_compute_distillation_metrics(
                self.agent,
                replay,
                device=device,
                temperature=float(self.learner_config.distill_temperature),
                forward_kl_coef=float(self.learner_config.forward_kl_coef),
                reverse_kl_coef=float(self.learner_config.reverse_kl_coef),
            )
            if distill_metrics is not None:
                loss = loss + distill_metrics["loss_forward_kl"] + distill_metrics["loss_reverse_kl"]
                metrics = {**metrics, **distill_metrics}

            self.latest_metrics = {key: float(val.detach().cpu().item()) for key, val in metrics.items()}
            self._record_update_metrics(metrics, batch_size=int(adv.shape[0]))
            self._maybe_log_train_seen_samples(
                metrics=metrics,
                adv=adv,
                ret=ret,
                batch_size=int(adv.shape[0]),
            )
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
