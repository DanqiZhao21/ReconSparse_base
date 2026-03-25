from __future__ import annotations

from typing import Any, Dict

import torch

from framework.algorithms.trajectory_policy_core import (
    agent_logp_from_replay_batch,
    compute_ppo_metrics,
    compute_ppo_objective,
    compute_reinforce_metrics,
    compute_reinforce_objective,
)
from framework.lightning_compat import L


class TrajectoryLightningModule(L.LightningModule):
    def __init__(
        self,
        *,
        agent: Any,
        optimizer: torch.optim.Optimizer,
        algo_kind: str,
        eta: float,
        clip_eps: float,
        vf_coef: float = 0.0,
        value_clip_eps: float = 0.0,
        kl_coef: float = 0.0,
        dual_clip: float | None = None,
        value_net: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.optimizer_ref = optimizer
        self.algo_kind = str(algo_kind)
        self.eta = float(eta)
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.value_clip_eps = float(value_clip_eps)
        self.kl_coef = float(kl_coef)
        self.dual_clip = None if dual_clip is None else float(dual_clip)
        self.policy_module = getattr(agent, "trainable_module", None)
        self.value_net = value_net
        self.latest_metrics: Dict[str, float] = {}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_ref

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
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
            eta=float(self.eta),
        )

        if self.algo_kind.startswith("ppo"):
            if self.value_net is None:
                raise RuntimeError("PPO Lightning module requires value_net")
            obs = batch["obs"].to(device=device, dtype=torch.float32)
            old_value = batch.get("old_value", None)
            if torch.is_tensor(old_value):
                old_value = old_value.to(device=device, dtype=torch.float32).view(-1)
            value_pred = self.value_net(obs).view(-1)
            ppo_loss = compute_ppo_objective(
                new_logp=new_logp,
                old_logp=old_logp,
                adv=adv,
                ret=ret,
                value_pred=value_pred,
                old_value=old_value,
                clip_eps=float(self.clip_eps),
                vf_coef=float(self.vf_coef),
                value_clip_eps=float(self.value_clip_eps),
                kl_coef=float(self.kl_coef),
                dual_clip=self.dual_clip,
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
            reinforce_old_logp = old_logp if self.algo_kind in {"reinforcepp", "reinforce_kl"} else None
            r_loss = compute_reinforce_objective(
                new_logp=new_logp,
                old_logp=reinforce_old_logp,
                adv=adv,
                clip_eps=float(self.clip_eps),
                kl_coef=float(self.kl_coef),
            )
            metrics = compute_reinforce_metrics(
                new_logp=new_logp,
                old_logp=reinforce_old_logp,
                adv=adv,
                loss=r_loss,
            )
            loss = r_loss.loss

        self.latest_metrics = {key: float(val.detach().cpu().item()) for key, val in metrics.items()}
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