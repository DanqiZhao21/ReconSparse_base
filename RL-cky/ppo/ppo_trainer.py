import logging
from typing import Dict, List, Union
import lightning as L
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan_plugin.modeling.types import (
    TargetsType
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from rift.ego.fine_tuner.optim.warmup_cos_lr import WarmupCosLR
from rift.ego.fine_tuner.ppo.ppo_finetuner import PPOContainer
from rift.ego.fine_tuner.utils import parse_trainable_rules

logger = logging.getLogger(__name__)


class LightningTrainer(L.LightningModule):
    def __init__(
        self,
        model: PPOContainer,
        params: DictConfig,
        trainable_layers: List[str],
        objective_aggregate_mode: str = "mean",
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        # Model
        self.model = model
        # Training parameters
        self.lr = params.lr
        self.min_lr = params.min_lr
        self.weight_decay = params.weight_decay
        self.epochs = params.epochs
        self.warmup_epochs = params.warmup_epochs
        self.frame_rate = params.frame_rate
        self.dt = 1.0 / self.frame_rate
        self.objective_aggregate_mode = objective_aggregate_mode
        self.trainable_layers = trainable_layers

        # PPO parameters
        self.clip_eps = params.ppo.clip_eps
        self.policy_weight = params.ppo.policy_weight
        self.value_weight = params.ppo.value_weight
        
        self.distill_weight = params.ppo.distill_weight
        self.kl_loss_weight = params.ppo.kl_loss_weight
        self.value_clip_eps = params.ppo.value_clip_eps
        self.critic_lr_multiplier = params.ppo.critic_lr_multiplier
        self.use_kl_term = params.use_kl_term
        self.T = params.temperature

        self.freeze_parameters(trainable_layers)

    def freeze_parameters(self, trainable_layers=[]):
        # freeze all param
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze specific layer
        patterns = parse_trainable_rules(trainable_layers)
        for name, param in self.model.named_parameters():
            if any(name.startswith(pfx) and (not sub or sub in name) for pfx, sub in patterns):
                param.requires_grad = True

    def _step(
        self, batch: Dict, prefix: str
    ) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        # current input output pair
        cur_feature = batch['ego_batch_cur_feature']
        cur_res = self.forward(cur_feature)
        pre_res = self.forward_pretrained(cur_feature)

        loss, stats = self.get_ppo_loss(cur_res, pre_res, batch)

        self._log_step(stats, prefix)

        return loss if self.training else 0.0

    def get_ppo_loss(self, cur_res, pre_res, batch):
        # extract the batch data
        state = batch['ego_batch_state']  # (bs, 4D)
        advantage = batch['ego_batch_advantage'].detach().view(-1)  # (bs,)
        target_return = batch['ego_batch_target_return'].detach().view(-1)  # (bs,)
        old_value = batch['ego_batch_value'].detach().view(-1)  # (bs,)
        old_log_prob = batch['ego_batch_action_log_prob'].detach().view(-1)  # (bs,)
        action_mode = batch['ego_batch_action_mode']  # (bs, 2)

        # the logits of the current actor
        logits = torch.stack([res["planning_logits"] for res in cur_res], dim=0)  # (bs, num_mode)

        bs, _ = logits.shape
        mode_idx = action_mode[:, 1].long()

        # derive the log-probability in the mode axis
        log_action_probs = F.log_softmax(logits / self.T, dim=1)  # (bs, num_mode)
        action_probs = torch.exp(log_action_probs)  # (bs, num_mode)
        # log-probability of the chosen action
        batch_index = torch.arange(bs, device=logits.device)
        cur_log_prob = log_action_probs[batch_index, mode_idx]  # (bs,)
        entropy = -(action_probs * log_action_probs).sum(dim=1).mean().detach()

        value = self.model.critic(state).view(-1)
        value_delta = value - old_value
        value_pred_clipped = old_value + value_delta.clamp(-self.value_clip_eps, self.value_clip_eps)
        value_loss_unclipped = F.smooth_l1_loss(value, target_return, reduction="none")
        value_loss_clipped = F.smooth_l1_loss(value_pred_clipped, target_return, reduction="none")
        value_term = torch.max(value_loss_unclipped, value_loss_clipped).mean()

        log_ratio = cur_log_prob - old_log_prob
        ratio = log_ratio.exp()
        L1 = advantage * ratio
        L2 = advantage * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_term = - torch.min(L1, L2).mean()

        # reverse KL: KL(student || teacher), always used as distillation loss
        teacher_logits = torch.stack([res["planning_logits"] for res in pre_res], dim=0)
        teacher_log_probs = F.log_softmax(teacher_logits / self.T, dim=1).detach()
        teacher_probs = torch.exp(teacher_log_probs)
        distill_term = F.kl_div(teacher_log_probs, action_probs, reduction="batchmean")

        # forward KL: KL(teacher || student), optional update constraint
        if self.use_kl_term:
            kl_term = F.kl_div(log_action_probs, teacher_probs, reduction="batchmean")
        else:
            kl_term = logits.new_zeros(())

        # total loss
        loss_policy = self.policy_weight * policy_term
        loss_value = self.value_weight * value_term
        loss_distill = self.distill_weight * distill_term
        loss_kl = self.kl_loss_weight * kl_term
        loss = loss_policy + loss_value + loss_distill + loss_kl

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
            value_clip_frac = (value_delta.abs() > self.value_clip_eps).float().mean()
            explained_variance = 1.0 - (
                torch.var(target_return - value, unbiased=False)
                / (torch.var(target_return, unbiased=False) + 1e-8)
            )
        
            stats = {
                "loss": loss.item(),  # total objective
                "loss_policy": loss_policy.item(),
                "loss_value": loss_value.item(),
                "loss_distill": loss_distill.item(),
                "loss_kl": loss_kl.item(),
                "entropy": entropy.item(),  # policy entropy (exploration level)
                "approx_kl": approx_kl.item(),  # policy update size proxy
                "clip_frac": clip_frac.item(),  # fraction of samples hitting PPO clip
                "value_clip_frac": value_clip_frac.item(),
                "explained_variance": explained_variance.item(),  # critic fit quality to returns
            }

        return loss, stats

    def _log_step(
        self,
        stats: Dict[str, float],
        prefix: str,
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param stats: dictionary of artifacts to log
        :param prefix: prefix prepended at each artifact's name
        """

        for key, value in stats.items():
            log_class = "loss" if key == "loss" or key.startswith("loss_") else "objective"
            self.log(
                f"{log_class}/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True if prefix == "train" and key == "loss" else False,
            )

    def training_step(
        self, batch: Dict, batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Dict, batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Dict, batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, cur_feature) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param data: features batch
        :return: model's predictions
        """
        return self.model(cur_feature)

    def forward_pretrained(self, cur_feature) -> TargetsType:
        """Forward through the frozen pretrained actor for KL regularization."""
        return self.model.forward_pretrained(cur_feature)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay_actor = set()
        no_decay_actor = set()
        decay_critic = set()
        no_decay_critic = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                
                # only contain the trainable param
                if not param.requires_grad:
                    continue
                is_critic = full_param_name.startswith("model.critic.")
                cur_decay = decay_critic if is_critic else decay_actor
                cur_no_decay = no_decay_critic if is_critic else no_decay_actor

                if "bias" in param_name:
                    cur_no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        cur_decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        cur_no_decay.add(full_param_name)
                    elif "in_proj_weight" in param_name:  # for attention modules in HiP-AD and SparseDrive
                        cur_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    cur_no_decay.add(full_param_name)
        # only contain the param requires grad
        param_dict = {param_name: param for param_name, param in self.named_parameters() if param.requires_grad}
        decay = decay_actor | decay_critic
        no_decay = no_decay_actor | no_decay_critic
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = []
        for names, wd, lr_scale in (
            (decay_actor, self.weight_decay, 1.0),
            (decay_critic, self.weight_decay, self.critic_lr_multiplier),
            (no_decay_actor, 0.0, 1.0),
            (no_decay_critic, 0.0, self.critic_lr_multiplier),
        ):
            if not names:
                continue
            optim_groups.append(
                {
                    "params": [param_dict[name] for name in sorted(names)],
                    "weight_decay": wd,
                    "lr_scale": lr_scale,
                }
            )

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay, eps=1e-5
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=self.min_lr,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
