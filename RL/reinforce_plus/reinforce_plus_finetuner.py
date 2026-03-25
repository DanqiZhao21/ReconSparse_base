#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : ppo_pluto.py
@Date    : 2025/01/27
'''
import torch
from torch import nn

from rift.ego.fine_tuner.base_finetuner import FineTuner
from rift.ego.fine_tuner.registry import register_fine_tuner


class ReinforcePlusContainer(nn.Module):
    """Wrap actor for Reinforce+ fine-tuning (no critic required)."""

    def __init__(self, actor: nn.Module, pretrained_actor: nn.Module = None):
        super().__init__()
        self.actor = actor
        self.pretrained_actor = pretrained_actor

    def train(self, mode: bool = True):
        super().train(mode)
        if self.pretrained_actor is not None:
            self.pretrained_actor.eval()
        return self

    def forward(self, cur_feature):
        return self.actor(**cur_feature)

    def forward_pretrained(self, cur_feature):
        """Forward pass using the frozen pretrained actor for KL regularization."""
        if self.pretrained_actor is None:
            raise ValueError("Pretrained actor is not provided.")

        with torch.no_grad():
            return self.pretrained_actor(**cur_feature)


@register_fine_tuner("reinforce_plus")
class ReinforcePlusFineTuner(FineTuner):
    name = 'reinforce_plus_finetuner'

    def __init__(self, config, ft_config, logger, extra_hydra_overrides=None):
        super().__init__(config, ft_config, logger, extra_hydra_overrides)
    
    def build_train_model(self):
        # wrap actor for lightning so we can freeze/unfreeze layers cleanly
        self.train_model = ReinforcePlusContainer(
            actor=self.actor_model,
            pretrained_actor=self.pretrained_actor_model,
        )
        # set train mode
        self.train_model.train()
