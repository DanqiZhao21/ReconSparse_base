#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : ppo_pluto.py
@Date    : 2025/01/27
'''
import hydra
import hydra._internal.instantiate._instantiate2

import torch
from torch import nn

from rift.ego.fine_tuner.base_finetuner import FineTuner
from rift.ego.fine_tuner.registry import register_fine_tuner
from rift.gym_carla.utils.net import CriticPPO
from rift.util.torch_util import CUDA

# Instantiation related symbols
instantiate = hydra._internal.instantiate._instantiate2.instantiate


class PPOContainer(nn.Module):
    """
    Wrap an existing actor model (from E2E_Agent.self.model) with a critic
    for PPO training. This module is used ONLY by the trainer; the fine-tuner
    will not perform inference with it.
    """
    def __init__(self, actor: nn.Module, critic: nn.Module, pretrained_actor: nn.Module = None):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.pretrained_actor = pretrained_actor

    def train(self, mode: bool = True):
        super().train(mode)
        if self.pretrained_actor is not None:
            self.pretrained_actor.eval()
        return self

    def forward(self, cur_feature):
        """
        Forward only consider the actor output
        """
        actor_output = self.actor(**cur_feature)
        return actor_output

    def forward_pretrained(self, cur_feature):
        """Forward pass using the frozen pretrained actor for KL regularization."""
        if self.pretrained_actor is None:
            raise ValueError("Pretrained actor is not provided.")

        with torch.no_grad():
            actor_output = self.pretrained_actor(**cur_feature)

        return actor_output


@register_fine_tuner("ppo")
class PPOFineTuner(FineTuner):
    name = 'ppo_finetuner'

    def __init__(self, config, ft_config, logger, extra_hydra_overrides=None):
        super().__init__(config, ft_config, logger, extra_hydra_overrides)
        self.hidden_dim = self.ft_config['hidden_dim']
        self.state_dim = self.ft_config['state_dim']
        self.action_dim = self.ft_config['action_dim']

        self.ppo_contrainer = None
    
    def build_train_model(self):
        # build the PPO container
        self.train_model = PPOContainer(
            actor=self.actor_model,
            critic=CUDA(CriticPPO(dims=self.hidden_dim, state_dim=self.state_dim, action_dim=self.action_dim)),
            pretrained_actor=self.pretrained_actor_model,
        )
        # set train mode
        self.train_model.train()
