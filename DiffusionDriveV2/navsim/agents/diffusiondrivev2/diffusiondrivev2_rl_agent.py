from typing import Any, List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
import copy
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config import TransfuserConfig

from navsim.agents.diffusiondrivev2.diffusiondrivev2_model_rl import V2TransfuserModel as TransfuserModel

from navsim.agents.diffusiondrivev2.transfuser_callback import TransfuserCallback 
from navsim.agents.diffusiondrivev2.transfuser_loss import transfuser_loss
from navsim.agents.diffusiondrivev2.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusiondrivev2.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
import torch.nn.functional as F
import numpy as np
import re

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)


class Diffusiondrivev2_Rl_Agent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(config)
        for name, param in self._transfuser_model.named_parameters():
            if not name.startswith("_trajectory_head"):
                param.requires_grad = False
        for name, module in self._transfuser_model.named_modules():
            if name and not name.startswith("_trajectory_head"):
                module.eval()
        self._transfuser_model._trajectory_head.train()
        self.init_from_pretrained()

    def init_from_pretrained(self):
        if self._checkpoint_path:
            checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove 'agent.' prefix from keys if present
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
            # Load state dict and get info about missing and unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        
        else:
            print("No checkpoint path provided. Initializing from scratch.")

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        # Evaluation should be tolerant to extra keys (model definition drifts across commits).
        # Otherwise Ray workers crash with "Unexpected key(s)".
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Override default compute_trajectory.

        The base AbstractAgent implementation assumes a single trajectory of shape (B,T,3).
        Our RL model returns multi-modal trajectories (e.g. (B,M,T,3)). For PDM evaluation we
        must pick ONE mode and return a Trajectory with poses shape (T,3).
        """
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            if predictions is None or (isinstance(predictions, dict) and predictions.get("trajectory", None) is None):
                raise RuntimeError("Agent forward did not return trajectory")

            traj_t = predictions["trajectory"]
            if not torch.is_tensor(traj_t):
                raise RuntimeError(f"trajectory must be a torch.Tensor, got {type(traj_t)!r}")
            traj_t = traj_t.detach().cpu()

            # Flatten any multi-modal structure into (B, M, T, 3)
            if traj_t.ndim == 2:
                # (T,3)
                poses_t = traj_t
            elif traj_t.ndim == 3:
                # (B,T,3) or (M,T,3)
                if int(traj_t.shape[-1]) != 3:
                    raise RuntimeError(f"trajectory last dim must be 3, got {tuple(traj_t.shape)}")
                if int(traj_t.shape[0]) == 1:
                    poses_t = traj_t[0]
                else:
                    # treat as (M,T,3) and pick mode 0
                    poses_t = traj_t[0]
            else:
                # (B, ..., T, 3) -> (B, M, T, 3)
                if int(traj_t.shape[-1]) != 3:
                    raise RuntimeError(f"trajectory last dim must be 3, got {tuple(traj_t.shape)}")
                B = int(traj_t.shape[0])
                T = int(traj_t.shape[-2])
                modes_t = traj_t.view(B, -1, T, 3)

                # Choose best mode by summed diffusion log-prob if provided.
                print("💗Multiple trajectory modes detected, selecting best mode based on log-prob.")
                best_idx = 0
                logp = predictions.get("log_probs", None) if isinstance(predictions, dict) else None
                if torch.is_tensor(logp):
                    logp_t = logp.detach().cpu()
                    if logp_t.ndim >= 3 and int(logp_t.shape[0]) == B:
                        lp_flat = logp_t.view(B, -1, int(logp_t.shape[-1]))
                        scores = lp_flat.sum(dim=-1)  # (B, M)
                        best_idx = int(torch.argmax(scores[0]).item())

                poses_t = modes_t[0, best_idx]

            poses = poses_t.numpy().astype(np.float32)

        return Trajectory(poses)


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, metric_cache=None, token=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features,targets=targets, eta=1.0, metric_cache=metric_cache, token=token)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        loss = predictions['loss']
        reward = predictions['reward']
        sub_rewards = predictions.get('sub_rewards', None)
        loss_dict = {'loss': loss, 'reward':reward}
        if sub_rewards is not None:
            loss_dict.update(sub_rewards) # add sub rewards to loss_dict if available
        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        optimizer_cfg = dict(type=self._config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self._config.weight_decay,
                            paramwise_cfg=self._config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self._config.scheduler_type,
                            milestones=self._config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._transfuser_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._transfuser_model.parameters()
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=10,
            warmup_epochs=1,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config)]
