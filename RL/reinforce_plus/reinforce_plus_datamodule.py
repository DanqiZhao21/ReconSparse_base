#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : datamodule.py
@Date    : 2024/12/15
'''
import numpy as np
import torch
from typing import Dict, List, Optional
from omegaconf import DictConfig
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from rift.gym_carla.buffer.ego_rollout_buffer import EgoRolloutBuffer
from rift.ego.fine_tuner.feature_decoder import BaseFeatureDecoder, build_feature_decoder

def compute_discounted_returns(rewards: torch.Tensor, undones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute Monte Carlo discounted returns."""
    returns = torch.zeros_like(rewards)
    running_return = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running_return = rewards[t] + gamma * running_return * undones[t]
        returns[t] = running_return
    return returns


class ReinforcePlusCollate:
    """Wrapper class that collates together multiple samples into a batch."""

    def __init__(self, feature_decoder: BaseFeatureDecoder, decode_workers: int) -> None:
        self._feature_decoder = feature_decoder
        self._decode_workers = decode_workers

    def _decode_cur_features(self, batch: List[Dict]) -> None:
        # extract features
        features = [sample["ego_obs"]["feature"] for sample in batch]
        # decode features
        decoded_feature = self._feature_decoder.decode_many(
            features,
            self._decode_workers,
        )
        return decoded_feature

    def __call__(
        self, batch: List[Dict]
    ) -> Dict:
        """
        Collate list of [dict] into batch
        :param batch: list of dict to be batched
        :return data already batched
        """
        bs = len(batch)
        assert bs > 0, "Batch size has to be greater than 0!"
        # extract ego data dict
        ego_cur_feature = self._decode_cur_features(batch)
        ego_return = [ego_data_dict['ego_return'] for ego_data_dict in batch]
        ego_advantage = [ego_data_dict['ego_advantage'] for ego_data_dict in batch]
        ego_actions_log_prob = [ego_data_dict['ego_action_log_prob'] for ego_data_dict in batch]
        ego_actions_mode = [ego_data_dict['ego_action_mode'] for ego_data_dict in batch]

        # collate ego data
        output = {
            'ego_batch_cur_feature': self._feature_decoder.collate_decoded_features(ego_cur_feature, samples_per_gpu=bs),
            'ego_batch_return': torch.stack(ego_return, dim=0),                     # [bs, 1]
            'ego_batch_advantage': torch.stack(ego_advantage, dim=0),               # [bs, 1]
            'ego_batch_action_log_prob': torch.stack(ego_actions_log_prob, dim=0),  # [bs, 1]
            'ego_batch_action_mode': torch.stack(ego_actions_mode, dim=0),          # [bs, 2]
        }

        return output


class ReinforcePlusDataset(Dataset):
    def __init__(self, cfg: DictConfig, buffer: EgoRolloutBuffer):
        self.buffer = buffer
        self.cfg = cfg
        self.data = {}

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer.sample(idx)


class ReinforcePlusDataModule(LightningDataModule):
    def __init__(self,cfg: DictConfig, buffer: EgoRolloutBuffer):
        super().__init__()
        self.buffer = buffer
        self.cfg = cfg

        self.gamma = cfg.gamma

        self.train_batch_size = cfg.train_batch_size
        self.val_batch_size = cfg.val_batch_size
        self.shuffle = cfg.shuffle
        self.num_workers = cfg.num_workers
        self.pin_memory = cfg.pin_memory
        self.persistent_workers = cfg.persistent_workers
        self.train_dataset = None
        # initialize feature decoder
        self.feature_decoder = build_feature_decoder(cfg.feature_decoder_type, cfg.mm_config_path)
        self._collate_fn = ReinforcePlusCollate(self.feature_decoder, cfg.decode_workers)

    def setup(self, stage: Optional[str] = None):
        assert self.buffer.buffer_full, 'The buffer should be full before training'

        # init the RLFT Dataset
        all_dataset = ReinforcePlusDataset(self.cfg, self.buffer)

        if stage == 'fit' or stage is None:
            self.train_dataset = all_dataset
        else:
            raise ValueError(f'CloseLoop fine-tuning currently only support ["fit"], got ${stage}.')

    def preprocess_buffer(self, model: torch.nn.Module):
        # extract necessary data from the buffer
        rewards = torch.from_numpy(np.stack(self.buffer.get_key_data('ego_reward'), axis=0)).float()
        undones = 1.0 - torch.from_numpy(np.stack(self.buffer.get_key_data('ego_done'), axis=0)).float()

        # compute discounted returns
        returns = compute_discounted_returns(rewards, undones, gamma=self.gamma)  # [bs]
        # global batch normalization of returns to get advantages
        mean_return = returns.mean()
        centered_returns = returns - mean_return
        std_return = centered_returns.std(unbiased=False) + 1e-8
        advantages = centered_returns / std_return  # [bs]

        # store the computed returns and advantages back to the buffer
        self.buffer.add_extra_data({
            'ego_return': returns.cpu(),
            'ego_advantage': advantages.cpu(),
        })

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return None
