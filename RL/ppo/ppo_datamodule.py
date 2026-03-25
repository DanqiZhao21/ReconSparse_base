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

from rift.ego.fine_tuner.utils import resolve_module_device
from rift.gym_carla.buffer.ego_rollout_buffer import EgoRolloutBuffer
from rift.ego.fine_tuner.feature_decoder import BaseFeatureDecoder, build_feature_decoder


def get_advantages_GAE(rewards, undones, values, next_values, unterminated, gamma=0.98, lambda_gae_adv=0.98):
    """
        unterminated: if the CBV collide with an object, then it is terminated
        undone: if the CBV is stuck or collide or max step will cause 'done'
        https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/AgentPPO.py
    """
    advantages = torch.empty_like(values)  # advantage value

    horizon_len = rewards.shape[0]

    advantage = torch.zeros_like(values[0])  # last advantage value by GAE (Generalized Advantage Estimate)

    for t in range(horizon_len - 1, -1, -1):
        delta = rewards[t] + unterminated[t] * gamma * next_values[t] - values[t]
        advantages[t] = advantage = delta + undones[t] * gamma * lambda_gae_adv * advantage
    return advantages


class PPOCollate:
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
        ego_cur_state = [ego_data_dict['ego_state'] for ego_data_dict in batch]
        ego_advantage = [ego_data_dict['ego_advantage'] for ego_data_dict in batch]
        ego_target_return = [ego_data_dict['ego_target_return'] for ego_data_dict in batch]
        ego_value = [ego_data_dict['ego_value'] for ego_data_dict in batch]
        ego_actions_log_prob = [ego_data_dict['ego_action_log_prob'] for ego_data_dict in batch]
        ego_actions_mode = [ego_data_dict['ego_action_mode'] for ego_data_dict in batch]

        # collate ego data
        output = {
            'ego_batch_cur_feature': self._feature_decoder.collate_decoded_features(ego_cur_feature, samples_per_gpu=bs),
            'ego_batch_state': torch.stack(ego_cur_state, dim=0),                   # [bs, 4D]
            'ego_batch_advantage': torch.stack(ego_advantage, dim=0),               # [bs]
            'ego_batch_target_return': torch.stack(ego_target_return, dim=0),       # [bs]
            'ego_batch_value': torch.stack(ego_value, dim=0),                       # [bs]
            'ego_batch_action_log_prob': torch.stack(ego_actions_log_prob, dim=0),  # [bs]
            'ego_batch_action_mode': torch.stack(ego_actions_mode, dim=0),          # [bs, 2]
        }

        return output


class PPODataset(Dataset):
    def __init__(self, cfg: DictConfig, buffer: EgoRolloutBuffer):
        self.buffer = buffer
        self.cfg = cfg
        self.data = {}

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer.sample(idx)


class PPODataModule(LightningDataModule):
    def __init__(self,cfg: DictConfig, buffer: EgoRolloutBuffer):
        super().__init__()
        self.buffer = buffer
        self.cfg = cfg

        self.gamma = cfg.gamma
        self.lambda_gae_adv = cfg.lambda_gae_adv

        self.train_batch_size = cfg.train_batch_size
        self.val_batch_size = cfg.val_batch_size
        self.shuffle = cfg.shuffle
        self.num_workers = cfg.num_workers
        self.pin_memory = cfg.pin_memory
        self.persistent_workers = cfg.persistent_workers
        self.train_dataset = None
        # initialize feature decoder
        self.feature_decoder = build_feature_decoder(cfg.feature_decoder_type, cfg.mm_config_path)
        self._collate_fn = PPOCollate(self.feature_decoder, cfg.decode_workers)

    def setup(self, stage: Optional[str] = None):
        assert self.buffer.buffer_full, 'The buffer should be full before training'

        # init the RLFT Dataset
        all_dataset = PPODataset(self.cfg, self.buffer)

        if stage == 'fit' or stage is None:
            self.train_dataset = all_dataset
        else:
            raise ValueError(f'CloseLoop fine-tuning currently only support ["fit"], got ${stage}.')

    def preprocess_buffer(self, model: torch.nn.Module):
        model_device = resolve_module_device(model)
        
        observations = self.buffer.get_key_data('ego_obs')
        next_observations = self.buffer.get_key_data('ego_next_obs')
        rewards = torch.from_numpy(np.stack(self.buffer.get_key_data('ego_reward'), axis=0)).to(model_device)  # (bs, )
        undones = 1.0 - torch.from_numpy(np.stack(self.buffer.get_key_data('ego_done'), axis=0)).float().to(model_device)  # (bs, )
        unterminated = 1.0 - torch.from_numpy(np.stack(self.buffer.get_key_data('ego_terminated'), axis=0)).float().to(model_device)  # (bs, )
        
        # critical feature for value net input
        cur_state = torch.stack([obs['critic_feature'] for obs in observations], dim=0).to(model_device)  # (bs, 4D)
        next_state = torch.stack([next_obs['critic_feature'] for next_obs in next_observations], dim=0).to(model_device)  # (bs, 4D)
        
        # critic model inference
        with torch.inference_mode():
            value = model.critic(cur_state)  # (bs, )
            next_value = model.critic(next_state)  # (bs, )

        # compute the advantages
        advantages = get_advantages_GAE(rewards, undones, value, next_value, unterminated, gamma=self.gamma, lambda_gae_adv=self.lambda_gae_adv)
        target_returns = advantages + value

        adv_std = advantages.std(unbiased=False) + 1e-5
        advantages = (advantages - advantages.mean()) / adv_std

        self.buffer.add_extra_data({
            'ego_state': cur_state.cpu(),               # [bs, 4D]
            'ego_advantage': advantages.cpu(),          # [bs]
            'ego_target_return': target_returns.cpu(),  # [bs]
            'ego_value': value.cpu(),                   # [bs]
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
