from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from framework.lightning_compat import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class _TrajectoryUpdateDataset(Dataset):
    def __init__(self, batch: Dict[str, Any], *, include_obs: bool) -> None:
        self._include_obs = bool(include_obs)
        self._obs = batch.get("obs_batch", None)
        self._old_logp = batch.get("old_logp", None)
        self._old_value = batch.get("old_value", None)
        self._adv = batch.get("adv", None)
        self._ret = batch.get("ret", None)
        self._replay = list(batch.get("replay", []))
        self._size = int(self._adv.shape[0]) if torch.is_tensor(self._adv) else len(self._replay)

    def __len__(self) -> int:
        return int(self._size)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "adv": self._adv[index],
            "ret": self._ret[index],
            "replay": self._replay[index],
        }
        if torch.is_tensor(self._old_logp) and int(self._old_logp.numel()) > 0:
            item["old_logp"] = self._old_logp[index]
        if torch.is_tensor(self._old_value) and int(self._old_value.numel()) > 0:
            item["old_value"] = self._old_value[index]
        if self._include_obs and torch.is_tensor(self._obs) and int(self._obs.shape[0]) > 0:
            item["obs"] = self._obs[index]
        return item


def _collate_trajectory_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "adv": torch.stack([sample["adv"] for sample in samples], dim=0),
        "ret": torch.stack([sample["ret"] for sample in samples], dim=0),
        "replay": [sample["replay"] for sample in samples],
    }
    if "old_logp" in samples[0]:
        out["old_logp"] = torch.stack([sample["old_logp"] for sample in samples], dim=0)
    if "old_value" in samples[0]:
        out["old_value"] = torch.stack([sample["old_value"] for sample in samples], dim=0)
    if "obs" in samples[0]:
        out["obs"] = torch.stack([sample["obs"] for sample in samples], dim=0)
    return out


class TrajectoryUpdateDataModule(LightningDataModule):
    def __init__(
        self,
        batch: Dict[str, Any],
        *,
        minibatch_size: int,
        ddp_enabled: bool,
        world_size: int,
        rank: int,
        seed: int,
        update_seed: int,
        include_obs: bool,
        use_distributed_sampler: bool,
    ) -> None:
        super().__init__()
        self._batch = batch
        self._minibatch_size = max(1, int(minibatch_size))
        self._ddp_enabled = bool(ddp_enabled)
        self._world_size = max(1, int(world_size))
        self._rank = int(rank)
        self._seed = int(seed)
        self._update_seed = int(update_seed)
        self._include_obs = bool(include_obs)
        self._use_distributed_sampler = bool(use_distributed_sampler)
        self._dataset: Optional[_TrajectoryUpdateDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        del stage
        self._dataset = _TrajectoryUpdateDataset(self._batch, include_obs=self._include_obs)

    def train_dataloader(self) -> DataLoader:
        if self._dataset is None:
            self.setup("fit")
        sampler = None
        shuffle = True
        if self._ddp_enabled and self._use_distributed_sampler:
            sampler = DistributedSampler(
                self._dataset,
                num_replicas=int(self._world_size),
                rank=int(self._rank),
                shuffle=True,
                drop_last=False,
                seed=int(self._seed) + int(self._update_seed) * 1000,
            )
            shuffle = False
        return DataLoader(
            self._dataset,
            batch_size=int(self._minibatch_size),
            shuffle=bool(shuffle),
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            collate_fn=_collate_trajectory_samples,
        )