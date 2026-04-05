from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import torch

from framework.batch import build_training_batch
from framework.io.buffer import BufferPaths, list_shards, read_int, stop_requested
from framework.io.shard_policy import (
    discard_incompatible_shards,
    discard_stale_shards,
    select_shards_for_update,
)
from framework.lightning.config import ActorLearnerLightningConfig
from framework.lightning.trajectory_datamodule import TrajectoryUpdateDataModule


class _EmptyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise IndexError(index)


class ActorLearnerUpdateDataModule(TrajectoryUpdateDataModule):
    def __init__(
        self,
        *,
        paths: BufferPaths,
        agent: Any,
        learner_config: ActorLearnerLightningConfig,
        device: torch.device,
        value_net: Optional[torch.nn.Module],
        ddp_enabled: bool,
        dist_module: Any,
        world_size: int,
        rank: int,
        stage_fn: Any,
        start_version: int,
    ) -> None:
        super().__init__(
            batch={},
            minibatch_size=int(learner_config.minibatch_size),
            ddp_enabled=bool(ddp_enabled),
            world_size=int(world_size),
            rank=int(rank),
            seed=int(learner_config.ddp_seed),
            update_seed=0,
            include_obs=bool(learner_config.include_obs),
            use_distributed_sampler=bool(learner_config.use_distributed_sampler),
        )
        self.paths = paths
        self.agent = agent
        self.learner_config = learner_config
        self.algo_key = str(learner_config.algo_kind)
        self.device = device
        self.gamma = float(learner_config.gamma)
        self.gae_lambda = float(learner_config.gae_lambda)
        self.value_net = value_net
        self.dist_module = dist_module
        self.rank = int(rank)
        self.mode = str(learner_config.mode).strip().lower()
        self.num_actors = int(learner_config.num_actors)
        self.shards_per_update = int(learner_config.shards_per_update)
        self.poll_s = float(learner_config.poll_s)
        self.max_shard_version_gap = int(learner_config.max_shard_version_gap)
        self.norm_eps = float(learner_config.norm_eps)
        self.inner_epochs = max(1, int(learner_config.inner_epochs))
        self.stage_fn = stage_fn
        self.start_version = int(start_version)

        self.current_selected: List[str] = []
        self.current_loaded = None
        self.current_wait_shards_s = 0.0
        self.current_load_shards_s = 0.0
        self.current_prepare_batch_s = 0.0
        self.current_weights_version = int(start_version)
        self.should_stop = False

    def current_inner_epoch_index(self) -> int:
        trainer = getattr(self, "trainer", None)
        current_epoch = int(getattr(trainer, "current_epoch", 0))
        return int(current_epoch % self.inner_epochs)

    def current_update_index(self) -> int:
        trainer = getattr(self, "trainer", None)
        current_epoch = int(getattr(trainer, "current_epoch", 0))
        return int(current_epoch // self.inner_epochs)

    def _is_new_update_epoch(self) -> bool:
        return self.current_inner_epoch_index() == 0 or self.current_loaded is None

    def _select_shards(self) -> List[str]:
        if self.rank != 0:
            selected: List[str] = []
        else:
            wait_t0 = time.time()
            self.stage_fn(f"[learner] stage1 collect: waiting shards mode={self.mode} actors={self.num_actors}")
            last_progress_t = 0.0
            selected = []
            while True:
                if stop_requested(self.paths):
                    break
                cur_ver_now = read_int(self.paths.version_file, default=self.start_version)
                self.current_weights_version = int(cur_ver_now)
                files = discard_stale_shards(
                    self.paths,
                    list_shards(self.paths),
                    cur_weights_version=int(cur_ver_now),
                    max_version_gap=int(self.max_shard_version_gap),
                )
                files = discard_incompatible_shards(
                    self.paths,
                    files,
                    agent=self.agent,
                    stage_fn=self.stage_fn,
                )
                selected = select_shards_for_update(
                    files,
                    mode=self.mode,
                    num_actors=self.num_actors,
                    shards_per_update=self.shards_per_update,
                )
                if len(selected) > 0:
                    break
                if time.time() - float(last_progress_t) >= 300.0:
                    last_progress_t = float(time.time())
                    self.stage_fn(
                        f"[learner] stage1 collect: have_shards={len(files)}/{max(1, int(self.shards_per_update))} (dir={self.paths.shards_dir})"
                    )
                time.sleep(self.poll_s)
            self.current_wait_shards_s = float(time.time() - wait_t0)

        if self._ddp_enabled:
            obj_list: List[Any] = [selected]
            self.dist_module.broadcast_object_list(obj_list, src=0)
            selected = obj_list[0]

        return list(selected)

    def train_dataloader(self):
        self._update_seed = int(self.trainer.current_epoch if self.trainer is not None else 0)
        if self._is_new_update_epoch():
            self.current_selected = self._select_shards()
            self.current_loaded = None
            self.current_load_shards_s = 0.0
            self.current_prepare_batch_s = 0.0

            if len(self.current_selected) == 0:
                self.should_stop = True
                return torch.utils.data.DataLoader(_EmptyDataset(), batch_size=1, shuffle=False, num_workers=0)

            load_t0 = time.time()
            self.current_loaded = build_training_batch(#framework/batch/actor_learner.py
                selected=self.current_selected,
                agent=self.agent,
                algo_key=self.algo_key,
                device=self.device,
                gamma=float(self.gamma),
                gae_lambda=float(self.gae_lambda),
                value_net=self.value_net,
                ddp_enabled=self._ddp_enabled,
                dist_module=self.dist_module,
                norm_eps=float(self.norm_eps),
            )
            self.current_load_shards_s = float(time.time() - load_t0)
            self.current_prepare_batch_s = 0.0
            self.should_stop = False
            self.stage_fn(
                f"[learner] stage2 train: selected_shards={len(self.current_selected)} inner_epochs={self.inner_epochs}"
            )
        else:
            self.current_load_shards_s = 0.0
            self.current_prepare_batch_s = 0.0
            self.should_stop = False

        self._batch = self.current_loaded.batch
        self._dataset = None
        return super().train_dataloader()
