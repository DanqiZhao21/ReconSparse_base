from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import torch

from framework.batch import build_training_batch
from framework.io.buffer import (
    BufferPaths,
    list_failed_actor_ids,
    list_shards,
    mark_stale_actor_heartbeats,
    read_int,
    stop_requested,
    write_actor_failure,
)
from framework.io.shard_policy import (
    discard_incompatible_shards,
    discard_stale_shards,
    resolve_async_shards_per_update,
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
        self.samples_per_update = int(getattr(learner_config, "samples_per_update", 0) or 0)
        self.max_inflight_per_actor = int(getattr(learner_config, "max_inflight_per_actor", 1))
        self.poll_s = float(learner_config.poll_s)
        self.shard_collect_timeout_s = float(getattr(learner_config, "shard_collect_timeout_s", 0.0))
        self.allow_partial_updates_after_timeout = bool(
            getattr(learner_config, "allow_partial_updates_after_timeout", False)
        )
        self.actor_heartbeat_timeout_s = float(getattr(learner_config, "actor_heartbeat_timeout_s", 0.0))
        self.actor_shard_stall_timeout_s = float(getattr(learner_config, "actor_shard_stall_timeout_s", 0.0))
        self.max_shard_version_lag = int(learner_config.max_shard_version_lag)
        self.norm_eps = float(learner_config.norm_eps)
        self.normalize_advantage = bool(getattr(learner_config, "normalize_advantage", True))
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

    def _actors_present(self, shard_files: List[str]) -> List[int]:
        present: set[int] = set()
        for fp in shard_files:
            name = os.path.basename(fp)
            if not name.startswith("actor"):
                continue
            tail = name[len("actor") :]
            actor_text = tail.split("_", 1)[0]
            try:
                present.add(int(actor_text))
            except Exception:
                continue
        return sorted(present)

    def _actor_id_from_shard_path(self, shard_path: str) -> Optional[int]:
        name = os.path.basename(shard_path)
        if not name.startswith("actor"):
            return None
        tail = name[len("actor") :]
        actor_text = tail.split("_", 1)[0]
        try:
            return int(actor_text)
        except Exception:
            return None

    def _mark_stalled_actor_shards(
        self,
        shard_files: List[str],
        *,
        failed_actor_ids: List[int],
        now: float,
    ) -> List[int]:
        timeout = float(self.actor_shard_stall_timeout_s)
        if timeout <= 0.0:
            return []

        failed = {int(actor_id) for actor_id in failed_actor_ids}
        progress: Dict[int, tuple[int, float]] = {}
        for fp in shard_files:
            actor_id = self._actor_id_from_shard_path(fp)
            if actor_id is None:
                continue
            try:
                mtime = float(os.path.getmtime(fp))
            except Exception:
                continue
            count, latest_mtime = progress.get(int(actor_id), (0, 0.0))
            progress[int(actor_id)] = (count + 1, max(float(latest_mtime), float(mtime)))

        marked: List[int] = []
        max_inflight = max(1, int(self.max_inflight_per_actor))
        for actor_id in range(int(self.num_actors)):
            aid = int(actor_id)
            if aid in failed:
                continue
            count, latest_mtime = progress.get(aid, (0, 0.0))
            if int(count) <= 0 or int(count) >= int(max_inflight):
                continue
            age_s = float(now) - float(latest_mtime)
            if age_s < timeout:
                continue
            write_actor_failure(
                self.paths,
                aid,
                message=(
                    "actor shard stall "
                    f"age_s={age_s:.2f} timeout_s={timeout:.2f} "
                    f"shards={int(count)}/{int(max_inflight)}"
                ),
            )
            marked.append(aid)

        if marked and callable(self.stage_fn):
            self.stage_fn(f"[learner] marked shard-stalled actor(s) failed: {marked}")
        return marked

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
            timed_out_actors: List[int] = []
            while True:
                if stop_requested(self.paths):
                    break
                if self.mode.startswith("async") and float(self.actor_heartbeat_timeout_s) > 0.0:
                    mark_stale_actor_heartbeats(
                        self.paths,
                        list(range(int(self.num_actors))),
                        timeout_s=float(self.actor_heartbeat_timeout_s),
                        now=float(time.time()),
                        stage_fn=self.stage_fn,
                    )
                failed_actor_ids = list_failed_actor_ids(self.paths) if self.mode.startswith("async") else []
                cur_ver_now = read_int(self.paths.version_file, default=self.start_version)
                self.current_weights_version = int(cur_ver_now)
                files = discard_stale_shards(
                    self.paths,
                    list_shards(self.paths),
                    cur_weights_version=int(cur_ver_now),
                    max_version_lag=int(self.max_shard_version_lag),
                )
                files = discard_incompatible_shards(
                    self.paths,
                    files,
                    agent=self.agent,
                    stage_fn=self.stage_fn,
                )
                effective_shards_per_update = int(self.shards_per_update)
                if self.mode.startswith("async"):
                    timed_out_actors = []
                    if float(self.actor_shard_stall_timeout_s) > 0.0:
                        marked_stalled_actors = self._mark_stalled_actor_shards(
                            files,
                            failed_actor_ids=list(failed_actor_ids),
                            now=float(time.time()),
                        )
                        if len(marked_stalled_actors) > 0:
                            failed_actor_ids = sorted(
                                {int(actor_id) for actor_id in failed_actor_ids}
                                | {int(actor_id) for actor_id in marked_stalled_actors}
                            )
                    effective_shards_per_update = resolve_async_shards_per_update(
                        requested_shards_per_update=int(self.shards_per_update),
                        num_actors=int(self.num_actors),
                        max_inflight_per_actor=int(self.max_inflight_per_actor),
                        failed_actor_ids=list(failed_actor_ids),
                    )
                    if effective_shards_per_update <= 0:
                        selected = []
                        break
                    if (
                        bool(self.allow_partial_updates_after_timeout)
                        and float(self.shard_collect_timeout_s) > 0.0
                        and len(files) > 0
                        and len(files) < int(effective_shards_per_update)
                        and (time.time() - float(wait_t0)) >= float(self.shard_collect_timeout_s)
                    ):
                        present_actor_ids = self._actors_present(files)
                        timed_out_actors = [
                            actor_id
                            for actor_id in range(int(self.num_actors))
                            if int(actor_id) not in present_actor_ids and int(actor_id) not in failed_actor_ids
                        ]
                        effective_shards_per_update = resolve_async_shards_per_update(
                            requested_shards_per_update=int(self.shards_per_update),
                            num_actors=int(self.num_actors),
                            max_inflight_per_actor=int(self.max_inflight_per_actor),
                            failed_actor_ids=list(failed_actor_ids) + list(timed_out_actors),
                        )
                selected = select_shards_for_update(
                    files,
                    mode=self.mode,
                    num_actors=self.num_actors,
                    shards_per_update=effective_shards_per_update,
                    samples_per_update=int(self.samples_per_update),
                )
                if len(selected) > 0:
                    break
                if time.time() - float(last_progress_t) >= 300.0:
                    last_progress_t = float(time.time())
                    failed_suffix = ""
                    if len(failed_actor_ids) > 0:
                        failed_suffix = f" failed_actors={failed_actor_ids}"
                    timeout_suffix = ""
                    if len(timed_out_actors) > 0:
                        timeout_suffix = f" timed_out_actors={timed_out_actors}"
                    self.stage_fn(
                        f"[learner] stage1 collect: have_shards={len(files)}/{max(1, int(effective_shards_per_update))}"
                        f"{failed_suffix}{timeout_suffix} (dir={self.paths.shards_dir})"
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
                normalize_advantage=bool(self.normalize_advantage),
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
