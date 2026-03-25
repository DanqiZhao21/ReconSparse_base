from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import torch

from framework.batch import build_training_batch
from framework.io.buffer import BufferPaths, list_shards, read_int, stop_requested
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
        algo_key: str,
        device: torch.device,
        gamma: float,
        gae_lambda: float,
        value_net: Optional[torch.nn.Module],
        value_optim: Optional[torch.optim.Optimizer],
        ddp_enabled: bool,
        dist_module: Any,
        world_size: int,
        rank: int,
        seed: int,
        minibatch_size: int,
        include_obs: bool,
        use_distributed_sampler: bool,
        mode: str,
        num_actors: int,
        shards_per_update: int,
        poll_s: float,
        max_shard_version_gap: int,
        norm_eps: float,
        stage_fn: Any,
        start_version: int,
    ) -> None:
        super().__init__(
            batch={},
            minibatch_size=int(minibatch_size),
            ddp_enabled=bool(ddp_enabled),
            world_size=int(world_size),
            rank=int(rank),
            seed=int(seed),
            update_seed=0,
            include_obs=bool(include_obs),
            use_distributed_sampler=bool(use_distributed_sampler),
        )
        self.paths = paths
        self.agent = agent
        self.algo_key = str(algo_key)
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.value_net = value_net
        self.value_optim = value_optim
        self.dist_module = dist_module
        self.rank = int(rank)
        self.mode = str(mode).strip().lower()
        self.num_actors = int(num_actors)
        self.shards_per_update = int(shards_per_update)
        self.poll_s = float(poll_s)
        self.max_shard_version_gap = int(max_shard_version_gap)
        self.norm_eps = float(norm_eps)
        self.stage_fn = stage_fn
        self.start_version = int(start_version)

        self.current_selected: List[str] = []
        self.current_loaded = None
        self.current_wait_shards_s = 0.0
        self.current_load_shards_s = 0.0
        self.current_prepare_batch_s = 0.0
        self.current_weights_version = int(start_version)
        self.should_stop = False

    @staticmethod
    def _parse_shard_weights_version(filename: str) -> Optional[int]:
        try:
            text = str(filename)
            start = text.find("_v")
            if start < 0:
                return None
            start += 2
            end = start
            while end < len(text) and text[end].isdigit():
                end += 1
            if end == start:
                return None
            return int(text[start:end])
        except Exception:
            return None

    def _filter_and_discard_stale_shards(self, shard_files: List[str], *, cur_weights_version: int) -> List[str]:
        max_gap = max(0, min(2, int(self.max_shard_version_gap)))
        upcoming = int(cur_weights_version) + 1
        min_ok = int(upcoming - max_gap)
        kept: List[str] = []
        stale: List[str] = []
        from framework.io.buffer import move_to_consumed

        for fp in shard_files:
            version = self._parse_shard_weights_version(os.path.basename(fp))
            if version is None:
                stale.append(fp)
                continue
            if int(version) < int(min_ok):
                stale.append(fp)
                continue
            if int(version) > int(cur_weights_version):
                continue
            kept.append(fp)
        for fp in stale:
            move_to_consumed(self.paths, fp)
        return kept

    def _filter_and_discard_incompatible_shards(self, shard_files: List[str]) -> List[str]:
        validator = getattr(self.agent, "replay_is_compatible", None)
        if not callable(validator):
            return shard_files

        from framework.io.buffer import move_to_consumed

        kept: List[str] = []
        dropped = 0
        for fp in shard_files:
            try:
                shard = torch.load(fp, map_location="cpu")
                replay = list(shard.get("replay", []))
                if len(replay) == 0:
                    kept.append(fp)
                    continue
                if all(bool(validator(rep)) for rep in replay):
                    kept.append(fp)
                    continue
            except Exception as exc:
                self.stage_fn(f"[learner] dropping incompatible shard {os.path.basename(fp)}: {exc}")
                move_to_consumed(self.paths, fp)
                dropped += 1
                continue

            self.stage_fn(f"[learner] dropping incompatible shard {os.path.basename(fp)} due to replay schema mismatch")
            move_to_consumed(self.paths, fp)
            dropped += 1

        if dropped > 0:
            self.stage_fn(f"[learner] discarded {dropped} incompatible shard(s) from {self.paths.shards_dir}")
        return kept

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
                files = self._filter_and_discard_stale_shards(
                    list_shards(self.paths),
                    cur_weights_version=int(cur_ver_now),
                )
                files = self._filter_and_discard_incompatible_shards(files)
                if self.mode.startswith("sync"):
                    have = set()
                    for fp in files:
                        name = os.path.basename(fp)
                        for actor_idx in range(self.num_actors):
                            if name.startswith(f"actor{actor_idx}_"):
                                have.add(actor_idx)
                    if len(have) >= self.num_actors:
                        per: Dict[int, str] = {}
                        for fp in files:
                            name = os.path.basename(fp)
                            for actor_idx in range(self.num_actors):
                                if name.startswith(f"actor{actor_idx}_") and actor_idx not in per:
                                    per[actor_idx] = fp
                        selected = [per[a] for a in sorted(per.keys())][: self.num_actors]
                        break
                else:
                    if len(files) >= max(1, int(self.shards_per_update)):
                        selected = files[: int(self.shards_per_update)]
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
        self.current_selected = self._select_shards()
        self.current_loaded = None
        self.current_load_shards_s = 0.0
        self.current_prepare_batch_s = 0.0

        if len(self.current_selected) == 0:
            self.should_stop = True
            return torch.utils.data.DataLoader(_EmptyDataset(), batch_size=1, shuffle=False, num_workers=0)

        load_t0 = time.time()
        self.current_loaded = build_training_batch(
            selected=self.current_selected,
            algo_key=self.algo_key,
            device=self.device,
            gamma=float(self.gamma),
            gae_lambda=float(self.gae_lambda),
            value_net=self.value_net,
            value_optim=self.value_optim,
            ddp_enabled=self._ddp_enabled,
            dist_module=self.dist_module,
            norm_eps=float(self.norm_eps),
        )
        self.current_load_shards_s = float(time.time() - load_t0)
        self.current_prepare_batch_s = 0.0
        self.should_stop = False
        self.stage_fn(f"[learner] stage2 train: selected_shards={len(self.current_selected)}")

        self._batch = self.current_loaded.batch
        self._dataset = None
        return super().train_dataloader()