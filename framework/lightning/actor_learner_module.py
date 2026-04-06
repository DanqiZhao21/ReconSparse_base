from __future__ import annotations

import os
import time
from typing import Any, Dict

from framework.io.buffer import move_to_consumed, prune_consumed, read_int, write_int
from framework.lightning.trajectory_module import TrajectoryLightningModule

try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore


class ActorLearnerLightningModule(TrajectoryLightningModule):
    def __init__(
        self,
        *,
        paths: Any,
        stage_fn: Any,
        ddp_enabled: bool,
        dist_module: Any,
        rank: int,
        wandb_enabled: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.paths = paths
        self.stage_fn = stage_fn
        self.ddp_enabled = bool(ddp_enabled)
        self.dist_module = dist_module
        self.rank = int(rank)
        self.wandb_enabled = bool(wandb_enabled)
        self.global_sample_step = 0
        self.global_train_seen_sample_step = 0
        self._update_train_t0 = 0.0
        self._latest_epoch_had_data = False

    def _inner_epochs(self) -> int:
        return max(1, int(self.learner_config.inner_epochs))

    def _inner_epoch_index(self) -> int:
        return int(self.current_epoch % self._inner_epochs())

    def _update_index(self) -> int:
        return int(self.current_epoch // self._inner_epochs())

    def _is_update_start(self) -> bool:
        return self._inner_epoch_index() == 0

    def _is_update_end(self) -> bool:
        return self._inner_epoch_index() == (self._inner_epochs() - 1)

    def on_train_epoch_start(self) -> None:
        datamodule = self.trainer.datamodule
        self._latest_epoch_had_data = bool(getattr(datamodule, "current_selected", []))
        if bool(getattr(datamodule, "should_stop", False)) or not self._latest_epoch_had_data:
            self.trainer.should_stop = True
            return
        if self._is_update_start():
            self._update_train_t0 = time.time()
            self._reset_update_metric_aggregates()
        if self.rank == 0 and self._is_update_start():
            with open(os.path.join(self.paths.root, "TRAINING_LOCK"), "w", encoding="utf-8") as handle:
                handle.write(f"training update={int(self._update_index())} time={time.time()}\n")

    def on_train_epoch_end(self) -> None:
        datamodule = self.trainer.datamodule
        if not self._latest_epoch_had_data:
            self.trainer.should_stop = True
            return

        if self.ddp_enabled and getattr(self.dist_module, "is_initialized", lambda: False)():
            self.dist_module.barrier()

        if not self._is_update_end():
            return

        if self.rank == 0:
            training_lock_file = os.path.join(self.paths.root, "TRAINING_LOCK")
            if os.path.exists(training_lock_file):
                try:
                    os.remove(training_lock_file)
                except Exception:
                    pass

            save_t0 = time.time()
            selected = list(getattr(datamodule, "current_selected", []))
            loaded = getattr(datamodule, "current_loaded", None)
            for fp in selected:
                move_to_consumed(self.paths, fp)
            prune_consumed(self.paths, keep_basenames={os.path.basename(fp) for fp in selected})
            cur_v = read_int(self.paths.version_file, default=1)
            new_v = int(cur_v) + 1
            try:
                self.agent.save_checkpoint(self.paths.latest_ckpt)
                write_int(self.paths.version_file, new_v)
            except Exception as exc:
                self.stage_fn(f"[learner] save/bump failed: {exc}")
            save_broadcast_s = float(time.time() - save_t0)

            train_time_s = float(time.time() - self._update_train_t0)
            update_time_s = float(getattr(datamodule, "current_wait_shards_s", 0.0)) + float(train_time_s)
            n = int(getattr(loaded, "num_samples", 0)) if loaded is not None else 0
            reward_sum = float(getattr(loaded, "reward_sum", 0.0)) if loaded is not None else 0.0
            reward_count = int(getattr(loaded, "reward_count", 0)) if loaded is not None else 0
            done_sum = float(getattr(loaded, "done_sum", 0.0)) if loaded is not None else 0.0
            done_count = int(getattr(loaded, "done_count", 0)) if loaded is not None else 0
            reward_mean = float(reward_sum) / float(max(1, reward_count))
            done_rate = float(done_sum) / float(max(1, done_count))
            ret = datamodule.current_loaded.batch["ret"]
            adv = datamodule.current_loaded.batch["adv"]
            metrics = self.aggregated_update_metrics()
            self.stage_fn(f"[learner] stage3 broadcast: ver={new_v}")
            self.stage_fn(
                f"[learner] update={int(self._update_index())} shards={len(selected)} "
                f"samples={n} ver={new_v} metrics={metrics}"
            )
            self.stage_fn(
                f"[learner] timing update={int(self._update_index())} "
                f"collect={float(getattr(datamodule, 'current_wait_shards_s', 0.0)):.2f}s "
                f"load={float(getattr(datamodule, 'current_load_shards_s', 0.0)):.2f}s "
                f"prepare={float(getattr(datamodule, 'current_prepare_batch_s', 0.0)):.2f}s "
                f"train={float(train_time_s):.2f}s save={float(save_broadcast_s):.2f}s "
                f"update={float(update_time_s):.2f}s "
                f"time_per_shard={float(train_time_s / float(max(1, len(selected)))):.2f}s"
            )

            if self.wandb_enabled and wandb is not None:
                global_sample_step = int(self.global_sample_step + n)
                payload: Dict[str, Any] = {
                    "update": int(self._update_index()),
                    "global_step": int(global_sample_step),
                    "global_sample_step": int(global_sample_step),
                    "global_train_seen_sample_step": int(self.global_train_seen_sample_step),
                    "weights_version": int(new_v),
                    "shards": int(len(selected)),
                    "samples": int(n),
                    "collect_time_s": float(getattr(datamodule, "current_wait_shards_s", 0.0)),
                    "load_shards_time_s": float(getattr(datamodule, "current_load_shards_s", 0.0)),
                    "prepare_batch_time_s": float(getattr(datamodule, "current_prepare_batch_s", 0.0)),
                    "train_time_s": float(train_time_s),
                    "update_time_s": float(update_time_s),
                    "save_broadcast_time_s": float(save_broadcast_s),
                    "reward_sum": float(reward_sum),
                    "reward_mean": float(reward_mean),
                    "done_rate": float(done_rate),
                    "ret_mean": float(ret.detach().mean().item()) if int(ret.numel()) > 0 else 0.0,
                    "ret_std": float(ret.detach().std(unbiased=False).item()) if int(ret.numel()) > 0 else 0.0,
                    "adv_std": float(adv.detach().std(unbiased=False).item()) if int(adv.numel()) > 0 else 0.0,
                    "time_per_sample_s": float(train_time_s / float(max(1, n))),
                    "time_per_shard_s": float(train_time_s / float(max(1, len(selected)))),
                }
                for key, value in metrics.items():
                    try:
                        payload[str(key)] = float(value)
                    except Exception:
                        continue
                update_view = {
                    "reward_sum": float(reward_sum),
                    "reward_mean": float(reward_mean),
                    "done_rate": float(done_rate),
                    "ret_mean": float(payload["ret_mean"]),
                    "ret_std": float(payload["ret_std"]),
                    "adv_std": float(payload["adv_std"]),
                    "samples": float(n),
                    "shards": float(len(selected)),
                    "weights_version": float(new_v),
                    "collect_time_s": float(payload["collect_time_s"]),
                    "load_shards_time_s": float(payload["load_shards_time_s"]),
                    "prepare_batch_time_s": float(payload["prepare_batch_time_s"]),
                    "train_time_s": float(payload["train_time_s"]),
                    "update_time_s": float(payload["update_time_s"]),
                    "save_broadcast_time_s": float(payload["save_broadcast_time_s"]),
                    "time_per_sample_s": float(payload["time_per_sample_s"]),
                    "time_per_shard_s": float(payload["time_per_shard_s"]),
                }
                update_view.update({key: float(value) for key, value in metrics.items()})
                for key, value in update_view.items():
                    payload[f"train_update/{key}"] = float(value)
                try:
                    self.global_sample_step += int(n)
                    wandb.log(payload)
                except Exception as exc:
                    self.stage_fn(f"[wandb] log failed: {exc}")

        if self.ddp_enabled and getattr(self.dist_module, "is_initialized", lambda: False)():
            self.dist_module.barrier()
