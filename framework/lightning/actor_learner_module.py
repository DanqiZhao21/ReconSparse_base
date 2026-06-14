from __future__ import annotations

import os
import time
from typing import Any, Dict

from framework.io.debug_retention import (
    archive_selected_shards_for_debug,
    copy_latest_to_history,
    should_retain_version,
)
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
            save_t0 = time.time()
            selected = list(getattr(datamodule, "current_selected", []))
            loaded = getattr(datamodule, "current_loaded", None)
            cur_v = read_int(self.paths.version_file, default=1)
            new_v = int(cur_v) + 1
            try:
                self.agent.save_checkpoint(self.paths.latest_ckpt)
                write_int(self.paths.version_file, new_v)
                retain_versions = int(getattr(self.learner_config, "debug_retain_versions", 0) or 0)
                if bool(getattr(self.learner_config, "debug_retain_ckpts", False)) and should_retain_version(
                    version=int(new_v),
                    retain_versions=int(retain_versions),
                ):
                    try:
                        history_path = copy_latest_to_history(self.paths, version=int(new_v))
                        self.stage_fn(f"[learner] debug retained checkpoint ver={int(new_v)} path={history_path}")
                    except Exception as exc:
                        self.stage_fn(f"[learner] debug checkpoint retention failed: {exc}")
                if (
                    bool(getattr(self.learner_config, "debug_retain_shards", False))
                    and len(selected) > 0
                    and should_retain_version(version=int(new_v), retain_versions=int(retain_versions))
                ):
                    try:
                        manifest_path = archive_selected_shards_for_debug(
                            self.paths,
                            selected=selected,
                            update_index=int(self._update_index()),
                            cur_version=int(cur_v),
                            new_version=int(new_v),
                        )
                        self.stage_fn(f"[learner] debug retained shards manifest={manifest_path}")
                    except Exception as exc:
                        self.stage_fn(f"[learner] debug shard retention failed: {exc}")
                for fp in selected:
                    move_to_consumed(self.paths, fp)
                prune_consumed(self.paths, keep_basenames={os.path.basename(fp) for fp in selected})
                if os.path.exists(training_lock_file):
                    try:
                        os.remove(training_lock_file)
                    except Exception:
                        pass
            except Exception as exc:
                self.stage_fn(f"[learner] save/bump failed: {exc}")
                raise
            save_broadcast_s = float(time.time() - save_t0)

            train_time_s = float(time.time() - self._update_train_t0)
            update_time_s = float(getattr(datamodule, "current_wait_shards_s", 0.0)) + float(train_time_s)
            n = int(getattr(loaded, "num_samples", 0)) if loaded is not None else 0
            reward_sum = float(getattr(loaded, "reward_sum", 0.0)) if loaded is not None else 0.0
            reward_count = int(getattr(loaded, "reward_count", 0)) if loaded is not None else 0
            done_sum = float(getattr(loaded, "done_sum", 0.0)) if loaded is not None else 0.0
            done_count = int(getattr(loaded, "done_count", 0)) if loaded is not None else 0
            reward_summary = dict(getattr(loaded, "reward_summary", {}) or {}) if loaded is not None else {}
            shard_outcomes = dict(getattr(loaded, "shard_outcomes", {}) or {}) if loaded is not None else {}
            reward_mean = float(reward_sum) / float(max(1, reward_count))
            done_rate = float(done_sum) / float(max(1, done_count))
            reward_summary_steps = float(max(1.0, float(reward_summary.get("step_count", 0.0))))
            shard_den = float(max(1, len(selected)))
            full_horizon_count = float(shard_outcomes.get("full_horizon_count", 0.0) or 0.0)
            env_done_count = float(shard_outcomes.get("env_done_count", 0.0) or 0.0)
            timeout_count = float(shard_outcomes.get("timeout_count", 0.0) or 0.0)
            forced_failure_count = float(shard_outcomes.get("forced_failure_count", 0.0) or 0.0)
            partial_nonterminal_count = float(shard_outcomes.get("partial_nonterminal_count", 0.0) or 0.0)
            shard_outcome_view = {
                "normal_end_rate": float(full_horizon_count + env_done_count + timeout_count) / shard_den,
                "forced_failure_rate": float(forced_failure_count) / shard_den,
                "full_horizon_rate": float(full_horizon_count) / shard_den,
                "env_done_rate": float(env_done_count) / shard_den,
                "timeout_rate": float(timeout_count) / shard_den,
                "partial_nonterminal_rate": float(partial_nonterminal_count) / shard_den,
            }
            reward_summary_view = {
                "positive_reward_mean": float(reward_summary.get("positive_reward_sum", 0.0)) / reward_summary_steps,
                "gated_positive_reward_mean": float(reward_summary.get("gated_positive_reward_sum", 0.0)) / reward_summary_steps,
                "cost_reward_mean": float(reward_summary.get("cost_reward_sum", 0.0)) / reward_summary_steps,
                "safety_gate_rate": float(reward_summary.get("safety_gate_active_count", 0.0)) / reward_summary_steps,
                "collision_gate_rate": float(reward_summary.get("collision_gate_count", 0.0)) / reward_summary_steps,
                "severe_tracking_lateral_gate_rate": float(reward_summary.get("severe_tracking_lateral_gate_count", 0.0)) / reward_summary_steps,
                "severe_tracking_yaw_gate_rate": float(reward_summary.get("severe_tracking_yaw_gate_count", 0.0)) / reward_summary_steps,
            }
            ret = datamodule.current_loaded.batch["ret"]
            adv = datamodule.current_loaded.batch["adv"]
            metrics = self.aggregated_update_metrics()
            timing_parts = self.aggregated_update_timing()
            self.stage_fn(f"[learner] stage3 broadcast: ver={new_v}")
            self.stage_fn(
                f"[learner] update={int(self._update_index())} shards={len(selected)} "
                f"samples={n} ver={new_v} metrics={metrics}"
            )
            self.stage_fn(
                f"[learner] reward_summary update={int(self._update_index())} "
                f"summary={reward_summary_view}"
            )
            self.stage_fn(
                f"[learner] shard_outcomes update={int(self._update_index())} "
                f"summary={shard_outcome_view}"
            )
            self.stage_fn(
                f"[learner] step_timing update={int(self._update_index())} parts={timing_parts}"
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
                    "progress/update": int(self._update_index()),
                    "progress/weights_version": int(new_v),
                    "progress/global_sample_step": int(global_sample_step),
                    "data/samples": int(n),
                    "data/shards": int(len(selected)),
                    "data/done_rate": float(done_rate),
                    "time/collect_s": float(getattr(datamodule, "current_wait_shards_s", 0.0)),
                    "time/load_shards_s": float(getattr(datamodule, "current_load_shards_s", 0.0)),
                    "time/train_s": float(train_time_s),
                    "time/update_s": float(update_time_s),
                    "time/save_broadcast_s": float(save_broadcast_s),
                    "reward/sum": float(reward_sum),
                    "reward/mean": float(reward_mean),
                    "reward/positive_mean": float(reward_summary_view["positive_reward_mean"]),
                    "reward/gated_positive_mean": float(reward_summary_view["gated_positive_reward_mean"]),
                    "reward/cost_mean": float(reward_summary_view["cost_reward_mean"]),
                    "reward_gate/safety_rate": float(reward_summary_view["safety_gate_rate"]),
                    "reward_gate/collision_rate": float(reward_summary_view["collision_gate_rate"]),
                    "reward_gate/severe_tracking_lateral_rate": float(
                        reward_summary_view["severe_tracking_lateral_gate_rate"]
                    ),
                    "reward_gate/severe_tracking_yaw_rate": float(
                        reward_summary_view["severe_tracking_yaw_gate_rate"]
                    ),
                    "shard/normal_end_rate": float(shard_outcome_view["normal_end_rate"]),
                    "shard/forced_failure_rate": float(shard_outcome_view["forced_failure_rate"]),
                    "shard/full_horizon_rate": float(shard_outcome_view["full_horizon_rate"]),
                    "shard/env_done_rate": float(shard_outcome_view["env_done_rate"]),
                    "shard/timeout_rate": float(shard_outcome_view["timeout_rate"]),
                    "shard/partial_nonterminal_rate": float(shard_outcome_view["partial_nonterminal_rate"]),
                    "batch/ret_mean": float(ret.detach().mean().item()) if int(ret.numel()) > 0 else 0.0,
                    "batch/ret_std": float(ret.detach().std(unbiased=False).item()) if int(ret.numel()) > 0 else 0.0,
                    "batch/adv_std": float(adv.detach().std(unbiased=False).item()) if int(adv.numel()) > 0 else 0.0,
                }
                for key, value in metrics.items():
                    try:
                        payload[f"optim/{key}"] = float(value)
                    except Exception:
                        continue
                try:
                    self.global_sample_step += int(n)
                    wandb.log(payload, step=int(payload["progress/update"]), commit=True)
                except Exception as exc:
                    self.stage_fn(f"[wandb] log failed: {exc}")

        if self.ddp_enabled and getattr(self.dist_module, "is_initialized", lambda: False)():
            self.dist_module.barrier()
