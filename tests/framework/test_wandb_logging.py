import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from framework.runner import logging as logging_mod


def test_wandb_init_defines_update_and_sample_metric_views(monkeypatch):
    calls = []

    fake_wandb = SimpleNamespace(
        init=lambda **_kwargs: object(),
        define_metric=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(logging_mod, "_WANDB_AVAILABLE", True)
    monkeypatch.setattr(logging_mod, "wandb", fake_wandb)

    enabled = logging_mod.wandb_init_if_enabled(
        {
            "train": {
                "wandb": {
                    "enabled": True,
                    "project": "unit-test",
                }
            }
        },
        role="learner",
        ddp_enabled=False,
        rank=0,
    )

    assert enabled is True
    assert (("update",), {}) in calls
    assert (("global_sample_step",), {}) in calls
    assert (("global_train_seen_sample_step",), {}) in calls
    assert (("train_update/*",), {"step_metric": "update"}) in calls
    assert (("train_seen_samples/*",), {"step_metric": "global_train_seen_sample_step"}) in calls
