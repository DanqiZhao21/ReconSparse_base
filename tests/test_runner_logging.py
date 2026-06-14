from __future__ import annotations

from types import SimpleNamespace

from framework.runner import logging as runner_logging


class _FakeWandb:
    def __init__(self) -> None:
        self.run = SimpleNamespace(url="https://wandb.ai/unit/project/runs/abc123")
        self.init_kwargs = None
        self.defined_metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def init(self, **kwargs):
        self.init_kwargs = dict(kwargs)

    def define_metric(self, *args, **kwargs) -> None:
        self.defined_metrics.append((tuple(args), dict(kwargs)))


def test_wandb_init_logs_resolved_run_url(monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    messages: list[str] = []

    monkeypatch.setattr(runner_logging, "wandb", fake_wandb)
    monkeypatch.setattr(runner_logging, "_WANDB_AVAILABLE", True)
    monkeypatch.setattr(runner_logging, "stage", lambda msg: messages.append(str(msg)))

    enabled = runner_logging.wandb_init_if_enabled(
        {
            "train": {
                "wandb": {
                    "enabled": True,
                    "project": "ReconDiff",
                    "entity": "2564380679-",
                }
            }
        },
        role="learner",
        ddp_enabled=False,
        rank=0,
    )

    assert enabled is True
    assert fake_wandb.init_kwargs["entity"] == "2564380679-"
    assert any("https://wandb.ai/unit/project/runs/abc123" in message for message in messages)


def test_wandb_init_generates_explicit_run_id_when_not_configured(monkeypatch) -> None:
    fake_wandb = _FakeWandb()

    monkeypatch.setattr(runner_logging, "wandb", fake_wandb)
    monkeypatch.setattr(runner_logging, "_WANDB_AVAILABLE", True)
    monkeypatch.setattr(runner_logging, "stage", lambda _msg: None)
    monkeypatch.setattr(runner_logging.time, "strftime", lambda _fmt: "20260610_031500")

    enabled = runner_logging.wandb_init_if_enabled(
        {
            "train": {
                "wandb": {
                    "enabled": True,
                    "project": "ReconDiff",
                }
            }
        },
        role="learner",
        ddp_enabled=False,
        rank=0,
    )

    assert enabled is True
    assert str(fake_wandb.init_kwargs["id"]).startswith("learner_20260610_031500_")


def test_wandb_init_defines_only_update_level_metrics(monkeypatch) -> None:
    fake_wandb = _FakeWandb()

    monkeypatch.setattr(runner_logging, "wandb", fake_wandb)
    monkeypatch.setattr(runner_logging, "_WANDB_AVAILABLE", True)
    monkeypatch.setattr(runner_logging, "stage", lambda _msg: None)

    enabled = runner_logging.wandb_init_if_enabled(
        {
            "train": {
                "wandb": {
                    "enabled": True,
                    "project": "ReconDiff",
                    "log_minibatch_metrics": True,
                    "log_legacy_raw_metrics": True,
                }
            }
        },
        role="learner",
        ddp_enabled=False,
        rank=0,
    )

    assert enabled is True
    defined = {str(args[0]) for args, _kwargs in fake_wandb.defined_metrics}
    assert "progress/update" in defined
    for namespace in ["data", "time", "optim", "reward", "reward_gate", "terminal", "batch"]:
        assert f"{namespace}/*" in defined
    assert "debug/train_seen_samples" not in defined
    assert "debug/minibatch/*" not in defined
    assert "train_update/*" not in defined
    assert "train_seen_samples/*" not in defined
    assert "global_step" not in defined
    assert "global_train_seen_sample_step" not in defined
