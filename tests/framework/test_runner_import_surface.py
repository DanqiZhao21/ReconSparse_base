import pathlib
import sys
from importlib import import_module

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from framework.batch.actor_learner import build_training_batch
from framework.io.buffer import BufferPaths, ensure_buffer_layout
from framework.runner.actor_runtime import actor_main
from framework.runner.agent_factory import build_agent
from framework.runner.config_normalization import normalize_actor_learner_cfg
from framework.runner.env_factory import build_actor_env, discover_scene_ids
from framework.runner.learner_factory import ValueNet, build_algorithm_bundle
from framework.runner.learner_runtime import learner_main
from framework.runner.orchestrator import orchestrator_main
from script.train_actor_learner_v2 import load_runner_entrypoints


def test_entrypoint_resolves_canonical_runner_helpers():
    entrypoints = load_runner_entrypoints()

    assert entrypoints["actor_main"] is actor_main
    assert entrypoints["learner_main"] is learner_main
    assert entrypoints["orchestrator_main"] is orchestrator_main
    assert entrypoints["normalize_actor_learner_cfg"] is normalize_actor_learner_cfg


def test_canonical_runner_modules_expose_expected_symbols(tmp_path):
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)

    assert callable(actor_main)
    assert callable(learner_main)
    assert callable(orchestrator_main)
    assert callable(normalize_actor_learner_cfg)
    assert callable(build_actor_env)
    assert callable(discover_scene_ids)
    assert callable(build_algorithm_bundle)
    assert callable(build_training_batch)
    assert ValueNet is not None
    assert paths.shards_dir.endswith("buffer/shards")


def test_build_agent_supports_dummy_smoke_backend(tmp_path):
    cfg = {
        "agent": {
            "type": "dummy",
            "ckpt": str(tmp_path / "dummy.ckpt"),
        },
        "train": {
            "policy_lr": 1e-3,
        },
    }

    agent = build_agent(cfg, device=torch.device("cpu"))
    logp = agent.logp_from_replay({"feature": 2.5})

    assert agent.trainable_module is not None
    assert torch.is_tensor(logp)
    assert logp.shape == torch.Size([])


def test_agent_package_lazy_exports_expected_symbols():
    agent_pkg = import_module("framework.agent")

    assert agent_pkg.Agent.__name__ == "Agent"
    assert agent_pkg.DummyPolicy.__name__ == "DummyPolicy"
    assert agent_pkg.DiffusionDriveV2Agent is agent_pkg.DiffusionDriveV2Policy
