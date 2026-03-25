# Framework Structure

This directory contains the active RL training infrastructure for ReconDreamer.

## Main Path

- `agent/`: policy adapters used by the trainer.
- `algorithms/`: PPO and ReinforcePP update implementations.
- `env_wrapper/`: simulator-facing wrappers and vector-environment utilities.
- `rewards/`: reward computation logic shared by environment wrappers.
- `rollout/`: actor-side rollout collection and shard creation.
- `batch/`: learner-side shard loading, return/advantage building, and batch assembly.
- `io/`: buffer and checkpoint IO helpers for actor-learner coordination.
- `runner/`: actor, learner, orchestrator, and launch-environment setup.
- `utils/`: shared support utilities such as gsplat backend and extension warmup.

## Entry Point

- `script/train_actor_learner_v2.py` is the thin training entrypoint.
- `framework.runner.actor_learner` contains the main runtime roles.

## Cleanup Rule

When adding new training code, prefer extending one of the modules above instead
of creating compatibility wrappers or dated backup files inside `framework/`.

If a helper is not part of the current actor-learner path, keep it outside this
package or document clearly why it must stay.