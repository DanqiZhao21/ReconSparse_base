"""
Standalone HUGSIM-ORI NuScenes evaluation for existing SparseDriveV2 ckpts.

Note:
--ckpts does not modify --hugsim-template in place. For each input ckpt, this
script writes a generated YAML under:
  <eval-output-root>/<run-name>/input_configs/
and replaces both:
  sparsedrive_v2_ckpt
  sparsedrive_v2_pretrain_ckpt
with the explicit ckpt path passed by --ckpts.

Typical 2-repeat / 88-scene run:

cd /root/clone/ReconDreamer-RL

PYTHONPATH=/root/clone/ReconDreamer-RL \
/root/miniconda3/envs/recondreamerNew-rl/bin/python -u script/evaluate_existing_sparsedrive_v2_ckpts.py \
  --ckpts /root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/20260612_collision_GRPO_latest.ckpt \
  --repeat-evals 2 \
  --max-scenes 88 \
  --slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7 \
  --run-name sparsedrive_v2_manual_eval_0612

Multiple ckpts in one evaluation batch:

PYTHONPATH=/root/clone/ReconDreamer-RL \
/root/miniconda3/envs/recondreamerNew-rl/bin/python -u script/evaluate_existing_sparsedrive_v2_ckpts.py \
  --ckpts /root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt \
  --repeat-evals 2 \
  --max-scenes 88 \
  --slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7 \
  --run-name sparsedrive_v2_compare_eval

Optional explicit HUGSIM template/output paths:

PYTHONPATH=/root/clone/ReconDreamer-RL \
/root/miniconda3/envs/recondreamerNew-rl/bin/python -u script/evaluate_existing_sparsedrive_v2_ckpts.py \
  --ckpts /root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt \
  --hugsim-template /root/clone/HUGSIM-ORI/configs/sim/nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml \
  --scenario-dir /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes \
  --eval-output-root /root/clone/HUGSIM-ORI/outputs/evaluate-auto \
  --repeat-evals 2 \
  --max-scenes 88 \
  --slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7 \
  --run-name sparsedrive_v2_manual_eval
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.smalltool.evaluateCache.evaluate_existing_sparsedrive_v2_ckpts import main


if __name__ == "__main__":
    raise SystemExit(main())
