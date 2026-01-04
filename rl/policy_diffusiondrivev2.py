import os
import sys
from typing import Dict, Tuple, Any

import numpy as np

# Ensure DiffusionDriveV2 is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DDV2_ROOT = os.path.join(REPO_ROOT, 'DiffusionDriveV2')
if DDV2_ROOT not in sys.path:
    sys.path.append(DDV2_ROOT)

try:
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_agent import Diffusiondrivev2_Sel_Agent
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
except Exception as e:
    Diffusiondrivev2_Sel_Agent = None
    TransfuserConfig = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class DiffusionDriveV2Policy:
    """
    Minimal RL policy wrapper around DiffusionDriveV2 selection agent.
    - Loads provided checkpoint for future fine-tuning.
    - Exposes `act(obs)` returning (ax, ay, flag) to match `RLReconEnv.step`.
    - Currently uses a simple fallback policy for action until full feature
      mapping (camera/lidar/status → model features) is integrated.
    """

    def __init__(self, x_anchor: int = 61, y_anchor: int = 61, ckpt_path: str | None = None, device: str | None = None):
        self.x_anchor = int(x_anchor)
        self.y_anchor = int(y_anchor)
        self.ckpt_path = ckpt_path
        self.device = device

        self._agent = None
        if _IMPORT_ERROR is not None:
            print(f"[DiffusionDriveV2Policy] Import error: {_IMPORT_ERROR}. Using fallback policy.")
        else:
            try:
                cfg = TransfuserConfig()
                # lr is irrelevant for inference-only; pass a small value
                self._agent = Diffusiondrivev2_Sel_Agent(config=cfg, lr=1e-4, checkpoint_path=self.ckpt_path)
                print(f"[DiffusionDriveV2Policy] Loaded DiffusionDriveV2 SEL agent from: {self.ckpt_path}")
            except Exception as e:
                print(f"[DiffusionDriveV2Policy] Failed to init agent ({e}). Using fallback policy.")
                self._agent = None

    def act(self, observation: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
        """
        Return anchor indices and flag for `RLReconEnv.step`.
        For now, returns a random anchor with flag=0 (use candidate anchors).
        Future: map observation to model features and use agent to score anchors.
        """
        #TODO:Placeholder: return empty metrics
        ax = np.random.randint(0, self.x_anchor)
        ay = np.random.randint(0, self.y_anchor)
        flag = 0
        return int(ax), int(ay), int(flag)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        #TODO: Placeholder: return empty metrics.
        return {"loss_pi": 0.0, "loss_v": 0.0}
