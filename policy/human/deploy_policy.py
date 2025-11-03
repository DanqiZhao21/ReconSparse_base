from typing import Dict, List, Any
import io
import logging
import traceback

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ----------------------------- Logging --------------------------------- #
logger = logging.getLogger("uvicorn.error")

# ----------------------------- Data Models ----------------------------- #
class Observation(BaseModel):
    back_left: list = Field(...)
    front_left: list = Field(...)
    front: list = Field(...)
    front_right: list = Field(...)
    back_right: list = Field(...)
    back: list = Field(...)

class ActionResponse(BaseModel):
    action: List[float]

# ----------------------------- Policy ---------------------------------- #
class Human:
    def __init__(self) -> None:
        self.observation_window: List[Dict[str, np.ndarray]] = []

    def update_observation_window(self, observation: Dict[str, np.ndarray]) -> None:
        self.observation_window.append(observation)

    @staticmethod
    def encode_obs(observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return observation

    def get_action(self, observation: Dict[str, np.ndarray], info: Dict) -> np.ndarray:
        return np.array([0, 0, True], dtype=np.float32)

policy = Human()

# ----------------------------- App ------------------------------------- #
app = FastAPI(title="Policy Server", version="1.0.0")

def serialize_item(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): serialize_item(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serialize_item(v) for v in x]
    return x


def build_obs_from_npz(npz) -> Dict[str, np.ndarray]:
    required_keys = ['back_left', 'front_left', 'front', 'front_right', 'back_right', 'back']
    for k in required_keys:
        if k not in npz:
            raise ValueError(f"missing key in npz: {k}")
    obs = {k: np.asarray(npz[k], dtype=np.float32) for k in required_keys}
    return obs

@app.post("/get_action/json", response_model=ActionResponse)
async def get_action_json(observation: Observation) -> ActionResponse:
    try:
        obs = {
            "back_left": np.asarray(observation.back_left, dtype=np.float32),
            "front_left": np.asarray(observation.front_left, dtype=np.float32),
            "front": np.asarray(observation.front, dtype=np.float32),
            "front_right": np.asarray(observation.front_right, dtype=np.float32),
            "back_right": np.asarray(observation.back_right, dtype=np.float32),
            "back": np.asarray(observation.back, dtype=np.float32),
        }
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid observation values: {e}")
    
    policy.update_observation_window(obs)

    try:
        action = policy.get_action(obs, info={})
    except Exception as e:
        logger.exception("policy.get_action failed")
        raise HTTPException(status_code=500, detail=f"policy.get_action error: {e}")

    return ActionResponse(action=serialize_item(action))

@app.post("/get_action/file")
async def get_action_file(obs: UploadFile = File(...)):
    if policy is None:
        raise HTTPException(status_code=500, detail="policy is not configured on server")

    try:
        contents = await obs.read()
        if not contents:
            raise HTTPException(status_code=400, detail="empty obs file")

        # 限制上传大小（示例 200 MB）
        max_bytes = 200 * 1024 * 1024
        if len(contents) > max_bytes:
            raise HTTPException(status_code=413, detail="payload too large")

        buf = io.BytesIO(contents)
        try:
            npz = np.load(buf, allow_pickle=False)
        except Exception as e:
            logger.exception("np.load failed")
            raise HTTPException(status_code=400, detail=f"np.load failed: {e}")

        try:
            obs_dict = build_obs_from_npz(npz)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        policy.update_observation_window(obs_dict)

        try:
            action = policy.get_action(obs_dict, info={})
        except Exception as e:
            logger.exception("policy.get_action failed")
            raise HTTPException(status_code=500, detail=f"policy.get_action error: {e}")

        return JSONResponse({"action": serialize_item(action)})

    except HTTPException:
        raise
    except Exception:
        logger.error("Unexpected server error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="internal server error")

@app.get("/healthz")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

# ----------------------------- Run ------------------------------------ #
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
