import sys
import os

# Make repo-local imports robust regardless of CWD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RENDER_ROOT = os.path.join(_REPO_ROOT, "reconsimulator", "render")
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
if _RENDER_ROOT not in sys.path:
    sys.path.append(_RENDER_ROOT)

import copy
import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Optional
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from reconsimulator.render.utils.misc import import_str
from reconsimulator.envs import nus_config as cfg

'''
定位 3DGS trainer 加载、渲染状态提取、几何工具。
'''
# ----------------------------- In-process trainer cache ----------------------------- #
# GPU renderer/model initialization is expensive and can easily dominate rollout time.
# In single-process vectorization (SerialVecEnv), multiple env instances may request
# the same (device, scene) trainer; cache it to avoid duplicate loads.
_SPLAT_CACHE: dict[tuple[str, int], tuple[object, int]] = {}
_SPLAT_CACHE_ORDER: list[tuple[str, int]] = []
_SPLAT_CACHE_MAX = 2


def clear_splat_cache() -> None:
    _SPLAT_CACHE.clear()
    _SPLAT_CACHE_ORDER.clear()


# ----------------------------- 路径工具 ----------------------------- #
def _ckpt_path(scene: int) -> str:
    return os.path.join(cfg.BASE_DATA_DIR, f"{scene:03d}", "3DGS_without_prior", "checkpoint_final.pth")


def _trainer_config_path() -> str:
    prefer = os.path.join(cfg.INFO_DIR, "config.yaml")
    fallback = os.path.join("assets", "nus", "others", "config.yaml")
    if os.path.exists(prefer):
        return prefer
    return fallback


def get_splat(device: str, scene: int, *, use_cache: bool = True):
    """
    加载重建 trainer 与时间步（统一路径风格 + 健壮性处理）
    """
    key = (str(device), int(scene))
    if bool(use_cache) and key in _SPLAT_CACHE:
        # Refresh LRU order
        try:
            _SPLAT_CACHE_ORDER.remove(key)
        except ValueError:
            pass
        _SPLAT_CACHE_ORDER.append(key)
        trainer, num_timesteps = _SPLAT_CACHE[key]
        return trainer, int(num_timesteps)

    ckpt_name = _ckpt_path(scene)
    if not os.path.exists(ckpt_name):
        raise FileNotFoundError(f"[get_splat] checkpoint not found: {ckpt_name}")

    checkpoint = torch.load(ckpt_name, map_location=device, weights_only=False)

    cfg_path = _trainer_config_path()
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"[get_splat] trainer config not found: {cfg_path}")

    conf = OmegaConf.load(cfg_path)
    conf = OmegaConf.merge(conf, OmegaConf.from_cli([]))

    try:
        embeds = checkpoint["models"]["CamPose"]["embeds.weight"]
        num_timesteps = embeds.shape[0] // 6
    except Exception as e:
        raise KeyError(
            "[get_splat] cannot infer num_timesteps from checkpoint; "
            "expect checkpoint['models']['CamPose']['embeds.weight']"
        ) from e

    recon_trainer = import_str(conf.trainer.type)(
        **conf.trainer,
        num_timesteps=num_timesteps,
        model_config=conf.model,
        num_train_images=num_timesteps * 6,
        num_full_images=num_timesteps * 6,
        device=device,
    )

    num_timesteps = (num_timesteps - 1) // 6 * 6
    recon_trainer.resume_from_checkpoint(ckpt_path=ckpt_name, load_only_model=True)

    if bool(use_cache):
        _SPLAT_CACHE[key] = (recon_trainer, int(num_timesteps))
        try:
            _SPLAT_CACHE_ORDER.remove(key)
        except ValueError:
            pass
        _SPLAT_CACHE_ORDER.append(key)
        # Evict oldest
        while len(_SPLAT_CACHE_ORDER) > int(_SPLAT_CACHE_MAX):
            old = _SPLAT_CACHE_ORDER.pop(0)
            try:
                _SPLAT_CACHE.pop(old, None)
            except Exception:
                pass

    return recon_trainer, num_timesteps


# ----------------------------- 视锥采样 ----------------------------- #
def _pixel_grid(device: str, img_height: int, img_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        x, y = torch.meshgrid(
            torch.arange(img_width, device=device),
            torch.arange(img_height, device=device),
            indexing="xy",
        )
    except TypeError:
        x = torch.arange(img_width, device=device)
        y = torch.arange(img_height, device=device)
        x, y = torch.meshgrid(x, y)
    return x.flatten(), y.flatten()


def build_sky_view_template(
    intrinsics: torch.Tensor,
    device: str,
    img_height: int,
    img_width: int,
) -> dict[str, torch.Tensor]:
    x, y = _pixel_grid(device, img_height, img_width)
    if len(intrinsics.shape) == 3:
        intrinsics = intrinsics[0]
    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsics[0, 2] + 0.5) / intrinsics[0, 0],
                (y - intrinsics[1, 2] + 0.5) / intrinsics[1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )
    direction_norm = torch.linalg.norm(camera_dirs, dim=-1, keepdims=True)
    return {
        "camera_dirs": camera_dirs,
        "direction_norm": direction_norm,
    }


def get_sky_view_from_template(
    c2w: torch.Tensor,
    template: dict[str, torch.Tensor],
    img_height: int,
    img_width: int,
):
    camera_dirs = template["camera_dirs"]
    direction_norm = template["direction_norm"]
    if len(c2w.shape) == 3:
        c2w = c2w[0]
    directions = torch.matmul(camera_dirs, c2w[:3, :3].transpose(0, 1))
    origins = torch.broadcast_to(c2w[:3, 3], directions.shape)
    viewdirs = directions / (direction_norm + 1e-8)
    origins = origins.reshape(img_height, img_width, 3)
    viewdirs = viewdirs.reshape(img_height, img_width, 3)
    direction_norm = direction_norm.reshape(img_height, img_width, 1)
    return origins, viewdirs, direction_norm


def get_sky_view(c2w: torch.Tensor,
                 intrinsics: torch.Tensor,
                 device: str,
                 img_height: int,
                 img_width: int):
    template = build_sky_view_template(intrinsics, device, img_height, img_width)
    return get_sky_view_from_template(c2w, template, img_height, img_width)


def get_rays(
    x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]

    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    viewdirs = directions / (direction_norm + 1e-8)
    return origins, viewdirs, direction_norm


# ----------------------------- 渲染状态 ----------------------------- #
def get_state(trainer, loaded_image_infos, loaded_cam_infos, now_frame: Optional[int] = None):
    device = next(trainer.parameters()).device if hasattr(trainer, "parameters") else torch.device("cuda")
    loaded_image_infos = move_to_device(loaded_image_infos, device)
    loaded_cam_infos = move_to_device(loaded_cam_infos, device)

    cam_infos1 = copy.deepcopy(loaded_cam_infos)
    image_infos1 = copy.deepcopy(loaded_image_infos)
    results = trainer(image_infos1, cam_infos1)

    normalized_rgb = results["rgb"].clamp(0.0, 1.0).detach().cpu().numpy()
    scaled_rgb = (normalized_rgb * 255).astype(np.uint8)
    return scaled_rgb


# ----------------------------- 设备搬运 ----------------------------- #
def move_to_device(data, device):
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


# ----------------------------- SLERP ----------------------------- #
def slerp(r1: R, r2: R, t: float) -> R:
    q1 = r1.as_quat()
    q2 = r2.as_quat()

    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2

    if np.abs(dot) > 0.9995:
        q = (1 - t) * q1 + t * q2
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        q = (np.sin((1 - t) * theta) / sin_theta) * q1 + (np.sin(t * theta) / sin_theta) * q2

    return R.from_quat(q)
