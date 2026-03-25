from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from framework.utils.gsplat_backend import ensure_gsplat_legacy_backend, should_use_legacy_gsplat


if should_use_legacy_gsplat():
    from gsplat.cuda_legacy._wrapper import (
        project_gaussians,
        rasterize_gaussians,
        spherical_harmonics as _legacy_spherical_harmonics,
    )

    def spherical_harmonics(*args, **kwargs):
        ensure_gsplat_legacy_backend()
        return _legacy_spherical_harmonics(*args, **kwargs)


    def rasterization(
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        sh_degree: Optional[int] = None,
        packed: bool = True,
        tile_size: int = 16,
        backgrounds: Optional[Tensor] = None,
        render_mode: str = 'RGB',
        sparse_grad: bool = False,
        absgrad: bool = False,
        rasterize_mode: str = 'classic',
        channel_chunk: int = 32,
        distributed: bool = False,
        ortho: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict]:
        ensure_gsplat_legacy_backend()
        del far_plane, radius_clip, packed, sparse_grad, absgrad, rasterize_mode, channel_chunk, distributed, ortho
        assert eps2d == 0.3, 'legacy gsplat hard-codes eps2d=0.3'
        assert render_mode in ['RGB', 'D', 'ED', 'RGB+D', 'RGB+ED'], render_mode
        C = len(viewmats)
        render_colors = []
        render_alphas = []
        means2d_all = []
        radii_all = []
        depths_all = []

        for cid in range(C):
            fx, fy = Ks[cid, 0, 0], Ks[cid, 1, 1]
            cx, cy = Ks[cid, 0, 2], Ks[cid, 1, 2]
            viewmat = viewmats[cid]

            means2d, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
                means3d=means,
                scales=scales,
                glob_scale=1.0,
                quats=quats,
                viewmat=viewmat,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_height=height,
                img_width=width,
                block_width=tile_size,
                clip_thresh=near_plane,
            )
            means2d_all.append(means2d)
            radii_all.append(radii)
            depths_all.append(depths)

            if colors.dim() == 3:
                c2w = viewmat.inverse()
                viewdirs = means - c2w[:3, 3]
                degree = int(math.sqrt(colors.shape[1]) - 1) if sh_degree is None else int(sh_degree)
                rgb = spherical_harmonics(degree, viewdirs, colors)
            else:
                rgb = colors

            if render_mode == 'RGB':
                payload = rgb
            elif render_mode in ('D', 'ED'):
                payload = depths[..., None]
            else:
                payload = torch.cat([rgb, depths[..., None]], dim=-1)

            if backgrounds is not None:
                background = backgrounds[cid]
                if payload.shape[-1] != background.shape[-1]:
                    pad = torch.zeros(payload.shape[-1] - background.shape[-1], device=background.device, dtype=background.dtype)
                    background = torch.cat([background, pad], dim=-1)
            else:
                background = torch.zeros(payload.shape[-1], device=means.device, dtype=payload.dtype)

            rendered, alpha = rasterize_gaussians(
                xys=means2d,
                depths=depths,
                radii=radii,
                conics=conics,
                num_tiles_hit=num_tiles_hit,
                colors=payload,
                opacity=opacities[..., None],
                img_height=height,
                img_width=width,
                block_width=tile_size,
                background=background,
                return_alpha=True,
            )
            alpha = alpha[..., None]
            if render_mode in ('ED', 'RGB+ED'):
                rendered[..., -1:] = rendered[..., -1:] / torch.clamp(alpha, min=1e-10)
            render_colors.append(rendered)
            render_alphas.append(alpha)

        meta = {
            'means2d': torch.stack(means2d_all, dim=0),
            'radii': torch.stack(radii_all, dim=0),
            'depths': torch.stack(depths_all, dim=0),
            'width': int(width),
            'height': int(height),
        }
        return torch.stack(render_colors, dim=0), torch.stack(render_alphas, dim=0), meta
else:
    from gsplat.cuda._wrapper import spherical_harmonics
    from gsplat.rendering import rasterization


__all__ = ['rasterization', 'should_use_legacy_gsplat', 'spherical_harmonics']
