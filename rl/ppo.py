from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


_CAM_KEYS = ["front_left", "front", "front_right", "back_left", "back", "back_right"]


def _obs_to_tensor(
    observation: Dict[str, np.ndarray],
    device: torch.device,
    *,
    image_size: Tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """Convert env dict-of-images (uint8 HWC) to a (1, 18, H, W) float tensor."""
    imgs: list[np.ndarray] = []
    for k in _CAM_KEYS:
        img = observation.get(k, None)
        if img is None:
            raise KeyError(f"Missing camera key in observation: {k}")
        imgs.append(img)
    arr = np.stack(imgs, axis=0)  # (6,H,W,3)
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    t = t.permute(0, 3, 1, 2) / 255.0  # (6,3,H,W)
    t = t.reshape(1, 18, t.shape[-2], t.shape[-1])
    if image_size is not None:
        t = F.interpolate(t, size=image_size, mode="bilinear", align_corners=False)
    return t


class _ActorCritic(nn.Module):
    def __init__(self, x_anchor: int, y_anchor: int, image_size: Tuple[int, int] = (64, 64)):
        super().__init__()
        self.x_anchor = int(x_anchor)
        self.y_anchor = int(y_anchor)
        self.image_size = tuple(image_size)

        self.conv = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 18, self.image_size[0], self.image_size[1])
            n_flat = int(self.conv(dummy).view(1, -1).shape[1])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
        )
        self.pi_x = nn.Linear(512, self.x_anchor)
        self.pi_y = nn.Linear(512, self.y_anchor)
        self.v = nn.Linear(512, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.fc(self.conv(obs))
        logits_x = self.pi_x(h)
        logits_y = self.pi_y(h)
        value = self.v(h).squeeze(-1)
        return logits_x, logits_y, value


@dataclass
class PPOBatch:
    obs: torch.Tensor  # (T, C, H, W)
    act_x: torch.Tensor  # (T,)
    act_y: torch.Tensor  # (T,)
    logp: torch.Tensor  # (T,)
    adv: torch.Tensor  # (T,)
    ret: torch.Tensor  # (T,)


class PPOAgent:
    """A minimal but working PPO agent for the MultiDiscrete anchor action space.

    Notes
    - This is a true policy-gradient update (PPO); it does NOT backprop through the env.
    - Action distribution is factored: pi(ax) * pi(ay). logprob = logp_x + logp_y.
    """

    def __init__(
        self,
        *,
        x_anchor: int = 61,
        y_anchor: int = 61,
        device: str | torch.device | None = None,
        lr: float = 2e-4,
        lr_value: float | None = None,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 1,
        minibatch_size: int = 64,
        image_size: Tuple[int, int] = (64, 64),
        guidance_weight: float = 0.0,
        guidance_sigma: float = 4.0,
    ):
        self.x_anchor = int(x_anchor)
        self.y_anchor = int(y_anchor)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.guidance_weight = float(guidance_weight)
        self.guidance_sigma = float(guidance_sigma)

        self.model = _ActorCritic(self.x_anchor, self.y_anchor, image_size=image_size).to(self.device)
        if lr_value is None:
            lr_value = lr
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(self.model.conv.parameters()) + list(self.model.fc.parameters()) + list(self.model.pi_x.parameters()) + list(self.model.pi_y.parameters()), "lr": float(lr)},
                {"params": list(self.model.v.parameters()), "lr": float(lr_value)},
            ]
        )

    # ------------------------- Public API ------------------------- #
    def act(self, observation: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
        """Evaluation-friendly action: choose argmax (deterministic)."""
        obs_t = _obs_to_tensor(observation, self.device)
        self.model.eval()
        with torch.no_grad():
            logits_x, logits_y, _ = self.model(obs_t)
            ax = int(torch.argmax(logits_x, dim=-1).item())
            ay = int(torch.argmax(logits_y, dim=-1).item())
        return ax, ay, 0

    def step(
        self,
        observation: Dict[str, np.ndarray],
        *,
        guidance: Tuple[int, int] | None = None,
        sample: bool = True,
    ) -> Tuple[Tuple[int, int, int], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or greedy) action and return (action, logp, value, entropy)."""
        obs_t = _obs_to_tensor(observation, self.device)
        self.model.train()

        logits_x, logits_y, value = self.model(obs_t)
        if guidance is not None and self.guidance_weight > 0.0:
            gx, gy = int(guidance[0]), int(guidance[1])
            logits_x = logits_x + self._gaussian_guidance_1d(self.x_anchor, gx)
            logits_y = logits_y + self._gaussian_guidance_1d(self.y_anchor, gy)

        dist_x = Categorical(logits=logits_x)
        dist_y = Categorical(logits=logits_y)

        if sample:
            ax = dist_x.sample()
            ay = dist_y.sample()
        else:
            ax = torch.argmax(logits_x, dim=-1)
            ay = torch.argmax(logits_y, dim=-1)

        logp = dist_x.log_prob(ax) + dist_y.log_prob(ay)
        ent = dist_x.entropy() + dist_y.entropy()
        action = (int(ax.item()), int(ay.item()), 0)
        return action, logp.squeeze(0), value.squeeze(0), ent.squeeze(0)

    def value(self, observation: Dict[str, np.ndarray]) -> float:
        obs_t = _obs_to_tensor(observation, self.device)
        self.model.eval()
        with torch.no_grad():
            _, _, v = self.model(obs_t)
        return float(v.item())

    def update(self, batch: PPOBatch) -> Dict[str, float]:
        """Run PPO updates; returns training metrics."""
        self.model.train()

        obs = batch.obs.to(self.device)
        act_x = batch.act_x.to(self.device)
        act_y = batch.act_y.to(self.device)
        old_logp = batch.logp.to(self.device)
        adv = batch.adv.to(self.device)
        ret = batch.ret.to(self.device)

        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n = obs.shape[0]
        idxs = torch.arange(n, device=self.device)

        last_loss_pi = 0.0
        last_loss_v = 0.0
        last_entropy = 0.0
        last_approx_kl = 0.0

        for _ in range(self.ppo_epochs):
            perm = idxs[torch.randperm(n)]
            for start in range(0, n, self.minibatch_size):
                mb = perm[start : start + self.minibatch_size]
                logits_x, logits_y, v = self.model(obs[mb])

                dist_x = Categorical(logits=logits_x)
                dist_y = Categorical(logits=logits_y)
                logp = dist_x.log_prob(act_x[mb]) + dist_y.log_prob(act_y[mb])
                entropy = (dist_x.entropy() + dist_y.entropy()).mean()

                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[mb]
                loss_pi = -(torch.min(surr1, surr2)).mean()

                loss_v = F.mse_loss(v, ret[mb])

                loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * entropy

                approx_kl = (old_logp[mb] - logp).mean().detach()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                last_loss_pi = float(loss_pi.item())
                last_loss_v = float(loss_v.item())
                last_entropy = float(entropy.item())
                last_approx_kl = float(approx_kl.item())

        return {
            "loss_pi": last_loss_pi,
            "loss_v": last_loss_v,
            "entropy": last_entropy,
            "approx_kl": last_approx_kl,
        }

    # ------------------------- Helpers ------------------------- #
    def _gaussian_guidance_1d(self, n: int, center: int) -> torch.Tensor:
        idx = torch.arange(n, device=self.device, dtype=torch.float32)
        c = float(np.clip(center, 0, n - 1))
        sigma = max(1e-6, self.guidance_sigma)
        g = -((idx - c) ** 2) / (2.0 * sigma * sigma)
        g = g * self.guidance_weight
        return g.view(1, -1)
