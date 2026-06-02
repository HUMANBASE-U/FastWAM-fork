from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PickCubeFastWAMSmall(nn.Module):
    """FastWAM-style PickCube policy with no test-time future imagination.

    Training uses an action loss plus a future-state auxiliary loss. Inference
    calls `predict_action_chunk`, which only uses the action head.
    """

    def __init__(
        self,
        obs_dim: int = 42,
        action_dim: int = 4,
        action_horizon: int = 8,
        future_horizon: int = 8,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.05,
        lambda_action: float = 1.0,
        lambda_future: float = 0.2,
        device: str = "cpu",
        model_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.future_horizon = int(future_horizon)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.lambda_action = float(lambda_action)
        self.lambda_future = float(lambda_future)
        self.device = torch.device(device)
        self.torch_dtype = model_dtype

        layers = [nn.LayerNorm(self.obs_dim), nn.Linear(self.obs_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(int(num_layers) - 1, 0)):
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        layers.append(nn.SiLU())
        self.encoder = nn.Sequential(*layers)
        self.action_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_horizon * self.action_dim),
        )
        self.future_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.future_horizon * self.obs_dim),
        )
        # Keep compatibility with earlier trainer conventions.
        self.dit = nn.ModuleDict(
            {
                "encoder": self.encoder,
                "action_head": self.action_head,
                "future_head": self.future_head,
            }
        )
        self.to(device=self.device, dtype=self.torch_dtype)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(obs)
        pred_action = self.action_head(z).view(obs.shape[0], self.action_horizon, self.action_dim)
        pred_future = self.future_head(z).view(obs.shape[0], self.future_horizon, self.obs_dim)
        return {"latent": z, "action": pred_action, "future_obs": pred_future}

    def training_loss(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs = sample["obs"].to(self.device, dtype=self.torch_dtype, non_blocking=True)
        action = sample["action"].to(self.device, dtype=self.torch_dtype, non_blocking=True)
        future_obs = sample["future_obs"].to(self.device, dtype=self.torch_dtype, non_blocking=True)
        out = self.forward(obs)
        loss_action = F.mse_loss(out["action"].float(), action.float())
        loss_future = F.mse_loss(out["future_obs"].float(), future_obs.float())
        loss_total = self.lambda_action * loss_action + self.lambda_future * loss_future
        return loss_total, {
            "loss_action": (self.lambda_action * loss_action).detach(),
            "loss_future": (self.lambda_future * loss_future).detach(),
        }

    @torch.no_grad()
    def predict_action_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device, dtype=self.torch_dtype)
        return self.forward(obs)["action"]

    def save_checkpoint(self, path: str, stats: dict[str, torch.Tensor] | None = None, step: int | None = None) -> str:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "step": step,
                "stats": stats,
                "config": {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "action_horizon": self.action_horizon,
                    "future_horizon": self.future_horizon,
                    "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim,
                    "lambda_action": self.lambda_action,
                    "lambda_future": self.lambda_future,
                },
            },
            path_obj,
        )
        return str(path_obj)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device)
        self.load_state_dict(payload["state_dict"], strict=True)
        return payload
