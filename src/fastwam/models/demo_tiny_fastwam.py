from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemoTinyWAMBackbone(nn.Module):
    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        proprio_dim: int,
        text_dim: int,
        hidden_dim: int,
        latent_dim: int,
        video_frames: int,
        video_loss_size: int,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.video_frames = int(video_frames)
        self.video_loss_size = int(video_loss_size)

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, hidden_dim),
            nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
        )
        self.proprio_encoder = nn.Sequential(
            nn.LayerNorm(proprio_dim),
            nn.Linear(proprio_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(hidden_dim, self.action_horizon * self.action_dim)
        self.video_head = nn.Linear(
            hidden_dim,
            3 * self.video_frames * self.video_loss_size * self.video_loss_size,
        )
        self.future_latent_head = nn.Linear(hidden_dim, self.latent_dim)
        self.action_from_latent_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_horizon * self.action_dim),
        )

    def encode_condition(
        self,
        video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        first_frame = video[:, :, 0]
        frame_feat = self.frame_encoder(first_frame)

        mask = context_mask.to(dtype=context.dtype).unsqueeze(-1)
        context_sum = (context * mask).sum(dim=1)
        context_count = mask.sum(dim=1).clamp(min=1.0)
        context_feat = self.context_encoder(context_sum / context_count)

        first_proprio = proprio[:, 0]
        proprio_feat = self.proprio_encoder(first_proprio)

        return self.fusion(torch.cat([frame_feat, context_feat, proprio_feat], dim=-1))

    def forward(
        self,
        video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode_condition(video, context, context_mask, proprio)
        pred_action = self.action_head(hidden).view(
            video.shape[0],
            self.action_horizon,
            self.action_dim,
        )
        pred_video_lowres = self.video_head(hidden).view(
            video.shape[0],
            3,
            self.video_frames,
            self.video_loss_size,
            self.video_loss_size,
        )
        pred_future_latent = self.future_latent_head(hidden)
        pred_action_from_latent = self.action_from_latent_head(pred_future_latent).view(
            video.shape[0],
            self.action_horizon,
            self.action_dim,
        )
        return {
            "action": pred_action,
            "video_lowres": pred_video_lowres,
            "future_latent": pred_future_latent,
            "action_from_latent": pred_action_from_latent,
        }


class DemoTinyFastWAM(nn.Module):
    """A tiny FastWAM proxy model for local smoke tests.

    This is not the paper model. It preserves the training surface that the
    repository trainer expects: a `.dit` trainable module, `training_loss`, and
    checkpoint helpers. Modes:
    - action_only: predict the action chunk from first frame, context, proprio.
    - joint_video_action: action loss plus low-resolution video auxiliary loss.
    - idm_proxy: predict a future latent first, then decode action from it.
    """

    def __init__(
        self,
        image_size: list[int] | tuple[int, int] = (64, 64),
        video_frames: int = 5,
        action_horizon: int = 8,
        action_dim: int = 7,
        proprio_dim: int = 8,
        context_len: int = 16,
        text_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 48,
        video_loss_size: int = 16,
        mode: str = "action_only",
        lambda_action: float = 1.0,
        lambda_video: float = 0.0,
        lambda_latent: float = 0.0,
        device: str = "cpu",
        model_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.image_size = tuple(int(v) for v in image_size)
        self.video_frames = int(video_frames)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.context_len = int(context_len)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.video_loss_size = int(video_loss_size)
        self.mode = str(mode)
        self.lambda_action = float(lambda_action)
        self.lambda_video = float(lambda_video)
        self.lambda_latent = float(lambda_latent)
        self.device = torch.device(device)
        self.torch_dtype = model_dtype

        if self.mode not in {"action_only", "joint_video_action", "idm_proxy"}:
            raise ValueError(f"Unsupported demo mode: {mode}")

        self.dit = DemoTinyWAMBackbone(
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            text_dim=self.text_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            video_frames=self.video_frames,
            video_loss_size=self.video_loss_size,
        )
        self.to(device=self.device, dtype=self.torch_dtype)

    def _move_batch(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        required = ("video", "action", "proprio", "context", "context_mask")
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"DemoTinyFastWAM sample missing keys: {missing}")

        batch = {
            "video": sample["video"].to(self.device, dtype=self.torch_dtype, non_blocking=True),
            "action": sample["action"].to(self.device, dtype=self.torch_dtype, non_blocking=True),
            "proprio": sample["proprio"].to(self.device, dtype=self.torch_dtype, non_blocking=True),
            "context": sample["context"].to(self.device, dtype=self.torch_dtype, non_blocking=True),
            "context_mask": sample["context_mask"].to(self.device, dtype=torch.bool, non_blocking=True),
        }
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            batch["action_is_pad"] = action_is_pad.to(self.device, dtype=torch.bool, non_blocking=True)
        return batch

    def _validate_batch(self, batch: dict[str, torch.Tensor]) -> None:
        video = batch["video"]
        action = batch["action"]
        proprio = batch["proprio"]
        context = batch["context"]
        context_mask = batch["context_mask"]
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[2] != self.video_frames:
            raise ValueError(f"Expected video_frames={self.video_frames}, got {video.shape[2]}")
        if action.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"`action` must be [B,{self.action_horizon},{self.action_dim}], got {tuple(action.shape)}"
            )
        if proprio.shape[1] < 1 or proprio.shape[2] != self.proprio_dim:
            raise ValueError(f"`proprio` must be [B,T,{self.proprio_dim}], got {tuple(proprio.shape)}")
        if context.shape[1:] != (self.context_len, self.text_dim):
            raise ValueError(
                f"`context` must be [B,{self.context_len},{self.text_dim}], got {tuple(context.shape)}"
            )
        if context_mask.shape[1] != self.context_len:
            raise ValueError(f"`context_mask` must have length {self.context_len}, got {tuple(context_mask.shape)}")

    def _action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        token_loss = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=-1)
        if action_is_pad is None:
            return token_loss.mean()
        valid = (~action_is_pad).to(device=token_loss.device, dtype=token_loss.dtype)
        return (token_loss * valid).sum() / valid.sum().clamp(min=1.0)

    def _video_lowres_target(self, video: torch.Tensor) -> torch.Tensor:
        bsz, channels, frames, _, _ = video.shape
        flattened = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, video.shape[3], video.shape[4])
        lowres = F.interpolate(
            flattened.float(),
            size=(self.video_loss_size, self.video_loss_size),
            mode="bilinear",
            align_corners=False,
        )
        return lowres.reshape(bsz, frames, channels, self.video_loss_size, self.video_loss_size).permute(0, 2, 1, 3, 4)

    def _future_latent_target(self, video: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(video[:, :, 1:].float(), output_size=(1, 4, 4)).flatten(1)
        if pooled.shape[1] == self.latent_dim:
            return pooled
        if pooled.shape[1] > self.latent_dim:
            return pooled[:, : self.latent_dim]
        pad = pooled.new_zeros(pooled.shape[0], self.latent_dim - pooled.shape[1])
        return torch.cat([pooled, pad], dim=1)

    def training_loss(self, sample: dict[str, Any], tiled: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        del tiled
        batch = self._move_batch(sample)
        self._validate_batch(batch)

        outputs = self.dit(
            video=batch["video"],
            context=batch["context"],
            context_mask=batch["context_mask"],
            proprio=batch["proprio"],
        )
        action_is_pad = batch.get("action_is_pad")

        if self.mode == "idm_proxy":
            pred_action = outputs["action_from_latent"]
        else:
            pred_action = outputs["action"]

        loss_action = self._action_loss(pred_action, batch["action"], action_is_pad)
        loss_total = self.lambda_action * loss_action
        loss_dict = {"loss_action": (self.lambda_action * loss_action).detach()}

        if self.mode == "joint_video_action" and self.lambda_video > 0:
            target_video = self._video_lowres_target(batch["video"])
            loss_video = F.mse_loss(outputs["video_lowres"].float(), target_video.float())
            loss_total = loss_total + self.lambda_video * loss_video
            loss_dict["loss_video"] = (self.lambda_video * loss_video).detach()

        if self.mode == "idm_proxy" and self.lambda_latent > 0:
            target_latent = self._future_latent_target(batch["video"])
            loss_latent = F.mse_loss(outputs["future_latent"].float(), target_latent.float())
            loss_total = loss_total + self.lambda_latent * loss_latent
            loss_dict["loss_latent"] = (self.lambda_latent * loss_latent).detach()

        return loss_total, loss_dict

    @torch.no_grad()
    def infer(self, *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("DemoTinyFastWAM only supports training_loss for tiny local reproduction.")

    def save_checkpoint(self, path: str, optimizer=None, step: int | None = None) -> str:
        del optimizer
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "step": step,
                "mode": self.mode,
                "config": {
                    "image_size": self.image_size,
                    "video_frames": self.video_frames,
                    "action_horizon": self.action_horizon,
                    "action_dim": self.action_dim,
                    "proprio_dim": self.proprio_dim,
                    "context_len": self.context_len,
                    "text_dim": self.text_dim,
                    "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim,
                    "video_loss_size": self.video_loss_size,
                },
            },
            path_obj,
        )
        return str(path_obj)

    def load_checkpoint(self, path: str, optimizer=None) -> dict[str, Any]:
        del optimizer
        payload = torch.load(path, map_location=self.device)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.load_state_dict(state_dict, strict=True)
        return payload if isinstance(payload, dict) else {"state_dict": state_dict}
