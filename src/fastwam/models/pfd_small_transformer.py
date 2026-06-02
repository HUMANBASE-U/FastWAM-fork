from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype)
        * -(torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype)) / max(half - 1, 1))
    )
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualAdapter(nn.Module):
    """Token-wise PFD residual adapter with zero-initialized output."""

    def __init__(self, action_dim: int, time_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.time_dim = int(time_dim)
        self.net = nn.Sequential(
            nn.Linear(action_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, v_base: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(tau.to(dtype=v_base.dtype), self.time_dim)
        t_emb = t_emb[:, None, :].expand(v_base.shape[0], v_base.shape[1], -1)
        return self.net(torch.cat([v_base, t_emb], dim=-1))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class StackCubePFDSmallTransformer(nn.Module):
    """Small PFD proxy model for StackCube.

    It reproduces the PFD mechanism on state-sequence tokens:
    - student mask: action tokens see current state token + action tokens
    - teacher mask: action tokens see all state tokens + action tokens
    - teacher/student share the same transformer parameters
    - residual adapter corrects the current-only student output
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        action_horizon: int = 8,
        state_horizon: int = 8,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.05,
        adapter_hidden_dim: int = 512,
        lambda_video: float = 1.0,
        lambda_gt: float = 1.0,
        lambda_res: float = 0.5,
        lambda_teacher: float = 0.1,
        use_pfd: bool = True,
        device: str = "cpu",
        model_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.state_horizon = int(state_horizon)
        self.num_state_tokens = self.state_horizon + 1
        self.hidden_dim = int(hidden_dim)
        self.use_pfd = bool(use_pfd)
        self.lambda_video = float(lambda_video)
        self.lambda_gt = float(lambda_gt)
        self.lambda_res = float(lambda_res)
        self.lambda_teacher = float(lambda_teacher)
        self.device = torch.device(device)
        self.torch_dtype = model_dtype

        self.state_proj = nn.Linear(self.obs_dim, hidden_dim)
        self.action_proj = nn.Linear(self.action_dim, hidden_dim)
        self.type_embed = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(self.num_state_tokens + self.action_horizon, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, self.action_dim)
        self.state_head = nn.Linear(hidden_dim, self.obs_dim)
        self.adapter = ResidualAdapter(self.action_dim, time_dim=64, hidden_dim=adapter_hidden_dim)
        # Compatibility with trainer patterns: the trainable backbone is still named `dit`.
        self.dit = nn.ModuleDict(
            {
                "state_proj": self.state_proj,
                "action_proj": self.action_proj,
                "blocks": self.blocks,
                "norm": self.norm,
                "action_head": self.action_head,
                "state_head": self.state_head,
            }
        )
        self.to(device=self.device, dtype=self.torch_dtype)

    def _attention_mask(self, teacher: bool, device: torch.device) -> torch.Tensor:
        total = self.num_state_tokens + self.action_horizon
        state_end = self.num_state_tokens
        mask = torch.zeros(total, total, device=device, dtype=torch.bool)
        if not teacher:
            # Prevent indirect future leakage: if the current-state token can
            # aggregate future-state tokens, action tokens could read future
            # information through that current token.
            mask[0, 1:state_end] = True
            action_rows = slice(state_end, total)
            future_state_cols = slice(1, state_end)
            mask[action_rows, future_state_cols] = True
        return mask

    def _encode_tokens(self, obs_seq: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        state_tokens = self.state_proj(obs_seq) + self.type_embed[0]
        action_tokens = self.action_proj(action_tokens) + self.type_embed[1]
        tokens = torch.cat([state_tokens, action_tokens], dim=1)
        return tokens + self.pos_embed[None, :, :]

    def _forward_masked(self, obs_seq: torch.Tensor, action_tokens: torch.Tensor, teacher: bool) -> dict[str, torch.Tensor]:
        x = self._encode_tokens(obs_seq, action_tokens)
        mask = self._attention_mask(teacher=teacher, device=x.device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.norm(x)
        state_x = x[:, : self.num_state_tokens]
        action_x = x[:, self.num_state_tokens :]
        return {
            "pred_action": self.action_head(action_x),
            "pred_state": self.state_head(state_x[:, 1:]),
        }

    def forward(self, obs_seq: torch.Tensor, action_tokens: torch.Tensor, tau: torch.Tensor, teacher: bool = False) -> dict[str, torch.Tensor]:
        out = self._forward_masked(obs_seq, action_tokens, teacher=teacher)
        if teacher:
            return out
        v_base = out["pred_action"]
        delta = self.adapter(v_base, tau) if self.use_pfd else torch.zeros_like(v_base)
        out["v_base"] = v_base
        out["delta_hat"] = delta
        out["v_final"] = v_base + delta
        return out

    def training_loss(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_seq = sample["obs_seq"].to(self.device, dtype=self.torch_dtype, non_blocking=True)
        action = sample["action"].to(self.device, dtype=self.torch_dtype, non_blocking=True)
        batch = action.shape[0]
        tau = torch.zeros(batch, device=self.device, dtype=self.torch_dtype)
        action_tokens = torch.zeros_like(action)

        student = self.forward(obs_seq, action_tokens, tau=tau, teacher=False)
        loss_video = F.mse_loss(student["pred_state"].float(), obs_seq[:, 1:].float())
        loss_gt = F.mse_loss(student["v_final"].float(), action.float())

        if self.use_pfd:
            with torch.no_grad():
                teacher = self.forward(obs_seq, action_tokens, tau=tau, teacher=True)
                v_teacher = teacher["pred_action"].detach()
                residual_target = (v_teacher - student["v_base"]).detach()
            loss_res = F.mse_loss(student["delta_hat"].float(), residual_target.float())
            loss_teacher = F.mse_loss(student["v_final"].float(), v_teacher.float())
            teacher_gap = F.mse_loss(student["v_base"].float(), v_teacher.float())
        else:
            loss_res = torch.zeros((), device=self.device)
            loss_teacher = torch.zeros((), device=self.device)
            teacher_gap = torch.zeros((), device=self.device)

        loss_total = (
            self.lambda_video * loss_video
            + self.lambda_gt * loss_gt
            + self.lambda_res * loss_res
            + self.lambda_teacher * loss_teacher
        )
        metrics = {
            "loss_video": (self.lambda_video * loss_video).detach(),
            "loss_gt": (self.lambda_gt * loss_gt).detach(),
            "loss_res": (self.lambda_res * loss_res).detach(),
            "loss_teacher": (self.lambda_teacher * loss_teacher).detach(),
            "teacher_gap": teacher_gap.detach(),
            "delta_norm": student["delta_hat"].detach().float().norm(dim=-1).mean(),
            "base_norm": student["v_base"].detach().float().norm(dim=-1).mean(),
        }
        return loss_total, metrics

    @torch.no_grad()
    def predict_action_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device, dtype=self.torch_dtype)
        obs_seq = torch.zeros((obs.shape[0], self.num_state_tokens, self.obs_dim), device=self.device, dtype=self.torch_dtype)
        obs_seq[:, 0] = obs
        action_tokens = torch.zeros((obs.shape[0], self.action_horizon, self.action_dim), device=self.device, dtype=self.torch_dtype)
        tau = torch.zeros(obs.shape[0], device=self.device, dtype=self.torch_dtype)
        return self.forward(obs_seq, action_tokens, tau=tau, teacher=False)["v_final"]

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
                    "state_horizon": self.state_horizon,
                    "hidden_dim": self.hidden_dim,
                    "use_pfd": self.use_pfd,
                    "lambda_video": self.lambda_video,
                    "lambda_gt": self.lambda_gt,
                    "lambda_res": self.lambda_res,
                    "lambda_teacher": self.lambda_teacher,
                },
            },
            path_obj,
        )
        return str(path_obj)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device)
        self.load_state_dict(payload["state_dict"], strict=True)
        return payload
