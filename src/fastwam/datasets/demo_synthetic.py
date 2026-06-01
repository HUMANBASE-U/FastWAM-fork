from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DemoSyntheticSpec:
    length: int
    video_frames: int
    image_size: tuple[int, int]
    action_horizon: int
    action_dim: int
    proprio_dim: int
    context_len: int
    text_dim: int
    num_tasks: int
    noise_std: float
    seed: int


class DemoSyntheticFastWAMDataset(Dataset):
    """Small in-memory dataset with FastWAM-like sample keys.

    The samples intentionally mimic the real training batch contract:
    video [3,T,H,W], action [A,D], proprio [A,P], context [L,E], masks, and
    padding flags. The content is synthetic and deterministic, so it is useful
    for smoke tests and tiny overfit runs without LIBERO/RoboTwin assets.
    """

    def __init__(
        self,
        length: int = 64,
        video_frames: int = 5,
        image_size: list[int] | tuple[int, int] = (64, 64),
        action_horizon: int = 8,
        action_dim: int = 7,
        proprio_dim: int = 8,
        context_len: int = 16,
        text_dim: int = 64,
        num_tasks: int = 4,
        noise_std: float = 0.01,
        seed: int = 123,
        pre_generate: bool = True,
    ):
        image_size = tuple(int(v) for v in image_size)
        if len(image_size) != 2:
            raise ValueError(f"`image_size` must be [H, W], got {image_size}")
        if video_frames <= 1 or video_frames % 4 != 1:
            raise ValueError("`video_frames` must be > 1 and satisfy T % 4 == 1.")
        if action_horizon % (video_frames - 1) != 0:
            raise ValueError("`action_horizon` must be divisible by video_frames - 1.")
        if action_dim <= 0 or proprio_dim <= 0 or context_len <= 0 or text_dim <= 0:
            raise ValueError("Action/proprio/context dimensions must be positive.")

        self.spec = DemoSyntheticSpec(
            length=int(length),
            video_frames=int(video_frames),
            image_size=image_size,
            action_horizon=int(action_horizon),
            action_dim=int(action_dim),
            proprio_dim=int(proprio_dim),
            context_len=int(context_len),
            text_dim=int(text_dim),
            num_tasks=int(num_tasks),
            noise_std=float(noise_std),
            seed=int(seed),
        )
        self.pre_generate = bool(pre_generate)
        self._samples = [self._make_sample(i) for i in range(self.spec.length)] if self.pre_generate else None

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def proprio_dim(self) -> int:
        return self.spec.proprio_dim

    @property
    def text_dim(self) -> int:
        return self.spec.text_dim

    def __len__(self) -> int:
        return self.spec.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        sample = self._samples[idx] if self._samples is not None else self._make_sample(idx)
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in sample.items()
        }

    def _make_sample(self, idx: int) -> dict[str, Any]:
        spec = self.spec
        gen = torch.Generator(device="cpu").manual_seed(spec.seed + int(idx))
        task_id = idx % spec.num_tasks

        init_xy = torch.rand(2, generator=gen) * 1.2 - 0.6
        velocity = torch.rand(2, generator=gen) * 0.5 - 0.25
        task_phase = torch.tensor(float(task_id) / max(spec.num_tasks - 1, 1))
        task_vec = self._task_context(task_id)

        action = self._make_action(init_xy, velocity, task_phase, gen)
        proprio = self._make_proprio(init_xy, velocity, task_phase)
        video = self._make_video(init_xy, velocity, task_phase, gen)

        prompt = f"demo task {task_id}: move a colored blob with synthetic robot actions"
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": prompt,
            "context": task_vec,
            "context_mask": torch.ones(spec.context_len, dtype=torch.bool),
            "image_is_pad": torch.zeros(spec.video_frames, dtype=torch.bool),
            "action_is_pad": torch.zeros(spec.action_horizon, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(spec.action_horizon, dtype=torch.bool),
            "metadata": {
                "idx": idx,
                "task_id": task_id,
                "synthetic": True,
            },
        }

    def _task_context(self, task_id: int) -> torch.Tensor:
        spec = self.spec
        pos = torch.linspace(0.0, 1.0, spec.context_len).unsqueeze(1)
        dim = torch.linspace(0.0, 1.0, spec.text_dim).unsqueeze(0)
        phase = float(task_id + 1)
        context = torch.sin((phase + 1.0) * pos * 3.14159) * torch.cos((phase + 0.5) * dim * 3.14159)
        return context.to(torch.float32)

    def _make_action(
        self,
        init_xy: torch.Tensor,
        velocity: torch.Tensor,
        task_phase: torch.Tensor,
        gen: torch.Generator,
    ) -> torch.Tensor:
        spec = self.spec
        t = torch.linspace(0.0, 1.0, spec.action_horizon)
        cols = [
            velocity[0] + 0.10 * torch.sin(2.0 * torch.pi * (t + task_phase)),
            velocity[1] + 0.10 * torch.cos(2.0 * torch.pi * (t + task_phase)),
            init_xy[0] + velocity[0] * t,
            init_xy[1] + velocity[1] * t,
            torch.sin(torch.pi * t + task_phase),
            torch.cos(torch.pi * t + task_phase),
            torch.where(t < 0.5, torch.full_like(t, -1.0), torch.ones_like(t)),
        ]
        action = torch.stack(cols, dim=-1)
        if spec.action_dim > action.shape[1]:
            pad = torch.zeros(spec.action_horizon, spec.action_dim - action.shape[1])
            action = torch.cat([action, pad], dim=1)
        action = action[:, : spec.action_dim]
        if spec.noise_std > 0:
            action = action + spec.noise_std * torch.randn(action.shape, generator=gen)
        return action.to(torch.float32)

    def _make_proprio(self, init_xy: torch.Tensor, velocity: torch.Tensor, task_phase: torch.Tensor) -> torch.Tensor:
        spec = self.spec
        t = torch.linspace(0.0, 1.0, spec.action_horizon)
        base = torch.stack(
            [
                init_xy[0] + velocity[0] * t,
                init_xy[1] + velocity[1] * t,
                velocity[0].expand_as(t),
                velocity[1].expand_as(t),
                task_phase.expand_as(t),
                t,
                torch.sin(2.0 * torch.pi * t),
                torch.cos(2.0 * torch.pi * t),
            ],
            dim=-1,
        )
        if spec.proprio_dim > base.shape[1]:
            pad = torch.zeros(spec.action_horizon, spec.proprio_dim - base.shape[1])
            base = torch.cat([base, pad], dim=1)
        return base[:, : spec.proprio_dim].to(torch.float32)

    def _make_video(
        self,
        init_xy: torch.Tensor,
        velocity: torch.Tensor,
        task_phase: torch.Tensor,
        gen: torch.Generator,
    ) -> torch.Tensor:
        spec = self.spec
        height, width = spec.image_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        frames = []
        for step in range(spec.video_frames):
            tau = float(step) / max(spec.video_frames - 1, 1)
            center = init_xy + velocity * tau * 2.0
            blob = torch.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / 0.08)
            stripe = 0.5 + 0.5 * torch.sin(6.0 * xx + task_phase * torch.pi + tau)
            channel_r = blob
            channel_g = stripe * (0.4 + 0.6 * blob)
            channel_b = 0.5 + 0.5 * torch.cos(4.0 * yy + tau * torch.pi)
            frame = torch.stack([channel_r, channel_g, channel_b], dim=0)
            frames.append(frame)
        video = torch.stack(frames, dim=1).clamp(0.0, 1.0)
        if spec.noise_std > 0:
            video = video + spec.noise_std * torch.randn(video.shape, generator=gen)
        return (video.clamp(0.0, 1.0) * 2.0 - 1.0).to(torch.float32)
