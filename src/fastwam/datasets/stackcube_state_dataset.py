from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class StackCubeStateSequenceDataset(Dataset):
    """State/action chunks from replayed ManiSkill StackCube demonstrations.

    The state sequence acts as a small, controllable proxy for video tokens:
    token 0 is the current observation, and tokens 1..K are privileged future
    observations that are available only during PFD training.
    """

    def __init__(
        self,
        traj_path: str = "~/.maniskill/demos/StackCube-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cpu.h5",
        action_horizon: int = 8,
        state_horizon: int = 8,
        max_trajectories: int | None = None,
        val_fraction: float = 0.1,
        split: str = "train",
        seed: int = 42,
        normalize: bool = True,
        stats: dict[str, torch.Tensor] | None = None,
    ):
        self.traj_path = Path(traj_path).expanduser()
        self.action_horizon = int(action_horizon)
        self.state_horizon = int(state_horizon)
        self.max_trajectories = None if max_trajectories is None else int(max_trajectories)
        self.val_fraction = float(val_fraction)
        self.split = str(split)
        self.seed = int(seed)
        self.normalize = bool(normalize)
        if self.split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train/val/all, got {split}")
        if not self.traj_path.exists():
            raise FileNotFoundError(f"StackCube trajectory file not found: {self.traj_path}")

        self.trajectories = self._load_trajectories()
        self.obs_dim = int(self.trajectories[0]["obs"].shape[-1])
        self.action_dim = int(self.trajectories[0]["actions"].shape[-1])
        self.indices = self._build_indices()
        self.stats = self._compute_stats() if stats is None else self._clone_stats(stats)

    def _clone_stats(self, stats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"obs_mean", "obs_std", "action_mean", "action_std"}
        missing = required.difference(stats)
        if missing:
            raise KeyError(f"Missing normalization stats: {sorted(missing)}")
        return {key: value.detach().cpu().float().clone() for key, value in stats.items()}

    def _load_trajectories(self) -> list[dict[str, np.ndarray]]:
        with h5py.File(self.traj_path, "r") as f:
            keys = sorted([k for k in f.keys() if k.startswith("traj_")], key=lambda k: int(k.split("_")[-1]))
            if self.max_trajectories is not None:
                keys = keys[: self.max_trajectories]
            rng = np.random.default_rng(self.seed)
            rng.shuffle(keys)
            if self.split != "all" and self.val_fraction > 0:
                n_val = max(1, int(round(len(keys) * self.val_fraction)))
                keys = keys[:n_val] if self.split == "val" else keys[n_val:]

            trajectories = []
            for key in keys:
                group = f[key]
                if "obs" not in group:
                    continue
                obs = np.asarray(group["obs"], dtype=np.float32)
                actions = np.asarray(group["actions"], dtype=np.float32)
                success = np.asarray(group.get("success", np.zeros(actions.shape[0], dtype=bool)))
                if obs.ndim != 2 or actions.ndim != 2:
                    continue
                if obs.shape[0] < actions.shape[0] + 1:
                    continue
                trajectories.append({"obs": obs, "actions": actions, "success": success, "key": key})
        if not trajectories:
            raise RuntimeError(f"No valid trajectories loaded from {self.traj_path}")
        return trajectories

    def _build_indices(self) -> list[tuple[int, int]]:
        indices = []
        min_tail = max(self.action_horizon, self.state_horizon)
        for traj_idx, traj in enumerate(self.trajectories):
            action_len = traj["actions"].shape[0]
            for step in range(max(action_len - min_tail + 1, 0)):
                indices.append((traj_idx, step))
        if not indices:
            raise RuntimeError("No valid chunk indices. Reduce action_horizon/state_horizon.")
        return indices

    def _compute_stats(self) -> dict[str, torch.Tensor]:
        obs = np.concatenate([traj["obs"][:-1] for traj in self.trajectories], axis=0)
        actions = np.concatenate([traj["actions"] for traj in self.trajectories], axis=0)
        return {
            "obs_mean": torch.from_numpy(obs.mean(axis=0).astype(np.float32)),
            "obs_std": torch.from_numpy((obs.std(axis=0) + 1e-6).astype(np.float32)),
            "action_mean": torch.from_numpy(actions.mean(axis=0).astype(np.float32)),
            "action_std": torch.from_numpy((actions.std(axis=0) + 1e-6).astype(np.float32)),
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        traj_idx, step = self.indices[idx]
        traj = self.trajectories[traj_idx]
        obs_seq = torch.from_numpy(traj["obs"][step : step + 1 + self.state_horizon]).float()
        action = torch.from_numpy(traj["actions"][step : step + self.action_horizon]).float()
        if self.normalize:
            obs_seq = (obs_seq - self.stats["obs_mean"]) / self.stats["obs_std"]
            action = (action - self.stats["action_mean"]) / self.stats["action_std"]
        return {
            "obs_seq": obs_seq,
            "action": action,
            "metadata": {
                "traj_key": traj["key"],
                "step": step,
                "split": self.split,
            },
        }

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return obs
        mean = self.stats["obs_mean"].to(obs.device, obs.dtype)
        std = self.stats["obs_std"].to(obs.device, obs.dtype)
        return (obs - mean) / std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return action
        mean = self.stats["action_mean"].to(action.device, action.dtype)
        std = self.stats["action_std"].to(action.device, action.dtype)
        return action * std + mean
