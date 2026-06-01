#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


def tensor_shape(value):
    return tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__


def main() -> None:
    os.chdir(ROOT)
    register_default_resolvers()
    overrides = ["task=demo_tiny_fastwam", "output_dir=./runs/demo_tiny_fastwam/smoke_test"]
    overrides.extend(sys.argv[1:])

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(cfg.output_dir)

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = instantiate(cfg.data.train)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(cfg.get("persistent_workers", False)) and int(cfg.num_workers) > 0,
    )
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.dit.parameters(), lr=float(cfg.learning_rate))

    batch = next(iter(loader))
    loss, loss_dict = model.training_loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Smoke-test loss is not finite: {loss}")

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print("SMOKE TEST OK")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"dataset_len: {len(dataset)}")
    print(f"batch video: {tensor_shape(batch['video'])}")
    print(f"batch action: {tensor_shape(batch['action'])}")
    print(f"batch proprio: {tensor_shape(batch['proprio'])}")
    print(f"batch context: {tensor_shape(batch['context'])}")
    print(f"loss: {float(loss.detach().cpu()):.6f}")
    print(f"loss_dict: {loss_dict}")
    print(f"grad_norm: {float(grad_norm):.6f}")
    if torch.cuda.is_available():
        print(f"gpu_mem_alloc_mb: {torch.cuda.memory_allocated() / (1024 ** 2):.1f}")
        print(f"gpu_mem_reserved_mb: {torch.cuda.memory_reserved() / (1024 ** 2):.1f}")
    else:
        print("gpu_mem: unavailable (torch.cuda.is_available() is False)")


if __name__ == "__main__":
    main()
