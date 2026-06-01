#!/usr/bin/env python
from __future__ import annotations

import csv
import multiprocessing.connection
import os
import subprocess
import sys
import time
from itertools import cycle
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


def multiprocessing_workers_supported() -> tuple[bool, str]:
    try:
        listener = multiprocessing.connection.Listener(("127.0.0.1", 0), family="AF_INET")
        listener.close()
    except Exception as exc:
        return False, repr(exc)
    return True, "ok"


def query_gpu() -> tuple[str, str]:
    if not torch.cuda.is_available():
        return "NA", "NA"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).strip()
    except Exception:
        return "NA", "NA"
    first = output.splitlines()[0].split(",")
    if len(first) < 2:
        return "NA", "NA"
    return first[0].strip(), first[1].strip()


def run_setting(cfg, setting: dict[str, object], device: str, model_dtype: torch.dtype) -> dict[str, object]:
    workers_ok, workers_reason = multiprocessing_workers_supported()
    if int(setting["num_workers"]) > 0 and not workers_ok:
        return {
            "setting": (
                f"bs={setting['batch_size']}, workers={setting['num_workers']}, "
                f"pin={setting['pin_memory']}, persistent={setting['persistent_workers']}"
            ),
            "step_time_s": "SKIP",
            "samples_per_s": "SKIP",
            "gpu_util_approx": "NA",
            "memory_used_mb": "NA",
            "torch_memory_alloc_mb": "NA",
            "torch_memory_reserved_mb": "NA",
            "final_loss": "SKIP",
            "notes": f"skipped: multiprocessing workers unavailable: {workers_reason}",
        }

    dataset = instantiate(cfg.data.train)
    loader = DataLoader(
        dataset,
        batch_size=int(setting["batch_size"]),
        shuffle=False,
        num_workers=int(setting["num_workers"]),
        pin_memory=bool(setting["pin_memory"]),
        persistent_workers=bool(setting["persistent_workers"]) and int(setting["num_workers"]) > 0,
    )
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.dit.parameters(), lr=float(cfg.learning_rate))

    max_steps = int(setting["steps"])
    warmup_steps = int(setting["warmup"])
    iterator = cycle(loader)
    step_times = []
    last_loss = None

    for step_idx in range(max_steps + warmup_steps):
        batch = next(iterator)
        start = time.perf_counter()
        loss, _ = model.training_loss(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        last_loss = float(loss.detach().cpu())
        if step_idx >= warmup_steps:
            step_times.append(elapsed)

    util, memory_used = query_gpu()
    memory_alloc = "NA"
    memory_reserved = "NA"
    if torch.cuda.is_available():
        memory_alloc = f"{torch.cuda.memory_allocated() / (1024 ** 2):.1f}"
        memory_reserved = f"{torch.cuda.memory_reserved() / (1024 ** 2):.1f}"

    avg_step_time = sum(step_times) / max(len(step_times), 1)
    return {
        "setting": (
            f"bs={setting['batch_size']}, workers={setting['num_workers']}, "
            f"pin={setting['pin_memory']}, persistent={setting['persistent_workers']}"
        ),
        "step_time_s": f"{avg_step_time:.4f}",
        "samples_per_s": f"{int(setting['batch_size']) / max(avg_step_time, 1e-9):.2f}",
        "gpu_util_approx": util,
        "memory_used_mb": memory_used,
        "torch_memory_alloc_mb": memory_alloc,
        "torch_memory_reserved_mb": memory_reserved,
        "final_loss": f"{last_loss:.6f}",
        "notes": "CUDA unavailable" if not torch.cuda.is_available() else "nvidia-smi snapshot",
    }


def main() -> None:
    os.chdir(ROOT)
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=demo_tiny_fastwam"])
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    misc.register_work_dir("./runs/demo_tiny_fastwam/benchmark")

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    settings = [
        {"batch_size": 1, "num_workers": 0, "pin_memory": False, "persistent_workers": False, "warmup": 2, "steps": 8},
        {"batch_size": 2, "num_workers": 0, "pin_memory": False, "persistent_workers": False, "warmup": 2, "steps": 8},
        {"batch_size": 4, "num_workers": 0, "pin_memory": False, "persistent_workers": False, "warmup": 2, "steps": 8},
        {"batch_size": 4, "num_workers": 2, "pin_memory": False, "persistent_workers": False, "warmup": 2, "steps": 8},
        {"batch_size": 4, "num_workers": 2, "pin_memory": True, "persistent_workers": True, "warmup": 2, "steps": 8},
    ]

    rows = [run_setting(cfg, setting, device, model_dtype) for setting in settings]
    out_path = ROOT / "runs/demo_tiny_fastwam/dataloader_benchmark.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"device: {device}")
    print(f"benchmark_csv: {out_path}")
    print("| setting | step_time | gpu_util_approx | memory_used | notes |")
    print("|---|---:|---:|---:|---|")
    for row in rows:
        step_time = row["step_time_s"]
        step_time_display = f"{step_time}s" if step_time != "SKIP" else "SKIP"
        print(
            f"| {row['setting']} | {step_time_display} | {row['gpu_util_approx']} | "
            f"{row['memory_used_mb']} MB | {row['notes']} |"
        )


if __name__ == "__main__":
    main()
