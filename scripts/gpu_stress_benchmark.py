#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import threading
import time
from pathlib import Path

import torch


def monitor_gpu(path: Path, interval: float, stop_event: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gpu_util_pct", "mem_util_pct", "memory_used_mib", "power_w"])
        while not stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                ).strip()
                first_line = out.splitlines()[0]
                writer.writerow([part.strip() for part in first_line.split(",")])
                f.flush()
            except Exception as exc:
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "NA", "NA", "NA", f"monitor_error:{exc!r}"])
                f.flush()
            stop_event.wait(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short, controlled CUDA matmul stress benchmark.")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile-csv", default="runs/demo_tiny_fastwam/gpu_stress_profile.csv")
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run outside the sandbox or on a GPU-visible shell.")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"device: {device}")
    print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"matrix_size: {args.size}x{args.size}")
    print(f"dtype: {args.dtype}")
    print(f"duration_target_s: {args.seconds}")

    a = torch.randn((args.size, args.size), device=device, dtype=dtype)
    b = torch.randn((args.size, args.size), device=device, dtype=dtype)
    c = torch.empty((args.size, args.size), device=device, dtype=dtype)
    torch.cuda.synchronize(device)

    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu,
        args=(Path(args.profile_csv), args.monitor_interval, stop_event),
        daemon=True,
    )
    monitor.start()

    warmup_iters = 3
    for _ in range(warmup_iters):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    iterations = 0
    while True:
        torch.mm(a, b, out=c)
        torch.cuda.synchronize(device)
        iterations += 1
        if time.perf_counter() - start >= args.seconds:
            break

    elapsed = time.perf_counter() - start
    stop_event.set()
    monitor.join(timeout=2.0)

    flops_per_iter = 2.0 * args.size**3
    tflops = (flops_per_iter * iterations) / max(elapsed, 1e-9) / 1e12
    alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)

    print("GPU STRESS OK")
    print(f"iterations: {iterations}")
    print(f"elapsed_s: {elapsed:.3f}")
    print(f"approx_tflops: {tflops:.2f}")
    print(f"gpu_mem_alloc_mb: {alloc_mb:.1f}")
    print(f"gpu_mem_reserved_mb: {reserved_mb:.1f}")
    print(f"profile_csv: {args.profile_csv}")


if __name__ == "__main__":
    main()
