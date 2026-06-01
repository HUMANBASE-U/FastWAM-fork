#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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


VARIANTS = [
    {
        "name": "action_only",
        "task": "demo_tiny_fastwam",
        "title": "Action-only WAM",
        "color": "#2563eb",
    },
    {
        "name": "joint_video_action",
        "task": "demo_tiny_joint",
        "title": "Action + video auxiliary",
        "color": "#16a34a",
    },
    {
        "name": "idm_proxy",
        "task": "demo_tiny_idm",
        "title": "IDM proxy latent -> action",
        "color": "#dc2626",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny FastWAM proxy variants and save result figures.")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset-length", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="docs/tiny_demo_results")
    parser.add_argument("--run-dir", default="runs/demo_tiny_fastwam/benchmark_suite")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(task: str, args: argparse.Namespace):
    overrides = [
        f"task={task}",
        f"batch_size={args.batch_size}",
        f"learning_rate={args.learning_rate}",
        f"data.train.length={args.dataset_length}",
        f"data.train.seed={args.seed}",
        f"model.hidden_dim={args.hidden_dim}",
        f"model.latent_dim={args.latent_dim}",
        "num_workers=0",
        "pin_memory=true",
        "persistent_workers=false",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def train_variant(variant: dict, args: argparse.Namespace, device: str, out_dir: Path, run_dir: Path):
    cfg = load_cfg(variant["task"], args)
    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset = instantiate(cfg.data.train)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.dit.parameters(), lr=float(cfg.learning_rate))

    rows = []
    data_iter = cycle(loader)
    start_wall = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = next(data_iter)
        step_start = time.perf_counter()
        loss, loss_dict = model.training_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start

        row = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "step_time_s": step_time,
            "samples_per_s": int(cfg.batch_size) / max(step_time, 1e-9),
        }
        for key, value in loss_dict.items():
            row[key] = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        rows.append(row)

    elapsed = time.perf_counter() - start_wall
    ckpt_path = run_dir / f"{variant['name']}.pt"
    model.save_checkpoint(str(ckpt_path), step=args.steps)

    csv_path = out_dir / f"{variant['name']}_loss.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()}, key=lambda x: ["step", "loss_total", "loss_action", "loss_video", "loss_latent", "step_time_s", "samples_per_s"].index(x) if x in ["step", "loss_total", "loss_action", "loss_video", "loss_latent", "step_time_s", "samples_per_s"] else 99)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "variant": variant,
        "cfg": cfg,
        "dataset": dataset,
        "model": model,
        "rows": rows,
        "csv_path": csv_path,
        "ckpt_path": ckpt_path,
        "elapsed_s": elapsed,
        "final_loss": rows[-1]["loss_total"],
        "final_action_loss": rows[-1].get("loss_action", float("nan")),
        "avg_samples_per_s": sum(row["samples_per_s"] for row in rows[-20:]) / min(20, len(rows)),
    }


def to_image(video_cthw: torch.Tensor) -> np.ndarray:
    frame = video_cthw.detach().float().cpu().clamp(-1.0, 1.0)
    frame = ((frame + 1.0) * 0.5).permute(1, 2, 0).numpy()
    return np.clip(frame, 0.0, 1.0)


def plot_dataset_overview(result: dict, out_dir: Path) -> None:
    sample = result["dataset"][0]
    video = sample["video"]
    action = sample["action"]
    context = sample["context"]

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.0, 1.1])
    for t in range(video.shape[1]):
        ax = fig.add_subplot(grid[0, t])
        ax.imshow(to_image(video[:, t]))
        ax.set_title(f"frame {t}")
        ax.axis("off")

    ax_action = fig.add_subplot(grid[1, :3])
    ax_action.plot(action[:, 0].numpy(), label="dx")
    ax_action.plot(action[:, 1].numpy(), label="dy")
    ax_action.plot(action[:, 6].numpy(), label="gripper")
    ax_action.set_title("Synthetic target action")
    ax_action.set_xlabel("action step")
    ax_action.grid(True, alpha=0.25)
    ax_action.legend()

    ax_ctx = fig.add_subplot(grid[1:, 3:])
    im = ax_ctx.imshow(context.numpy(), aspect="auto", cmap="viridis")
    ax_ctx.set_title("Mock text/context embedding")
    ax_ctx.set_xlabel("embedding dim")
    ax_ctx.set_ylabel("token")
    fig.colorbar(im, ax=ax_ctx, fraction=0.046, pad=0.04)

    ax_note = fig.add_subplot(grid[2, :3])
    ax_note.axis("off")
    ax_note.text(
        0.0,
        0.8,
        "Tiny dataset contract:\n"
        "video [3,T,H,W], action [T,D], proprio [T,P], context [L,E]\n"
        "This is synthetic data for validating the training chain, not robot skill data.",
        fontsize=11,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_overview.png", dpi=180)
    plt.close(fig)


def plot_loss_curves(results: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for result in results:
        rows = result["rows"]
        steps = [row["step"] for row in rows]
        total = [row["loss_total"] for row in rows]
        axes[0].plot(steps, total, label=result["variant"]["title"], color=result["variant"]["color"])
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    styles = {"loss_action": "-", "loss_video": "--", "loss_latent": ":"}
    for result in results:
        rows = result["rows"]
        steps = [row["step"] for row in rows]
        for key, style in styles.items():
            values = [row.get(key) for row in rows if row.get(key) is not None]
            if not values:
                continue
            key_steps = [row["step"] for row in rows if row.get(key) is not None]
            axes[1].plot(
                key_steps,
                values,
                linestyle=style,
                color=result["variant"]["color"],
                label=f"{result['variant']['name']}:{key}",
            )
    axes[1].set_title("Component losses")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("weighted loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def predict_action(result: dict, sample: dict):
    model = result["model"]
    model.eval()
    batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
    moved = model._move_batch(batch)
    outputs = model.dit(
        video=moved["video"],
        context=moved["context"],
        context_mask=moved["context_mask"],
        proprio=moved["proprio"],
    )
    if model.mode == "idm_proxy":
        pred_action = outputs["action_from_latent"]
    else:
        pred_action = outputs["action"]
    return moved, outputs, pred_action[0].detach().float().cpu()


def plot_action_predictions(results: list[dict], out_dir: Path) -> None:
    sample = results[0]["dataset"][0]
    target = sample["action"].detach().float().cpu()
    dims = list(range(target.shape[1]))

    fig, axes = plt.subplots(len(results), len(dims), figsize=(2.6 * len(dims), 3.0 * len(results)), sharex=True)
    if len(results) == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, result in enumerate(results):
        pred = predict_action(result, sample)[2]
        for col_idx, dim in enumerate(dims):
            ax = axes[row_idx, col_idx]
            ax.plot(target[:, dim].numpy(), color="black", linewidth=2.0, label="target")
            ax.plot(pred[:, dim].numpy(), color=result["variant"]["color"], linestyle="--", label="pred")
            ax.set_title(f"{result['variant']['name']}\naction dim {dim}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("value")
            if row_idx == len(results) - 1:
                ax.set_xlabel("step")
            if row_idx == 0 and col_idx == len(dims) - 1:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "action_prediction_all_models.png", dpi=180)
    plt.close(fig)


def plot_joint_video_prediction(results: list[dict], out_dir: Path) -> None:
    joint = next(result for result in results if result["variant"]["name"] == "joint_video_action")
    sample = joint["dataset"][0]
    moved, outputs, _ = predict_action(joint, sample)
    target = joint["model"]._video_lowres_target(moved["video"])[0].detach().cpu()
    pred = outputs["video_lowres"][0].detach().float().cpu().clamp(-1.0, 1.0)

    frames = target.shape[1]
    fig, axes = plt.subplots(2, frames, figsize=(2.4 * frames, 5))
    for t in range(frames):
        axes[0, t].imshow(to_image(target[:, t]))
        axes[0, t].set_title(f"target frame {t}")
        axes[0, t].axis("off")
        axes[1, t].imshow(to_image(pred[:, t]))
        axes[1, t].set_title(f"pred frame {t}")
        axes[1, t].axis("off")
    fig.suptitle("Joint video/action tiny model: low-resolution video auxiliary output")
    fig.tight_layout()
    fig.savefig(out_dir / "joint_video_prediction.png", dpi=180)
    plt.close(fig)


def plot_summary_table(results: list[dict], args: argparse.Namespace, device: str, out_dir: Path) -> None:
    headers = ["model", "final total loss", "final action loss", "avg samples/s", "checkpoint"]
    rows = [
        [
            result["variant"]["name"],
            f"{result['final_loss']:.5f}",
            f"{result['final_action_loss']:.5f}",
            f"{result['avg_samples_per_s']:.1f}",
            str(result["ckpt_path"]),
        ]
        for result in results
    ]

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.axis("off")
    ax.set_title(
        f"Tiny FastWAM benchmark summary | device={device} | steps={args.steps} | batch={args.batch_size}",
        pad=16,
    )
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_summary.png", dpi=180)
    plt.close(fig)


def write_summary_md(results: list[dict], args: argparse.Namespace, device: str, out_dir: Path) -> None:
    lines = [
        "# Tiny FastWAM Benchmark Results",
        "",
        "This directory contains generated figures for the toy benchmark path.",
        "",
        f"- device: `{device}`",
        f"- steps per model: `{args.steps}`",
        f"- batch size: `{args.batch_size}`",
        f"- dataset length: `{args.dataset_length}`",
        f"- hidden dim: `{args.hidden_dim}`",
        "",
        "| model | final total loss | final action loss | avg samples/s |",
        "|---|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['variant']['name']} | {result['final_loss']:.6f} | "
            f"{result['final_action_loss']:.6f} | {result['avg_samples_per_s']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Generated figures:",
            "",
            "- `dataset_overview.png`",
            "- `loss_curves.png`",
            "- `action_prediction_all_models.png`",
            "- `joint_video_prediction.png`",
            "- `benchmark_summary.png`",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    out_dir = ROOT / args.out_dir
    run_dir = ROOT / args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(run_dir))
    register_default_resolvers()
    set_seed(args.seed)
    device = choose_device(args.device)

    results = []
    for variant in VARIANTS:
        print(f"[train] {variant['name']} on {device}")
        results.append(train_variant(variant, args, device, out_dir, run_dir))

    plot_dataset_overview(results[0], out_dir)
    plot_loss_curves(results, out_dir)
    plot_action_predictions(results, out_dir)
    plot_joint_video_prediction(results, out_dir)
    plot_summary_table(results, args, device, out_dir)
    write_summary_md(results, args, device, out_dir)

    print("TINY BENCHMARK SUITE OK")
    print(f"out_dir: {out_dir}")
    for result in results:
        print(
            f"{result['variant']['name']}: final_loss={result['final_loss']:.6f} "
            f"csv={result['csv_path']}"
        )


if __name__ == "__main__":
    main()
