from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.pickcube_state_dataset import PickCubeStateDataset
from fastwam.models.pickcube_fastwam_small import PickCubeFastWAMSmall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a small FastWAM-style PickCube policy. "
            "Training uses action + future-state losses; evaluation uses only the action head."
        )
    )
    parser.add_argument("--traj-path", default="~/.maniskill/demos/PickCube-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cpu.h5")
    parser.add_argument("--run-dir", default="runs/pickcube_fastwam")
    parser.add_argument("--docs-dir", default="docs/pickcube_fastwam_results")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-trajectories", type=int, default=511)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--future-horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lambda-future", type=float, default=0.2)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--execute-horizon", type=int, default=1)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision during training.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def bool_scalar(value: Any) -> bool:
    return bool(round(scalar(value)))


def to_numpy_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        raise TypeError("This PickCube pipeline expects obs_mode='state', not a dict observation.")
    if isinstance(obs, torch.Tensor):
        arr = obs.detach().cpu().numpy()
    else:
        arr = np.asarray(obs)
    arr = arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def render_frame(env: Any) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    frame = frame.astype(np.uint8)
    if frame.shape[0] >= 512 and frame.shape[1] >= 512:
        frame = frame[::2, ::2]
    return frame


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_loss_curves(history: list[dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    steps = [row["step"] for row in history]
    plt.figure(figsize=(8, 5))
    for key in ["train_loss", "train_action_loss", "train_future_loss", "val_loss"]:
        values = [row.get(key, np.nan) for row in history]
        if np.isfinite(values).any():
            plt.plot(steps, values, label=key)
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.title("PickCube FastWAM-small loss curves")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_eval_summary(metrics: dict[str, Any], out_path: Path) -> None:
    labels = ["success_rate", "avg_return", "avg_episode_len", "avg_is_grasped"]
    values = [
        float(metrics.get("success_rate", 0.0)),
        float(metrics.get("avg_return", 0.0)),
        float(metrics.get("avg_episode_len", 0.0)) / max(float(metrics.get("max_episode_steps", 50)), 1.0),
        float(metrics.get("avg_is_grasped", 0.0)),
    ]
    plt.figure(figsize=(7, 4))
    colors = ["#2a9d8f", "#457b9d", "#8d99ae", "#e76f51"]
    plt.bar(labels, values, color=colors)
    plt.ylim(0, max(1.0, max(values) * 1.15))
    plt.ylabel("normalized score")
    plt.title("PickCube evaluation summary")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_datasets(args: argparse.Namespace) -> tuple[PickCubeStateDataset, PickCubeStateDataset]:
    train_ds = PickCubeStateDataset(
        traj_path=args.traj_path,
        action_horizon=args.action_horizon,
        future_horizon=args.future_horizon,
        max_trajectories=args.max_trajectories,
        val_fraction=args.val_fraction,
        split="train",
        seed=args.seed,
        normalize=True,
    )
    val_ds = PickCubeStateDataset(
        traj_path=args.traj_path,
        action_horizon=args.action_horizon,
        future_horizon=args.future_horizon,
        max_trajectories=args.max_trajectories,
        val_fraction=args.val_fraction,
        split="val",
        seed=args.seed,
        normalize=True,
        stats=train_ds.stats,
    )
    return train_ds, val_ds


def make_model(args: argparse.Namespace, train_ds: PickCubeStateDataset) -> PickCubeFastWAMSmall:
    return PickCubeFastWAMSmall(
        obs_dim=train_ds.obs_dim,
        action_dim=train_ds.action_dim,
        action_horizon=args.action_horizon,
        future_horizon=args.future_horizon,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        lambda_future=args.lambda_future,
        device=args.device,
    )


@torch.no_grad()
def evaluate_val_loss(model: PickCubeFastWAMSmall, loader: DataLoader, max_batches: int = 20) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss, _ = model.training_loss(batch)
        losses.append(float(loss.detach().cpu().item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(args: argparse.Namespace, model: PickCubeFastWAMSmall, train_ds: PickCubeStateDataset, val_ds: PickCubeStateDataset) -> list[dict[str, float]]:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))
    history: list[dict[str, float]] = []
    iterator = iter(loader)
    t0 = time.perf_counter()
    last_log_t = t0
    last_logged_step = 0
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and args.device.startswith("cuda")):
            loss, metrics = model.training_loss(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 100 == 0 or step == args.steps:
            now = time.perf_counter()
            elapsed = max(now - last_log_t, 1e-6)
            steps_per_sec = (step - last_logged_step) / elapsed
            last_log_t = now
            last_logged_step = step
            val_loss = evaluate_val_loss(model, val_loader, max_batches=10) if (step == 1 or step % 500 == 0 or step == args.steps) else float("nan")
            memory_alloc = torch.cuda.memory_allocated() / 1024**2 if args.device.startswith("cuda") and torch.cuda.is_available() else 0.0
            memory_reserved = torch.cuda.memory_reserved() / 1024**2 if args.device.startswith("cuda") and torch.cuda.is_available() else 0.0
            row = {
                "step": float(step),
                "train_loss": float(loss.detach().cpu().item()),
                "train_action_loss": float(metrics["loss_action"].detach().cpu().item()),
                "train_future_loss": float(metrics["loss_future"].detach().cpu().item()),
                "val_loss": val_loss,
                "steps_per_sec": steps_per_sec,
                "gpu_mem_alloc_mb": memory_alloc,
                "gpu_mem_reserved_mb": memory_reserved,
            }
            history.append(row)
            print(
                f"step={step:05d} loss={row['train_loss']:.5f} "
                f"action={row['train_action_loss']:.5f} future={row['train_future_loss']:.5f} "
                f"val={row['val_loss']:.5f} steps/s={steps_per_sec:.2f} "
                f"mem={memory_alloc:.1f}/{memory_reserved:.1f}MB",
                flush=True,
            )

    ckpt_path = run_dir / "pickcube_fastwam_small.pt"
    model.save_checkpoint(str(ckpt_path), stats=train_ds.stats, step=args.steps)
    write_history_csv(run_dir / "loss_history.csv", history)
    return history


def load_checkpoint_if_available(args: argparse.Namespace, model: PickCubeFastWAMSmall) -> dict[str, Any] | None:
    ckpt_path = Path(args.run_dir) / "pickcube_fastwam_small.pt"
    if not ckpt_path.exists():
        return None
    return model.load_checkpoint(str(ckpt_path))


def evaluate_policy(args: argparse.Namespace, model: PickCubeFastWAMSmall, train_ds: PickCubeStateDataset) -> dict[str, Any]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    docs_dir = Path(args.docs_dir)
    videos_dir = docs_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    if action_low.size != train_ds.action_dim:
        action_low = np.full(train_ds.action_dim, -1.0, dtype=np.float32)
        action_high = np.full(train_ds.action_dim, 1.0, dtype=np.float32)

    episode_rows = []
    success_video_path = None
    last_video_path = None
    model.eval()
    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        frames = [render_frame(env)]
        ep_return = 0.0
        ep_success = False
        ep_is_grasped = False
        ep_is_obj_placed = False
        steps = 0
        done = False
        while not done and steps < args.max_episode_steps:
            obs_np = to_numpy_obs(obs)
            obs_tensor = torch.from_numpy(obs_np).to(args.device).float()
            obs_norm = train_ds.normalize_obs(obs_tensor)
            pred_norm = model.predict_action_chunk(obs_norm)[0]
            pred_actions = train_ds.denormalize_action(pred_norm).detach().cpu().numpy()
            for action in pred_actions[: max(1, args.execute_horizon)]:
                action = np.clip(action.astype(np.float32), action_low, action_high)
                obs, reward, terminated, truncated, info = env.step(action[None, :])
                ep_return += scalar(reward)
                steps += 1
                ep_success = ep_success or bool_scalar(info.get("success", False))
                ep_is_grasped = ep_is_grasped or bool_scalar(info.get("is_grasped", False))
                ep_is_obj_placed = ep_is_obj_placed or bool_scalar(info.get("is_obj_placed", False))
                if steps % max(1, args.render_every) == 0:
                    frames.append(render_frame(env))
                done = bool_scalar(terminated) or bool_scalar(truncated) or ep_success or steps >= args.max_episode_steps
                if done:
                    break

        video_name = f"episode_{ep:03d}_{'success' if ep_success else 'fail'}.mp4"
        video_path = videos_dir / video_name
        imageio.mimsave(video_path, frames, fps=args.video_fps, macro_block_size=1)
        last_video_path = str(video_path)
        if ep_success and success_video_path is None:
            success_copy = videos_dir / "success_episode.mp4"
            imageio.mimsave(success_copy, frames, fps=args.video_fps, macro_block_size=1)
            success_video_path = str(success_copy)
        episode_rows.append(
            {
                "episode": ep,
                "success": int(ep_success),
                "return": ep_return,
                "episode_len": steps,
                "is_grasped": int(ep_is_grasped),
                "is_obj_placed": int(ep_is_obj_placed),
                "video": str(video_path),
            }
        )
        print(f"eval episode={ep:03d} success={ep_success} return={ep_return:.3f} len={steps} video={video_path}", flush=True)

    env.close()
    metrics = {
        "env_id": "PickCube-v1",
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pos",
        "action_space": "4D [dx, dy, dz, gripper]",
        "test_time_future_imagination": False,
        "num_eval_episodes": args.eval_episodes,
        "max_episode_steps": args.max_episode_steps,
        "successes": int(sum(row["success"] for row in episode_rows)),
        "success_rate": float(np.mean([row["success"] for row in episode_rows])) if episode_rows else 0.0,
        "avg_return": float(np.mean([row["return"] for row in episode_rows])) if episode_rows else 0.0,
        "avg_episode_len": float(np.mean([row["episode_len"] for row in episode_rows])) if episode_rows else 0.0,
        "avg_is_grasped": float(np.mean([row["is_grasped"] for row in episode_rows])) if episode_rows else 0.0,
        "avg_is_obj_placed": float(np.mean([row["is_obj_placed"] for row in episode_rows])) if episode_rows else 0.0,
        "success_video": success_video_path,
        "last_eval_video": last_video_path,
        "episodes": episode_rows,
    }
    docs_dir.mkdir(parents=True, exist_ok=True)
    with (docs_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (docs_dir / "eval_episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)
    return metrics


def write_docs(args: argparse.Namespace, history: list[dict[str, float]], metrics: dict[str, Any]) -> None:
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    loss_png = docs_dir / "loss_curves.png"
    eval_png = docs_dir / "eval_summary.png"
    plot_loss_curves(history, loss_png)
    plot_eval_summary(metrics, eval_png)
    readme = docs_dir / "README.md"
    final_loss = history[-1]["train_loss"] if history else None
    final_val = next((row["val_loss"] for row in reversed(history) if np.isfinite(row["val_loss"])), None)
    readme.write_text(
        "\n".join(
            [
                "# PickCube FastWAM-small Benchmark",
                "",
                "This is a student-scale FastWAM-style benchmark on ManiSkill `PickCube-v1`.",
                "Training uses behavior cloning action prediction plus a future-state auxiliary loss.",
                "Evaluation disables test-time future imagination and uses only `predict_action_chunk()`.",
                "",
                "## Key Result",
                "",
                f"- success_rate: {metrics.get('success_rate', 0.0):.3f}",
                f"- successes: {metrics.get('successes', 0)}/{metrics.get('num_eval_episodes', 0)}",
                f"- avg_return: {metrics.get('avg_return', 0.0):.3f}",
                f"- avg_episode_len: {metrics.get('avg_episode_len', 0.0):.2f}",
                f"- final_train_loss: {final_loss if final_loss is not None else 'n/a'}",
                f"- final_val_loss: {final_val if final_val is not None else 'n/a'}",
                "",
                "## Artifacts",
                "",
                f"- Loss curves: `{loss_png.name}`",
                f"- Evaluation summary: `{eval_png.name}`",
                f"- Metrics JSON: `metrics.json`",
                f"- Episode table: `eval_episodes.csv`",
                f"- Success video: `{metrics.get('success_video')}`",
                f"- Last evaluation video: `{metrics.get('last_eval_video')}`",
                "",
                "## What This Is Not",
                "",
                "This is not a full reproduction of the original Fast-WAM paper.",
                "It is a small PickCube imitation benchmark for verifying data -> model -> loss -> eval -> video.",
                "",
                "## Command",
                "",
                "```bash",
                "CUDA_VISIBLE_DEVICES=0 python experiments/pickcube_fastwam/run_pickcube_fastwam_pipeline.py --amp",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    Path(args.docs_dir).mkdir(parents=True, exist_ok=True)
    print(json.dumps(vars(args), indent=2), flush=True)
    train_ds, val_ds = make_datasets(args)
    print(
        f"dataset train_chunks={len(train_ds)} val_chunks={len(val_ds)} "
        f"obs_dim={train_ds.obs_dim} action_dim={train_ds.action_dim}",
        flush=True,
    )
    model = make_model(args, train_ds)
    history: list[dict[str, float]] = []
    if not args.skip_train:
        history = train(args, model, train_ds, val_ds)
    else:
        load_checkpoint_if_available(args, model)
        loss_path = Path(args.run_dir) / "loss_history.csv"
        if loss_path.exists():
            with loss_path.open() as f:
                history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    metrics: dict[str, Any] = {}
    if not args.skip_eval:
        metrics = evaluate_policy(args, model, train_ds)
    else:
        metrics_path = Path(args.docs_dir) / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
    if history and metrics:
        write_docs(args, history, metrics)
    print("DONE")
    if metrics:
        print(f"success_rate={metrics.get('success_rate', 0.0):.3f} successes={metrics.get('successes', 0)}/{metrics.get('num_eval_episodes', 0)}")
        print(f"success_video={metrics.get('success_video')}")
        print(f"last_eval_video={metrics.get('last_eval_video')}")


if __name__ == "__main__":
    main()
