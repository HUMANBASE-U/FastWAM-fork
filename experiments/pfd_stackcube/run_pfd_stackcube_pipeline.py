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

from fastwam.datasets.stackcube_state_dataset import StackCubeStateSequenceDataset
from fastwam.models.pfd_small_transformer import StackCubePFDSmallTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate StackCube PFD-small or current-only baseline.")
    parser.add_argument("--variant", choices=["pfd", "baseline"], default="pfd")
    parser.add_argument("--traj-path", default="~/.maniskill/demos/StackCube-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cpu.h5")
    parser.add_argument("--run-root", default="runs/pfd_stackcube")
    parser.add_argument("--docs-root", default="docs/pfd_stackcube_results")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-trajectories", type=int, default=506)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--state-horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--adapter-hidden-dim", type=int, default=512)
    parser.add_argument("--lambda-video", type=float, default=1.0)
    parser.add_argument("--lambda-gt", type=float, default=1.0)
    parser.add_argument("--lambda-res", type=float, default=0.5)
    parser.add_argument("--lambda-teacher", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--execute-horizon", type=int, default=1)
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
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
    return float(np.asarray(value).reshape(-1)[0])


def bool_scalar(value: Any) -> bool:
    return bool(round(scalar(value)))


def to_numpy_obs(obs: Any) -> np.ndarray:
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


def make_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = Path(args.run_root) / args.variant
    docs_dir = Path(args.docs_root) / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "videos").mkdir(parents=True, exist_ok=True)
    return run_dir, docs_dir


def make_datasets(args: argparse.Namespace) -> tuple[StackCubeStateSequenceDataset, StackCubeStateSequenceDataset]:
    train_ds = StackCubeStateSequenceDataset(
        traj_path=args.traj_path,
        action_horizon=args.action_horizon,
        state_horizon=args.state_horizon,
        max_trajectories=args.max_trajectories,
        val_fraction=args.val_fraction,
        split="train",
        seed=args.seed,
        normalize=True,
    )
    val_ds = StackCubeStateSequenceDataset(
        traj_path=args.traj_path,
        action_horizon=args.action_horizon,
        state_horizon=args.state_horizon,
        max_trajectories=args.max_trajectories,
        val_fraction=args.val_fraction,
        split="val",
        seed=args.seed,
        normalize=True,
        stats=train_ds.stats,
    )
    return train_ds, val_ds


def make_model(args: argparse.Namespace, train_ds: StackCubeStateSequenceDataset) -> StackCubePFDSmallTransformer:
    return StackCubePFDSmallTransformer(
        obs_dim=train_ds.obs_dim,
        action_dim=train_ds.action_dim,
        action_horizon=args.action_horizon,
        state_horizon=args.state_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        lambda_video=args.lambda_video,
        lambda_gt=args.lambda_gt,
        lambda_res=(args.lambda_res if args.variant == "pfd" else 0.0),
        lambda_teacher=(args.lambda_teacher if args.variant == "pfd" else 0.0),
        use_pfd=args.variant == "pfd",
        device=args.device,
    )


@torch.no_grad()
def val_loss(model: StackCubePFDSmallTransformer, loader: DataLoader, max_batches: int = 10) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss, _ = model.training_loss(batch)
        losses.append(float(loss.detach().cpu().item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace, model: StackCubePFDSmallTransformer, train_ds: StackCubeStateSequenceDataset, val_ds: StackCubeStateSequenceDataset, run_dir: Path) -> list[dict[str, float]]:
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
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
    last_t = time.perf_counter()
    last_step = 0
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0 or step == args.steps:
            now = time.perf_counter()
            steps_per_sec = (step - last_step) / max(now - last_t, 1e-6)
            last_t = now
            last_step = step
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu().item()),
                "val_loss": val_loss(model, val_loader) if step == 1 or step % 500 == 0 or step == args.steps else float("nan"),
                "grad_norm": float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                "steps_per_sec": steps_per_sec,
                "gpu_mem_alloc_mb": torch.cuda.memory_allocated() / 1024**2 if args.device.startswith("cuda") and torch.cuda.is_available() else 0.0,
            }
            for key, value in metrics.items():
                row[key] = float(value.detach().cpu().item())
            history.append(row)
            print(
                f"{args.variant} step={step:05d} loss={row['loss']:.5f} "
                f"gt={row.get('loss_gt', 0):.5f} video={row.get('loss_video', 0):.5f} "
                f"res={row.get('loss_res', 0):.5f} teacher={row.get('loss_teacher', 0):.5f} "
                f"val={row['val_loss']:.5f} delta={row.get('delta_norm', 0):.5f} "
                f"steps/s={steps_per_sec:.2f}",
                flush=True,
            )
    model.save_checkpoint(str(run_dir / "checkpoint.pt"), stats=train_ds.stats, step=args.steps)
    write_csv(run_dir / "loss_history.csv", history)
    return history


def evaluate(args: argparse.Namespace, model: StackCubePFDSmallTransformer, train_ds: StackCubeStateSequenceDataset, docs_dir: Path) -> dict[str, Any]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make(
        "StackCube-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    model.eval()
    rows = []
    success_video = None
    last_video = None
    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + 1000 + ep)
        frames = [render_frame(env)]
        done = False
        ep_return = 0.0
        ep_success = False
        ep_is_grasped = False
        ep_is_cubeA_on_cubeB = False
        steps = 0
        while not done and steps < args.max_episode_steps:
            obs_np = to_numpy_obs(obs)
            obs_tensor = torch.from_numpy(obs_np).to(args.device).float()
            obs_norm = train_ds.normalize_obs(obs_tensor)
            pred_norm = model.predict_action_chunk(obs_norm)[0]
            pred_actions = train_ds.denormalize_action(pred_norm).detach().cpu().numpy()
            for action in pred_actions[: max(1, args.execute_horizon)]:
                action = np.clip(action.astype(np.float32), low, high)
                if args.binarize_gripper and action.shape[0] > 0:
                    action[-1] = 1.0 if action[-1] >= 0 else -1.0
                obs, reward, terminated, truncated, info = env.step(action[None, :])
                ep_return += scalar(reward)
                steps += 1
                ep_success = ep_success or bool_scalar(info.get("success", False))
                ep_is_grasped = ep_is_grasped or bool_scalar(info.get("is_cubeA_grasped", info.get("is_grasped", False)))
                ep_is_cubeA_on_cubeB = ep_is_cubeA_on_cubeB or bool_scalar(info.get("is_cubeA_on_cubeB", False))
                frames.append(render_frame(env))
                done = bool_scalar(terminated) or bool_scalar(truncated) or ep_success or steps >= args.max_episode_steps
                if done:
                    break
        video_path = docs_dir / "videos" / f"episode_{ep:03d}_{'success' if ep_success else 'fail'}.mp4"
        imageio.mimsave(video_path, frames, fps=args.video_fps, macro_block_size=1)
        last_video = str(video_path)
        if ep_success and success_video is None:
            success_path = docs_dir / "videos" / "success_episode.mp4"
            imageio.mimsave(success_path, frames, fps=args.video_fps, macro_block_size=1)
            success_video = str(success_path)
        rows.append(
            {
                "episode": ep,
                "success": int(ep_success),
                "return": ep_return,
                "episode_len": steps,
                "is_grasped": int(ep_is_grasped),
                "is_cubeA_on_cubeB": int(ep_is_cubeA_on_cubeB),
                "video": str(video_path),
            }
        )
        print(f"eval {args.variant} ep={ep:03d} success={ep_success} return={ep_return:.3f} len={steps}", flush=True)
    env.close()
    metrics = {
        "variant": args.variant,
        "env_id": "StackCube-v1",
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pos",
        "binarize_gripper": bool(args.binarize_gripper),
        "pfd_teacher_used_at_inference": False,
        "pfd_future_tokens_used_at_inference": False,
        "num_eval_episodes": args.eval_episodes,
        "successes": int(sum(r["success"] for r in rows)),
        "success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
        "avg_return": float(np.mean([r["return"] for r in rows])) if rows else 0.0,
        "avg_episode_len": float(np.mean([r["episode_len"] for r in rows])) if rows else 0.0,
        "avg_is_grasped": float(np.mean([r["is_grasped"] for r in rows])) if rows else 0.0,
        "avg_is_cubeA_on_cubeB": float(np.mean([r["is_cubeA_on_cubeB"] for r in rows])) if rows else 0.0,
        "success_video": success_video,
        "last_eval_video": last_video,
        "episodes": rows,
    }
    (docs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (docs_dir / "eval_episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def plot_history(history: list[dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    steps = [r["step"] for r in history]
    plt.figure(figsize=(9, 5))
    for key in ["loss", "loss_gt", "loss_video", "loss_res", "loss_teacher", "val_loss"]:
        vals = np.asarray([r.get(key, np.nan) for r in history], dtype=float)
        if np.isfinite(vals).any():
            plt.plot(steps, vals, label=key)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("StackCube PFD-small losses")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_report(args: argparse.Namespace, history: list[dict[str, float]], metrics: dict[str, Any], docs_dir: Path) -> None:
    plot_history(history, docs_dir / "loss_curves.png")
    lines = [
        f"# StackCube {args.variant.upper()} Small Result",
        "",
        "This is a small PFD proxy benchmark on ManiSkill `StackCube-v1`.",
        "It uses state-sequence tokens as controllable video-token proxies.",
        "",
        "## Result",
        "",
        f"- success_rate: {metrics.get('success_rate', 0.0):.3f}",
        f"- successes: {metrics.get('successes', 0)}/{metrics.get('num_eval_episodes', 0)}",
        f"- avg_return: {metrics.get('avg_return', 0.0):.3f}",
        f"- avg_episode_len: {metrics.get('avg_episode_len', 0.0):.2f}",
        f"- avg_is_grasped: {metrics.get('avg_is_grasped', 0.0):.3f}",
        f"- avg_is_cubeA_on_cubeB: {metrics.get('avg_is_cubeA_on_cubeB', 0.0):.3f}",
        f"- teacher_used_at_inference: {metrics.get('pfd_teacher_used_at_inference')}",
        f"- future_tokens_used_at_inference: {metrics.get('pfd_future_tokens_used_at_inference')}",
        "",
        "## Artifacts",
        "",
        "- `loss_curves.png`",
        "- `metrics.json`",
        "- `eval_episodes.csv`",
        f"- success_video: `{metrics.get('success_video')}`",
        f"- last_eval_video: `{metrics.get('last_eval_video')}`",
        "",
    ]
    (docs_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir, docs_dir = make_dirs(args)
    print(json.dumps(vars(args), indent=2), flush=True)
    train_ds, val_ds = make_datasets(args)
    print(f"dataset train_chunks={len(train_ds)} val_chunks={len(val_ds)} obs_dim={train_ds.obs_dim} action_dim={train_ds.action_dim}", flush=True)
    model = make_model(args, train_ds)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params} ({n_params/1e6:.3f}M)", flush=True)
    history: list[dict[str, float]] = []
    if not args.skip_train:
        history = train(args, model, train_ds, val_ds, run_dir)
    else:
        payload = model.load_checkpoint(str(run_dir / "checkpoint.pt"))
        if "stats" in payload and payload["stats"] is not None:
            train_ds.stats = payload["stats"]
        loss_path = run_dir / "loss_history.csv"
        if loss_path.exists():
            with loss_path.open() as f:
                history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    metrics: dict[str, Any] = {}
    if not args.skip_eval:
        metrics = evaluate(args, model, train_ds, docs_dir)
    else:
        metrics_path = docs_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
    if history and metrics:
        write_csv(docs_dir / "loss_history.csv", history)
        write_report(args, history, metrics, docs_dir)
    print("DONE")
    if metrics:
        print(f"success_rate={metrics.get('success_rate', 0.0):.3f} successes={metrics.get('successes', 0)}/{metrics.get('num_eval_episodes', 0)}")
        print(f"success_video={metrics.get('success_video')}")
        print(f"last_eval_video={metrics.get('last_eval_video')}")


if __name__ == "__main__":
    main()
