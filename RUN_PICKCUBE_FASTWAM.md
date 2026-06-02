# PickCube FastWAM-small Runbook

## Core Idea

This benchmark turns FastWAM into a concrete, student-scale robot task:
`PickCube-v1` in ManiSkill.

The model is still FastWAM-style:

```text
state obs_t -> encoder -> latent z_t
z_t -> action head -> action chunk
z_t -> future-state head -> auxiliary future-state prediction
```

Training uses two losses:

```text
L = L_action + 0.2 * L_future
```

Evaluation uses only the action head:

```text
obs_t -> encoder -> action head -> robot action
```

So this run follows the route you requested: no test-time future imagination.

## What Is Different From The Original Paper

This is not a full paper reproduction.

| Part | Original Fast-WAM | This PickCube demo |
|---|---|---|
| Environment | LIBERO / RoboTwin | ManiSkill `PickCube-v1` |
| Observation | multi-camera video / image tokens | 42D simulator state |
| Model size | large Wan/DiT-style backbone | small MLP encoder + heads |
| Dataset | large robot datasets | small PickCube demonstrations |
| Training objective | paper-scale world/action model | action BC + future-state auxiliary loss |
| Inference | Fast-WAM variants | action head only, no future imagination |
| Goal | paper metrics | runnable benchmark, success rate, videos |

## Action Space

The policy outputs a 4D action for `pd_ee_delta_pos`:

```text
[dx, dy, dz, gripper]
```

This is not direct motor voltage. It is a high-level end-effector delta command.
ManiSkill's controller converts it into lower-level robot motion.

## Data

The demo uses ManiSkill PickCube demonstrations under:

```text
~/.maniskill/demos/PickCube-v1/rl/
```

The state replay file used for training is:

```text
~/.maniskill/demos/PickCube-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cpu.h5
```

The repo does not track this dataset. It is a local cache.

## How To Rebuild The Data

Download the small PickCube demo:

```bash
python -m mani_skill.utils.download_demo PickCube-v1
```

Replay a state-observation slice:

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5 \
  --sim-backend physx_cpu \
  -o state \
  --save-traj \
  --use-env-states \
  --count 512 \
  --num-envs 1
```

## How To Run

Run the full train/eval/video pipeline on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pickcube_fastwam.sh
```

Short smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/pickcube_fastwam/run_pickcube_fastwam_pipeline.py \
  --steps 2 \
  --batch-size 64 \
  --num-workers 0 \
  --max-trajectories 32 \
  --skip-eval \
  --run-dir runs/pickcube_fastwam_smoke
```

## Outputs

Training artifacts are written to:

```text
runs/pickcube_fastwam/
```

Tracked result artifacts are written to:

```text
docs/pickcube_fastwam_results/
```

Important files:

```text
docs/pickcube_fastwam_results/loss_curves.png
docs/pickcube_fastwam_results/eval_summary.png
docs/pickcube_fastwam_results/metrics.json
docs/pickcube_fastwam_results/eval_episodes.csv
docs/pickcube_fastwam_results/videos/success_episode.mp4
docs/pickcube_fastwam_results/videos/episode_019_success.mp4
```

## Current Result

```text
eval episodes: 20
successes: 15
success rate: 0.75
avg return: 8.283
avg episode length: 23.00
avg grasped: 0.95
avg object placed: 0.80
```

## GPU Diagnosis

This benchmark does use GPU, but it does not fill a 4090.

Observed profile:

```text
GPU 0 avg util: 5.06%
GPU 0 peak util: 7.00%
GPU 0 max memory used by nvidia-smi: 880 MiB
```

Reason:

```text
42D state input + small MLP + small batch = tiny CUDA kernels.
The GPU is active, but the computation is too small to saturate the card.
```

For this benchmark, low utilization is expected. To increase utilization, use
larger batch size, larger hidden dimension, visual observations, or multiple
parallel environments. That would make the experiment heavier, so this run keeps
the beginner demo small.

## Next Experiments

1. Compare `lambda_future=0.0` vs `lambda_future=0.2` to test whether the future-state auxiliary loss helps action success.
2. Replace state input with RGB image input for a more realistic FastWAM-style visual model.
3. Add an IDM proxy variant where predicted future latent is fed into an action decoder, then compare with this no-imagination route.
