# PFD StackCube Small Reproduction

## What This Implements

This is a small-scale reproduction of the core method in:

`Privileged Foresight Distillation: Zero-Cost Future Correction for World Action Models`

Links:

- arXiv: https://arxiv.org/abs/2604.25859
- HTML: https://arxiv.org/html/2604.25859v2
- code page mentioned by the paper: https://github.com/PengchengFang-cs/PFD

This repo implementation is intentionally not a full paper-scale reproduction.
It implements the key PFD mechanism on ManiSkill `StackCube-v1`:

```text
current/future state sequence tokens + action tokens
  -> shared transformer backbone
  -> student forward with current-only mask
  -> teacher forward with privileged future-token mask
  -> residual adapter
  -> current-only inference
```

## What Is Simplified

| Component | PFD paper / Fast-WAM scale | This demo |
|---|---|---|
| visual backbone | Wan/Fast-WAM video DiT | small transformer |
| observations | video/latent tokens | 48D StackCube state tokens |
| teacher/student | shared backbone, two masks | same |
| residual adapter | MLP, zero-init output | same idea |
| inference | current-only + adapter | same |
| full LIBERO/RoboTwin | yes | no |

The state sequence acts as a controlled proxy for video tokens. Token 0 is the
current observation. Tokens 1..K are privileged future observations that the
teacher can attend during training, but the student and inference path cannot.

## Data

The training data comes from ManiSkill's open StackCube demonstrations:

```bash
python -m mani_skill.utils.download_demo StackCube-v1
```

The downloaded data lives outside the repo:

```text
~/.maniskill/demos/StackCube-v1/
```

The state replay file used by this demo is:

```text
~/.maniskill/demos/StackCube-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cpu.h5
```

It was generated with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5 \
  --sim-backend physx_cpu \
  -o state \
  --save-traj \
  --use-env-states \
  --count 512 \
  --num-envs 1
```

Replay result:

```text
506/512 demonstrations saved
obs shape: [T+1, 48]
action shape: [T, 4]
```

## Model

File:

```text
src/fastwam/models/pfd_small_transformer.py
```

The model has about `8.63M` parameters.

Student mask:

```text
action tokens can attend:
  - current state token
  - action tokens

action tokens cannot attend:
  - future state tokens

current state token also cannot attend future state tokens,
to avoid indirect future leakage.
```

Teacher mask:

```text
action tokens can attend:
  - current state token
  - future state tokens
  - action tokens
```

Adapter:

```text
delta_hat = g_phi(v_base, tau)
v_final = v_base + delta_hat
```

The final adapter layer is zero-initialized, so initially:

```text
delta_hat = 0
v_final = v_base
```

## Loss

The PFD-small loss follows the paper structure:

```text
L = lambda_video L_video
  + lambda_gt L_gt
  + lambda_res L_res
  + lambda_teacher L_teacher
```

Default weights:

```text
lambda_video = 1.0
lambda_gt = 1.0
lambda_res = 0.5
lambda_teacher = 0.1
```

In this proxy:

```text
L_video   = future state-token reconstruction loss
L_gt      = action prediction loss against demo action
L_res     = residual adapter loss against detach(v_teacher - v_base)
L_teacher = v_final loss against detach(v_teacher)
```

## How To Run

Full PFD-small train/eval:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pfd_stackcube.sh \
  --steps 8000 \
  --batch-size 512 \
  --lr 0.0001 \
  --eval-episodes 20 \
  --binarize-gripper
```

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/pfd_stackcube/run_pfd_stackcube_pipeline.py \
  --variant pfd \
  --steps 2 \
  --batch-size 32 \
  --num-workers 0 \
  --max-trajectories 32 \
  --skip-eval \
  --device cuda:0
```

Current-only baseline config is provided but not run in the first PFD-only pass:

```text
configs/fastwam_baseline_small.yaml
```

## Current Result

Tracked result directory:

```text
docs/pfd_stackcube_results_binarized/pfd/
```

Training:

```text
steps: 8000
batch size: 512
final train loss: 0.18483
final gt/action loss: 0.09439
final video loss: 0.00333
final weighted residual loss: 0.07259
final weighted teacher loss: 0.01452
final val loss: 0.14542
```

Evaluation with gripper binarization:

```text
episodes: 20
successes: 2
success rate: 0.10
avg return: 15.836
avg episode length: 46.80
```

Raw continuous-gripper evaluation reached `0/20`, but several episodes achieved
grasp or cube-on-cube without satisfying the final static success condition.
The binarized gripper evaluation is the reported runnable result.

## Artifacts

```text
docs/pfd_stackcube_results_binarized/pfd/loss_curves.png
docs/pfd_stackcube_results_binarized/pfd/metrics.json
docs/pfd_stackcube_results_binarized/pfd/eval_episodes.csv
docs/pfd_stackcube_results_binarized/pfd/gpu_profile_summary.json
docs/pfd_stackcube_results_binarized/pfd/videos/success_episode.mp4
docs/pfd_stackcube_results_binarized/pfd/videos/episode_019_fail.mp4
```

## GPU Profile

Single RTX 4090 was used:

```text
GPU 0 avg util: 34.42%
GPU 0 max util: 51.00%
GPU 0 max memory used: 2057 MiB
GPU 1 avg util: 0.00%
```

This model is still too small to justify two-GPU distributed training. Two-GPU
training should be tested only when we connect PFD to the original Fast-WAM
Wan/ActionDiT backbone or move to RGB/latent tokens with a larger model.

## Next Steps

1. Run the provided baseline config for a real PFD-vs-current-only comparison.
2. Replace state tokens with RGB/latent tokens to reduce state shortcut.
3. Add partial fine-tuning on original Fast-WAM: adapter + last 1-2 action/video blocks.
4. Tune residual weights; this first run shows residual/delta norms can be noisy.
