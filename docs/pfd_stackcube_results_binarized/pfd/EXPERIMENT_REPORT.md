# StackCube PFD-small Experiment Report

## Setup

- Task: ManiSkill `StackCube-v1`
- Data: open ManiSkill demonstrations downloaded by `download_demo StackCube-v1`
- Model: `StackCubePFDSmallTransformer`
- Parameters: about 8.63M
- Observation: 48D state sequence tokens
- Action: 4D `pd_ee_delta_pos`
- Inference: current-only, no teacher mask, no future tokens

## PFD Mechanism Check

- Teacher and student share the same transformer backbone.
- Student uses a current-only attention mask.
- Teacher uses a privileged future-state-token attention mask.
- Teacher output is detached.
- Residual target is `detach(v_teacher - v_base)`.
- Adapter input uses live `v_base`.
- Adapter output layer is zero-initialized.

Smoke test verified:

```text
initial delta_norm: 0.00000
forward/backward: passed
checkpoint save: passed
```

## Training Result

```text
steps: 8000
batch size: 512
learning rate: 1e-4
final loss: 0.18483
final gt/action loss: 0.09439
final video loss: 0.00333
final residual loss: 0.07259
final teacher loss: 0.01452
final val loss: 0.14542
```

The residual branch is active but noisy. This is visible in `loss_res`,
`teacher_gap`, and `delta_norm`.

## Evaluation Result

Continuous gripper:

```text
successes: 0/20
avg grasped: 0.50
avg cubeA-on-cubeB: 0.35
```

Binarized gripper:

```text
successes: 2/20
success rate: 0.10
```

The successful videos are tracked under:

```text
docs/pfd_stackcube_results_binarized/pfd/videos/
```

## Interpretation

This is a successful first PFD pipeline run, not a strong final policy.

What worked:

- PFD double-mask training runs.
- Adapter zero-init behaves correctly.
- Current-only inference runs without future tokens.
- StackCube simulator evaluation runs.
- At least one successful video is generated.

What remains weak:

- Success rate is low.
- State-token proxy can still learn shortcuts.
- The residual target is noisy.
- We have not yet run the current-only baseline ablation.

## Recommended Next Experiment

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/pfd_stackcube/run_pfd_stackcube_pipeline.py \
  --variant baseline \
  --steps 8000 \
  --batch-size 512 \
  --lr 0.0001 \
  --eval-episodes 20 \
  --binarize-gripper
```

Then compare:

```text
baseline success rate
PFD success rate
loss_gt
avg_is_grasped
avg_is_cubeA_on_cubeB
```
