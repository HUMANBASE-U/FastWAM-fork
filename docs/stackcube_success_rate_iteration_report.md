# StackCube Success-Rate Iteration Report

Date: 2026-06-02

## Question

The original StackCube state-token PFD demo had very low success. The goal was to determine whether the cause was model size, training steps, missing observations, poor data/control mode, or PFD residual instability, then push the runnable benchmark toward 80%+ success.

## Key Finding

The main bottleneck was not GPU utilization or only parameter count. The largest factor was the action/control interface and the supervision target.

`pd_joint_delta_pos` is the best local StackCube control mode because the provided PPO expert reaches 80%+ on the same evaluation distribution. Direct behavior cloning from fixed H5 actions reached only 30%, but distilling the PPO actor improved the learned transformer policy to 74%.

## Results

| Route | Control mode | Policy | Success |
|---|---|---:|---:|
| Joint-delta BC | `pd_joint_delta_pos` | transformer | 15/50, 30% |
| PPO-distilled action-only | `pd_joint_delta_pos` | 19.3M transformer | 37/50, 74% |
| PFD residual ablation | `pd_joint_delta_pos` | PPO-distilled PFD | 23/50, 46% |
| Frozen PPO-backbone route | `pd_joint_delta_pos` | PPO actor in same eval pipeline | 41/50, 82% |

## What Changed

- Added phase embeddings to the state/action token transformer.
- Added PPO teacher distillation mode.
- Added `pd_joint_delta_pos` replayed state dataset support.
- Added evaluation modes for model-only, PPO-backbone, and model/PPO blend.
- Increased the strongest student to 19.3M parameters and trained action-only with `lambda_video=0.0`.

## Interpretation

The best pure learned transformer policy is currently 74%, not 80%.

The 82% run uses the frozen PPO expert as a pretrained backbone / teacher-policy route. This is a realistic next-step framing for a student benchmark: start from a working policy backbone, then train small adapters or PFD-style residual corrections.

Direct PFD residual training did not help in this proxy benchmark. It made `delta_hat` large and unstable, increasing action error and reducing success to 46%. The next PFD step should use staged adapter training and residual gating.

## Artifacts

- 30% joint-delta BC:
  `docs/pfd_stackcube_joint_delta_baseline/baseline/`
- 74% PPO-distilled transformer:
  `docs/pfd_stackcube_ppo_distill_actiononly_baseline/baseline/`
- 46% PFD residual ablation:
  `docs/pfd_stackcube_ppo_distill_actiononly_pfd/pfd/`
- 82% frozen PPO-backbone:
  `docs/pfd_stackcube_ppo_expert_backbone_eval/baseline/`

## Next Step

Implement staged PFD:

1. Start from the 74% PPO-distilled transformer or the frozen PPO backbone.
2. Freeze the base policy.
3. Train only a zero-initialized residual adapter.
4. Add a residual gate initialized near zero.
5. Add DAgger-style states collected from the student and labeled by PPO.
