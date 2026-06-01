# Tiny FastWAM Benchmark Results

This directory contains generated figures for the toy benchmark path.

- device: `cuda:0`
- steps per model: `160`
- batch size: `128`
- dataset length: `1024`
- hidden dim: `256`

| model | final total loss | final action loss | avg samples/s |
|---|---:|---:|---:|
| action_only | 0.000183 | 0.000183 | 38400.5 |
| joint_video_action | 0.002219 | 0.000208 | 34657.1 |
| idm_proxy | 0.002855 | 0.001346 | 20609.1 |

Generated figures:

- `dataset_overview.png`
- `loss_curves.png`
- `action_prediction_all_models.png`
- `joint_video_prediction.png`
- `benchmark_summary.png`
