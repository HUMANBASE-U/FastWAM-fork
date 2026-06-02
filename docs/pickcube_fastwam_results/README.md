# PickCube FastWAM-small Benchmark

This is a student-scale FastWAM-style benchmark on ManiSkill `PickCube-v1`.
Training uses behavior cloning action prediction plus a future-state auxiliary loss.
Evaluation disables test-time future imagination and uses only `predict_action_chunk()`.

## Key Result

- success_rate: 0.750
- successes: 15/20
- avg_return: 8.283
- avg_episode_len: 23.00
- avg_is_grasped: 0.950
- avg_is_obj_placed: 0.800
- final_train_loss: 0.009945960715413094
- final_val_loss: 0.017375178821384906
- test_time_future_imagination: false
- GPU 0 average utilization during this small run: 5.06%
- GPU 0 peak utilization during this small run: 7.00%

## Artifacts

- Loss curves: `loss_curves.png`
- Evaluation summary: `eval_summary.png`
- Metrics JSON: `metrics.json`
- Episode table: `eval_episodes.csv`
- GPU profile: `gpu_profile.csv`
- GPU profile summary: `gpu_profile_summary.json`
- Success video: `docs/pickcube_fastwam_results/videos/success_episode.mp4`
- Last evaluation video: `docs/pickcube_fastwam_results/videos/episode_019_success.mp4`

## What This Is Not

This is not a full reproduction of the original Fast-WAM paper.
It is a small PickCube imitation benchmark for verifying data -> model -> loss -> eval -> video.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/pickcube_fastwam/run_pickcube_fastwam_pipeline.py --amp
```
