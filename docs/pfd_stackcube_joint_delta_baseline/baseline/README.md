# StackCube BASELINE Small Result

This is a small PFD proxy benchmark on ManiSkill `StackCube-v1`.
It uses state-sequence tokens as controllable video-token proxies.

## Result

- success_rate: 0.300
- successes: 15/50
- avg_return: 10.314
- avg_episode_len: 39.82
- avg_is_grasped: 0.860
- avg_is_cubeA_on_cubeB: 0.420
- teacher_used_at_inference: False
- future_tokens_used_at_inference: False

## Artifacts

- `loss_curves.png`
- `metrics.json`
- `eval_episodes.csv`
- success_video: `docs/pfd_stackcube_joint_delta_baseline/baseline/videos/success_episode.mp4`
- last_eval_video: `docs/pfd_stackcube_joint_delta_baseline/baseline/videos/episode_049_fail.mp4`
