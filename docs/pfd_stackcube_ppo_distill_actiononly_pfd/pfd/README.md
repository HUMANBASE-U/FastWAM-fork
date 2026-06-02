# StackCube PFD Small Result

This is a small PFD proxy benchmark on ManiSkill `StackCube-v1`.
It uses state-sequence tokens as controllable video-token proxies.

## Result

- success_rate: 0.460
- successes: 23/50
- avg_return: 12.078
- avg_episode_len: 35.06
- avg_is_grasped: 1.000
- avg_is_cubeA_on_cubeB: 0.640
- teacher_used_at_inference: False
- future_tokens_used_at_inference: False

## Artifacts

- `loss_curves.png`
- `metrics.json`
- `eval_episodes.csv`
- success_video: `docs/pfd_stackcube_ppo_distill_actiononly_pfd/pfd/videos/success_episode.mp4`
- last_eval_video: `docs/pfd_stackcube_ppo_distill_actiononly_pfd/pfd/videos/episode_049_fail.mp4`
