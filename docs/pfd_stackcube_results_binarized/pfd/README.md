# StackCube PFD Small Result

This is a small PFD proxy benchmark on ManiSkill `StackCube-v1`.
It uses state-sequence tokens as controllable video-token proxies.

## Result

- success_rate: 0.100
- successes: 2/20
- avg_return: 15.547
- avg_episode_len: 46.80
- avg_is_grasped: 0.500
- avg_is_cubeA_on_cubeB: 0.350
- teacher_used_at_inference: False
- future_tokens_used_at_inference: False

## Artifacts

- `loss_curves.png`
- `metrics.json`
- `eval_episodes.csv`
- success_video: `docs/pfd_stackcube_results_binarized/pfd/videos/success_episode.mp4`
- last_eval_video: `docs/pfd_stackcube_results_binarized/pfd/videos/episode_019_fail.mp4`
