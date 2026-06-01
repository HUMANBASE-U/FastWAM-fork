# FastWAM Tiny Demo Runbook

## 1. What This Demo Does

This is a tiny local reproduction path for the FastWAM repo. It does not load
Wan2.2, T5, ActionDiT, LIBERO, or RoboTwin. Instead, it keeps the same training
shape:

```text
Hydra config
  -> synthetic dataset
  -> DataLoader
  -> tiny FastWAM-like model
  -> training_loss
  -> backward
  -> optimizer step
  -> trainer logging/checkpoint
```

The synthetic dataset emits FastWAM-like keys:

```text
video:        [B, 3, T, H, W]
action:       [B, action_horizon, action_dim]
proprio:      [B, action_horizon, proprio_dim]
context:      [B, context_len, text_dim]
context_mask: [B, context_len]
```

The default tiny task is an action-only baseline. Two proxy variants are also
available:

- `demo_tiny_fastwam`: action-only WAM proxy.
- `demo_tiny_joint`: action plus low-resolution future-video auxiliary loss.
- `demo_tiny_idm`: future-latent prediction followed by action decoding.

## 2. Difference From Full Paper Reproduction

Full Fast-WAM training uses large LIBERO/RoboTwin datasets, Wan2.2/T5 assets,
ActionDiT preprocessing, Accelerate/Deepspeed, and multi-GPU training.

This tiny demo only proves the local engineering chain:

```text
dataset -> dataloader -> model forward -> loss -> backward -> optimizer -> checkpoint
```

It is a proxy experiment scaffold, not a paper-metric reproduction.

## 3. Environment Setup

Use the current Python through `python -m pip`. In this session, bare `pip`
points to a different Python version, so avoid it.

Minimal commands used for the tiny path:

```bash
python -m pip install -e . --no-deps
python -m pip install hydra-core==1.3.2 accelerate==1.12.0
python -m pip install termcolor==2.5.0
```

Full project dependencies are intentionally not installed for this demo.

## 4. Run The Smoke Test

```bash
python scripts/smoke_test_tiny_fastwam.py
```

Expected output includes:

```text
SMOKE TEST OK
batch video: (8, 3, 5, 64, 64)
batch action: (8, 8, 7)
loss: ...
```

Variant smoke tests:

```bash
python scripts/smoke_test_tiny_fastwam.py task=demo_tiny_joint output_dir=./runs/demo_tiny_fastwam/smoke_joint
python scripts/smoke_test_tiny_fastwam.py task=demo_tiny_idm output_dir=./runs/demo_tiny_fastwam/smoke_idm
```

## 5. Run Tiny Training

Default action-only training:

```bash
bash scripts/run_tiny_train.sh
```

Useful overrides:

```bash
MAX_STEPS=100 BATCH_SIZE=8 NUM_WORKERS=0 bash scripts/run_tiny_train.sh
```

The wrapper saves:

```text
runs/demo_tiny_fastwam/tiny_train_log.txt
runs/demo_tiny_fastwam/gpu_profile.csv
runs/demo_tiny_fastwam/tiny_train/checkpoints/
```

You can also call the original trainer entry directly:

```bash
python scripts/train.py task=demo_tiny_fastwam
```

For proxy variants:

```bash
python scripts/train.py task=demo_tiny_joint max_steps=60
python scripts/train.py task=demo_tiny_idm max_steps=60
```

## 6. GPU Profile

The training wrapper starts:

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1
```

and writes:

```text
runs/demo_tiny_fastwam/gpu_profile.csv
```

In the current session, CUDA is unavailable:

```text
nvidia-smi cannot communicate with the NVIDIA driver
torch.cuda.is_available() == False
```

So GPU utilization cannot be measured until the server session exposes a usable
GPU driver/device.

For Codex specifically, default sandboxed commands may not expose `/dev/nvidia*`.
Approved outside-sandbox commands can see the GPUs. On this server, both RTX
4090s were visible outside the sandbox.

GPU 0 tiny training example:

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_DIR=runs/demo_tiny_fastwam/tiny_train_gpu \
LOG_PATH=runs/demo_tiny_fastwam/tiny_train_gpu_log.txt \
GPU_PROFILE_PATH=runs/demo_tiny_fastwam/gpu_profile_gpu0.csv \
MAX_STEPS=100 \
BATCH_SIZE=128 \
NUM_WORKERS=0 \
PIN_MEMORY=true \
bash scripts/run_tiny_train.sh
```

Larger tiny proxy example:

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_DIR=runs/demo_tiny_fastwam/tiny_train_gpu_big_nosync \
LOG_PATH=runs/demo_tiny_fastwam/tiny_train_gpu_big_nosync_log.txt \
GPU_PROFILE_PATH=runs/demo_tiny_fastwam/gpu_profile_gpu0_big_nosync.csv \
MAX_STEPS=50 \
BATCH_SIZE=512 \
NUM_WORKERS=0 \
PIN_MEMORY=true \
bash scripts/run_tiny_train.sh \
  data.train.length=4096 \
  model.hidden_dim=1024 \
  model.latent_dim=256
```

The tiny proxy model is too small to saturate a 4090. To verify raw GPU health:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gpu_stress_benchmark.py \
  --seconds 12 \
  --size 8192 \
  --dtype float16 \
  --profile-csv runs/demo_tiny_fastwam/gpu_stress_profile_gpu0.csv
```

Observed result on GPU 0:

```text
tiny train max util: about 11-15%
matmul stress max util: 100%
matmul stress max power: about 488 W
```

Interpretation: the GPU is fine; the tiny demo is just too small to fill it.

## 7. Dataloader / Step-Time Benchmark

```bash
python scripts/benchmark_tiny_dataloader.py
```

Output CSV:

```text
runs/demo_tiny_fastwam/dataloader_benchmark.csv
```

Current result: `num_workers>0` is skipped because multiprocessing worker
sockets are blocked in this sandbox/session. On a normal GPU job shell, rerun
the benchmark and compare `num_workers=0/2/4`, `pin_memory=true/false`, and
larger batch sizes.

## 8. Extending To Real LIBERO / RoboTwin

Do not start with the full dataset. The next practical step is a tiny real-data
adapter that returns the same keys as `DemoSyntheticFastWAMDataset`:

```text
video
action
proprio
prompt
context
context_mask
image_is_pad
action_is_pad
proprio_is_pad
```

Recommended order:

1. Obtain one small LIBERO trajectory or one preprocessed sample.
2. Write a `demo_real_slice` dataset that maps it to the same tensor contract.
3. Run `scripts/smoke_test_tiny_fastwam.py task=...`.
4. Only then consider the official Wan/FastWAM model path and text embedding cache.

## 9. Current Open Problems

- CUDA is not visible in this session.
- Official Python/Torch versions do not match the README exactly.
- Full project dependencies are incomplete by design.
- The tiny model is a proxy model, not the Wan2.2 FastWAM architecture.
- DataLoader multiprocessing is blocked in this sandbox/session.

## 10. Next Best Experiments

1. Action-only vs joint auxiliary:
   train `demo_tiny_fastwam` and `demo_tiny_joint`, compare action loss.

2. Imagine-then-act proxy:
   train `demo_tiny_idm` and compare with action-only.

3. Real-data slice:
   replace the synthetic generator with one real LIBERO trajectory while keeping
   the same batch keys.

## 11. Generated Toy Benchmark Figures

The tracked result figures live in:

```text
docs/tiny_demo_results/
```

Regenerate them with:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_tiny_benchmark_suite.py \
  --steps 160 \
  --batch-size 128 \
  --dataset-length 1024 \
  --hidden-dim 256 \
  --latent-dim 64 \
  --device auto
```

Generated files:

- `docs/tiny_demo_results/dataset_overview.png`
  Shows the synthetic video frames, target action trajectory, and mock text embedding.

- `docs/tiny_demo_results/loss_curves.png`
  Shows total loss and component losses for action-only, joint video/action, and IDM proxy.

- `docs/tiny_demo_results/action_prediction_all_models.png`
  Compares predicted vs target action dimensions for all three tiny models.

- `docs/tiny_demo_results/joint_video_prediction.png`
  Shows low-resolution target future frames and the joint model's predicted frames.

- `docs/tiny_demo_results/benchmark_summary.png`
  Compact table of final losses and throughput.

How to interpret the results:

- `action_only` gets the lowest action loss because it only spends capacity on action prediction.
- `joint_video_action` has a higher total loss because it also pays the video auxiliary loss.
- `idm_proxy` is harder because action prediction must pass through a learned future latent bottleneck.
- These are toy proxy experiments. They validate the research scaffold, not full Fast-WAM paper performance.
