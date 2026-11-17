# Apple Silicon port: Phase 0 (setup and baseline)

Goal of this phase: get the existing Mac build path working, then measure two
things for three reference configs: the env-only throughput ceiling on the Mac,
and stock CUDA training results on an NVIDIA box.

## Machines

| | Mac (target) | hansolo (CUDA reference) |
|---|---|---|
| Chip | Apple M4 Max, 12P + 4E cores, 40-core GPU | RTX 5090 32 GB, 32 cores |
| Memory | 128 GB unified | |
| OS / toolchain | macOS 27.2, Apple clang 21, Metal 4 | Ubuntu 24.04, CUDA 12.8, clang 18, driver 580 |
| CPU features | BF16, I8MM, SME2 (`sysctl hw.optional.arm`) | |

hansolo is shared. During these runs a teammate's process held about 16 GB of
VRAM, so the CUDA numbers below may be pessimistic.

## Build fixes

- `build.sh` used `${ENV^^}` (bash 4 syntax). macOS `/bin/bash` is 3.2, so
  every build failed with "bad substitution". It now uses `tr`.
- `brew install libomp` is required on the Mac. `brew --prefix libomp` prints
  a path even when the package isn't installed, so the script doesn't catch it.
- Building the CUDA trainer without sudo (hansolo):
  - `nvcc` no longer requires ccache.
  - `OMP_LIB` can be overridden; use `OMP_LIB=-lgomp`.
  - New `NCCL_HOME` override, pointing at NCCL unpacked from the
    `nvidia-nccl-cu12` wheel.

  Build command:
  `CUDA_HOME=/usr/local/cuda OMP_LIB=-lgomp NCCL_HOME=~/opt/nccl-wheel/nvidia/nccl ./build.sh ENV`

`./build.sh ENV --cpu` on the Mac: 77 of 85 envs build. The 8 failures:

| env | reason | Mac-specific? |
|---|---|---|
| robot_arm | CUDA-only physics, by design | no |
| osrs | shared library, not an env | no |
| matsci | needs LAMMPS (`lammps/library.h`) | no |
| osrs_colosseum, osrs_inferno, osrs_pvp, osrs_zulrah | data download returns 404 | probably not (unchecked on Linux) |
| nethack | vendored fast-nle fails under Apple clang (implicit declaration of `nle_lnc_bits`) | possibly |

## Reference configs

| config | policy hidden x layers | params | total_agents | minibatch | total_timesteps |
|---|---|---|---|---|---|
| breakout | 32 x 4 | 16.2K | 4096 | 65536 | 55M |
| pacman | 256 x 6 | 1.3M | 8192 | 16384 | 205M |
| tetris | 512 x 2 (config says 2.20258, a sweep result) | 1.7M | 4096 | 65536 | 1.92B |

## Env-only ceiling on the Mac

Measured with `tests/bench_env_sps.{c,sh}`: the trainer's rollout env loop
without a model, random actions, 2048 measured steps after 128 warmup steps.

| config | 1 thread | 4 | 8 | 12 |
|---|---|---|---|---|
| breakout | 27.1M | 75.2M | **83.3M** | 81.1M |
| pacman | 7.3M | 25.2M | 46.1M | **47.8M** |
| tetris | 15.1M | 47.1M | **63.1M** | 60.0M |

Throughput stops improving at about 8 threads.

## CUDA training baseline (hansolo, stock trainer, bf16, default config)

| config | wall | SPS | final score | final perf | time split |
|---|---|---|---|---|---|
| breakout | 2.4 s | 22.9M | 853 | 0.987 | rollout 76% (model 16%, env 36%, copy 20%) / train 23% |
| pacman | 51 s | 4.1M | 187 | 0.780 | rollout 45% (model 7%, env 17%) / train 54% |
| tetris | 9m56s | 3.2M | 29980 | 0.200 | rollout 57% (model 42%, env 10%) / train 42% |

Logs are on hansolo in `~/runs/*_cuda_bf16.log`.

## What this tells us

- On this Mac the envs are fast: 12–20x the 5090's end-to-end training SPS
  for pacman and tetris. The speed limit on the Mac will be model compute
  (rollout forward + PPO train), not envs.
- On the 5090, model work (rollout forward plus training) is 60–85% of wall
  time for the 256/512-hidden configs.
- Next, Phase 1 (CPU backend, fp32) has to reproduce these learning
  results. Phase 2 benchmarks it against the env ceiling above to decide
  whether a Metal backend is needed.

## Reproduce

Mac:

    brew install libomp
    ./build.sh breakout --cpu            # fetches raylib
    ./tests/bench_env_sps.sh breakout
    ./build/bench_env_sps_breakout --vec.num_threads=8 --bench.steps=2048 --bench.warmup=128

hansolo:

    CUDA_HOME=/usr/local/cuda OMP_LIB=-lgomp NCCL_HOME=~/opt/nccl-wheel/nvidia/nccl \
        PATH=/usr/local/cuda/bin:$PATH ./build.sh breakout && ./puffer train
