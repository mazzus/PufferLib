# PufferLib on Apple Silicon

The trainer runs natively on Apple Silicon Macs, with its compute on the GPU
through Metal. It is the same single-source trainer as the CUDA build, and
CUDA builds are unaffected.

## Quick start (macOS)

    brew install libomp
    ./build.sh breakout --cpu     # once: fetches raylib; builds the play binary
    ./build.sh breakout           # trainer, Metal GPU backend (default on macOS)
    ./puffer train                # run from the repo root

- `./build.sh ENV --no-metal` builds the CPU-only trainer (Accelerate +
  OpenMP).
- Training is fp32 (`--float` is implied). Checkpoints use the same flat
  fp32 format as CUDA and load in either direction.
- `PUF_CPU_PROFILE=1 ./puffer train ...` prints a per-kernel profile, flush
  counts and GPU busy time.

## Status (M4 Max, 40-core GPU)

| config | Metal (protocol SPS) | CPU-only | RTX 5090 bf16 |
|---|---|---|---|
| breakout | ~2.3M | ~480K | 22.9M |
| pacman | ~320K | ~42K | 4.1M |
| tetris | ~190K | ~54K | 3.2M |

- **Coverage:** 70 of 85 envs train (see PHASE5.md). Envs with custom CUDA
  policy nets (asteroids, minimal, craftax, nmmo3, osrs_*, nethack) are not
  supported yet.
- **Correctness:** checked against CUDA. Forward pass and single-minibatch
  losses match to float precision for discrete, multi-head and continuous
  action spaces. Full training runs reach the same scores (breakout,
  pacman).
- **Not available:** the Protein hyperparameter `sweep` (needs cuSOLVER),
  multi-GPU, bf16.

## How it works

| piece | file |
|---|---|
| backend switch | `src/puf_backend.h` |
| CUDA API on the host (memory, launches, cuBLAS/cuRAND, stubs) | `src/puf_cuda_cpu.h` |
| Metal runtime (arena, command buffers, MPS GEMMs, dispatch) | `src/puf_metal.h`, `src/puf_metal.mm` |
| MSL kernel ports | `src/puf_kernels.metal` |
| per-kernel port declarations | `PUF_METAL_PORT(...)` next to each CUDA kernel |

Phase write-ups: PHASE0.md (baseline), PHASE1.md (CPU backend, parity),
PHASE3.md (Metal backend), PHASE4.md (performance), PHASE5.md (coverage,
correctness).

## Tests and tools

| script | what |
|---|---|
| `tests/test_cpu_backend.sh` | GEMM wrappers, block reduce, Philox (CPU backend) |
| `tests/test_metal_kernels.sh [ENV]` | every Metal kernel vs its CPU emulation, GEMM paths |
| `tests/oracle_train.sh ENV` (+ `METAL=1`) | one rollout + one train epoch dump; compare with `tests/oracle_compare.py` against a CUDA dump |
| `tests/bench_train_sps.sh BIN STEPS N` | end-to-end SPS protocol |
| `tests/bench_learner.sh ENV` (+ `METAL=1`) | learner-only epochs (low-noise A/B) |
| `tests/bench_env_sps.sh ENV` | env-only ceiling |
| `tests/smoke_train_all.sh` | brief training of every built env |
