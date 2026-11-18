# Apple Silicon port: Phase 3 (Metal GPU backend)

Goal: run the trainer's compute on the M4 Max GPU while keeping the
single-source trainer and its CUDA numerics.

## Design

`./build.sh ENV` on macOS now builds the Metal backend by default
(`--no-metal` gives the CPU-only trainer from Phase 1). It is the CPU
backend (`src/puf_cuda_cpu.h`) plus `-DPUF_TRAIN_METAL`:

- **Memory:** all `cudaMalloc` memory is sub-allocated from shared-storage
  `MTLBuffer` chunks (the arena). On unified memory, host code (the CPU
  environments, uploads) reads and writes it directly.
- **Ordering:** each OS thread records into its own command buffer on its
  own queue, in issue order, which is CUDA stream order. The arena is
  hazard-tracked, so GPU commands serialize as on a CUDA stream. The dW side
  stream lands in the same command buffer.
- **GEMMs:** `cublasGemmEx` maps to `MPSMatrixMultiplication` on arena
  offsets. Column-major cuBLAS operands are mapped to row-major MPS views;
  kernel objects are cached per thread, because MPS encoding is not
  thread-safe.
- **Copies and memsets** within the arena are GPU blits.
- **Kernels:** `src/puf_kernels.metal` holds MSL ports of the 25 kernels on
  the default-net training path. Each port is line for line, with the same
  reduction trees and the same Philox generator.
  - A `PUF_METAL_PORT(kernel, "msl_name", (params), (args))` line next to
    each CUDA kernel defines a typed overload. It marshals the kernel's
    parameters into the MSL argument struct (natural alignment, arena
    pointers as GPU addresses inside structs), and overloaded kernels
    resolve exactly as in CUDA. The line is empty under CUDA.
  - Unported kernels, host kernels, host transfers and synchronisation
    flush the thread's command buffer and then run on the CPU, so the
    trainer stays correct at every step. Today that is about one flush per
    rollout step (actions go back to the CPU environments) plus a few per
    epoch.
- **Library:** compiled at runtime in safe-math mode, with the environment
  constants (`PUF_NUM_ATNS`, `PUF_MAX_HEAD_A`) passed as macros.

Shared-source changes remain no-ops under CUDA. CUDA bf16 breakout on the
RTX 5090 still trains to the identical 853.029.

## Correctness

- **Per-kernel differential test** (`tests/test_metal_kernels.sh`): each
  ported kernel on random inputs, Metal vs the CPU emulation of the CUDA
  kernel.
  - Bit-exact: advantage, Muon chain, element-wise kernels.
  - At most 4.3e-7 relative: MinGRU gate and scans, PPO loss, sampling
    (identical samples over 3 steps).
- **Oracle vs CUDA fp32** (`METAL=1 tests/oracle_train.sh pacman`): forward
  6.3e-7, RNN state 5.3e-7, one-minibatch losses 2.2e-8 (max relative).
- **Learning:**

| run | Metal | CUDA fp32/bf16 reference |
|---|---|---|
| breakout async=0, seed 73 | 849 / 0.983 | 843 / 0.975 (fp32) |
| breakout async=1, seeds 73/1/2 | 845, 849, 859 | 11 seeds 658–861 (fp32) |
| pacman, full 205M steps | **190.7 / 0.795** | 187.2 / 0.780 (bf16) |

## Throughput

Protocol: `tests/bench_train_sps.sh`, steps / wall time, median of 2, idle
machine, default config.

| config | Metal | CPU backend | speed-up | FLOP ceiling (Metal fp32) | RTX 5090 bf16 |
|---|---|---|---|---|---|
| breakout (55M) | ~1.45M SPS | ~480K | 3.0× | — (latency-bound) | 22.9M |
| pacman (20M) | ~262K SPS | ~42K | 6.2× | ~330K | 4.1M |
| tetris (20M) | ~153K SPS | ~54K | 2.8× | ~180K | 3.2M |

pacman and tetris run at about 80–85% of the fp32 FLOP ceiling estimated in
Phase 1. Further gains need lower precision (half/bf16 GEMMs), fewer GPU
passes (fusion), or both. That is Phase 4.

## Tools

- `PUF_CPU_PROFILE=1 ./puffer train ...`: per-kernel host time, GEMM
  encode time, and flush/GEMM/dispatch/blit counts.
- `tests/spike_metal.mm`: standalone checks of the Metal mechanisms used.
