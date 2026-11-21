# Apple Silicon port: Phase 1 (CPU trainer backend)

Goal: train end to end on the Mac in fp32 and match the CUDA trainer's
numerics and learning results.

## Design

The trainer stays a single CUDA C++ source. Every kernel launch goes through
`PUF_LAUNCH(kernel, grid, block[, smem[, stream]])(args)` and every device
operation goes through the CUDA runtime / cuBLAS / cuRAND API.
`src/puf_backend.h` chooses the implementation:

- **CUDA (default):** the real toolkit headers; `PUF_LAUNCH` is `<<<...>>>`.
- **CPU (`-DPUF_TRAIN_CPU`, default on macOS):** `src/puf_cuda_cpu.h`
  implements the subset of the CUDA API the trainer uses, on the host.
  - Device memory is host memory, and every call is synchronous. Events
    record wall time, so the dashboard's timing breakdown still works.
  - Kernels run as host loops: blocks run in parallel with OpenMP, and the
    threads within a block run one after another. `block_reduce_sum`, the
    only cooperative pattern, runs on the block's last thread using the same
    tree order as CUDA.
  - The hot kernels (MinGRU gate and both scans) have host-native versions,
    with the hidden dimension as the vectorised inner loop.
  - cuBLAS `GemmEx` maps to Accelerate `cblas_sgemm`; cuRAND maps to a
    Philox4x32-10 implementation.
  - NCCL, NVML, NVTX and CUDA graphs are stubs. Graphs are forced off.
  - Protein (the sweep) needs cuSOLVER, so `sweep` exits with a message.

Changes to the shared source are no-ops under CUDA. After every change,
CUDA bf16 breakout on the RTX 5090 trains to the **identical** final score
(853.029).

Build and run on macOS: `brew install libomp; ./build.sh ENV --cpu` (fetches
raylib), then `./build.sh ENV && ./puffer train`.

## Correctness

**Parity oracle** (`tests/oracle_train.{cu,sh}`, `tests/oracle_compare.py`).
It uses identical host-generated weights and a synthetic rollout, then runs
one rollout forward pass and one train epoch. The table shows max relative
difference, CUDA (RTX 5090, fp32) vs the Mac CPU backend:

| config | rollout decoder output | rollout RNN state | losses, 1 minibatch | losses, full epoch | weight update |
|---|---|---|---|---|---|
| breakout (hidden 32) | 2.4e-7 | 1.4e-7 | — | 2.3e-6 (3 mb) | 2.6e-2 |
| pacman (hidden 256) | 7.7e-7 | 6.0e-7 | **1.1e-8** | 3.5e-3 (57 mb) | 8.7e-2 |
| tetris (hidden 512) | 1.0e-6 | 5.3e-7 | — | 5.2e-5 (21 mb) | 1.1e-2 |

The forward pass and a single minibatch's losses agree at float precision.
The weight-update difference is the same size when the Mac is compared with
itself after a 1e-7 relative perturbation of the input weights (breakout
2.7e-2, pacman 8.4e-2, tetris 1.1e-2). That is Muon's Newton–Schulz
iteration amplifying rounding, not a backend bug. Loss differences over a
full epoch grow from that minibatch-to-minibatch divergence.

**Unit tests** (`tests/test_cpu_backend.sh`):
- all three `puf_mm*` GEMM wrappers against naive loops, with non-square
  shapes and alpha/beta;
- block reduce;
- Philox uniform and normal moments, and stream independence.

**Learning** (breakout, fp32, default config, 55M steps). Final score /
perf:

| | CUDA fp32 (RTX 5090) | Mac CPU backend |
|---|---|---|
| async=1 | 11 seeds: 658–861 (median 840) | 8 seeds: 573–858 (median ~830) |
| async=0 | 843 / 0.975 (seed 73) | 4 seeds: 825–860 (859, 860, 847, 825) |

Async training is noisy on both backends, and the Mac distribution sits
inside the CUDA one.

**Thread safety:** ThreadSanitizer, async=1, 1M steps, with OpenMP disabled
(TSan does not understand libomp). About 7 epochs, 0 reports. The OpenMP
paths were not covered.

## Throughput (CPU backend)

Protocol (`tests/bench_train_sps.sh`): fixed step count, SPS = steps / wall
time (including startup), 2 repeats, idle machine, default config.

| config | steps | runs (SPS) | median | CUDA 5090 bf16 | env-only ceiling on Mac |
|---|---|---|---|---|---|
| breakout | 55M (full) | 430K, 538K | **~480K** | 22.9M | 83M |
| pacman | 5M | 43.9K, 39.6K | **~42K** | 4.1M | 48M |
| tetris | 3M | 51.8K, 55.2K | **~54K** | 3.2M | 63M |

Run-to-run variation is large (±10–20%) even on an idle machine. The async
rollout and learner threads compete for cores and for the CPU's matrix units.
The dashboard's SPS figure reads higher than this protocol because it
excludes startup and is not averaged over the run.

Profile (`PUF_CPU_PROFILE=1`, pacman, summed over threads): matrix
multiplies take about 40%. The rest is the MinGRU gate and scans and the
emulated element-wise kernels (`add_kernel`, `zero_term_state`, Muon
element-wise). Emulated kernels pay for macOS thread-local-storage lookups on
every simulated thread.

## Phase 2 decision: build a Metal backend

GEMM throughput at the trainer's shapes (`tests/bench_gemm_mac.mm`):

| shape (M x N x K) | Accelerate (CPU) | MPS fp32 (GPU), batched command buffer |
|---|---|---|
| breakout rollout 4096x96x32 | 20 µs, 1.2 TF | 68 µs, 0.4 TF |
| pacman rollout 8192x768x256 | 2.0 ms, 1.6 TF | 0.32 ms, 10.2 TF |
| pacman train 16384x768x256 | 2.6 ms, 2.5 TF | 0.86 ms, 7.5 TF |
| tetris train 65536x1536x512 | 41 ms, 2.5 TF | 13 ms, 7.8 TF |

From about hidden 256 upward, the GPU runs the dominant GEMMs 3–6× faster
than the CPU. GEMM is already about 40% of CPU-backend time, so even a
zero-overhead CPU backend is capped near 2–2.5 TFLOP/s. **Decision: build a
Metal backend for hidden ≥ 256.** For breakout-sized models the CPU wins on
GEMM latency, and the end-to-end comparison is still open. It depends on
the Metal per-step dispatch and sync latency that Phase 3 measures first.

Expected ceilings, from model FLOPs (rollout forward + 3 × forward ×
replay_ratio + Muon) and measured GEMM throughput:

| config | FLOP / env step | 5090 bf16 (measured) | Metal fp32 at ~5.6 TF (ceiling) | CPU at ~2 TF (ceiling) |
|---|---|---|---|---|
| pacman | ~16.7 MFLOP | 4.1M SPS (~68 TF) | ~330K SPS | ~120K SPS |
| tetris | ~31 MFLOP | 3.2M SPS (~100 TF) | ~180K SPS | ~65K SPS |

Even a perfect Metal fp32 backend will be 10–20× slower than an RTX 5090 on
these models. bf16/fp16 on the GPU is a later optimisation; MPS fp16 was
only up to 1.4× faster than fp32 at these shapes.
