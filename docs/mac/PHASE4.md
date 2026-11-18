# Apple Silicon port: Phase 4 (Metal performance)

## Where the time went (after Phase 3)

`PUF_CPU_PROFILE=1` with GPU timing (`puf_metal_times`) on pacman:

- **The learner is the critical path.** In async training, rollout work is
  about 0.27 s per epoch and the learner about 1.9 s.
- **Encoding and GPU execution were serial.** The learner encoded a whole
  epoch into one command buffer, then committed and waited. About 0.5 s of
  that 1.9 s was host-side MPS encoding, at roughly 75 µs per
  `MPSMatrixMultiplication` encode. Caching `MPSMatrix` objects does not
  change that; the cost is inside MPS.
- **Avoidable flushes:** `puf_stamp` (3 per epoch) and small host-to-device
  copies (lr, ent_coef) each forced a full wait.

## Changes

- **Overlapped encode and execute:** command buffers are committed every
  `PUF_METAL_COMMIT_EVERY` (default 16) commands without waiting. Flushes
  wait only for the last committed buffer; same-queue order keeps
  correctness.
- **`PUF_HOST_NOFLUSH(kernel)`:** host kernels that touch only memory the
  GPU never uses (`puf_stamp`) skip the flush.
- **Staged uploads:** small host-to-device copies go through a per-thread
  arena staging ring plus a blit (`puf_metal_upload`).
- **`gemm_tile`:** an MSL `simdgroup_matrix` GEMM for small shapes, kept as
  an option (`PUF_METAL_SMALL_GEMM`) but **off by default**. It saves
  encode time but runs slower than MPS on the GPU (pacman learner 391K vs
  388K at ≤256³, 314K at ≤256×768×256; breakout learner 2.8M vs 5.0M).

## Measuring

End-to-end async training SPS varies ±10–20% between identical runs. The
cause is GPU sharing between the rollout and learner threads, and thermal
state. `tests/bench_learner.{cu,sh}` times learner epochs on a synthetic
rollout (no envs, async=0); repeats agree within about 5%, so it is the tool
for A/B tests.

| pacman learner (bench_learner, 10 epochs, 2 repeats) | learner SPS |
|---|---|
| Phase 3 behaviour (`COMMIT_EVERY=1e6`, all MPS) | ~332K |
| commit every 16, all MPS (**default**) | ~388K (+17%) |
| commit every 16, gemm_tile ≤ 256³ | ~391K |
| commit every 64, gemm_tile ≤ 256³ | ~364K |
| commit every 16, gemm_tile ≤ 256×768×256 | ~314K |

End to end (`tests/bench_train_sps.sh`, median of 2):

| config | Phase 3 | Phase 4 | ceiling (fp32 FLOPs) |
|---|---|---|---|
| breakout (55M) | ~1.45M | ~1.29M (runs 0.99–1.34M; within noise) | latency-bound |
| pacman (20M) | ~262K | ~265K | ~330K |
| tetris (20M) | ~153K | ~164K | ~180K |

Learning is unchanged: breakout 845 / 0.979 with the new defaults.

## Status and next steps

pacman and tetris run at about 80–90% of the fp32 FLOP ceiling. The rollout
and learner share the GPU, so end-to-end gains are smaller than learner
gains. What's left:

- **Lower precision:** half or bf16 GEMM operands with fp32 accumulation. MPS
  fp16 measured at most 1.4× fp32 at these shapes. This needs the bf16
  storage path (`precision_t`) on Metal.
- **Fusion:** fold the MinGRU gate into the GEMM epilogue, and batch Muon's
  per-layer Newton–Schulz GEMMs (layers share shapes) to cut dispatches.

## Update: async comparison and fresh numbers

After the `find_chunk` fix (`tests/bench_train_sps.sh`, breakout 3
repeats, others 2):

| config | async=0 | async=1 (default) |
|---|---|---|
| breakout | 2.13M | **2.29M** (repeats within 0.3%) |
| pacman | 300K | **321K** |
| tetris | 164K | **190K** |

async=1 is faster on every config, so it stays the default.

These are much higher and much steadier than the earlier end-to-end
numbers in this document (breakout 1.1–1.3M, pacman ~265K, tetris
~164K). No change in between explains a jump that size. The earlier runs
most likely shared the machine with background load (builds, indexing of
build outputs), so treat them as pessimistic. tetris now slightly exceeds
the rough fp32 "ceiling" estimate (~180K), which shows that estimate was
conservative.
