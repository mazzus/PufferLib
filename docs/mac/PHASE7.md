# Apple Silicon port: Phase 7 (GPU profile, GEMM tuning)

Goal: find the remaining SPS in the bf16 learner. Tetris is the most
learner-bound config, so it is the main case below.

## GPU time per kernel

`PUF_CPU_PROFILE` measures host time. On a GPU-bound learner that time
misleads. In bf16 tetris, three tiny Muon GEMMs (8x8x8, 512x8x8, 8x8x512)
seemed to take ~1.2 ms each to encode, 60% of host time. A `sample` of the
process showed why: the host was blocked in `commandBufferWithUnretainedReferences`
waiting for a free command buffer (back-pressure from the GPU). Whichever
call opens the next command buffer pays that wait.

`PUF_METAL_GPU_PROFILE=1` gives each dispatch or blit its own command
buffer, then attributes `GPUEndTime - GPUStartTime` to the kernel name, or
to GEMM name, shape and split count. The split-K reduce is listed on its
own line.

bf16 tetris, share of GPU time:

| | share |
|---|---|
| GEMMs | ~85% |
| of which the four 65536-row minibatch GEMMs (forward, dX, dW) | ~75% |
| MinGRU scan forward + backward | ~8% |
| everything else | ~7% |

## GEMMs are near the hardware peak

`tests/bench_gemm_shapes.cpp` times the backend's real GEMM path, split-K
logic included, at shapes taken from the profile. It uses wall time over
back-to-back calls. Summed per-command-buffer GPU spans overcount when
consecutive buffers overlap on the GPU.

| shape (cuBLAS m, n, k) | bf16 TFLOPS |
|---|---|
| TN 1536 x 65536 x 512 (forward) | 15.0 |
| NN 512 x 65536 x 1536 (dX) | 14.5 |
| NT 512 x 1536 x 65536 (dW, split-K) | 13.8 |
| NN 4096 x 4096 x 4096 (reference) | 14.8 |

bf16 runs at about the fp32 rate on this GPU (M4 Max, 40 cores). The large
GEMMs reach ~14–15 TFLOPS, which is close to what this kernel gets on a
large square problem. fp32 MPP reaches 13.1 TFLOPS on the forward shape,
against 8.2 for MPS. So tile tuning has little left to win. The tetris
learner is compute-bound on GEMMs running near peak.

## Change: split-K target 320 → 2048 threadgroups

The dW GEMMs (K = 65536) have few output tiles, for example 384 for
1536x512. At the old target (320 threadgroups) they did not split at all,
which left the GPU under-occupied. With a target of 2048 they split 3–6
ways.

GEMM time at each config's shapes, weighted by calls per epoch:

| config | old target (320) | new target (2048) | change |
|---|---|---|---|
| tetris | 10.70 s | 10.32 s | −3.5% |
| pacman | 6.14 s | 6.04 s | −1.6% |
| breakout | 172 ms | 172 ms | neutral |

Learner-only tetris (interleaved, 3 rounds, on battery): the best epoch
improves ~3% in each round (1.153/1.185/1.239 s vs 1.186/1.213/1.280 s);
the mean epoch is within noise.

Parity is unchanged:

- Kernel tests pass in fp32 and bf16.
- The tetris fp32 oracle against CUDA stays at the same level (worst rel
  1.1e-2 vs 1.2e-2, over 21 chained minibatches).
- A 1e-7 weight perturbation moves the losses more (1e-3) than the new
  summation order does.

`PUF_GEMM_SPLIT_TARGET` still overrides it.

## Tests

`tests/test_metal_gemm.mm` drove the GEMM kernels directly with the old
argument-buffer convention, so it had stopped working. It is removed:

- Its correctness checks (ragged shapes, alpha/beta, all layouts, split-K)
  are covered by `tests/test_metal_kernels.sh`, through the real path.
- Its MPS comparison is covered by `bench_gemm_shapes --f32` with
  `PUF_METAL_GEMM=mps`.

## What's left for SPS

- **Learner:** at peak GEMM throughput, more learner SPS needs fewer FLOPs
  (model or minibatch config, which changes the algorithm) or a faster
  GPU.
- **MinGRU scans (~8%):** the only non-GEMM kernels worth another look.
- **breakout:** the learner is a small share there. Its end-to-end SPS is
  bound by the env and rollout.

## Review fixes

From a review of the whole stack before merging:

- **Residency:** the Metal runtime makes arena chunks resident for every
  compute encoder. Before, a chunk allocated while an encoder was open
  (for example a grown split-K workspace) was not made resident for that
  encoder.
- **GPU errors:** `puf_metal_flush` checks the command buffer status itself,
  so a GPU error aborts before the host reads the results.
- **Late defines:** `puf_metal_define` after the kernel library is compiled
  is fatal instead of silently ignored.
- **NetHack:** the `sample_logits`, `cache_imp_and_v` and `ppo_loss_compute`
  Metal ports are disabled for `PUFFER_NETHACK` builds, because the MSL
  versions lack the NetHack head logic. Host emulation runs instead.
- **Protein test:** `src/protein.cu` defines a plain-CUDA `PUF_LAUNCH`
  fallback, so `tests/test_protein_numerics.cu` builds again on CUDA. Its
  output is byte-identical to stock 5.0. Both fail the checked-in golden
  file the same way (157 of 412 records), so that file is stale upstream.

CUDA breakout is still bit-identical (853.029). Metal bf16 breakout is
unchanged (845.035), and the kernel tests pass in fp32 and bf16.
