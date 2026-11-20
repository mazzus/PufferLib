# Apple Silicon port: Phase 6 (MPP GEMMs, bf16)

Two changes, both aimed at SPS:

1. GEMMs on Metal Performance Primitives (MSL 4 tensor ops, macOS 26+)
   instead of MPSMatrixMultiplication.
2. bf16 storage for activations and gradients, with fp32 master weights.
   This is now the Mac default, as it is on CUDA (`./build.sh ENV`;
   `--float` for fp32).

Result: **bf16 is about 1.4–1.5x faster end to end on configs where the
learner dominates (pacman, tetris) and about 1.1x on breakout, where env
and rollout dominate. It learns the same as fp32.**

## MPP GEMMs

`src/puf_gemm.metal` contains row-major nt/tn/nn/tt GEMMs built on
`mpp::tensor_ops::matmul2d`. Accumulation is fp32, and storage is fp32 or
bf16. The file is compiled at runtime as a separate MSL 4.0 library.

- **Encoding cost:** about 1 µs per GEMM, against 17–200 µs for an MPS
  encode. That cost dominated small-model learners.
- **Split-K:** small outputs with a long K (weight gradients) use split-K.
  Each split writes fp32 partials to a workspace, and a reduce kernel
  applies alpha/beta.
- **Long K:** a barrier-synced K loop (BK 128) keeps the simdgroups close
  in K, following the MPP programming guide.
- **Fallback:** fp32 falls back to MPS if the library fails to build, and
  `PUF_METAL_GEMM=mps` forces MPS. bf16 requires MPP and exits at startup
  without it.
- **Learner-only fp32 speed vs MPS:** breakout ~2.1x, pacman ~1.4x,
  tetris neutral (interleaved A/B).

## bf16

`precision_t` is `__nv_bfloat16`, as on CUDA:

- On the host it is an emulated struct with round-to-nearest-even
  conversion.
- In MSL it is `bfloat`. `puf_kernels.metal` is compiled with
  `-DPUF_BF16`, and every kernel reads and writes precision_t storage
  through `to_float`/`from_float` at the same points as the CUDA kernel.
  This includes the per-step rounding of the MinGRU scan state and the
  rounding of continuous actions before their logprob.
- GEMMs: an all-`CUDA_R_16BF` `cublasGemmEx` call goes to the bf16 MPP
  kernels (bf16 in, fp32 accumulate, bf16 out).
- Muon keeps fp32 momentum and updates the fp32 masters, and the params
  are re-cast to bf16 after each step (same as CUDA).
- The advantage kernel's vector width becomes 8, so the horizon must be a
  multiple of 8 in bf16 (as on CUDA).

### Correctness

**Kernel tests:** `tests/test_metal_kernels.sh ENV --bf16` compares every
Metal kernel with its host emulation. Most are bit-exact. A new
Newton–Schulz test runs Muon's orthogonalization through the Metal GEMM
path against a host reference that rounds at the same points:

| shape | bf16 rel L2 | fp32 rel L2 |
|---|---|---|
| 256x768 | 2.7e-3 | 1.5e-6 |
| 768x256 | 3.0e-3 | 1.5e-6 |

**Oracle (`BF16=1 tests/oracle_train.sh`):** one rollout forward and one
train epoch on identical weights and data, Metal bf16 against CUDA bf16
(RTX 5090):

| env | forward | rollout state | losses (rel) |
|---|---|---|---|
| pacman | 99.5% bit-identical | 99.97% bit-identical | 4.6e-6 |
| drone (continuous) | n/a | 100% bit-identical | 7e-5 |

**Muon update gap and cuBLAS:** the weight update after one epoch differed
by 24% rel L2 (pacman). This was localized to Newton–Schulz, which
amplifies small input differences; the pre-NS gradient matched to 1.1e-3.
The cause is on the CUDA side:

- With `CUBLAS_DEFAULT_MATH`, cuBLAS may do bf16 split-K reductions in
  reduced precision.
- With `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` set (a temporary
  test edit on the CUDA machine, since reverted), the gradient matches to
  1.1e-4 and the update to 3.9%. That is comparable to the 2.6%
  cross-backend noise of fp32.
- The Metal kernels always reduce in fp32, so in bf16 the Mac is slightly
  more accurate than default CUDA. The CUDA build is unchanged; setting
  that math mode there would be a behavior change for upstream to decide.

For scale, bf16 vs fp32 on the same backend differs by 8.3e-3 in the
pacman forward pass and 1.3e-4 in the losses. drone differs by 0.31 in the
forward pass, and by the same amount on CUDA, so that comes from the env
itself.

### Learning

Metal fp32 vs Metal bf16 (breakout and pacman at full length; tetris at
150M of its 1.9B steps):

| env | steps | fp32 score | bf16 score |
|---|---|---|---|
| breakout (seeds 1/2/3) | 55M | 857 / 828 / 854 | 860 / 837 / 829 |
| pacman | 205M (full) | perf 0.829 (score 199.0) | perf 0.832 (score 199.7) |
| tetris | 150M | 79.6 | 83.8 |
| chain_mdp, uchar obs (seeds 1–4, best perf) | 5M | 0.28 / 0 / 0.002 / 0 | 0.38 / 0 / 0 / 0 |

bf16 learns the same as fp32: within seed spread on breakout (3 seeds),
and within a single-seed comparison on pacman and tetris. Envs with uchar
observations train in bf16 too; the obs cast to bf16 is exact for 0–255.

### Speed

End to end, steps / wall time (startup included):

| env | steps | fp32 | bf16 | speed-up |
|---|---|---|---|---|
| breakout | 55M | 2.96M, 2.95M | 3.21M, 3.22M | 1.09x |
| pacman | 205M | 325K (631 s) | 458K (448 s) | 1.41x |
| tetris | 150M | 194K, 201K (775 s, 748 s) | 299K, 280K (502 s, 536 s) | 1.47x |

Learner only (`tests/bench_learner.sh`, interleaved A/B, ratios; absolute
times drifted with power state):

| env | fp32 → bf16 epoch time | speed-up |
|---|---|---|
| breakout | 0.0114–0.0120 s → 0.0082–0.0084 s | ~1.4x |
| pacman | 0.99–1.29 s → 0.76–0.93 s | ~1.3–1.4x |
| tetris | 2.28–2.85 s → 1.61–1.83 s | ~1.4–1.55x |

The gain comes from halved memory traffic in the element-wise kernels
(MinGRU scan, Muon, PPO loss) and from the bf16 MPP GEMMs. In the bf16
profile every hot kernel runs on Metal (`PUF_CPU_PROFILE=1`; the last host
kernel, the float `transpose_102`, is now ported). breakout gains least
because its tiny learner is a small share of wall time; the rollout and
env dominate.

### Env coverage in bf16

Every env with an fp32 Mac trainer (73) was rebuilt with `--bf16` and
smoke-trained for 45 s (the `tests/smoke_train_all.sh` criterion). All 73
build. 70 train, the same 70 as fp32. The three that don't fail the same
way they do in fp32 (PHASE5.md):

- drive: needs the map dataset.
- tactical: needs a 93 GB arena.
- template: hits a missing log key after 38 epochs.

No env hit the bf16 horizon constraint, and no losses went NaN.

## Not done

- **MPP tile tuning:** for example tetris dW shapes, Morton order and
  static extents.
- `tests/test_metal_gemm.mm` still uses the older argument-buffer
  convention.
