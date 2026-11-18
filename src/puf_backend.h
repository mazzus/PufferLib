// Trainer compute backend. The trainer is single-source CUDA C++. All device
// API calls and kernel launches go through the CUDA runtime/cuBLAS/cuRAND API
// and PUF_LAUNCH, so the same source builds for:
//   - CUDA (default): real CUDA toolkit headers, PUF_LAUNCH is <<<...>>>
//   - CPU (-DPUF_TRAIN_CPU): src/puf_cuda_cpu.h implements the used subset of
//     the CUDA API on the host, and PUF_LAUNCH runs kernels as host loops.
// Not to be confused with PUF_BACKEND / PUF_CPU / PUF_GPU in pufferenv.h,
// which select where the *environment* runs.
#pragma once

#ifdef PUF_TRAIN_CPU
#include "puf_cuda_cpu.h"
#else
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <nccl.h>
#include <nvml.h>
#include <nvtx3/nvToolsExt.h>
// PUF_LAUNCH(kernel, grid, block[, smem[, stream]])(args...)
#define PUF_LAUNCH(kernel, ...) kernel<<<__VA_ARGS__>>>
// Dynamic shared memory declaration (extern __shared__ T name[]).
#define PUF_DYN_SHARED(T, name) extern __shared__ T name[]
// True when kernels run as host loops (no CUDA graphs, no cooperative blocks).
constexpr bool PUF_HOST_KERNELS = false;
#define PUF_HOST_KERNEL_SCOPE(name)
constexpr bool PUF_HOST_MINGRU = false;
#define PUF_METAL_PORT(kernel, msl, params, args)
#endif
