// C API of the Metal backend (src/puf_metal.mm). Used by src/puf_cuda_cpu.h
// when built with -DPUF_TRAIN_METAL. Plain C/C++ (no Objective-C), so the
// trainer itself stays C++.
//
// Model
//   - All device memory comes from shared-storage MTLBuffer chunks (the arena).
//     On Apple Silicon unified memory, host code can read and write it
//     directly after the GPU work touching it has completed.
//   - Each OS thread records GPU work into its own command buffer, on its own
//     command queue, in issue order (CUDA stream order). The arena is a
//     hazard-tracked resource, so consecutive GPU commands are serialized.
//   - puf_metal_flush() commits the calling thread's command buffer and waits.
//     Anything that touches arena memory from the host must flush first.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void puf_metal_init(void);
// Preprocessor macro for the kernel library (call before the first kernel use).
void puf_metal_define(const char* name, long value);
// Zeroed, 256-byte aligned arena allocation. Never NULL (aborts when out of memory).
void* puf_metal_alloc(size_t bytes);
void puf_metal_free(void* p);
// True if p points into the arena.
int puf_metal_owns(const void* p);
// GPU virtual address for an arena pointer (0 for NULL; aborts if not in arena).
uint64_t puf_metal_gpu_address(const void* p);

// Commit the calling thread's pending GPU work and wait for completion.
void puf_metal_flush(void);
// GPU blit copy / byte fill within the arena (recorded, not waited for).
void puf_metal_copy(void* dst, const void* src, size_t bytes);
void puf_metal_fill(void* dst, int value, size_t bytes);

// Column-major GEMM with cublasGemmEx semantics, fp32, all operands in the
// arena: C(m,n) = alpha * op(A) @ op(B) + beta * C.
void puf_metal_sgemm(int trans_a, int trans_b, int m, int n, int k,
    float alpha, const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc);

// Compute dispatch of a kernel from the Metal kernel library. args/args_len
// is the kernel's argument struct (bound with setBytes at index 0).
// Returns 0 if the library has no kernel of that name.
int puf_metal_has_kernel(const char* name);
void puf_metal_dispatch(const char* name, uint32_t grid_x, uint32_t grid_y,
    uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z,
    const void* args, size_t args_len);

// Counters for profiling (process-wide).
void puf_metal_stats(long* flushes, long* gemms, long* dispatches, long* blits);

#ifdef __cplusplus
}
#endif
