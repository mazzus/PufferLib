// CPU implementation of the subset of the CUDA API used by the trainer
// (src/pufferl.cu, src/algo.cu, src/ocean.cu). Selected with -DPUF_TRAIN_CPU
// via src/puf_backend.h. Compile the .cu sources as C++17 with clang++.
//
// Execution model
//   - "Device" memory is host memory. Every API call runs synchronously on the
//     calling thread, so stream ordering, events and synchronization hold
//     trivially. Events record wall-clock time so the dashboard's timing
//     breakdown still works.
//   - PUF_LAUNCH runs a kernel as a host loop: blocks (optionally in parallel
//     with OpenMP) and, within a block, threads sequentially in threadIdx order.
//     threadIdx/blockIdx/blockDim/gridDim are thread_local.
//   - Threads of a block do not run concurrently, so __syncthreads() is not
//     supported (it aborts). The only cooperative pattern in the trainer is the
//     trailing block_reduce_sum; the version here runs on the block's last
//     thread with the same tree order as CUDA, which gives identical results.
//   - CUDA graphs are not supported (calls abort); the trainer forces
//     cudagraphs off when PUF_HOST_KERNELS is true.
//   - cuBLAS GEMM maps to Accelerate (macOS) or CBLAS sgemm, fp32 only.
//   - cuRAND Philox4x32-10 device API and host generators are implemented
//     here. Streams are not bit-identical to cuRAND.
//   - NCCL, NVML, NVTX and the profiler API are stubs (single process only).
#pragma once

#ifndef PRECISION_FLOAT
#error "PUF_TRAIN_CPU requires PRECISION_FLOAT (build with --float)"
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// Classic CBLAS sgemm (Accelerate on macOS, OpenBLAS etc. elsewhere).
// Declared here rather than via <Accelerate/Accelerate.h>, whose MacTypes.h
// defines Byte and clashes with the trainer's Byte tensor type.
enum { PufCblasColMajor = 102, PufCblasNoTrans = 111, PufCblasTrans = 112 };
extern "C" void cblas_sgemm(int order, int transa, int transb, int m, int n,
    int k, float alpha, const float* a, int lda, const float* b, int ldb,
    float beta, float* c, int ldc);

#ifdef __APPLE__
typedef unsigned long ulong;
#endif

constexpr bool PUF_HOST_KERNELS = true;

// Metal backend (-DPUF_TRAIN_METAL): device memory is a shared MTLBuffer arena,
// GEMMs/copies/fills run on the GPU, and ported kernels dispatch to Metal.
// Anything else runs on the host after flushing the thread's GPU work.
#ifdef PUF_TRAIN_METAL
#include "puf_metal.h"
#define PUF_METAL_FLUSH() puf_metal_flush()
// MinGRU uses the GPU-portable kernels (ported to Metal) instead of host versions
constexpr bool PUF_HOST_MINGRU = false;
#else
#define PUF_METAL_FLUSH() ((void)0)
constexpr bool PUF_HOST_MINGRU = true;
#endif

[[noreturn]] static inline void puf_cpu_fatal(const char* what) {
    fprintf(stderr, "PUF_TRAIN_CPU: unsupported: %s\n", what);
    abort();
}

// Qualifiers
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __shared__ static thread_local
#define __launch_bounds__(...)

// Vector and index types
struct uint3 { unsigned int x, y, z; };
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};
struct float4 { float x, y, z, w; };
struct alignas(16) uint4 { unsigned int x, y, z, w; };
struct uint2 { unsigned int x, y; };
static inline float4 make_float4(float x, float y, float z, float w) {
    return {x, y, z, w};
}

inline thread_local uint3 threadIdx = {0, 0, 0};
inline thread_local uint3 blockIdx = {0, 0, 0};
inline thread_local dim3 blockDim = dim3(1, 1, 1);
inline thread_local dim3 gridDim = dim3(1, 1, 1);
inline thread_local void* puf_cpu_dyn_smem_ptr = NULL;
inline thread_local size_t puf_cpu_dyn_smem_size = 0;

#define PUF_DYN_SHARED(T, name) T* name = (T*)puf_cpu_dyn_smem_ptr

static inline void __syncthreads() {
    puf_cpu_fatal("__syncthreads (cooperative kernel) in CPU emulation");
}

// Device math intrinsics
template <class T> static inline T __ldg(const T* p) { return *p; }
static inline float __expf(float x) { return expf(x); }
static inline float __logf(float x) { return logf(x); }
// Plain auto: returns by value (decltype(a < b ? a : b) would be a dangling
// reference to a parameter when A == B).
template <class A, class B>
static inline auto min(A a, B b) { return a < b ? a : b; }
template <class A, class B>
static inline auto max(A a, B b) { return a < b ? b : a; }

// bf16 storage type (only for the GPU-env obs widen overload; fp32 training)
struct __nv_bfloat16 { uint16_t x; };
static inline float __bfloat162float(__nv_bfloat16 b) {
    uint32_t bits = (uint32_t)b.x << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
static inline __nv_bfloat16 __float2bfloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
    return {(uint16_t)((bits + rounding) >> 16)};
}

// Deterministic block reduce, emulation version. Same tree order as the CUDA
// version in pufferl.cu (which is skipped via this guard). Threads run
// sequentially, so the last thread of the block does the whole reduction once
// every thread has written its smem slot.
#define PUF_BLOCK_REDUCE_SUM
static inline void block_reduce_sum(
        float* smem, float* out, int tid, int nthreads, int nchan) {
    if (tid != nthreads - 1) {
        return;
    }
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        for (int t = 0; t < s; t++) {
            for (int c = 0; c < nchan; c++) {
                smem[c * nthreads + t] += smem[c * nthreads + t + s];
            }
        }
    }
    for (int c = 0; c < nchan; c++) {
        out[c] = smem[c * nthreads];
    }
}

// Runtime API
typedef int cudaError_t;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind {
    cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice, cudaMemcpyDefault,
};
enum { cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2, cudaHostAllocPortable = 1 };
enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal, cudaStreamCaptureModeThreadLocal,
    cudaStreamCaptureModeRelaxed,
};
struct PufCpuStream { int id; };
struct PufCpuEvent { double t; };
typedef PufCpuStream* cudaStream_t;
typedef PufCpuEvent* cudaEvent_t;
typedef struct PufCpuGraph* cudaGraph_t;
typedef struct PufCpuGraphExec* cudaGraphExec_t;

static inline double puf_cpu_now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static inline unsigned long long puf_cpu_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static inline cudaError_t puf_cpu_host_malloc(void** p, size_t n) {
    void* mem = NULL;
    if (posix_memalign(&mem, 64, n ? n : 64) != 0) {
        return 2;
    }
    memset(mem, 0, n);
    *p = mem;
    return cudaSuccess;
}
static inline cudaError_t puf_cpu_malloc(void** p, size_t n) {
#ifdef PUF_TRAIN_METAL
    *p = puf_metal_alloc(n);
    return cudaSuccess;
#endif
    void* mem = NULL;
    if (posix_memalign(&mem, 64, n ? n : 64) != 0) {
        return 2;
    }
    memset(mem, 0, n);
    *p = mem;
    return cudaSuccess;
}
template <class T> static inline cudaError_t cudaMalloc(T** p, size_t n) {
    return puf_cpu_malloc((void**)p, n);
}
template <class T> static inline cudaError_t cudaHostAlloc(T** p, size_t n, unsigned) {
    return puf_cpu_host_malloc((void**)p, n);
}
static inline cudaError_t cudaFree(void* p) {
#ifdef PUF_TRAIN_METAL
    if (puf_metal_owns(p)) {
        puf_metal_free(p);
        return cudaSuccess;
    }
#endif
    free(p);
    return cudaSuccess;
}
static inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
    if (!n) return cudaSuccess;
#ifdef PUF_TRAIN_METAL
    if (puf_metal_owns(d) && puf_metal_owns(s)) {
        puf_metal_copy(d, s, n);
        return cudaSuccess;
    }
    puf_metal_flush();
#endif
    memmove(d, s, n);
    return cudaSuccess;
}
static inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n,
        cudaMemcpyKind k, cudaStream_t = 0) {
    return cudaMemcpy(d, s, n, k);
}
static inline cudaError_t cudaMemset(void* p, int v, size_t n) {
    if (!n) return cudaSuccess;
#ifdef PUF_TRAIN_METAL
    if (puf_metal_owns(p)) {
        puf_metal_fill(p, v, n);
        return cudaSuccess;
    }
#endif
    memset(p, v, n);
    return cudaSuccess;
}
static inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t = 0) {
    return cudaMemset(p, v, n);
}
static inline cudaError_t cudaMemGetInfo(size_t* free_b, size_t* total_b) {
    *free_b = 0;
    *total_b = 0;
    return cudaSuccess;
}
static inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
static inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }
static inline cudaError_t cudaDeviceSynchronize() { PUF_METAL_FLUSH(); return cudaSuccess; }
static inline cudaError_t cudaProfilerStart() { return cudaSuccess; }
static inline cudaError_t cudaProfilerStop() { return cudaSuccess; }

static inline cudaError_t cudaStreamCreate(cudaStream_t* s) {
    *s = new PufCpuStream{1};
    return cudaSuccess;
}
static inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) {
    return cudaStreamCreate(s);
}
static inline cudaError_t cudaStreamSynchronize(cudaStream_t) { PUF_METAL_FLUSH(); return cudaSuccess; }
static inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) {
    return cudaSuccess;
}
static inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
    *e = new PufCpuEvent{0.0};
    return cudaSuccess;
}
static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) {
    return cudaEventCreate(e);
}
static inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
static inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t = 0) {
    e->t = puf_cpu_now_s();
    return cudaSuccess;
}
static inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    *ms = (float)((b->t - a->t) * 1000.0);
    return cudaSuccess;
}

static inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
    puf_cpu_fatal("cudaStreamBeginCapture (CUDA graphs)");
}
static inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) {
    puf_cpu_fatal("cudaStreamEndCapture (CUDA graphs)");
}
static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long) {
    puf_cpu_fatal("cudaGraphInstantiate (CUDA graphs)");
}
static inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) {
    puf_cpu_fatal("cudaGraphLaunch (CUDA graphs)");
}
static inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }

// Optional profiler: PUF_CPU_PROFILE=1 prints per-kernel / per-GEMM-shape wall
// time at exit (summed over all threads that launch).
#include <atomic>
#include <mutex>
struct PufProfEntry { const char* name; char key[96]; double sec; long calls; };
inline std::mutex puf_prof_mu;
inline PufProfEntry puf_prof_tab[512];
inline int puf_prof_n = 0;
inline int puf_prof_on = -1;
static inline void puf_prof_dump() {
    std::lock_guard<std::mutex> g(puf_prof_mu);
    double total = 0;
    for (int i = 0; i < puf_prof_n; i++) total += puf_prof_tab[i].sec;
    // simple selection sort by time
    for (int i = 0; i < puf_prof_n; i++) {
        for (int j = i + 1; j < puf_prof_n; j++) {
            if (puf_prof_tab[j].sec > puf_prof_tab[i].sec) {
                PufProfEntry t = puf_prof_tab[i];
                puf_prof_tab[i] = puf_prof_tab[j];
                puf_prof_tab[j] = t;
            }
        }
    }
    fprintf(stderr, "\nPUF_CPU_PROFILE (total %.2fs)\n", total);
#ifdef PUF_TRAIN_METAL
    long fl, gm, dp, bl;
    puf_metal_stats(&fl, &gm, &dp, &bl);
    fprintf(stderr, "metal: flushes=%ld gemms=%ld dispatches=%ld blits=%ld\n", fl, gm, dp, bl);
#endif
    for (int i = 0; i < puf_prof_n && i < 40; i++) {
        PufProfEntry* e = &puf_prof_tab[i];
        fprintf(stderr, "%8.3fs %5.1f%% %9ld  %s\n", e->sec, 100 * e->sec / total,
            e->calls, e->key);
    }
}
static inline bool puf_prof_enabled() {
    if (puf_prof_on < 0) {
        const char* v = getenv("PUF_CPU_PROFILE");
        puf_prof_on = v && v[0] == '1';
        if (puf_prof_on) atexit(puf_prof_dump);
    }
    return puf_prof_on;
}
static inline void puf_prof_add(const char* key, double sec) {
    std::lock_guard<std::mutex> g(puf_prof_mu);
    for (int i = 0; i < puf_prof_n; i++) {
        if (strcmp(puf_prof_tab[i].key, key) == 0) {
            puf_prof_tab[i].sec += sec;
            puf_prof_tab[i].calls++;
            return;
        }
    }
    if (puf_prof_n < 512) {
        PufProfEntry* e = &puf_prof_tab[puf_prof_n++];
        snprintf(e->key, sizeof(e->key), "%s", key);
        e->sec = sec;
        e->calls = 1;
    }
}

// Entry of a host-native kernel (reads/writes device memory on the host).
struct PufProfScope {
    const char* name;
    double t0;
    PufProfScope(const char* n) : name(n), t0(0.0) {
        PUF_METAL_FLUSH();
        t0 = puf_prof_enabled() ? puf_cpu_now_s() : 0.0;
    }
    ~PufProfScope() {
        if (puf_prof_on == 1) puf_prof_add(name, puf_cpu_now_s() - t0);
    }
};
#define PUF_HOST_KERNEL_SCOPE(name) PufProfScope _puf_prof_scope(name)

// Kernel launch
struct PufLaunchCfg {
    dim3 grid, block;
    size_t smem;
};
static inline PufLaunchCfg puf_cfg(dim3 grid, dim3 block, size_t smem = 0,
        cudaStream_t = 0) {
    return {grid, block, smem};
}

// Metal kernel ports. A ported kernel gets one overload of puf_metal_try for
// its kernel id (declared next to the CUDA kernel via PUF_METAL_PORT). The
// overload has the kernel's exact parameter types, so overloaded kernels
// resolve as in CUDA, and marshals the parameters (in order, natural
// alignment, arena pointers as GPU addresses) into the MSL argument struct.
// The variadic fallback (worst conversion rank) means "not ported".
constexpr uint64_t puf_hash(const char* s) {
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h = (h ^ (uint64_t)(unsigned char)*s++) * 1099511628211ull;
    }
    return h;
}
template <uint64_t H> struct PufKernelId {};
struct PufLaunchCfg;
template <uint64_t H>
static inline bool puf_metal_try(PufKernelId<H>, const PufLaunchCfg&, ...) {
    return false;
}

struct PufArgWriter {
    alignas(16) unsigned char buf[2048];
    size_t len = 0;
    bool ok = true;
    void align(size_t a) { len = (len + a - 1) & ~(a - 1); }
    void raw(const void* p, size_t n, size_t a) {
        align(a);
        if (len + n > sizeof(buf)) { ok = false; return; }
        memcpy(buf + len, p, n);
        len += n;
    }
    template <class... A> bool put(const A&... a);
};
#ifdef PUF_TRAIN_METAL
template <class T> static inline void puf_put(PufArgWriter& w, T* p) {
    if (p != NULL && !puf_metal_owns(p)) {
        w.ok = false;  // host pointer: cannot run on the GPU
    }
    uint64_t a = puf_metal_gpu_address(w.ok ? p : NULL);
    w.raw(&a, 8, 8);
}
#endif
static inline void puf_put(PufArgWriter& w, int v) { w.raw(&v, 4, 4); }
static inline void puf_put(PufArgWriter& w, unsigned int v) { w.raw(&v, 4, 4); }
static inline void puf_put(PufArgWriter& w, float v) { w.raw(&v, 4, 4); }
static inline void puf_put(PufArgWriter& w, bool v) { w.raw(&v, 1, 1); }
static inline void puf_put(PufArgWriter& w, long v) { w.raw(&v, 8, 8); }
static inline void puf_put(PufArgWriter& w, long long v) { w.raw(&v, 8, 8); }
static inline void puf_put(PufArgWriter& w, unsigned long v) { w.raw(&v, 8, 8); }
static inline void puf_put(PufArgWriter& w, unsigned long long v) { w.raw(&v, 8, 8); }
template <class... A> bool PufArgWriter::put(const A&... a) {
    (puf_put(*this, a), ...);
    align(16);
    return ok;
}
#define PUF_UNPAREN(...) __VA_ARGS__
#ifdef PUF_TRAIN_METAL
// PUF_METAL_PORT(kernel, "msl_name", (param decls), (param names))
#define PUF_METAL_PORT(kernel, msl, params, args) \
    static inline bool puf_metal_try(PufKernelId<puf_hash(#kernel)>, \
            const PufLaunchCfg& _c, PUF_UNPAREN params) { \
        PufArgWriter _w; \
        if (!_w.put args) return false; \
        return puf_metal_launch(msl, _c, _w); \
    }
#else
#define PUF_METAL_PORT(kernel, msl, params, args)
#endif

// Below this many total threads, run blocks on the calling thread.
#ifndef PUF_CPU_PAR_MIN_THREADS
#define PUF_CPU_PAR_MIN_THREADS 32768
#endif

#ifdef PUF_TRAIN_METAL
// Set to force every kernel onto the CPU emulation (differential tests).
inline bool puf_metal_force_cpu = false;
static inline bool puf_metal_launch(const char* msl, const PufLaunchCfg& c, PufArgWriter& w) {
    if (puf_metal_force_cpu || !puf_metal_has_kernel(msl)) {
        return false;
    }
    puf_metal_dispatch(msl, c.grid.x, c.grid.y, c.grid.z, c.block.x, c.block.y,
        c.block.z, w.buf, w.len);
    return true;
}
#endif

template <uint64_t H, class F> struct PufLaunch {
    PufLaunchCfg c;
    F f;
    const char* name;

    template <class... A> void run_block(long b, A&... a) {
        dim3 g = c.grid, bd = c.block;
        gridDim = g;
        blockDim = bd;
        if (c.smem > puf_cpu_dyn_smem_size) {
            free(puf_cpu_dyn_smem_ptr);
            puf_cpu_dyn_smem_ptr = malloc(c.smem);
            puf_cpu_dyn_smem_size = c.smem;
        }
        blockIdx = {(unsigned)(b % g.x), (unsigned)((b / g.x) % g.y),
            (unsigned)(b / ((long)g.x * g.y))};
        for (unsigned tz = 0; tz < bd.z; tz++) {
            for (unsigned ty = 0; ty < bd.y; ty++) {
                for (unsigned tx = 0; tx < bd.x; tx++) {
                    threadIdx = {tx, ty, tz};
                    f(a...);
                }
            }
        }
    }

    template <class... A> void operator()(A... a) {
        bool prof = puf_prof_enabled();
        double t0 = prof ? puf_cpu_now_s() : 0.0;
#ifdef PUF_TRAIN_METAL
        if (puf_metal_try(PufKernelId<H>{}, c, a...)) {
            if (prof) {
                char key[96];
                snprintf(key, sizeof(key), "metal %s (encode)", name);
                puf_prof_add(key, puf_cpu_now_s() - t0);
            }
            return;
        }
#endif
        PUF_METAL_FLUSH();
        if (prof) t0 = puf_cpu_now_s();
        launch(a...);
        if (prof) {
            puf_prof_add(name, puf_cpu_now_s() - t0);
        }
    }

    template <class... A> void launch(A&... a) {
        long nblocks = (long)c.grid.x * c.grid.y * c.grid.z;
        long nthreads = (long)c.block.x * c.block.y * c.block.z;
        if (nblocks <= 0 || nthreads <= 0) {
            return;
        }
        if (nblocks > 1 && nblocks * nthreads >= PUF_CPU_PAR_MIN_THREADS
                && !omp_in_parallel()) {
            #pragma omp parallel for schedule(static)
            for (long b = 0; b < nblocks; b++) {
                run_block(b, a...);
            }
        } else {
            for (long b = 0; b < nblocks; b++) {
                run_block(b, a...);
            }
        }
    }
};
template <uint64_t H, class F> static inline PufLaunch<H, F> puf_launch(PufLaunchCfg c,
        F f, const char* name) {
    return PufLaunch<H, F>{c, f, name};
}
// PUF_LAUNCH(kernel, grid, block[, smem[, stream]])(args...)
#define PUF_LAUNCH(kernel, ...) \
    puf_launch<puf_hash(#kernel)>(puf_cfg(__VA_ARGS__), \
        [](auto&... _puf_a) { kernel(_puf_a...); }, #kernel)

// cuBLAS (fp32 only)
typedef struct PufCpuCublas* cublasHandle_t;
typedef int cublasStatus_t;
enum { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
enum cudaDataType_t { CUDA_R_32F = 0, CUDA_R_16BF = 14 };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F = 68 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT = -1 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0 };
static inline cublasStatus_t cublasCreate(cublasHandle_t* h) { *h = NULL; return 0; }
static inline cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) { return 0; }
static inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) { return 0; }
static inline cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) { return 0; }
// Column-major, same argument semantics as cublasGemmEx.
static inline cublasStatus_t cublasGemmEx(cublasHandle_t,
        cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
        const void* alpha, const void* A, cudaDataType_t at, int lda,
        const void* B, cudaDataType_t bt, int ldb, const void* beta,
        void* C, cudaDataType_t ct, int ldc, cublasComputeType_t, cublasGemmAlgo_t) {
    if (at != CUDA_R_32F || bt != CUDA_R_32F || ct != CUDA_R_32F) {
        puf_cpu_fatal("cublasGemmEx with non-fp32 operands");
    }
    if (m <= 0 || n <= 0) {
        return 0;
    }
    bool prof = puf_prof_enabled();
    double t0 = prof ? puf_cpu_now_s() : 0.0;
#ifdef PUF_TRAIN_METAL
    puf_metal_sgemm(ta == CUBLAS_OP_T, tb == CUBLAS_OP_T, m, n, k,
        *(const float*)alpha, (const float*)A, lda, (const float*)B, ldb,
        *(const float*)beta, (float*)C, ldc);
    if (prof) {
        char key[96];
        snprintf(key, sizeof(key), "mps sgemm %c%c m=%d n=%d k=%d (encode)",
            ta == CUBLAS_OP_T ? 'T' : 'N', tb == CUBLAS_OP_T ? 'T' : 'N', m, n, k);
        puf_prof_add(key, puf_cpu_now_s() - t0);
    }
    return 0;
#endif
    cblas_sgemm(PufCblasColMajor,
        ta == CUBLAS_OP_T ? PufCblasTrans : PufCblasNoTrans,
        tb == CUBLAS_OP_T ? PufCblasTrans : PufCblasNoTrans,
        m, n, k, *(const float*)alpha, (const float*)A, lda,
        (const float*)B, ldb, *(const float*)beta, (float*)C, ldc);
    if (prof) {
        char key[96];
        snprintf(key, sizeof(key), "sgemm %c%c m=%d n=%d k=%d",
            ta == CUBLAS_OP_T ? 'T' : 'N', tb == CUBLAS_OP_T ? 'T' : 'N', m, n, k);
        puf_prof_add(key, puf_cpu_now_s() - t0);
    }
    return 0;
}

// cuRAND: Philox4x32-10 (Salmon et al. 2011), counter = (ctr, subsequence),
// key = seed. Layout and semantics follow cuRAND's device API; outputs are not
// guaranteed to be bit-identical to cuRAND.
// alignas(16): same 64-byte layout as the MSL struct (uint4 is 16-aligned there)
struct alignas(16) curandStatePhilox4_32_10_t {
    uint4 ctr;
    uint4 output;
    uint2 key;
    unsigned int state;
    int boxmuller_flag;
    float boxmuller_extra;
};

static inline void puf_philox_round(uint4& c, uint2 k) {
    uint64_t p0 = (uint64_t)0xD2511F53u * c.x;
    uint64_t p1 = (uint64_t)0xCD9E8D57u * c.z;
    uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
    uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
    c = {hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0};
}
static inline uint4 puf_philox4x32_10(uint4 c, uint2 k) {
    for (int i = 0; i < 10; i++) {
        puf_philox_round(c, k);
        k.x += 0x9E3779B9u;
        k.y += 0xBB67AE85u;
    }
    return c;
}
static inline void puf_philox_incr(uint4& c) {
    if (++c.x) return;
    if (++c.y) return;
    if (++c.z) return;
    ++c.w;
}
static inline void curand_init(unsigned long long seed,
        unsigned long long subsequence, unsigned long long offset,
        curandStatePhilox4_32_10_t* s) {
    s->key = {(unsigned)seed, (unsigned)(seed >> 32)};
    s->ctr = {0, 0, (unsigned)subsequence, (unsigned)(subsequence >> 32)};
    s->state = 0;
    s->boxmuller_flag = 0;
    s->boxmuller_extra = 0.0f;
    unsigned long long blocks = offset / 4;
    s->ctr.x += (unsigned)blocks;
    if (s->ctr.x < (unsigned)blocks) s->ctr.y++;
    s->ctr.y += (unsigned)(blocks >> 32);
    s->output = puf_philox4x32_10(s->ctr, s->key);
    s->state = (unsigned)(offset % 4);
}
static inline unsigned int curand(curandStatePhilox4_32_10_t* s) {
    unsigned int r;
    switch (s->state) {
        case 0: r = s->output.x; break;
        case 1: r = s->output.y; break;
        case 2: r = s->output.z; break;
        default: r = s->output.w; break;
    }
    if (++s->state == 4) {
        puf_philox_incr(s->ctr);
        s->output = puf_philox4x32_10(s->ctr, s->key);
        s->state = 0;
    }
    return r;
}
#define PUF_2POW32_INV 2.3283064e-10f
static inline float puf_u32_to_uniform(unsigned int x) {
    return x * PUF_2POW32_INV + PUF_2POW32_INV / 2.0f;  // (0, 1]
}
static inline float curand_uniform(curandStatePhilox4_32_10_t* s) {
    return puf_u32_to_uniform(curand(s));
}
static inline void puf_box_muller(unsigned int a, unsigned int b, float* z0, float* z1) {
    float u = puf_u32_to_uniform(a);
    float v = b * PUF_2POW32_INV * 6.2831855f + PUF_2POW32_INV * 3.1415927f;
    float r = sqrtf(-2.0f * logf(u));
    *z0 = r * sinf(v);
    *z1 = r * cosf(v);
}
static inline float curand_normal(curandStatePhilox4_32_10_t* s) {
    if (s->boxmuller_flag) {
        s->boxmuller_flag = 0;
        return s->boxmuller_extra;
    }
    unsigned int a = curand(s), b = curand(s);
    float z0, z1;
    puf_box_muller(a, b, &z0, &z1);
    s->boxmuller_flag = 1;
    s->boxmuller_extra = z1;
    return z0;
}

// cuRAND host generator API
typedef int curandStatus_t;
enum curandRngType_t { CURAND_RNG_PSEUDO_DEFAULT = 100 };
struct PufCpuCurandGen { unsigned long long seed; };
typedef PufCpuCurandGen* curandGenerator_t;
static inline curandStatus_t curandCreateGenerator(curandGenerator_t* g, curandRngType_t) {
    *g = new PufCpuCurandGen{0};
    return 0;
}
static inline curandStatus_t curandDestroyGenerator(curandGenerator_t g) {
    delete g;
    return 0;
}
static inline curandStatus_t curandSetPseudoRandomGeneratorSeed(
        curandGenerator_t g, unsigned long long seed) {
    g->seed = seed;
    return 0;
}
static inline curandStatus_t curandGenerateUniform(curandGenerator_t g, float* out, size_t n) {
    PUF_METAL_FLUSH();
    curandStatePhilox4_32_10_t s;
    curand_init(g->seed, 0, 0, &s);
    for (size_t i = 0; i < n; i++) {
        out[i] = curand_uniform(&s);
    }
    return 0;
}
static inline curandStatus_t curandGenerateNormal(curandGenerator_t g, float* out,
        size_t n, float mean, float stddev) {
    PUF_METAL_FLUSH();
    curandStatePhilox4_32_10_t s;
    curand_init(g->seed, 0, 0, &s);
    for (size_t i = 0; i + 1 < n + 1; i += 2) {
        unsigned int a = curand(&s), b = curand(&s);
        float z0, z1;
        puf_box_muller(a, b, &z0, &z1);
        out[i] = mean + stddev * z0;
        if (i + 1 < n) {
            out[i + 1] = mean + stddev * z1;
        }
    }
    return 0;
}

// NCCL (single process only)
struct ncclUniqueId { char internal[128]; };
typedef struct PufCpuNcclComm* ncclComm_t;
typedef int ncclResult_t;
enum ncclDataType_t { ncclFloat = 7, ncclBfloat16 = 9 };
enum ncclRedOp_t { ncclSum = 0, ncclAvg = 4 };
static inline ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
    memset(id, 0, sizeof(*id));
    return 0;
}
static inline ncclResult_t ncclCommInitRank(ncclComm_t*, int, ncclUniqueId, int) {
    puf_cpu_fatal("NCCL multi-process training");
}
static inline ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t,
        ncclRedOp_t, ncclComm_t, cudaStream_t) {
    puf_cpu_fatal("ncclAllReduce");
}
static inline ncclResult_t ncclCommDestroy(ncclComm_t) { return 0; }

// NVML
typedef struct PufCpuNvmlDevice* nvmlDevice_t;
typedef int nvmlReturn_t;
struct nvmlUtilization_t { unsigned int gpu, memory; };
static inline nvmlReturn_t nvmlInit() { return 0; }
static inline nvmlReturn_t nvmlShutdown() { return 0; }
static inline nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int, nvmlDevice_t* d) {
    *d = NULL;
    return 0;
}
static inline nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t, nvmlUtilization_t* u) {
    u->gpu = 0;
    u->memory = 0;
    return 0;
}

// NVTX
static inline int nvtxRangePushA(const char*) { return 0; }
static inline int nvtxRangePop() { return 0; }
