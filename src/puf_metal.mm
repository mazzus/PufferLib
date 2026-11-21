// Metal backend implementation. See src/puf_metal.h for the model.
// Objective-C++ (ARC). Built and linked by build.sh --metal.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "puf_metal.h"

[[noreturn]] static void puf_metal_fatal(const char* what) {
    fprintf(stderr, "PUF_TRAIN_METAL: %s\n", what);
    abort();
}

struct Chunk {
    id<MTLBuffer> buf;
    char* base;
    size_t cap;
    size_t used;
};

static id<MTLDevice> g_dev = nil;
static id<MTLLibrary> g_lib = nil;
static id<MTLLibrary> g_gemm_lib = nil;  // src/puf_gemm.metal (MPP); nil = MPS GEMMs
static std::mutex g_mu;
static std::vector<Chunk> g_chunks;           // guarded by g_mu
static std::vector<id<MTLResource>> g_res;   // all chunk buffers, for useResources
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipes;
static NSMutableDictionary* g_macros = nil;
static std::atomic<long> g_flushes{0}, g_gemm_n{0}, g_dispatch_n{0}, g_blit_n{0};
static std::mutex g_time_mu;
static double g_gpu_s = 0, g_wait_s = 0;  // guarded by g_time_mu

static const size_t kChunk = (size_t)1 << 30;

// Per-thread recording state (CUDA stream order within a thread).
struct ThreadCtx {
    id<MTLCommandQueue> queue = nil;
    id<MTLCommandBuffer> cb = nil;
    id<MTLCommandBuffer> last = nil;  // last committed, not yet waited for
    int pending = 0;                  // commands recorded into cb
    char* staging = nullptr;          // host->device upload ring (arena)
    size_t staging_off = 0;
    id<MTLComputeCommandEncoder> compute = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    // MPS kernel objects are not safe to encode from several threads at once,
    // so each thread caches its own (rollout workers encode concurrently).
    std::unordered_map<std::string, MPSMatrixMultiplication*> gemms;
    std::unordered_map<std::string, MPSMatrix*> mats;
    void* gemm_ws = nullptr;          // split-K fp32 partials (arena)
    size_t gemm_ws_bytes = 0;
};

// MPSMatrix wrappers for (buffer, offset, shape, rowBytes), cached per thread:
// the trainer reuses the same arena views every step.
static MPSMatrix* cached_matrix(ThreadCtx* c, id<MTLBuffer> buf, size_t off,
        int rows, int cols, size_t row_bytes) {
    char key[96];
    snprintf(key, sizeof(key), "%p %zu %d %d %zu", (__bridge void*)buf, off, rows, cols, row_bytes);
    auto it = c->mats.find(key);
    if (it != c->mats.end()) return it->second;
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
        columns:cols rowBytes:row_bytes dataType:MPSDataTypeFloat32];
    MPSMatrix* m = [[MPSMatrix alloc] initWithBuffer:buf offset:off descriptor:d];
    c->mats[key] = m;
    return m;
}
static thread_local ThreadCtx* tl_ctx = nullptr;

static ThreadCtx* ctx() {
    if (!tl_ctx) {
        puf_metal_init();
        tl_ctx = new ThreadCtx();
        tl_ctx->queue = [g_dev newCommandQueue];
    }
    if (!tl_ctx->cb) {
        tl_ctx->cb = [tl_ctx->queue commandBufferWithUnretainedReferences];
    }
    return tl_ctx;
}

// Commit the current command buffer without waiting, so the GPU executes it
// while the host keeps encoding. Same-queue command buffers run in order.
static int g_commit_every = 0;
static long g_small_gemm = 0;  // M*N*K at or below which gemm_tile replaces MPS
// MPP GEMM tuning (env PUF_GEMM_SPLIT_TARGET / _MIN_KC / _KLOOP_MIN)
static long g_split_target = 2048; // threadgroups to aim for (~50 per core, 40 cores)
static int g_split_min_kc = 512;   // minimum K per split
static int g_kloop_min = 1024;     // K per threadgroup at which the barrier K loop is used
static void commit_nowait(ThreadCtx* c);
// PUF_METAL_GPU_PROFILE=1: every dispatch/blit gets its own command buffer and
// its GPU time is attributed to its label (kernel name, or GEMM name + shape).
// Serializes the GPU, so absolute speed drops; shares are what to read.
static int g_gpu_prof = 0;
static thread_local const char* tl_label = nullptr;
static void gpu_prof_commit(ThreadCtx* c, const char* label);
static void count_command(ThreadCtx* c) {
    if (++c->pending >= g_commit_every) commit_nowait(c);
}

static void end_encoders(ThreadCtx* c) {
    if (c->compute) {
        [c->compute endEncoding];
        c->compute = nil;
    }
    if (c->blit) {
        [c->blit endEncoding];
        c->blit = nil;
    }
}

static void load_library() {
    const char* path = getenv("PUF_METAL_KERNELS");
    if (!path) path = "src/puf_kernels.metal";
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path]
        encoding:NSUTF8StringEncoding error:nil];
    if (!src) {
        // Fatal: silently running every kernel on the CPU would hide a broken install.
        fprintf(stderr, "PUF_TRAIN_METAL: kernel library not found at %s (run from the repo "
            "root or set PUF_METAL_KERNELS)\n", path);
        abort();
    }
    MTLCompileOptions* opts = [MTLCompileOptions new];
    if (g_macros) opts.preprocessorMacros = g_macros;
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeSafe;
        opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError* err = nil;
    g_lib = [g_dev newLibraryWithSource:src options:opts error:&err];
    if (!g_lib) {
        fprintf(stderr, "PUF_TRAIN_METAL: kernel library compile failed:\n%s\n",
            [[err description] UTF8String]);
        abort();
    }
}

// MPP GEMM library (MSL 4, macOS 26+). Optional: if it is missing or fails
// to compile, fp32 GEMMs use MPS and bf16 GEMMs are unavailable.
// PUF_METAL_GEMM=mps forces MPS.
static void load_gemm_library() {
    const char* force = getenv("PUF_METAL_GEMM");
    if (force && strcmp(force, "mps") == 0) return;
    const char* path = getenv("PUF_METAL_GEMM_KERNELS");
    if (!path) path = "src/puf_gemm.metal";
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path]
        encoding:NSUTF8StringEncoding error:nil];
    if (!src) {
        fprintf(stderr, "PUF_TRAIN_METAL: %s not found; using MPS GEMMs\n", path);
        return;
    }
    if (@available(macOS 26.0, *)) {
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        NSError* err = nil;
        g_gemm_lib = [g_dev newLibraryWithSource:src options:opts error:&err];
        if (!g_gemm_lib) {
            fprintf(stderr, "PUF_TRAIN_METAL: %s failed to compile; using MPS GEMMs\n%s\n",
                path, [[err description] UTF8String]);
        }
    } else {
        fprintf(stderr, "PUF_TRAIN_METAL: MPP GEMMs need macOS 26; using MPS GEMMs\n");
    }
}

extern "C" int puf_metal_has_mpp_gemm(void) {
    puf_metal_init();
    return g_gemm_lib != nil;
}

extern "C" void puf_metal_define(const char* name, long value) {
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    if (!g_macros) g_macros = [NSMutableDictionary new];
    g_macros[[NSString stringWithUTF8String:name]] = @(value);
}

extern "C" void puf_metal_init(void) {
    static std::once_flag once;
    std::call_once(once, [] {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) puf_metal_fatal("no Metal device");
        const char* ce = getenv("PUF_METAL_COMMIT_EVERY");
        g_commit_every = ce ? atoi(ce) : 16;
        const char* sg = getenv("PUF_METAL_SMALL_GEMM");
        g_small_gemm = sg ? atol(sg) : 0;  // gemm_tile measured slower than MPS on the GPU
        if (const char* v = getenv("PUF_GEMM_SPLIT_TARGET")) g_split_target = atol(v);
        if (const char* v = getenv("PUF_GEMM_MIN_KC")) g_split_min_kc = atoi(v);
        if (const char* v = getenv("PUF_GEMM_KLOOP_MIN")) g_kloop_min = atoi(v);
        if (const char* v = getenv("PUF_METAL_GPU_PROFILE")) g_gpu_prof = atoi(v);
        if (g_commit_every < 1) g_commit_every = 1;
        load_library();
        load_gemm_library();
    });
}

extern "C" void* puf_metal_alloc(size_t bytes) {
    puf_metal_init();
    size_t n = bytes ? (bytes + 255) & ~(size_t)255 : 256;
    std::lock_guard<std::mutex> g(g_mu);
    for (Chunk& c : g_chunks) {
        if (c.cap - c.used >= n) {
            char* p = c.base + c.used;
            c.used += n;
            memset(p, 0, n);
            return p;
        }
    }
    size_t cap = n > kChunk ? n : kChunk;
    id<MTLBuffer> buf = [g_dev newBufferWithLength:cap options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "PUF_TRAIN_METAL: arena chunk allocation of %.2f GB failed "
            "(maxBufferLength %.1f GB, arena so far %zu chunks)\n", cap / 1e9,
            g_dev.maxBufferLength / 1e9, g_chunks.size());
        abort();
    }
    Chunk c = {buf, (char*)buf.contents, cap, n};
    g_chunks.push_back(c);
    g_res.push_back(buf);
    memset(c.base, 0, n);
    return c.base;
}

// Bump allocator: frees are ignored (the trainer allocates up front).
extern "C" void puf_metal_free(void*) {}

// Returns the chunk by value: another thread's puf_metal_alloc may grow
// g_chunks (reallocating it) once the lock is released.
static bool find_chunk(const void* p, Chunk* out) {
    const char* q = (const char*)p;
    std::lock_guard<std::mutex> g(g_mu);
    for (const Chunk& c : g_chunks) {
        if (q >= c.base && q < c.base + c.cap) {
            *out = c;
            return true;
        }
    }
    return false;
}

extern "C" int puf_metal_owns(const void* p) {
    Chunk c;
    return p != nullptr && find_chunk(p, &c);
}

extern "C" uint64_t puf_metal_gpu_address(const void* p) {
    if (!p) return 0;
    Chunk c;
    if (!find_chunk(p, &c)) puf_metal_fatal("pointer is not in the Metal arena");
    return c.buf.gpuAddress + (uint64_t)((const char*)p - c.base);
}

static void commit_nowait(ThreadCtx* c) {
    if (!c->cb) return;
    end_encoders(c);
    [c->cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "PUF_TRAIN_METAL: command buffer error: %s\n",
                [[done.error description] UTF8String]);
            abort();
        }
        std::lock_guard<std::mutex> g(g_time_mu);
        g_gpu_s += done.GPUEndTime - done.GPUStartTime;
    }];
    [c->cb commit];
    c->last = c->cb;
    c->cb = nil;
    c->pending = 0;
}

extern "C" void puf_metal_flush(void) {
    ThreadCtx* c = tl_ctx;
    if (!c || (!c->cb && !c->last)) return;
    commit_nowait(c);
    CFTimeInterval w0 = CACurrentMediaTime();
    [c->last waitUntilCompleted];
    c->last = nil;
    {
        std::lock_guard<std::mutex> g(g_time_mu);
        g_wait_s += CACurrentMediaTime() - w0;
    }
    g_flushes++;
}

struct GpuProfEntry { double s = 0; long n = 0; };
static std::map<std::string, GpuProfEntry>* g_gpu_prof_map = nullptr;
static void gpu_prof_dump() {
    std::lock_guard<std::mutex> g(g_time_mu);
    if (!g_gpu_prof_map) return;
    std::vector<std::pair<std::string, GpuProfEntry>> v(g_gpu_prof_map->begin(), g_gpu_prof_map->end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.s > b.second.s; });
    double total = 0;
    for (auto& e : v) total += e.second.s;
    fprintf(stderr, "\nPUF_METAL_GPU_PROFILE (GPU time %.2fs, one command buffer per command)\n", total);
    for (auto& e : v) {
        if (e.second.s < total * 1e-3) break;
        fprintf(stderr, "  %8.3fs %5.1f%% %8ld %9.1fus  %s\n", e.second.s, 100 * e.second.s / total,
            e.second.n, 1e6 * e.second.s / e.second.n, e.first.c_str());
    }
}
static void gpu_prof_commit(ThreadCtx* c, const char* label) {
    std::string key = label;
    {
        std::lock_guard<std::mutex> g(g_time_mu);
        if (!g_gpu_prof_map) {
            g_gpu_prof_map = new std::map<std::string, GpuProfEntry>();
            atexit(gpu_prof_dump);
        }
    }
    [c->cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        std::lock_guard<std::mutex> g(g_time_mu);
        GpuProfEntry& e = (*g_gpu_prof_map)[key];
        e.s += done.GPUEndTime - done.GPUStartTime;
        e.n++;
    }];
    commit_nowait(c);
}

// Small host->device copy without a flush: stage the bytes in a per-thread
// arena ring and record a GPU blit. Returns 0 if too large.
extern "C" int puf_metal_upload(void* dst, const void* src, size_t bytes) {
    const size_t ring = (size_t)4 << 20;
    if (bytes > ring / 4) return 0;
    ThreadCtx* c = ctx();
    if (!c->staging) c->staging = (char*)puf_metal_alloc(ring);
    size_t off = (c->staging_off + 255) & ~(size_t)255;
    if (off + bytes > ring) {
        puf_metal_flush();  // earlier staged blits must be done before reuse
        c = ctx();
        off = 0;
    }
    memcpy(c->staging + off, src, bytes);
    c->staging_off = off + bytes;
    puf_metal_copy(dst, c->staging + off, bytes);
    return 1;
}

static id<MTLBlitCommandEncoder> blit_encoder(ThreadCtx* c) {
    if (c->compute) {
        [c->compute endEncoding];
        c->compute = nil;
    }
    if (!c->blit) c->blit = [c->cb blitCommandEncoder];
    return c->blit;
}

extern "C" void puf_metal_copy(void* dst, const void* src, size_t bytes) {
    if (!bytes) return;
    Chunk cd, cs;
    if (!find_chunk(dst, &cd) || !find_chunk(src, &cs)) puf_metal_fatal("puf_metal_copy outside arena");
    ThreadCtx* c = ctx();
    [blit_encoder(c) copyFromBuffer:cs.buf sourceOffset:(const char*)src - cs.base
        toBuffer:cd.buf destinationOffset:(char*)dst - cd.base size:bytes];
    g_blit_n++;
    if (g_gpu_prof) gpu_prof_commit(c, "blit copy");
    else count_command(c);
}

extern "C" void puf_metal_fill(void* dst, int value, size_t bytes) {
    if (!bytes) return;
    Chunk cd;
    if (!find_chunk(dst, &cd)) puf_metal_fatal("puf_metal_fill outside arena");
    ThreadCtx* c = ctx();
    size_t off = (char*)dst - cd.base;
    [blit_encoder(c) fillBuffer:cd.buf range:NSMakeRange(off, bytes) value:(uint8_t)value];
    g_blit_n++;
    if (g_gpu_prof) gpu_prof_commit(c, "blit fill");
    else count_command(c);
}

// MPP GEMM (src/puf_gemm.metal). Takes cuBLAS column-major arguments:
// col-major C(m,n) = op(A) op(B) is row-major Cr(n,m) = op(B)^T op(A)^T, i.e. a
// row-major GEMM with M=n, N=m, K=k, left=B (transposed iff tb), right=A
// (transposed iff ta). bf16: A, B, C are bf16 (fp32 accumulation).
// Split-K (small outputs with long K, e.g. weight gradients) writes fp32
// partials to a per-thread workspace, then a reduce kernel applies alpha/beta.
static bool mpp_gemm(int bf16, int ta, int tb, int m, int n, int k, float alpha,
        const void* A, int lda, const void* B, int ldb, float beta, void* C, int ldc) {
    if (!g_gemm_lib || k <= 0) return false;
    const int M = n, N = m, K = k, TM = 64, TN = 32, BK = 128;
    long tiles = (long)((N + TN - 1) / TN) * ((M + TM - 1) / TM);
    int splits = 1;
    if (tiles < g_split_target && K >= 2 * g_split_min_kc) {
        long s = (g_split_target + tiles - 1) / tiles;
        long smax = K / g_split_min_kc;
        if (s > smax) s = smax;
        if (s > 64) s = 64;
        splits = (int)s;
    }
    int k_chunk = K;
    if (splits > 1) {
        k_chunk = ((K + splits - 1) / splits + BK - 1) / BK * BK;
        splits = (K + k_chunk - 1) / k_chunk;
    }
    const char* lay = tb ? (ta ? "tt" : "tn") : (ta ? "nt" : "nn");
    char name[64];
    snprintf(name, sizeof(name), "puf_gemm_%s_%s%s%s", lay, bf16 ? "bf16" : "f32",
        (splits > 1 && bf16) ? "_part" : "", k_chunk >= g_kloop_min ? "_k" : "");
    char label[128];
    if (g_gpu_prof) {
        snprintf(label, sizeof(label), "%s M=%d N=%d K=%d splits=%d", name, M, N, K, splits);
        tl_label = label;
    }
    void* out = C;
    if (splits > 1) {
        ThreadCtx* c = ctx();
        size_t need = (size_t)splits * M * N * sizeof(float);
        if (c->gemm_ws_bytes < need) {
            // Bump arena: the old workspace is simply abandoned (sizes settle quickly).
            c->gemm_ws = puf_metal_alloc(need);
            c->gemm_ws_bytes = need;
        }
        out = c->gemm_ws;
    }
    struct {
        uint64_t a, b, c;
        int M, N, K, lda, ldb, ldc;
        float alpha, beta;
        int splits, k_chunk;
    } args = {puf_metal_gpu_address(B), puf_metal_gpu_address(A), puf_metal_gpu_address(out),
        M, N, K, ldb, lda, ldc, alpha, beta, splits, k_chunk};
    puf_metal_dispatch(name, (N + TN - 1) / TN, (M + TM - 1) / TM, splits, 128, 1, 1,
        &args, sizeof(args));
    if (splits > 1) {
        struct {
            uint64_t w, c;
            int M, N, K, lda, ldb, ldc;
            float alpha, beta;
            int splits, k_chunk;
        } r = {args.c, puf_metal_gpu_address(C), M, N, K, ldb, lda, ldc, alpha, beta,
            splits, k_chunk};
        long mn = (long)M * N;
        if (g_gpu_prof) strncat(label, " reduce", sizeof(label) - strlen(label) - 1);
        puf_metal_dispatch(bf16 ? "puf_gemm_splitk_reduce_bf16" : "puf_gemm_splitk_reduce_f32",
            (uint32_t)((mn + 255) / 256), 1, 1, 256, 1, 1, &r, sizeof(r));
    }
    tl_label = nullptr;
    g_gemm_n++;
    return true;
}

extern "C" void puf_metal_gemm_bf16(int ta, int tb, int m, int n, int k,
        float alpha, const void* A, int lda, const void* B, int ldb,
        float beta, void* C, int ldc) {
    if (m <= 0 || n <= 0) return;
    if (!mpp_gemm(1, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)) {
        puf_metal_fatal("bf16 GEMM needs the MPP GEMM library (src/puf_gemm.metal, macOS 26+)");
    }
}

// Column-major X (rows x cols, ld) is row-major X^T (cols x rows, rowBytes ld*4).
extern "C" void puf_metal_sgemm(int ta, int tb, int m, int n, int k,
        float alpha, const float* A, int lda, const float* B, int ldb,
        float beta, float* C, int ldc) {
    if (m <= 0 || n <= 0) return;
    Chunk ca, cbk, cc;
    if (!find_chunk(A, &ca) || !find_chunk(B, &cbk) || !find_chunk(C, &cc)) {
        puf_metal_fatal("sgemm operand outside arena");
    }
    if (mpp_gemm(0, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)) return;
    // col-major C(m,n) = op(A) op(B)  <=>  row-major Cr(n,m) = op(B)^T op(A)^T
    if ((long)m * n * k <= g_small_gemm && puf_metal_has_kernel("gemm_tile")) {
        struct {
            uint64_t L, R, C;
            int M, N, K, ldl, ldr, ldc, transL, transR;
            float alpha, beta;
        } args = {puf_metal_gpu_address(B), puf_metal_gpu_address(A),
            puf_metal_gpu_address(C), n, m, k, ldb, lda, ldc, tb, ta, alpha, beta};
        puf_metal_dispatch("gemm_tile", (m + 31) / 32, (n + 31) / 32, 1, 128, 1, 1,
            &args, sizeof(args));
        g_gemm_n++;
        return;
    }
    ThreadCtx* c = ctx();
    end_encoders(c);
    // Left = B as row-major Br (colsB x rowsB); op(B)^T = tb ? Br^T : Br.
    int rowsB = tb ? n : k, colsB = tb ? k : n;
    int rowsA = ta ? k : m, colsA = ta ? m : k;
    MPSMatrix* ml = cached_matrix(c, cbk.buf, (const char*)B - cbk.base,
        colsB, rowsB, (size_t)ldb * 4);
    MPSMatrix* mr = cached_matrix(c, ca.buf, (const char*)A - ca.base,
        colsA, rowsA, (size_t)lda * 4);
    MPSMatrix* mc = cached_matrix(c, cc.buf, (const char*)C - cc.base,
        n, m, (size_t)ldc * 4);
    char key[128];
    snprintf(key, sizeof(key), "%d%d %d %d %d %a %a", tb, ta, n, m, k, alpha, beta);
    MPSMatrixMultiplication* mm;
    auto it = c->gemms.find(key);
    if (it == c->gemms.end()) {
        mm = [[MPSMatrixMultiplication alloc] initWithDevice:g_dev
            transposeLeft:tb ? YES : NO transposeRight:ta ? YES : NO
            resultRows:n resultColumns:m interiorColumns:k alpha:alpha beta:beta];
        c->gemms[key] = mm;
    } else {
        mm = it->second;
    }
    [mm encodeToCommandBuffer:c->cb leftMatrix:ml rightMatrix:mr resultMatrix:mc];
    g_gemm_n++;
    if (g_gpu_prof) gpu_prof_commit(c, "mps sgemm");
    else count_command(c);
}

static id<MTLComputePipelineState> pipeline(const char* name) {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = g_pipes.find(name);
    if (it != g_pipes.end()) return it->second;
    id<MTLComputePipelineState> p = nil;
    NSString* fname = [NSString stringWithUTF8String:name];
    id<MTLFunction> f = g_lib ? [g_lib newFunctionWithName:fname] : nil;
    if (!f && g_gemm_lib) f = [g_gemm_lib newFunctionWithName:fname];
    {
        if (f) {
            NSError* err = nil;
            p = [g_dev newComputePipelineStateWithFunction:f error:&err];
            if (!p) {
                fprintf(stderr, "PUF_TRAIN_METAL: pipeline %s: %s\n", name,
                    [[err description] UTF8String]);
                abort();
            }
        }
    }
    g_pipes[name] = p;
    return p;
}

extern "C" int puf_metal_has_kernel(const char* name) {
    puf_metal_init();
    return pipeline(name) != nil;
}

extern "C" void puf_metal_dispatch(const char* name, uint32_t gx, uint32_t gy,
        uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz,
        const void* args, size_t args_len) {
    id<MTLComputePipelineState> p = pipeline(name);
    if (!p) puf_metal_fatal("dispatch of missing kernel");
    if ((size_t)bx * by * bz > p.maxTotalThreadsPerThreadgroup) {
        puf_metal_fatal("threadgroup larger than pipeline maximum");
    }
    ThreadCtx* c = ctx();
    if (c->blit) {
        [c->blit endEncoding];
        c->blit = nil;
    }
    if (!c->compute) {
        c->compute = [c->cb computeCommandEncoder];
        std::lock_guard<std::mutex> g(g_mu);
        [c->compute useResources:g_res.data() count:g_res.size()
            usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    [c->compute setComputePipelineState:p];
    [c->compute setBytes:args length:args_len atIndex:0];
    [c->compute dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
        threadsPerThreadgroup:MTLSizeMake(bx, by, bz)];
    g_dispatch_n++;
    if (g_gpu_prof) gpu_prof_commit(c, tl_label ? tl_label : name);
    else count_command(c);
}

extern "C" void puf_metal_times(double* gpu_s, double* wait_s) {
    std::lock_guard<std::mutex> g(g_time_mu);
    *gpu_s = g_gpu_s;
    *wait_s = g_wait_s;
}

extern "C" void puf_metal_stats(long* flushes, long* gemms, long* dispatches, long* blits) {
    *flushes = g_flushes;
    *gemms = g_gemm_n;
    *dispatches = g_dispatch_n;
    *blits = g_blit_n;
}
