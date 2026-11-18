// Metal backend implementation. See src/puf_metal.h for the model.
// Objective-C++ (ARC). Built and linked by build.sh --metal.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <QuartzCore/QuartzCore.h>

#include <atomic>
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
static void commit_nowait(ThreadCtx* c);
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
        fprintf(stderr, "PUF_TRAIN_METAL: no kernel library at %s; all kernels run on CPU\n", path);
        return;
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
        if (g_commit_every < 1) g_commit_every = 1;
        load_library();
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
    if (!buf) puf_metal_fatal("arena chunk allocation failed");
    Chunk c = {buf, (char*)buf.contents, cap, n};
    g_chunks.push_back(c);
    g_res.push_back(buf);
    memset(c.base, 0, n);
    return c.base;
}

// Bump allocator: frees are ignored (the trainer allocates up front).
extern "C" void puf_metal_free(void*) {}

static const Chunk* find_chunk(const void* p) {
    const char* q = (const char*)p;
    std::lock_guard<std::mutex> g(g_mu);
    for (const Chunk& c : g_chunks) {
        if (q >= c.base && q < c.base + c.cap) return &c;
    }
    return nullptr;
}

extern "C" int puf_metal_owns(const void* p) {
    return p != nullptr && find_chunk(p) != nullptr;
}

extern "C" uint64_t puf_metal_gpu_address(const void* p) {
    if (!p) return 0;
    const Chunk* c = find_chunk(p);
    if (!c) puf_metal_fatal("pointer is not in the Metal arena");
    return c->buf.gpuAddress + (uint64_t)((const char*)p - c->base);
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
    const Chunk* cd = find_chunk(dst);
    const Chunk* cs = find_chunk(src);
    if (!cd || !cs) puf_metal_fatal("puf_metal_copy outside arena");
    ThreadCtx* c = ctx();
    [blit_encoder(c) copyFromBuffer:cs->buf sourceOffset:(const char*)src - cs->base
        toBuffer:cd->buf destinationOffset:(char*)dst - cd->base size:bytes];
    g_blit_n++;
    count_command(c);
}

extern "C" void puf_metal_fill(void* dst, int value, size_t bytes) {
    if (!bytes) return;
    const Chunk* cd = find_chunk(dst);
    if (!cd) puf_metal_fatal("puf_metal_fill outside arena");
    ThreadCtx* c = ctx();
    size_t off = (char*)dst - cd->base;
    [blit_encoder(c) fillBuffer:cd->buf range:NSMakeRange(off, bytes) value:(uint8_t)value];
    g_blit_n++;
    count_command(c);
}

// Column-major X (rows x cols, ld) is row-major X^T (cols x rows, rowBytes ld*4).
extern "C" void puf_metal_sgemm(int ta, int tb, int m, int n, int k,
        float alpha, const float* A, int lda, const float* B, int ldb,
        float beta, float* C, int ldc) {
    if (m <= 0 || n <= 0) return;
    const Chunk* ca = find_chunk(A);
    const Chunk* cbk = find_chunk(B);
    const Chunk* cc = find_chunk(C);
    if (!ca || !cbk || !cc) puf_metal_fatal("sgemm operand outside arena");
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
    MPSMatrix* ml = cached_matrix(c, cbk->buf, (const char*)B - cbk->base,
        colsB, rowsB, (size_t)ldb * 4);
    MPSMatrix* mr = cached_matrix(c, ca->buf, (const char*)A - ca->base,
        colsA, rowsA, (size_t)lda * 4);
    MPSMatrix* mc = cached_matrix(c, cc->buf, (const char*)C - cc->base,
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
    count_command(c);
}

static id<MTLComputePipelineState> pipeline(const char* name) {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = g_pipes.find(name);
    if (it != g_pipes.end()) return it->second;
    id<MTLComputePipelineState> p = nil;
    if (g_lib) {
        id<MTLFunction> f = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]];
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
    count_command(c);
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
