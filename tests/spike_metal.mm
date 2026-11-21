// Metal backend feasibility spike. Answers, by test:
//  1. Can a kernel dereference GPU addresses stored inside a by-value struct
//     (setBytes), with all memory sub-allocated from one shared arena buffer?
//  2. Is maxBufferLength large enough for one arena?
//  3. Are MPS GEMMs on arena offsets correct for the three puf_mm* forms
//     (incl. alpha/beta and the same matrix as both operands, as Muon does)?
//  4. Commit+wait latency of one rollout-step-shaped command buffer.
//  5. Two command queues driven from two threads on the same arena.
//
//   clang++ -std=c++17 -O2 -fobjc-arc tests/spike_metal.mm -framework Metal \
//     -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
//     -o build/spike_metal && ./build/spike_metal
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <time.h>

extern "C" void cblas_sgemm(int order, int transa, int transb, int m, int n,
    int k, float alpha, const float* a, int lda, const float* b, int ldb,
    float beta, float* c, int ldc);

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
    else { printf("ok:   "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Prec { device float* data; long shape[8]; };
struct Two { Prec a; Prec b; float scale; int n; };

// 1: pointers inside a by-value (constant) struct
kernel void axpy_struct(constant Two& t [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if ((int)i < t.n) t.b.data[i] = t.b.data[i] + t.scale * t.a.data[i];
}
// small elementwise kernel used for latency
kernel void gate_like(device float* out [[buffer(0)]], constant int& n [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {
    if ((int)i < n) { float x = out[i]; out[i] = 1.0f / (1.0f + exp(-x)) * x; }
}
)MSL";

struct Arena {
    id<MTLBuffer> buf;
    char* base;
    size_t used, cap;
    void* alloc(size_t n) {
        size_t off = (used + 255) & ~(size_t)255;
        if (off + n > cap) { fprintf(stderr, "arena full\n"); abort(); }
        used = off + n;
        return base + off;
    }
    uint64_t gpu(const void* p) { return buf.gpuAddress + (uint64_t)((const char*)p - base); }
    size_t off(const void* p) { return (size_t)((const char*)p - base); }
};

struct PrecHost { uint64_t data; int64_t shape[8]; };
struct TwoHost { PrecHost a; PrecHost b; float scale; int n; };

static void fill(float* p, long n, unsigned s) {
    for (long i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; p[i] = (float)(s >> 8) / 16777216.0f - 0.5f; }
}

// C(MxN) = alpha * opA(A) @ opB(B) + beta*C in row-major, via MPS, on arena offsets
static void mps_gemm(id<MTLDevice> dev, id<MTLCommandBuffer> cb, Arena& ar,
        bool ta, bool tb, int M, int N, int K, const float* A, const float* B, float* C,
        float alpha, float beta) {
    int ar_ = ta ? K : M, ac = ta ? M : K;
    int br = tb ? N : K, bc = tb ? K : N;
    MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:ar_ columns:ac rowBytes:ac * 4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* db = [MPSMatrixDescriptor matrixDescriptorWithRows:br columns:bc rowBytes:bc * 4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * 4 dataType:MPSDataTypeFloat32];
    MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:ar.buf offset:ar.off(A) descriptor:da];
    MPSMatrix* mb = [[MPSMatrix alloc] initWithBuffer:ar.buf offset:ar.off(B) descriptor:db];
    MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:ar.buf offset:ar.off(C) descriptor:dc];
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:ta transposeRight:tb resultRows:M resultColumns:N interiorColumns:K
        alpha:alpha beta:beta];
    [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
}

static void cpu_gemm(bool ta, bool tb, int M, int N, int K, const float* A, const float* B,
        float* C, float alpha, float beta) {
    // row-major C = op(A) op(B)  ==  col-major C^T = op(B)^T op(A)^T
    int lda = ta ? M : K, ldb = tb ? K : N;
    cblas_sgemm(102, tb ? 112 : 111, ta ? 112 : 111, N, M, K, alpha, B, ldb, A, lda, beta, C, N);
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s, maxBufferLength=%.1f GB, recommendedMaxWorkingSet=%.1f GB, argBufTier=%d\n",
            [[dev name] UTF8String], dev.maxBufferLength / 1e9,
            dev.recommendedMaxWorkingSetSize / 1e9, (int)dev.argumentBuffersSupport);
        CHECK(dev.maxBufferLength >= (size_t)4 << 30, "maxBufferLength >= 4 GB");
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&err];
        if (!lib) { printf("compile error: %s\n", [[err description] UTF8String]); return 1; }
        id<MTLComputePipelineState> p_axpy = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"axpy_struct"] error:&err];
        id<MTLComputePipelineState> p_gate = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gate_like"] error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];

        Arena ar;
        ar.cap = (size_t)1 << 30;
        ar.buf = [dev newBufferWithLength:ar.cap options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        ar.base = (char*)ar.buf.contents;
        ar.used = 0;

        // 1. struct-embedded GPU pointers
        {
            int n = 100003;
            float* a = (float*)ar.alloc(n * 4);
            float* b = (float*)ar.alloc(n * 4);
            fill(a, n, 1); fill(b, n, 2);
            float* ref = (float*)malloc(n * 4);
            for (int i = 0; i < n; i++) ref[i] = b[i] + 0.5f * a[i];
            TwoHost t = {};
            t.a.data = ar.gpu(a); t.a.shape[0] = n;
            t.b.data = ar.gpu(b); t.b.shape[0] = n;
            t.scale = 0.5f; t.n = n;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:p_axpy];
            [enc useResource:ar.buf usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [enc setBytes:&t length:sizeof(t) atIndex:0];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double md = 0;
            for (int i = 0; i < n; i++) md = fmax(md, fabs(ref[i] - b[i]));
            CHECK(md < 1e-6, "GPU pointers in by-value struct (setBytes) deref, maxdiff=%g", md);
        }

        // 3. MPS GEMM forms on arena offsets vs cblas
        {
            struct G { bool ta, tb; int M, N, K; float alpha, beta; bool same; const char* name; } gs[] = {
                {false, true, 130, 96, 258, 1.0f, 0.0f, false, "puf_mm    (A @ B^T)"},
                {true, false, 96, 33, 517, 0.75f, 0.5f, false, "puf_mm_tn (A^T @ B)"},
                {false, false, 64, 300, 7, -1.25f, 0.5f, false, "puf_mm_nn (A @ B)"},
                {false, true, 96, 96, 288, 1.0f, 0.0f, true, "muon gram  X @ X^T (same buffer)"},
                {true, false, 64, 64, 300, 1.0f, 0.0f, true, "muon gram  X^T @ X (same buffer)"},
            };
            for (G g : gs) {
                long na = (long)g.M * g.K, nb = (long)g.N * g.K, nc = (long)g.M * g.N;
                if (g.same) nb = na;
                float* A = (float*)ar.alloc(na * 4);
                float* B = g.same ? A : (float*)ar.alloc(nb * 4);
                float* C = (float*)ar.alloc(nc * 4);
                fill(A, na, 3); if (!g.same) fill(B, nb, 4); fill(C, nc, 5);
                float* R = (float*)malloc(nc * 4);
                memcpy(R, C, nc * 4);
                cpu_gemm(g.ta, g.tb, g.M, g.N, g.K, A, B, R, g.alpha, g.beta);
                id<MTLCommandBuffer> cb = [q commandBuffer];
                mps_gemm(dev, cb, ar, g.ta, g.tb, g.M, g.N, g.K, A, B, C, g.alpha, g.beta);
                [cb commit];
                [cb waitUntilCompleted];
                double md = 0, mx = 0;
                for (long i = 0; i < nc; i++) { md = fmax(md, fabs(R[i] - C[i])); mx = fmax(mx, fabs(R[i])); }
                CHECK(md / mx < 1e-5, "MPS %s M=%d N=%d K=%d rel maxdiff=%g", g.name, g.M, g.N, g.K, md / mx);
                free(R);
            }
        }

        // 4. rollout-step latency: 6 GEMMs + 10 small dispatches in one command buffer
        struct Lat { const char* name; int B, H, L, obs; } lats[] = {
            {"breakout-like B=4096 H=32 L=4", 4096, 32, 4, 118},
            {"pacman-like   B=2048 H=256 L=6", 2048, 256, 6, 291},
            {"pacman-like   B=8192 H=256 L=6", 8192, 256, 6, 291},
        };
        for (Lat l : lats) {
            float* x = (float*)ar.alloc((long)l.B * l.obs * 4);
            float* we = (float*)ar.alloc((long)l.H * l.obs * 4);
            float* h = (float*)ar.alloc((long)l.B * l.H * 4);
            float* wl = (float*)ar.alloc((long)3 * l.H * l.H * 4);
            float* comb = (float*)ar.alloc((long)l.B * 3 * l.H * 4);
            fill(x, (long)l.B * l.obs, 6); fill(we, (long)l.H * l.obs, 7); fill(wl, 3L * l.H * l.H, 8);
            int n = l.B * l.H;
            double best = 1e9, sum = 0;
            int reps = 50;
            for (int r = 0; r < reps + 5; r++) {
                double t0 = now_s();
                id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
                mps_gemm(dev, cb, ar, false, true, l.B, l.H, l.obs, x, we, h, 1, 0);
                for (int i = 0; i < l.L; i++) {
                    mps_gemm(dev, cb, ar, false, true, l.B, 3 * l.H, l.H, h, wl, comb, 1, 0);
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:p_gate];
                    [enc setBuffer:ar.buf offset:ar.off(h) atIndex:0];
                    [enc setBytes:&n length:4 atIndex:1];
                    for (int k = 0; k < 2; k++)
                        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
                [cb commit];
                [cb waitUntilCompleted];
                double dt = now_s() - t0;
                if (r >= 5) { best = fmin(best, dt); sum += dt; }
            }
            printf("info: step latency %s: mean %.1f us, best %.1f us (encode+commit+wait)\n",
                l.name, sum / reps * 1e6, best * 1e6);
        }

        // 5. two queues from two threads on the same arena
        {
            int n = 1 << 20;
            float* a = (float*)ar.alloc(n * 4);
            float* b = (float*)ar.alloc(n * 4);
            float* c = (float*)ar.alloc(n * 4);
            float* d = (float*)ar.alloc(n * 4);
            fill(a, n, 9); fill(c, n, 10);
            memset(b, 0, n * 4); memset(d, 0, n * 4);
            auto work = [&](float* src, float* dst, int iters) {
                id<MTLCommandQueue> q2 = [dev newCommandQueue];
                for (int it = 0; it < iters; it++) {
                    TwoHost t = {};
                    t.a.data = ar.gpu(src); t.b.data = ar.gpu(dst); t.scale = 1.0f; t.n = n;
                    id<MTLCommandBuffer> cb = [q2 commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:p_axpy];
                    [enc useResource:ar.buf usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                    [enc setBytes:&t length:sizeof(t) atIndex:0];
                    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }
            };
            std::thread t1([&] { @autoreleasepool { work(a, b, 200); } });
            std::thread t2([&] { @autoreleasepool { work(c, d, 200); } });
            t1.join(); t2.join();
            double md = 0;
            for (int i = 0; i < n; i++) {
                md = fmax(md, fabs(b[i] - 200.0f * a[i]));
                md = fmax(md, fabs(d[i] - 200.0f * c[i]));
            }
            CHECK(md < 1e-3, "two queues / two threads on one arena, maxdiff=%g", md);
        }
        printf(failures ? "SPIKE: %d FAILED\n" : "SPIKE: all checks passed\n", failures);
    }
    return failures != 0;
}
