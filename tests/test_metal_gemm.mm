// Tests and benchmarks src/puf_gemm.metal (MPP matmul2d GEMMs) standalone:
//   1. correctness vs a double reference for nt/tn/nn, fp32 and bf16, with
//      alpha/beta and ragged sizes, run twice (checks overwrite semantics)
//   2. speed vs MPSMatrixMultiplication fp32 at the trainer's shapes
//
//   clang++ -std=c++17 -O2 -fobjc-arc tests/test_metal_gemm.mm -framework Metal \
//       -framework MetalPerformanceShaders -framework Foundation \
//       -o build/test_metal_gemm && ./build/test_metal_gemm [--bench]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

struct PufGemmArgs { int M, N, K; float alpha, beta; int splits, k_chunk; };

// Split-K policy (mirrors puf_metal.mm): enough threadgroups to fill the GPU,
// each split at least MIN_KC deep. FORCE_SPLITS overrides for tests.
static int g_force_splits = 0;
static int choose_splits(int M, int N, int K) {
    if (g_force_splits) return g_force_splits;
    long tiles = (long)((N + 31) / 32) * ((M + 63) / 64);
    static long target = getenv("SPLIT_TARGET") ? atol(getenv("SPLIT_TARGET")) : 320;
    static int min_kc = getenv("MIN_KC") ? atoi(getenv("MIN_KC")) : 512;
    if (tiles >= target || K < 2 * min_kc) return 1;
    long s = (target + tiles - 1) / tiles;
    long smax = K / min_kc;
    if (s > smax) s = smax;
    if (s > 64) s = 64;
    return (int)(s < 1 ? 1 : s);
}

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLLibrary> lib;

static id<MTLComputePipelineState> pipe(const char* name) {
    NSError* e = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!f) { printf("missing %s\n", name); exit(1); }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&e];
    if (!ps) { printf("pipeline %s: %s\n", name, [[e description] UTF8String]); exit(1); }
    return ps;
}

static id<MTLBuffer> g_ws;
static bool g_bench_bf16 = false;
static id<MTLComputePipelineState> pipe(const char* name);
static int g_kloop_min = getenv("KLOOP_MIN") ? atoi(getenv("KLOOP_MIN")) : 1024;
static id<MTLComputePipelineState> kpipe(int layout, bool bf16, bool part, bool kloop) {
    static NSMutableDictionary* cache = [NSMutableDictionary new];
    const char* lay[3] = {"nt", "tn", "nn"};
    NSString* name = [NSString stringWithFormat:@"puf_gemm_%s_%s%s%s", lay[layout],
        bf16 ? "bf16" : "f32", (part && bf16) ? "_part" : "", kloop ? "_k" : ""];
    id<MTLComputePipelineState> ps = cache[name];
    if (!ps) { ps = pipe([name UTF8String]); cache[name] = ps; }
    return ps;
}
static void encode(id<MTLCommandBuffer> cb, int layout, bool bf16,
        id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c, PufGemmArgs p) {
    p.splits = choose_splits(p.M, p.N, p.K);
    int bk = 128;
    p.k_chunk = p.splits > 1 ? ((p.K + p.splits - 1) / p.splits + bk - 1) / bk * bk : p.K;
    if (p.splits > 1) p.splits = (p.K + p.k_chunk - 1) / p.k_chunk;
    id<MTLComputePipelineState> ps = kpipe(layout, bf16, p.splits > 1, p.k_chunk >= g_kloop_min);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    size_t need = (size_t)p.splits * p.M * p.N * 4;
    if (p.splits > 1 && (!g_ws || g_ws.length < need))
        g_ws = [dev newBufferWithLength:need options:MTLResourceStorageModePrivate];
    [enc setBuffer:p.splits > 1 ? g_ws : c offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((p.N + 31) / 32, (p.M + 63) / 64, p.splits)
        threadsPerThreadgroup:MTLSizeMake(ps.threadExecutionWidth * 4, 1, 1)];
    if (p.splits > 1) {
        [enc setComputePipelineState:pipe(bf16 ? "puf_gemm_splitk_reduce_bf16" : "puf_gemm_splitk_reduce_f32")];
        [enc setBuffer:g_ws offset:0 atIndex:0];
        [enc setBuffer:c offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        long mn = (long)p.M * p.N;
        [enc dispatchThreads:MTLSizeMake(mn, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc endEncoding];
}

// layout: 0 nt, 1 tn, 2 nn. Element (i,k) of op(A) and (k,j) of op(B).
static float opA(const float* A, int layout, int M, int K, int i, int k) {
    return layout == 1 ? A[(long)k * M + i] : A[(long)i * K + k];
}
static float opB(const float* B, int layout, int N, int K, int k, int j) {
    return layout == 0 ? B[(long)j * K + k] : B[(long)k * N + j];
}

static int check(int layout, bool bf16, int M, int N, int K, float alpha, float beta) {
    const char* names[2][3] = {{"puf_gemm_nt_f32", "puf_gemm_tn_f32", "puf_gemm_nn_f32"},
                               {"puf_gemm_nt_bf16", "puf_gemm_tn_bf16", "puf_gemm_nn_bf16"}};
    size_t es = bf16 ? 2 : 4;
    long na = (long)M * K, nb = (long)N * K, nc = (long)M * N;
    float* A = (float*)malloc(na * 4); float* B = (float*)malloc(nb * 4); float* C0 = (float*)malloc(nc * 4);
    for (long i = 0; i < na; i++) A[i] = sinf(i * 0.37f + 0.1f);
    for (long i = 0; i < nb; i++) B[i] = cosf(i * 0.11f) * 0.1f;
    for (long i = 0; i < nc; i++) C0[i] = sinf(i * 0.05f);
    if (bf16) {
        for (long i = 0; i < na; i++) A[i] = bf2f(f2bf(A[i]));
        for (long i = 0; i < nb; i++) B[i] = bf2f(f2bf(B[i]));
        for (long i = 0; i < nc; i++) C0[i] = bf2f(f2bf(C0[i]));
    }
    id<MTLBuffer> a = [dev newBufferWithLength:na * es options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [dev newBufferWithLength:nb * es options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [dev newBufferWithLength:nc * es options:MTLResourceStorageModeShared];
    auto put = [&](id<MTLBuffer> buf, const float* src, long n) {
        if (bf16) { uint16_t* d = (uint16_t*)buf.contents; for (long i = 0; i < n; i++) d[i] = f2bf(src[i]); }
        else memcpy(buf.contents, src, n * 4);
    };
    auto get = [&](long i) -> double {
        return bf16 ? bf2f(((uint16_t*)c.contents)[i]) : ((float*)c.contents)[i];
    };
    put(a, A, na); put(b, B, nb);
    int bad = 0;
    for (int run = 0; run < 2; run++) {
        put(c, C0, nc);  // beta reads C; reset so both runs compute the same thing
        id<MTLCommandBuffer> cb = [q commandBuffer];
        encode(cb, layout, bf16, a, b, c, {M, N, K, alpha, beta, 1, K});
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { printf("error: %s\n", [[cb.error description] UTF8String]); return 1; }
        double maxrel = 0;
        for (long i = 0; i < M; i++) for (long j = 0; j < N; j++) {
            double ref = 0, mag = 0;
            for (int k = 0; k < K; k++) {
                double t = (double)opA(A, layout, M, K, i, k) * opB(B, layout, N, K, k, j);
                ref += t; mag += fabs(t);
            }
            ref = alpha * ref + beta * C0[i * N + j];
            mag = fabs(alpha) * mag + fabs(beta * C0[i * N + j]);
            double rel = fabs(get(i * N + j) - ref) / (mag + 1e-30);
            if (rel > maxrel) maxrel = rel;
        }
        double tol = bf16 ? 8e-3 : 1e-6;
        bool ok = maxrel < tol;
        if (!ok) bad++;
        if (run == 1 || !ok)
            printf("%s %-17s M=%4d N=%4d K=%4d a=%.2f b=%.2f splits=%d run%d maxrel %.1e\n", ok ? "PASS" : "FAIL",
                names[bf16][layout], M, N, K, alpha, beta, choose_splits(M, N, K), run, maxrel);
    }
    free(A); free(B); free(C0);
    return bad;
}

struct Shape { const char* name; int layout, M, N, K; };

static void bench(Shape s) {
    const char* names[3] = {"puf_gemm_nt_f32", "puf_gemm_tn_f32", "puf_gemm_nn_f32"};
    long na = (long)s.M * s.K, nb = (long)s.N * s.K, nc = (long)s.M * s.N;
    id<MTLBuffer> a = [dev newBufferWithLength:na * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [dev newBufferWithLength:nb * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [dev newBufferWithLength:nc * 4 options:MTLResourceStorageModeShared];
    // MPS: row-major mapping, transposeLeft for tn, transposeRight for nt
    int ar = s.layout == 1 ? s.K : s.M, ac = s.layout == 1 ? s.M : s.K;
    int br = s.layout == 0 ? s.N : s.K, bc = s.layout == 0 ? s.K : s.N;
    MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:a descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:ar columns:ac rowBytes:ac * 4 dataType:MPSDataTypeFloat32]];
    MPSMatrix* mb = [[MPSMatrix alloc] initWithBuffer:b descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:br columns:bc rowBytes:bc * 4 dataType:MPSDataTypeFloat32]];
    MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:c descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:s.M columns:s.N rowBytes:s.N * 4 dataType:MPSDataTypeFloat32]];
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:s.layout == 1 transposeRight:s.layout == 0 resultRows:s.M resultColumns:s.N
        interiorColumns:s.K alpha:1.0 beta:0.0];
    double flop = 2.0 * s.M * s.N * s.K;
    int iters = flop > 5e10 ? 20 : (flop > 1e9 ? 100 : 400);
    int pcb = iters < 50 ? iters : 50;
    double t_mps = 0, t_mpp = 0, e_mps = 0, e_mpp = 0;
    for (int rep = 0; rep < 2; rep++) {
        t_mps = 0; t_mpp = 0; e_mps = 0; e_mpp = 0;
        for (int i = 0; i < iters; i += pcb) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            double t0 = now_s();
            for (int j = 0; j < pcb; j++) [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
            e_mps += now_s() - t0;
            [cb commit]; [cb waitUntilCompleted];
            t_mps += now_s() - t0;
            cb = [q commandBuffer];
            t0 = now_s();
            for (int j = 0; j < pcb; j++) encode(cb, s.layout, g_bench_bf16, a, b, c, {s.M, s.N, s.K, 1.0f, 0.0f, 1, s.K});
            e_mpp += now_s() - t0;
            [cb commit]; [cb waitUntilCompleted];
            t_mpp += now_s() - t0;
        }
    }
    printf("%-34s %s s%-2d MPS %6.2f TF (enc %5.1f us)  MPP %6.2f TF (enc %4.1f us)  %5.2fx\n", s.name,
        names[s.layout] + 9, choose_splits(s.M, s.N, s.K), flop / (t_mps / iters) / 1e12, e_mps / iters * 1e6,
        flop / (t_mpp / iters) / 1e12, e_mpp / iters * 1e6, t_mps / t_mpp);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        NSError* e = nil;
        NSString* src = [NSString stringWithContentsOfFile:@"src/puf_gemm.metal" encoding:NSUTF8StringEncoding error:&e];
        if (!src) { printf("run from the repo root\n"); return 1; }
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.languageVersion = MTLLanguageVersion4_0;
        lib = [dev newLibraryWithSource:src options:opt error:&e];
        if (!lib) { printf("compile: %s\n", [[e description] UTF8String]); return 1; }
        int bad = 0;
        int sizes[][3] = {{64, 32, 16}, {100, 37, 29}, {1, 7, 300}, {130, 70, 1}, {257, 129, 65}};
        for (int bf = 0; bf < 2; bf++)
            for (int layout = 0; layout < 3; layout++)
                for (auto& sz : sizes) {
                    bad += check(layout, bf, sz[0], sz[1], sz[2], 1.0f, 0.0f);
                    bad += check(layout, bf, sz[0], sz[1], sz[2], -0.7f, 0.3f);
                }
        int ksizes[][3] = {{96, 32, 3000}, {70, 33, 1100}, {64, 64, 4096}};
        for (int fs : {2, 3, 7})
            for (int bf = 0; bf < 2; bf++)
                for (int layout = 0; layout < 3; layout++)
                    for (auto& sz : ksizes) {
                        g_force_splits = fs;
                        bad += check(layout, bf, sz[0], sz[1], sz[2], -0.7f, 0.3f);
                        g_force_splits = 0;
                    }
        printf("%s (%d failures)\n", bad ? "FAILED" : "ALL PASSED", bad);
        if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
            Shape shapes[] = {
                {"breakout fwd 65536x96x32", 0, 65536, 96, 32},
                {"breakout dW 96x32x65536", 1, 96, 32, 65536},
                {"breakout dX 65536x32x96", 2, 65536, 32, 96},
                {"pacman rollout 8192x768x256", 0, 8192, 768, 256},
                {"pacman fwd 16384x768x256", 0, 16384, 768, 256},
                {"pacman dW 768x256x16384", 1, 768, 256, 16384},
                {"pacman dX 16384x256x768", 2, 16384, 256, 768},
                {"muon gram 256x256x768", 0, 256, 256, 768},
                {"muon nn 256x768x256", 2, 256, 768, 256},
                {"tetris fwd 65536x1536x512", 0, 65536, 1536, 512},
                {"tetris dW 1536x512x65536", 1, 1536, 512, 65536},
                {"tetris dX 65536x512x1536", 2, 65536, 512, 1536},
            };
            for (Shape s : shapes) bench(s);
        }
        return bad != 0;
    }
}
