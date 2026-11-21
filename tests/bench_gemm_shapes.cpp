// Time of the Metal backend's GEMM path (puf_metal_gemm_bf16 /
// puf_metal_sgemm, i.e. the MPP kernels with their split-K logic) at given
// cuBLAS-style shapes. Shapes are read from stdin, one per line, in the form
// PUF_CPU_PROFILE prints them, e.g. "TN m=1536 n=65536 k=512", optionally
// followed by a call count used to weight the total.
//
//   clang++ -std=c++17 -O2 -fobjc-arc -x objective-c++ src/puf_metal.mm -x c++ \
//       tests/bench_gemm_shapes.cpp -framework Metal -framework MetalPerformanceShaders \
//       -framework Foundation -framework QuartzCore -o build/bench_gemm_shapes
//   ./build/bench_gemm_shapes [--f32] < shapes.txt
// Env: PUF_METAL_GEMM_KERNELS=alt.metal to A/B a kernel variant; the
// PUF_GEMM_* knobs of src/puf_metal.mm apply.
#include "../src/puf_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    bool f32 = argc > 1 && strcmp(argv[1], "--f32") == 0;
    size_t es = f32 ? 4 : 2;
    char line[256];
    double total = 0, total_flop = 0;
    printf("%-28s %6s %10s %8s %10s\n", "shape", "calls", "us/call", "TFLOPS", "weighted ms");
    while (fgets(line, sizeof(line), stdin)) {
        char ta, tb;
        int m, n, k;
        long calls = 1;
        if (sscanf(line, " %c%c m=%d n=%d k=%d %ld", &ta, &tb, &m, &n, &k, &calls) < 5) continue;
        int TA = ta == 'T', TB = tb == 'T';
        int lda = TA ? k : m, ldb = TB ? n : k, ldc = m;
        void* A = puf_metal_alloc((size_t)m * k * es);
        void* B = puf_metal_alloc((size_t)k * n * es);
        void* C = puf_metal_alloc((size_t)m * n * es);
        // Fill with small values (bit patterns of 0x3c00 bf16 ~ 0.0078; fp32 ~ tiny)
        puf_metal_fill(A, 0x3c, (size_t)m * k * es);
        puf_metal_fill(B, 0x3c, (size_t)k * n * es);
        puf_metal_flush();
        double flop = 2.0 * m * n * k;
        int reps = (int)(2e10 / flop);
        if (reps < 3) reps = 3;
        if (reps > 2000) reps = 2000;
        auto run = [&](int r) {
            for (int i = 0; i < r; i++) {
                if (f32) {
                    puf_metal_sgemm(TA, TB, m, n, k, 1.0f, (const float*)A, lda,
                        (const float*)B, ldb, 0.0f, (float*)C, ldc);
                } else {
                    puf_metal_gemm_bf16(TA, TB, m, n, k, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
                }
            }
            puf_metal_flush();
        };
        run(2);  // pipeline compile, workspace
        double best = 1e30;
        for (int t = 0; t < 3; t++) {
            // Wall time over many back-to-back GEMMs: summed per-command-buffer
            // GPU spans overcount when consecutive buffers overlap on the GPU.
            double t0 = wall_now();
            run(reps);
            double us = (wall_now() - t0) / reps * 1e6;
            if (us < best) best = us;
        }
        double tf = flop / (best * 1e-6) / 1e12;
        total += best * calls / 1e3;
        total_flop += flop * calls;
        printf("%c%c m=%-6d n=%-6d k=%-6d %6ld %10.1f %8.2f %10.2f\n", ta, tb, m, n, k, calls,
            best, tf, best * calls / 1e3);
    }
    printf("total weighted time %.1f ms (%.2f TFLOPS overall)\n", total,
        total_flop / (total * 1e-3) / 1e12);
    return 0;
}
