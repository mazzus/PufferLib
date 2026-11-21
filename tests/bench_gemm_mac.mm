// GEMM throughput on Apple Silicon at the trainer's real shapes:
// CPU (Accelerate cblas_sgemm, uses AMX/SME) vs GPU (MPSMatrixMultiplication,
// fp32 and fp16). Feeds the Phase 2 backend decision.
//
//   clang++ -std=c++17 -O2 -fobjc-arc tests/bench_gemm_mac.mm \
//       -framework Accelerate -framework Metal -framework MetalPerformanceShaders \
//       -framework Foundation -o build/bench_gemm_mac && ./build/bench_gemm_mac
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" void cblas_sgemm(int order, int transa, int transb, int m, int n,
    int k, float alpha, const float* a, int lda, const float* b, int ldb,
    float beta, float* c, int ldc);

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

struct Shape { const char* name; int M, N, K; };

// C(M,N) = A(M,K) @ B(N,K)^T  (puf_mm: activations x weight^T)
static double bench_cpu(Shape s, int iters) {
    float* A = (float*)malloc((size_t)s.M * s.K * 4);
    float* B = (float*)malloc((size_t)s.N * s.K * 4);
    float* C = (float*)malloc((size_t)s.M * s.N * 4);
    for (long i = 0; i < (long)s.M * s.K; i++) A[i] = (i % 7) * 0.01f;
    for (long i = 0; i < (long)s.N * s.K; i++) B[i] = (i % 5) * 0.01f;
    // row-major A @ B^T == col-major C^T = B @ A^T : (N x M) = (N x K)(K x M)
    cblas_sgemm(102, 112, 111, s.N, s.M, s.K, 1.0f, B, s.K, A, s.K, 0.0f, C, s.N);
    double t0 = now_s();
    for (int i = 0; i < iters; i++) {
        cblas_sgemm(102, 112, 111, s.N, s.M, s.K, 1.0f, B, s.K, A, s.K, 0.0f, C, s.N);
    }
    double dt = (now_s() - t0) / iters;
    free(A); free(B); free(C);
    return dt;
}

static double bench_gpu(id<MTLDevice> dev, id<MTLCommandQueue> q, Shape s,
        MPSDataType dt_type, int iters, int per_cb) {
    size_t es = dt_type == MPSDataTypeFloat32 ? 4 : 2;
    id<MTLBuffer> a = [dev newBufferWithLength:(size_t)s.M * s.K * es options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [dev newBufferWithLength:(size_t)s.N * s.K * es options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [dev newBufferWithLength:(size_t)s.M * s.N * es options:MTLResourceStorageModeShared];
    MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:s.M columns:s.K rowBytes:s.K * es dataType:dt_type];
    MPSMatrixDescriptor* db = [MPSMatrixDescriptor matrixDescriptorWithRows:s.N columns:s.K rowBytes:s.K * es dataType:dt_type];
    MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:s.M columns:s.N rowBytes:s.N * es dataType:dt_type];
    MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:a descriptor:da];
    MPSMatrix* mb = [[MPSMatrix alloc] initWithBuffer:b descriptor:db];
    MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:c descriptor:dc];
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:NO transposeRight:YES resultRows:s.M resultColumns:s.N
        interiorColumns:s.K alpha:1.0 beta:0.0];
    for (int w = 0; w < 3; w++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
        [cb commit];
        [cb waitUntilCompleted];
    }
    double t0 = now_s();
    for (int i = 0; i < iters; i += per_cb) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (int j = 0; j < per_cb; j++) {
            [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
        }
        [cb commit];
        [cb waitUntilCompleted];
    }
    return (now_s() - t0) / iters;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("GPU: %s\n", [[dev name] UTF8String]);
        Shape shapes[] = {
            {"breakout rollout mingru 4096x96x32", 4096, 96, 32},
            {"breakout train mingru 65536x96x32", 65536, 96, 32},
            {"pacman rollout mingru 8192x768x256", 8192, 768, 256},
            {"pacman train mingru 16384x768x256", 16384, 768, 256},
            {"tetris rollout mingru 4096x1536x512", 4096, 1536, 512},
            {"tetris train mingru 65536x1536x512", 65536, 1536, 512},
        };
        printf("%-40s %8s %8s | batched cb TF: %8s %8s %8s\n", "shape", "cpu_TF",
            "gpu32_TF", "fp32", "fp16", "bf16");
        for (Shape s : shapes) {
            double flop = 2.0 * s.M * s.N * s.K;
            int iters = flop > 5e10 ? 10 : (flop > 1e9 ? 50 : 500);
            int pcb = iters < 50 ? iters : 50;
            double tc = bench_cpu(s, iters);
            double tg = bench_gpu(dev, q, s, MPSDataTypeFloat32, iters, 1);
            // Many GEMMs in one command buffer: the trainer's steady state
            double b32 = bench_gpu(dev, q, s, MPSDataTypeFloat32, iters, pcb);
            double b16 = bench_gpu(dev, q, s, MPSDataTypeFloat16, iters, pcb);
            double bbf = 1e9;  // MPSMatrix has no bf16
            printf("%-40s %8.2f %8.2f | %22.2f %8.2f %8.2f\n", s.name,
                flop / tc / 1e12, flop / tg / 1e12, flop / b32 / 1e12,
                flop / b16 / 1e12, flop / bbf / 1e12);
        }
    }
    return 0;
}
