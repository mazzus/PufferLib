// bf16 GEMM options on Apple Silicon at the trainer's shapes, vs fp32 MPS.
// MPSMatrixMultiplication has no bf16, so this measures:
//   - MetalPerformancePrimitives matmul2d (MSL 4, tensor ops), bf16 x bf16 ->
//     float or bf16, and the same kernel with fp32 / half operands
//   - MPSGraph matmul with bf16 tensors (executable, encoded into a command buffer)
// All GEMMs are C(M,N) = A(M,K) @ B(N,K)^T, row major (the trainer's puf_mm).
//
//   clang++ -std=c++17 -O2 -fobjc-arc tests/bench_gemm_bf16.mm -framework Metal \
//       -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//       -framework Foundation -o build/bench_gemm_bf16 && ./build/bench_gemm_bf16
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
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

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename TA, typename TC, int TM, int TN>
inline void mm_nt(device TA* a, device TA* b, device TC* c, int M, int N, int K,
        uint2 tgid) {
    tensor<device TA, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
    tensor<device TA, dextents<int32_t, 2>, tensor_inline> B(b, dextents<int32_t, 2>(K, N));
    tensor<device TC, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
        false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tgid.y * TM);
    auto mB = B.slice(0, tgid.x * TN);
    auto mC = C.slice(tgid.x * TN, tgid.y * TM);
    op.run(mA, mB, mC);
}

#define MM(name, TA, TC, TM, TN) \
kernel void name(device TA* a [[buffer(0)]], device TA* b [[buffer(1)]], \
        device TC* c [[buffer(2)]], constant int3& mnk [[buffer(3)]], \
        uint2 tgid [[threadgroup_position_in_grid]]) { \
    mm_nt<TA, TC, TM, TN>(a, b, c, mnk.x, mnk.y, mnk.z, tgid); \
}
MM(mm_bf_f_64x32, bfloat, float, 64, 32)
MM(mm_bf_bf_64x32, bfloat, bfloat, 64, 32)
MM(mm_bf_f_64x64, bfloat, float, 64, 64)
MM(mm_bf_bf_64x64, bfloat, bfloat, 64, 64)
MM(mm_bf_bf_128x64, bfloat, bfloat, 128, 64)
MM(mm_h_h_64x64, half, half, 64, 64)
MM(mm_f_f_64x64, float, float, 64, 64)
)MSL";

struct Shape { const char* name; int M, N, K; };

static id<MTLBuffer> buf(id<MTLDevice> dev, size_t n) {
    return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
}

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

// Fill A, B with bf16 values; also return fp32 copies for a reference check.
static void fill(float* af, float* bf, uint16_t* ab, uint16_t* bb, Shape s) {
    for (long i = 0; i < (long)s.M * s.K; i++) { float v = sinf(i * 0.37f) * 0.5f; ab[i] = f2bf(v); af[i] = bf2f(ab[i]); }
    for (long i = 0; i < (long)s.N * s.K; i++) { float v = cosf(i * 0.11f) * 0.05f; bb[i] = f2bf(v); bf[i] = bf2f(bb[i]); }
}

static double run_mps32(id<MTLDevice> dev, id<MTLCommandQueue> q, Shape s, int iters, int pcb) {
    id<MTLBuffer> a = buf(dev, (size_t)s.M * s.K * 4), b = buf(dev, (size_t)s.N * s.K * 4), c = buf(dev, (size_t)s.M * s.N * 4);
    MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:a descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:s.M columns:s.K rowBytes:s.K * 4 dataType:MPSDataTypeFloat32]];
    MPSMatrix* mb = [[MPSMatrix alloc] initWithBuffer:b descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:s.N columns:s.K rowBytes:s.K * 4 dataType:MPSDataTypeFloat32]];
    MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:c descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:s.M columns:s.N rowBytes:s.N * 4 dataType:MPSDataTypeFloat32]];
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:YES resultRows:s.M resultColumns:s.N interiorColumns:s.K alpha:1.0 beta:0.0];
    double t0 = 0;
    for (int rep = 0; rep < 2; rep++) {
        t0 = now_s();
        for (int i = 0; i < iters; i += pcb) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            for (int j = 0; j < pcb; j++) [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    return (now_s() - t0) / iters;
}

// Returns seconds per GEMM; *err gets max relative error vs fp32 reference (bf16 inputs).
static double run_mpp(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib,
        const char* fn, int tm, int tn, int esz_in, int esz_out, Shape s, int iters, int pcb,
        double* err, double* enc_us) {
    NSError* e = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&e];
    if (!ps) { printf("pipeline %s: %s\n", fn, [[e description] UTF8String]); return 1; }
    id<MTLBuffer> a = buf(dev, (size_t)s.M * s.K * esz_in), b = buf(dev, (size_t)s.N * s.K * esz_in), c = buf(dev, (size_t)s.M * s.N * esz_out);
    float* af = (float*)malloc((size_t)s.M * s.K * 4); float* bf = (float*)malloc((size_t)s.N * s.K * 4);
    uint16_t* ab = (uint16_t*)malloc((size_t)s.M * s.K * 2); uint16_t* bb = (uint16_t*)malloc((size_t)s.N * s.K * 2);
    fill(af, bf, ab, bb, s);
    if (esz_in == 2 && strstr(fn, "_h_")) {
        __fp16* ha = (__fp16*)a.contents; __fp16* hb = (__fp16*)b.contents;
        for (long i = 0; i < (long)s.M * s.K; i++) ha[i] = (__fp16)af[i];
        for (long i = 0; i < (long)s.N * s.K; i++) hb[i] = (__fp16)bf[i];
    } else if (esz_in == 2) {
        memcpy(a.contents, ab, (size_t)s.M * s.K * 2); memcpy(b.contents, bb, (size_t)s.N * s.K * 2);
    } else {
        memcpy(a.contents, af, (size_t)s.M * s.K * 4); memcpy(b.contents, bf, (size_t)s.N * s.K * 4);
    }
    int mnk[3] = {s.M, s.N, s.K};
    MTLSize grid = MTLSizeMake((s.N + tn - 1) / tn, (s.M + tm - 1) / tm, 1);
    MTLSize tg = MTLSizeMake(ps.threadExecutionWidth * 4, 1, 1);
    auto encode = [&](id<MTLCommandBuffer> cb) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:a offset:0 atIndex:0]; [enc setBuffer:b offset:0 atIndex:1]; [enc setBuffer:c offset:0 atIndex:2];
        [enc setBytes:mnk length:12 atIndex:3];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    };
    { id<MTLCommandBuffer> cb = [q commandBuffer]; encode(cb); [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { printf("%s error: %s\n", fn, [[cb.error description] UTF8String]); return 1; } }
    // Reference check on a sample of outputs
    double maxrel = 0;
    for (int t = 0; t < 512; t++) {
        long i = (long)(((unsigned)t * 2654435761u) % (unsigned)s.M), j = (long)(((unsigned)t * 40503u) % (unsigned)s.N);
        double ref = 0, mag = 0;
        for (int k = 0; k < s.K; k++) { ref += (double)af[i * s.K + k] * bf[j * s.K + k]; mag += fabs((double)af[i * s.K + k] * bf[j * s.K + k]); }
        double got;
        if (esz_out == 4) got = ((float*)c.contents)[i * s.N + j];
        else if (strstr(fn, "_h_")) got = (double)((__fp16*)c.contents)[i * s.N + j];
        else got = bf2f(((uint16_t*)c.contents)[i * s.N + j]);
        double rel = fabs(got - ref) / (mag + 1e-30);
        if (rel > maxrel) maxrel = rel;
    }
    *err = maxrel;
    free(af); free(bf); free(ab); free(bb);
    double t0 = 0, tenc = 0;
    for (int rep = 0; rep < 2; rep++) {
        t0 = now_s(); tenc = 0;
        for (int i = 0; i < iters; i += pcb) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            double te = now_s();
            for (int j = 0; j < pcb; j++) encode(cb);
            tenc += now_s() - te;
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    *enc_us = tenc / iters * 1e6;
    return (now_s() - t0) / iters;
}

static double run_graph(id<MTLDevice> dev, id<MTLCommandQueue> q, Shape s, int iters, int pcb, double* enc_us) {
    MPSGraph* g = [MPSGraph new];
    MPSGraphTensor* A = [g placeholderWithShape:@[@(s.M), @(s.K)] dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor* B = [g placeholderWithShape:@[@(s.N), @(s.K)] dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor* Bt = [g transposeTensor:B dimension:0 withDimension:1 name:nil];
    MPSGraphTensor* C = [g matrixMultiplicationWithPrimaryTensor:A secondaryTensor:Bt name:nil];
    id<MTLBuffer> a = buf(dev, (size_t)s.M * s.K * 2), b = buf(dev, (size_t)s.N * s.K * 2), c = buf(dev, (size_t)s.M * s.N * 2);
    MPSGraphTensorData* da = [[MPSGraphTensorData alloc] initWithMTLBuffer:a shape:@[@(s.M), @(s.K)] dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData* db = [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:@[@(s.N), @(s.K)] dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData* dc = [[MPSGraphTensorData alloc] initWithMTLBuffer:c shape:@[@(s.M), @(s.N)] dataType:MPSDataTypeBFloat16];
    MPSGraphDevice* gd = [MPSGraphDevice deviceWithMTLDevice:dev];
    MPSGraphShapedType* ta = [[MPSGraphShapedType alloc] initWithShape:@[@(s.M), @(s.K)] dataType:MPSDataTypeBFloat16];
    MPSGraphShapedType* tb = [[MPSGraphShapedType alloc] initWithShape:@[@(s.N), @(s.K)] dataType:MPSDataTypeBFloat16];
    MPSGraphExecutable* ex = [g compileWithDevice:gd feeds:@{A: ta, B: tb} targetTensors:@[C] targetOperations:nil compilationDescriptor:nil];
    double t0 = 0, tenc = 0;
    for (int rep = 0; rep < 2; rep++) {
        t0 = now_s(); tenc = 0;
        for (int i = 0; i < iters; i += pcb) {
            MPSCommandBuffer* cb = [MPSCommandBuffer commandBufferFromCommandQueue:q];
            double te = now_s();
            for (int j = 0; j < pcb; j++)
                [ex encodeToCommandBuffer:cb inputsArray:@[da, db] resultsArray:@[dc] executionDescriptor:nil];
            tenc += now_s() - te;
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    *enc_us = tenc / iters * 1e6;
    return (now_s() - t0) / iters;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.languageVersion = MTLLanguageVersion4_0;
        NSError* e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&e];
        if (!lib) { printf("compile: %s\n", [[e description] UTF8String]); return 1; }
        printf("GPU: %s\n", [[dev name] UTF8String]);
        Shape shapes[] = {
            {"breakout train 65536x96x32", 65536, 96, 32},
            {"pacman train fwd 16384x768x256", 16384, 768, 256},
            {"pacman train dW 768x256x16384", 768, 256, 16384},
            {"tetris train fwd 65536x1536x512", 65536, 1536, 512},
            {"tetris rollout 4096x1536x512", 4096, 1536, 512},
        };
        struct K { const char* fn; int tm, tn, ein, eout; } ks[] = {
            {"mm_bf_f_64x32", 64, 32, 2, 4}, {"mm_bf_bf_64x32", 64, 32, 2, 2},
            {"mm_bf_f_64x64", 64, 64, 2, 4}, {"mm_bf_bf_64x64", 64, 64, 2, 2},
            {"mm_bf_bf_128x64", 128, 64, 2, 2},
            {"mm_h_h_64x64", 64, 64, 2, 2}, {"mm_f_f_64x64", 64, 64, 4, 4},
        };
        for (Shape s : shapes) {
            double flop = 2.0 * s.M * s.N * s.K;
            int iters = flop > 5e10 ? 20 : (flop > 1e9 ? 100 : 500);
            int pcb = iters < 50 ? iters : 50;
            double t32 = run_mps32(dev, q, s, iters, pcb);
            printf("\n%s\n  %-22s %7.2f TF\n", s.name, "MPS fp32 (baseline)", flop / t32 / 1e12);
            for (K k : ks) {
                double err = 0, enc = 0;
                double t = run_mpp(dev, q, lib, k.fn, k.tm, k.tn, k.ein, k.eout, s, iters, pcb, &err, &enc);
                printf("  %-22s %7.2f TF  %5.2fx  enc %5.1f us  maxrel %.1e\n", k.fn, flop / t / 1e12, t32 / t, enc, err);
            }
            double enc = 0;
            double tg = run_graph(dev, q, s, iters, pcb, &enc);
            printf("  %-22s %7.2f TF  %5.2fx  enc %5.1f us\n", "MPSGraph bf16", flop / tg / 1e12, t32 / tg, enc);
        }
    }
    return 0;
}
