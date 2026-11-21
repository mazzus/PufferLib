// GEMMs for the Metal backend, built on Metal Performance Primitives
// (MSL 4 tensor ops, macOS 26+). Compiled at runtime as a separate library;
// if it fails to build, puf_metal.mm falls back to MPSMatrixMultiplication.
//
// Row-major C(M,N) = alpha * op(A) @ op(B) + beta * C, leading dims lda/ldb/ldc:
//   nt: A(M,K) @ B(N,K)^T   tn: A(K,M)^T @ B(K,N)   nn: A(M,K) @ B(K,N)   tt
// T is the storage type of A, B and C; accumulation is always fp32.
// Arguments: one constant struct at buffer 0 holding GPU addresses
// (the backend's bindless convention; all arena buffers are made resident).
//
// Dispatch: threadgroups (ceil(N/TN), ceil(M/TM), splits), 4 simdgroups each.
// splits > 1 (split-K, for small outputs with a long K such as weight
// gradients): each z-slice handles a K range and writes fp32 partials to a
// workspace (alpha 1, beta 0); puf_gemm_splitk_reduce_* then applies
// alpha/beta and writes C.
//
// The K loop runs in BK chunks with a threadgroup barrier per chunk so the
// simdgroups stay close in K and share cache (MPP programming guide 2.3.4).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct PufGemmArgs {
    int M, N, K;
    int lda, ldb, ldc;
    float alpha, beta;
    int splits;     // K splits (grid z); 1 = direct
    int k_chunk;    // K per split (multiple of BK except the last)
};

template <typename T, typename TO>
struct PufGemmParams {
    device T* a;
    device T* b;
    device TO* c;   // C, or the [splits, M, N] fp32 workspace when splits > 1
    PufGemmArgs p;
};

template <typename T, typename TO, bool TA, bool TB, int TM, int TN, int BK>
inline void puf_gemm_impl(constant PufGemmParams<T, TO>& P, uint3 tgid) {
    device T* a = P.a;
    device T* b = P.b;
    device TO* c = P.c;
    constant PufGemmArgs& p = P.p;
    using ext = dextents<int32_t, 2>;
    using str = array<int32_t, 2>;
    const int M = p.M, N = p.N, K = p.K;
    const int k0 = tgid.z * p.k_chunk;
    const int kc = min(p.k_chunk, K - k0);
    if (kc <= 0) return;
    const int row0 = tgid.y * TM, col0 = tgid.x * TN;
    // Extents/strides list the innermost (contiguous) dimension first. Each
    // tensor covers only this split's K range.
    tensor<device T, ext, tensor_inline> A(
        TA ? a + (long)k0 * p.lda : a + k0,
        TA ? ext(M, kc) : ext(kc, M), str{1, p.lda});
    tensor<device T, ext, tensor_inline> B(
        TB ? b + k0 : b + (long)k0 * p.ldb,
        TB ? ext(kc, N) : ext(N, kc), str{1, p.ldb});
    // Split-K writes slice z of a packed [splits, M, N] fp32 workspace.
    const bool part = p.splits > 1;
    tensor<device TO, ext, tensor_inline> C(part ? c + (long)tgid.z * M * N : c,
        ext(N, M), str{1, part ? N : p.ldc});
    // BK == 0: one run over the whole K (short K); else a barrier-synced K loop.
    constexpr int DK = BK ? BK : static_cast<int>(dynamic_extent);
    constexpr auto desc = matmul2d_descriptor(TM, TN, DK, TA, TB, false,
        BK ? matmul2d_descriptor::mode::multiply_accumulate
           : matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA0 = TA ? A.slice(row0, 0) : A.slice(0, row0);
    auto mB0 = TB ? B.slice(0, col0) : B.slice(col0, 0);
    auto acc = op.template get_destination_cooperative_tensor<
        decltype(mA0), decltype(mB0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    if (BK == 0) {
        op.run(mA0, mB0, acc);
    } else {
        for (int k = 0; k < kc; k += BK) {
            threadgroup_barrier(mem_flags::mem_none);
            auto tA = TA ? A.slice(row0, k) : A.slice(k, row0);
            auto tB = TB ? B.slice(k, col0) : B.slice(col0, k);
            op.run(tA, tB, acc);
        }
    }

    auto mC = C.slice(col0, row0);
    auto out = op.template get_destination_cooperative_tensor<
        decltype(mA0), decltype(mB0), TO>();
    float alpha = p.splits > 1 ? 1.0f : p.alpha;
    float beta = p.splits > 1 ? 0.0f : p.beta;
    if (beta != 0.0f) {
        auto old = op.template get_destination_cooperative_tensor<
            decltype(mA0), decltype(mB0), TO>();
        old.load(mC);
        #pragma unroll
        for (uint16_t i = 0; i < out.get_capacity(); ++i)
            out[i] = (TO)(alpha * acc[i] + beta * (float)old[i]);
    } else {
        #pragma unroll
        for (uint16_t i = 0; i < out.get_capacity(); ++i)
            out[i] = (TO)(alpha * acc[i]);
    }
    out.store(mC);
}

#define PUF_GEMM(name, T, TO, TA, TB, TM, TN, BK) \
kernel void name(constant PufGemmParams<T, TO>& P [[buffer(0)]], \
        uint3 tgid [[threadgroup_position_in_grid]]) { \
    puf_gemm_impl<T, TO, TA, TB, TM, TN, BK>(P, tgid); \
}

// Direct (C is T) and split-K partial (C is the fp32 workspace) variants;
// _k: barrier-synced K loop (BK 128) for long K.
PUF_GEMM(puf_gemm_nt_f32, float, float, false, true, 64, 32, 0)
PUF_GEMM(puf_gemm_nt_bf16, bfloat, bfloat, false, true, 64, 32, 0)
PUF_GEMM(puf_gemm_nt_bf16_part, bfloat, float, false, true, 64, 32, 0)
PUF_GEMM(puf_gemm_nt_f32_k, float, float, false, true, 64, 32, 128)
PUF_GEMM(puf_gemm_nt_bf16_k, bfloat, bfloat, false, true, 64, 32, 128)
PUF_GEMM(puf_gemm_nt_bf16_part_k, bfloat, float, false, true, 64, 32, 128)
PUF_GEMM(puf_gemm_tn_f32, float, float, true, false, 64, 32, 0)
PUF_GEMM(puf_gemm_tn_bf16, bfloat, bfloat, true, false, 64, 32, 0)
PUF_GEMM(puf_gemm_tn_bf16_part, bfloat, float, true, false, 64, 32, 0)
PUF_GEMM(puf_gemm_tn_f32_k, float, float, true, false, 64, 32, 128)
PUF_GEMM(puf_gemm_tn_bf16_k, bfloat, bfloat, true, false, 64, 32, 128)
PUF_GEMM(puf_gemm_tn_bf16_part_k, bfloat, float, true, false, 64, 32, 128)
PUF_GEMM(puf_gemm_nn_f32, float, float, false, false, 64, 32, 0)
PUF_GEMM(puf_gemm_nn_bf16, bfloat, bfloat, false, false, 64, 32, 0)
PUF_GEMM(puf_gemm_nn_bf16_part, bfloat, float, false, false, 64, 32, 0)
PUF_GEMM(puf_gemm_nn_f32_k, float, float, false, false, 64, 32, 128)
PUF_GEMM(puf_gemm_nn_bf16_k, bfloat, bfloat, false, false, 64, 32, 128)
PUF_GEMM(puf_gemm_nn_bf16_part_k, bfloat, float, false, false, 64, 32, 128)
PUF_GEMM(puf_gemm_tt_f32, float, float, true, true, 64, 32, 0)
PUF_GEMM(puf_gemm_tt_bf16, bfloat, bfloat, true, true, 64, 32, 0)
PUF_GEMM(puf_gemm_tt_bf16_part, bfloat, float, true, true, 64, 32, 0)
PUF_GEMM(puf_gemm_tt_f32_k, float, float, true, true, 64, 32, 128)
PUF_GEMM(puf_gemm_tt_bf16_k, bfloat, bfloat, true, true, 64, 32, 128)
PUF_GEMM(puf_gemm_tt_bf16_part_k, bfloat, float, true, true, 64, 32, 128)

// C = alpha * sum_z W[z] + beta * C over M x N (W packed, C with ldc).
template <typename T>
struct PufGemmReduceParams {
    device const float* w;
    device T* c;
    PufGemmArgs p;
};
template <typename T>
inline void puf_gemm_reduce(constant PufGemmReduceParams<T>& R, uint i) {
    constant PufGemmArgs& p = R.p;
    long mn = (long)p.M * p.N;
    if (i >= mn) return;
    float s = 0.0f;
    for (int z = 0; z < p.splits; z++) s += R.w[z * mn + i];
    long ci = (long)(i / p.N) * p.ldc + i % p.N;
    float v = p.alpha * s;
    if (p.beta != 0.0f) v += p.beta * (float)R.c[ci];
    R.c[ci] = (T)v;
}
kernel void puf_gemm_splitk_reduce_f32(constant PufGemmReduceParams<float>& R [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
    puf_gemm_reduce<float>(R, i);
}
kernel void puf_gemm_splitk_reduce_bf16(constant PufGemmReduceParams<bfloat>& R [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
    puf_gemm_reduce<bfloat>(R, i);
}
