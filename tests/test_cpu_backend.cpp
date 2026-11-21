// Unit tests for the CPU trainer backend (src/puf_cuda_cpu.h).
// Build + run: ./tests/test_cpu_backend.sh
#include <initializer_list>
#include "pufferl.cu"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; \
    printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static float frand(uint32_t* s) {
    *s = *s * 1664525u + 1013904223u;
    return (float)(*s >> 8) / 16777216.0f * 2.0f - 1.0f;
}

static Prec make(int64_t r, int64_t c, uint32_t* seed) {
    Prec t = {.shape = {r, c}};
    cudaMalloc(&t.data, r * c * sizeof(float));
    for (int64_t i = 0; i < r * c; i++) t.data[i] = frand(seed);
    return t;
}

static float maxdiff(const float* a, const float* b, long n) {
    float m = 0;
    for (long i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

// out(M,N) = alpha * a(M,K) @ b(N,K)^T + beta * out
static void test_mm(int M, int N, int K, float alpha, float beta) {
    uint32_t s = 1234 + M * 7 + N * 13 + K;
    Prec a = make(M, K, &s), b = make(N, K, &s), out = make(M, N, &s);
    float* ref = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        double acc = 0;
        for (int k = 0; k < K; k++) acc += (double)a.data[i * K + k] * b.data[j * K + k];
        ref[i * N + j] = alpha * acc + beta * out.data[i * N + j];
    }
    puf_mm(&a, &b, &out, 0, alpha, beta);
    float d = maxdiff(ref, out.data, M * N);
    CHECK(d < 1e-4f * K, "puf_mm M=%d N=%d K=%d maxdiff=%g", M, N, K, d);
}

// out(M,N) = alpha * a(K,M)^T @ b(K,N) + beta * out
static void test_mm_tn(int M, int N, int K, float alpha, float beta) {
    uint32_t s = 99 + M * 7 + N * 13 + K;
    Prec a = make(K, M, &s), b = make(K, N, &s), out = make(M, N, &s);
    float* ref = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        double acc = 0;
        for (int k = 0; k < K; k++) acc += (double)a.data[k * M + i] * b.data[k * N + j];
        ref[i * N + j] = alpha * acc + beta * out.data[i * N + j];
    }
    puf_mm_tn(&a, &b, &out, 0, alpha, beta);
    float d = maxdiff(ref, out.data, M * N);
    CHECK(d < 1e-4f * K, "puf_mm_tn M=%d N=%d K=%d maxdiff=%g", M, N, K, d);
}

// out(M,N) = alpha * a(M,K) @ b(K,N) + beta * out
static void test_mm_nn(int M, int N, int K, float alpha, float beta) {
    uint32_t s = 7 + M * 7 + N * 13 + K;
    Prec a = make(M, K, &s), b = make(K, N, &s), out = make(M, N, &s);
    float* ref = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        double acc = 0;
        for (int k = 0; k < K; k++) acc += (double)a.data[i * K + k] * b.data[k * N + j];
        ref[i * N + j] = alpha * acc + beta * out.data[i * N + j];
    }
    puf_mm_nn(&a, &b, &out, 0, alpha, beta);
    float d = maxdiff(ref, out.data, M * N);
    CHECK(d < 1e-4f * K, "puf_mm_nn M=%d N=%d K=%d maxdiff=%g", M, N, K, d);
}

__global__ void reduce_test_kernel(float* partials, const float* src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        sum += src[i];
    }
    sdata[tid] = sum;
    block_reduce_sum(sdata, &partials[blockIdx.x], tid, blockDim.x, 1);
}

static void test_block_reduce() {
    int n = 100003, blocks = 37;
    float* src;
    float* part;
    cudaMalloc(&src, n * sizeof(float));
    cudaMalloc(&part, blocks * sizeof(float));
    uint32_t s = 5;
    double total = 0;
    for (int i = 0; i < n; i++) { src[i] = frand(&s); total += src[i]; }
    PUF_LAUNCH(reduce_test_kernel, blocks, 256, 0, 0)(part, src, n);
    double got = 0;
    for (int b = 0; b < blocks; b++) got += part[b];
    CHECK(fabs(got - total) < 1e-2, "block_reduce got=%f want=%f", got, total);
}

static void test_rng() {
    curandStatePhilox4_32_10_t st;
    curand_init(42, 3, 0, &st);
    int n = 400000;
    double su = 0, su2 = 0, sn = 0, sn2 = 0;
    float umin = 2, umax = -1;
    for (int i = 0; i < n; i++) {
        float u = curand_uniform(&st);
        su += u; su2 += u * u;
        umin = fminf(umin, u); umax = fmaxf(umax, u);
        float z = curand_normal(&st);
        sn += z; sn2 += z * z;
    }
    double mu = su / n, vu = su2 / n - mu * mu;
    double mz = sn / n, vz = sn2 / n - mz * mz;
    CHECK(umin > 0.0f && umax <= 1.0f, "uniform range [%g, %g]", umin, umax);
    CHECK(fabs(mu - 0.5) < 0.005 && fabs(vu - 1.0 / 12) < 0.005, "uniform mean %f var %f", mu, vu);
    CHECK(fabs(mz) < 0.01 && fabs(vz - 1.0) < 0.01, "normal mean %f var %f", mz, vz);
    // Distinct subsequences must give distinct streams
    curandStatePhilox4_32_10_t a, b;
    curand_init(42, 0, 0, &a);
    curand_init(42, 1, 0, &b);
    int same = 0;
    for (int i = 0; i < 64; i++) same += curand(&a) == curand(&b);
    CHECK(same < 4, "subsequences overlap (%d/64 equal)", same);
}

int main() {
    cublas_init_handle();
    int shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {64, 17, 33}, {130, 96, 258}, {7, 300, 2}};
    for (auto& sh : shapes) {
        for (float beta : {0.0f, 0.5f}) {
            test_mm(sh[0], sh[1], sh[2], 1.0f, beta);
            test_mm_tn(sh[0], sh[1], sh[2], 0.75f, beta);
            test_mm_nn(sh[0], sh[1], sh[2], -1.25f, beta);
        }
    }
    test_block_reduce();
    test_rng();
    printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures != 0;
}
