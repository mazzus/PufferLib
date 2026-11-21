// Per-kernel differential test: Metal kernel vs the CPU emulation of the same
// CUDA kernel, on random inputs. Build + run: ./tests/test_metal_kernels.sh
#include <initializer_list>
#include <vector>
#include "pufferl.cu"

static int failures = 0;
static uint32_t g_seed = 1;
static float urand() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float)(g_seed >> 8) / 16777216.0f * 2.0f - 1.0f;
}

// A set of arena buffers; run() clones them so both paths see identical input.
// Each buffer is either precision_t (bf16 in bf16 builds) or float, matching
// the kernel parameter it feeds; P()/F() give the typed pointer.
struct Buf {
    void* v; long n; bool f32;
    size_t esz() const { return f32 ? sizeof(float) : sizeof(precision_t); }
    precision_t* P(long off = 0) const { assert(!f32); return (precision_t*)v + off; }
    float* F() const { assert(f32); return (float*)v; }
    float get(long i) const { return f32 ? ((float*)v)[i] : to_float(((precision_t*)v)[i]); }
    void set(long i, float x) const {
        if (f32) ((float*)v)[i] = x; else ((precision_t*)v)[i] = from_float(x);
    }
};
struct Case {
    std::vector<Buf> bufs;
    Buf add(long n, float lo = -1, float hi = 1, bool f32 = false) {
        Buf b = {NULL, n, f32};
        cudaMalloc(&b.v, n * b.esz());
        for (long i = 0; i < n; i++) b.set(i, lo + (hi - lo) * 0.5f * (urand() + 1.0f));
        bufs.push_back(b);
        return b;
    }
    Buf zeros(long n) { return add(n, 0, 0); }
    Buf addf(long n, float lo = -1, float hi = 1) { return add(n, lo, hi, true); }
    Buf zerosf(long n) { return add(n, 0, 0, true); }
};

// Copy every buffer of a case into a fresh set; returns the pointer mapping.
static std::vector<Buf> clone(const Case& c) {
    std::vector<Buf> out;
    for (const Buf& b : c.bufs) {
        Buf o = b;
        cudaMalloc(&o.v, b.n * b.esz());
        memcpy(o.v, b.v, b.n * b.esz());
        out.push_back(o);
    }
    return out;
}

// bf16: both paths compute in fp32 and round once, but libm vs Metal exp/log
// can differ in the last fp32 bit and flip a bf16 rounding (2^-8 relative).
static double ptol(double tol) { return USE_BF16 ? fmax(tol, 2e-2) : tol; }

// f(bufs) launches the kernel(s) on the given buffer set.
template <class F> static void diff(const char* name, Case& c, F f, double tol) {
    std::vector<Buf> cpu = clone(c), gpu = clone(c);
    puf_metal_force_cpu = true;
    f(cpu);
    cudaDeviceSynchronize();
    puf_metal_force_cpu = false;
    f(gpu);
    cudaDeviceSynchronize();
    double worst = 0;
    for (size_t i = 0; i < cpu.size(); i++) {
        double scale = 1e-30, md = 0;
        for (long j = 0; j < cpu[i].n; j++) {
            double a = cpu[i].get(j), b = gpu[i].get(j);
            if (std::isnan(a) != std::isnan(b)) md = INFINITY;
            else if (!std::isnan(a)) md = fmax(md, fabs(a - b));
            scale = fmax(scale, fabs(a));
        }
        worst = fmax(worst, md / scale);
    }
    tol = ptol(tol);
    bool ok = worst <= tol;
    failures += !ok;
    printf("%s %-28s max rel diff %.3e (tol %.0e)\n", ok ? "ok  " : "FAIL", name, worst, tol);
}

static Prec P(precision_t* p, int64_t a, int64_t b = 0, int64_t c = 0) {
    return {.data = p, .shape = {a, b, c}};
}

// Env action layout: A = total logit columns (sum of head sizes; continuous
// heads are size 1), continuous if every head is size 1.
static const int ACT_SZ[] = ACT_SIZES;
static int act_total() { int a = 0; for (int h = 0; h < NUM_ATNS; h++) a += ACT_SZ[h]; return a; }
static bool act_continuous() { for (int h = 0; h < NUM_ATNS; h++) if (ACT_SZ[h] != 1) return false; return true; }
static void fill_actions(Buf act, long rows) {
    for (long i = 0; i < rows; i++)
        for (int h = 0; h < NUM_ATNS; h++)
            act.set(i * NUM_ATNS + h, ACT_SZ[h] == 1 ? urand() : (float)((i * 7 + h * 3) % ACT_SZ[h]));
}

int main() {
    cublas_init_handle();
    const int B = 96, T = 32, H = 40, A = act_total();
    const bool CONT = act_continuous();
    printf("env %s: NUM_ATNS=%d A_total=%d %s, precision %s\n", PUFFER_ENV_NAME, NUM_ATNS, A,
        CONT ? "continuous" : "discrete", USE_BF16 ? "bf16" : "fp32");

    {   // mingru_gate
        Case c;
        c.zeros(B * H); c.zeros(B * H);
        c.add(B * 3 * H, -4, 4); c.add(B * H); c.add(B * H);
        diff("mingru_gate", c, [&](std::vector<Buf>& b) {
            PUF_LAUNCH(mingru_gate, grid_size(B * H), BLOCK_SIZE)(
                b[0].P(), b[1].P(), b[2].P(), b[3].P(), b[4].P(), H, B);
        }, 1e-5);
    }
    {   // mingru scans fwd + bwd
        Case c;
        c.add((long)B * T * 3 * H, -4, 4);  // 0 combined
        c.add(B * H);                       // 1 state
        c.add((long)B * T * H);             // 2 input
        Buf term = c.zeros((long)B * T);   // 3 terminals
        for (long i = 0; i < (long)B * T; i++) term.set(i, (i % 7 == 3) ? 1.0f : 0.0f);
        for (int k = 0; k < 6; k++) c.zeros((long)B * T * 3 * H);  // 4..9 scan bufs
        c.add((long)B * T * H);             // 10 grad_out
        c.add(B * H);                       // 11 grad_next_state
        auto scan_of = [&](std::vector<Buf>& b) {
            PrefixScan s = {};
            s.combined_ptr = b[0].P(); s.state_ptr = b[1].P(); s.input_ptr = b[2].P();
            s.terminals_ptr = b[3].P(); s.B = B; s.T = T; s.H = H;
            s.scan_h = P(b[4].P(), B, T, H); s.out = P(b[5].P(), B, T, H);
            s.next_state = P(b[6].P(), B, 1, H); s.grad_combined = P(b[7].P(), B, T, 3 * H);
            s.grad_state = P(b[8].P(), B, 1, H); s.grad_input = P(b[9].P(), B, T, H);
            return s;
        };
        diff("mingru_scan_forward+backward", c, [&](std::vector<Buf>& b) {
            PrefixScan s = scan_of(b);
            PUF_LAUNCH(mingru_scan_forward, grid_size(B * H), BLOCK_SIZE)(s);
            PUF_LAUNCH(mingru_scan_backward, grid_size(B * H), BLOCK_SIZE)(s, b[10].P(), b[11].P());
        }, 1e-5);
    }
    {   // sample_logits: same Philox streams -> identical actions
        Case c;
        int N = 4099;
        c.add((long)N * (A + 1), -3, 3);   // 0 dec_out
        c.zerosf((long)N * NUM_ATNS);      // 1 actions
        c.zerosf((long)N * NUM_ATNS);      // 2 env actions
        c.zeros(N); c.zeros(N);            // 3 logprobs, 4 values
        c.add((long)N * A, 1, 1);          // 5 mask (all ones)
        long rng_floats = (long)N * sizeof(curandStatePhilox4_32_10_t) / 4;
        c.zerosf(rng_floats);              // 6 rng states
        c.add(NUM_ATNS, -1.0f, 0.5f);      // 7 logstd (continuous only)
        int* act_sizes;
        cudaMalloc(&act_sizes, sizeof(int) * NUM_ATNS);
        int as[] = ACT_SIZES;
        memcpy(act_sizes, as, sizeof(as));
        curandStatePhilox4_32_10_t* st = (curandStatePhilox4_32_10_t*)c.bufs[6].v;
        for (int i = 0; i < N; i++) curand_init(7, i, 0, &st[i]);
        diff("sample_logits (x3 steps)", c, [&](std::vector<Buf>& b) {
            for (int step = 0; step < 3; step++) {
                Prec logstd = CONT ? P(b[7].P(), 1, NUM_ATNS) : Prec{};
                PUF_LAUNCH(sample_logits, grid_size(N), BLOCK_SIZE)(
                    P(b[0].P(), N, A + 1), logstd, act_sizes, b[1].F(), b[2].F(), b[3].P(),
                    b[4].P(), (curandStatePhilox4_32_10_t*)b[6].v, b[5].P(), A);
            }
        }, 1e-5);
    }
    {   // puff_advantage
        Case c;
        int Nr = 100, Tr = 32;
        c.add(Nr * Tr); c.add(Nr * Tr);
        Buf d = c.zeros(Nr * Tr);
        for (int i = 0; i < Nr * Tr; i++) d.set(i, (i % 11 == 0) ? 1.0f : 0.0f);
        c.add(Nr * Tr, 0.5f, 1.5f); c.zeros(Nr * Tr); c.zeros(Nr * Tr);
        diff("puff_advantage", c, [&](std::vector<Buf>& b) {
            PUF_LAUNCH(puff_advantage, (Nr + 63) / 64, 64)(b[0].P(), b[1].P(), b[2].P(), b[3].P(),
                b[4].P(), b[5].P(), 0.97f, 0.95f, 1.0f, 1.0f, Nr, Tr);
        }, 1e-5);
    }
    {   // muon norm chain
        Case c;
        int n = 50000;
        c.add(n); c.zerosf(256); c.zerosf(1); c.zerosf(n); Buf lr = c.zerosf(1); lr.set(0, 0.01f);
        c.addf(n);
        diff("muon sum_sq/clip/norm/update", c, [&](std::vector<Buf>& b) {
            int blocks = min(grid_size(n), 256);
            PUF_LAUNCH(muon_sum_sq_partials, blocks, 256)(b[1].F(), b[0].P(), n);
            PUF_LAUNCH(muon_sum_sq_reduce, 1, 256)(b[2].F(), b[1].F(), blocks);
            PUF_LAUNCH(muon_clip_nesterov, grid_size(n), BLOCK_SIZE)(
                b[3].F(), b[0].P(), b[2].F(), 1.0f, 1e-6f, 0.9f, n);
            PUF_LAUNCH(muon_l2_normalize, grid_size(n), BLOCK_SIZE)(b[0].P(), b[2].F(), 1e-7f, n);
            PUF_LAUNCH(muon_store_update, grid_size(n), BLOCK_SIZE)(b[0].P(), b[0].P(), 1.5f, n);
            PUF_LAUNCH(muon_weight_update, grid_size(n), BLOCK_SIZE)(b[5].F(), b[0].P(), b[4].F(), 0.0f, n);
        }, 1e-5);
    }
    {   // cache_imp_and_v + ppo_loss_compute + reduce
        Case c;
        int Nb = 64, Tb = 32, NT = Nb * Tb;
        c.add((long)NT * (A + 1), -3, 3);     // 0 dec
        Buf act = c.zerosf((long)NT * NUM_ATNS);  // 1 actions
        fill_actions(act, NT);
        c.add(NT, -1.5f, -0.5f);               // 2 old logprobs
        c.add((long)NT * A, 1, 1);             // 3 mask
        c.zeros(NT); c.zeros(NT);              // 4 imp, 5 value
        c.zerosf((long)NT * A);                // 6 logps / grad_logits
        c.zerosf(NT);                          // 7 new_lp / grad_values
        c.add(NT); c.add(NT); c.add(NT);       // 8 adv, 9 values, 10 returns
        int grid = (NT + PPO_THREADS - 1) / PPO_THREADS;
        c.zerosf((long)grid * LOSS_N);         // 11 partials
        c.zerosf(NUM_LOSSES);                  // 12 losses
        Buf ent = c.zerosf(1); ent.set(0, 0.01f);  // 13 ent_coef
        c.add(NUM_ATNS, -1.0f, 0.5f);          // 14 logstd (continuous)
        c.zerosf((long)NT * NUM_ATNS);         // 15 grad_logstd (continuous)
        c.zeros(NUM_ATNS);                     // 16 summed logstd grad
        int* act_sizes;
        cudaMalloc(&act_sizes, sizeof(int) * NUM_ATNS);
        int as[] = ACT_SIZES;
        memcpy(act_sizes, as, sizeof(as));
        diff("cache_imp + ppo_loss", c, [&](std::vector<Buf>& b) {
            Prec dec = P(b[0].P(), Nb, Tb, A + 1);
            Prec logstd = CONT ? P(b[14].P(), 1, NUM_ATNS) : Prec{};
            PUF_LAUNCH(cache_imp_and_v, grid_size(NT), BLOCK_SIZE)(dec, b[1].F(), b[2].P(),
                b[3].P(), logstd, act_sizes, b[4].P(), b[5].P(), b[6].F(), b[7].F());
            PPOKernelArgs ka = {
                .grad_logits = b[6].F(), .grad_logstd = CONT ? b[15].F() : NULL,
                .grad_values_pred = b[7].F(),
                .logits = b[0].P(), .logstd = CONT ? b[14].P() : NULL, .values_pred = b[0].P(A),
                .act_sizes = act_sizes, .action_mask = b[3].P(), .num_atns = NUM_ATNS,
                .clip_coef = 0.2f, .vf_clip_coef = 0.5f, .vf_coef = 0.5f,
                .ent_coef = b[13].F(), .T_seq = Tb, .A_total = A, .N = Nb,
                .is_continuous = CONT,
            };
            PPOGraphArgs ga = {.imp = b[4].P(), .actions = b[1].F(), .old_logprobs = b[2].P(),
                .advantages = b[8].P(), .values = b[9].P(), .returns = b[10].P()};
            PUF_LAUNCH(ppo_loss_compute, grid, PPO_THREADS)(b[11].F(), ka, ga);
            PUF_LAUNCH(ppo_loss_reduce, 1, LOSS_N)(b[12].F(), b[11].F(), grid);
            if (CONT) {  // decoder backward: logstd grad summed over rows
                PUF_LAUNCH(sum_rows_to_precision_kernel, grid_size(NUM_ATNS), BLOCK_SIZE)(
                    b[16].P(), b[15].F(), NT, NUM_ATNS);
            }
        }, 1e-5);
    }
    {   // small elementwise kernels
        Case c;
        int L = 3, Na = 200, Hh = 24, n = L * Na * Hh;
        c.add(n);                                   // 0 state
        Buf t = c.zerosf(Na);                       // 1 terminals
        for (int i = 0; i < Na; i++) t.set(i, (i % 5 == 0));
        c.zeros(2L * n);                            // 2 snapshot dst
        c.add(n); c.add(n);                         // 3,4 add dst/src
        c.zeros(n); c.addf(n); c.zeros(n); c.addf(n); // 5..8 cast_rew_term
        c.zeros(n); c.add(n);                       // 9,10 transpose
        diff("zero_term/snapshot/add/cast/T", c, [&](std::vector<Buf>& b) {
            Prec st = P(b[0].P(), L, Na, Hh);
            Float term = {.data = b[1].F(), .shape = {Na}};
            PUF_LAUNCH(zero_term_state, grid_size(n), BLOCK_SIZE)(st, term, 0, 0, Na);
            PUF_LAUNCH(snapshot_state, grid_size(n), BLOCK_SIZE)(P(b[2].P(), L, 2 * Na, Hh), st, 7, Na);
            PUF_LAUNCH(add_kernel, grid_size(n), BLOCK_SIZE)(b[3].P(), b[4].P(), n);
            PUF_LAUNCH(cast_rew_term, grid_size(n), BLOCK_SIZE)(b[5].P(), b[6].F(), b[7].P(), b[8].F(), n);
            PUF_LAUNCH(transpose_102, grid_size(n), BLOCK_SIZE)(b[9].P(), (const precision_t*)b[10].P(), L, Na, Hh);
            PUF_LAUNCH(clamp_precision_kernel, grid_size(n), BLOCK_SIZE)(b[10].P(), -0.5f, 0.5f, n);
        }, 1e-6);
    }
    {   // casts between precision_t, float and uchar
        Case c;
        int n = 5000;
        c.zeros(n); c.addf(n, -3, 3);               // 0,1 cast_p_f
        c.zerosf(n); c.add(n, -3, 3);               // 2,3 float <- precision
        c.zeros(n);                                 // 4 precision <- uchar
        Buf u = c.zerosf((n + 3) / 4);              // 5 uchar src (packed)
        for (int i = 0; i < n; i++) ((unsigned char*)u.v)[i] = (unsigned char)(i * 37);
        c.zerosf((n + 3) / 4);                      // 6 uchar dst
        c.add(n, 0, 200);                           // 7 precision src for uchar
        c.zerosf(n);                                // 8 float transpose dst
        diff("casts", c, [&](std::vector<Buf>& b) {
            PUF_LAUNCH(cast, grid_size(n), BLOCK_SIZE)(b[0].P(), b[1].F(), n);
#ifndef PRECISION_FLOAT
            PUF_LAUNCH(cast, grid_size(n), BLOCK_SIZE)(b[2].F(), b[3].P(), n);
            PUF_LAUNCH(transpose_102, grid_size(n), BLOCK_SIZE)(b[8].F(), (const float*)b[1].F(), 5, 40, 25);
#endif
            PUF_LAUNCH(cast, grid_size(n), BLOCK_SIZE)(b[4].P(), (unsigned char*)b[5].v, n);
            PUF_LAUNCH(cast, grid_size(n), BLOCK_SIZE)((unsigned char*)b[6].v, b[7].P(), n);
        }, 1e-6);
    }
    {   // GEMM wrappers on the Metal path vs a double host reference on the
        // same (precision_t-rounded) inputs. bf16 output rounding: 2^-8 rel.
        const double gtol = USE_BF16 ? 1e-2 : 1e-5;
        int shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {64, 17, 33}, {130, 96, 258},
            {256, 256, 256}, {300, 768, 256}, {2048, 768, 256},
            {96, 32, 8192}, {256, 96, 3000}};  // long K: split-K + K loop on MPP
        for (auto& sh : shapes) {
            int M = sh[0], N = sh[1], K = sh[2];
            for (int form = 0; form < 3; form++) {
                for (float beta : {0.0f, 0.5f}) {
                    float alpha = form == 2 ? -1.25f : 0.75f;
                    // form 0: puf_mm a(M,K) b(N,K)^T; 1: puf_mm_tn a(K,M)^T b(K,N); 2: puf_mm_nn
                    long na = (long)M * K, nb = (long)N * K, nc = (long)M * N;
                    precision_t *A, *Bm, *C;
                    cudaMalloc(&A, na * sizeof(precision_t));
                    cudaMalloc(&Bm, nb * sizeof(precision_t));
                    cudaMalloc(&C, nc * sizeof(precision_t));
                    for (long i = 0; i < na; i++) A[i] = from_float(urand());
                    for (long i = 0; i < nb; i++) Bm[i] = from_float(urand());
                    for (long i = 0; i < nc; i++) C[i] = from_float(urand());
                    double* ref = (double*)malloc(nc * sizeof(double));
                    Prec pa, pb, pc = P(C, M, N);
                    if (form == 0) { pa = P(A, M, K); pb = P(Bm, N, K); }
                    else if (form == 1) { pa = P(A, K, M); pb = P(Bm, K, N); }
                    else { pa = P(A, M, K); pb = P(Bm, K, N); }
                    // host reference, row-major
                    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
                        double acc = 0;
                        for (int k = 0; k < K; k++) {
                            float av = to_float(form == 1 ? A[(long)k * M + i] : A[(long)i * K + k]);
                            float bv = to_float(form == 0 ? Bm[(long)j * K + k] : Bm[(long)k * N + j]);
                            acc += (double)av * bv;
                        }
                        ref[(long)i * N + j] = alpha * acc + beta * to_float(C[(long)i * N + j]);
                    }
                    if (form == 0) puf_mm(&pa, &pb, &pc, 0, alpha, beta);
                    else if (form == 1) puf_mm_tn(&pa, &pb, &pc, 0, alpha, beta);
                    else puf_mm_nn(&pa, &pb, &pc, 0, alpha, beta);
                    cudaDeviceSynchronize();
                    double md = 0, mx = 1e-30;
                    for (long i = 0; i < nc; i++) {
                        md = fmax(md, fabs(ref[i] - to_float(C[i])));
                        mx = fmax(mx, fabs(ref[i]));
                    }
                    bool ok = md / mx < gtol;
                    failures += !ok;
                    if (!ok) printf("FAIL gemm form=%d M=%d N=%d K=%d beta=%.1f rel %.3e\n",
                        form, M, N, K, beta, md / mx);
                    free(ref);
                    cudaFree(A); cudaFree(Bm); cudaFree(C);
                }
            }
        }
        // Muon gram: same buffer as both operands
        precision_t* X;
        int R = 96, Cc = 288;
        cudaMalloc(&X, (long)R * Cc * sizeof(precision_t));
        for (long i = 0; i < (long)R * Cc; i++) X[i] = from_float(urand());
        precision_t* G;
        cudaMalloc(&G, (long)R * R * sizeof(precision_t));
        Prec px = P(X, R, Cc), pg = P(G, R, R);
        puf_mm(&px, &px, &pg, 0);
        cudaDeviceSynchronize();
        double md = 0;
        for (int i = 0; i < R; i++) for (int j = 0; j < R; j++) {
            double acc = 0;
            for (int k = 0; k < Cc; k++)
                acc += (double)to_float(X[i * Cc + k]) * to_float(X[j * Cc + k]);
            md = fmax(md, fabs(acc - to_float(G[i * R + j])) / fmax(1.0, fabs(acc)));
        }
        failures += md >= gtol;
        printf("%s gemm forms x shapes (%s, %s), muon gram rel %.3e\n",
            md < gtol ? "ok  " : "FAIL", puf_metal_has_mpp_gemm() ? "MPP" : "MPS",
            USE_BF16 ? "bf16" : "fp32", md);
    }
    {   // Muon Newton-Schulz (5 steps, wide and tall) on the GEMM path vs a host
        // reference that rounds to precision_t at the same points.
        for (auto rc : {std::pair<int, int>{256, 768}, std::pair<int, int>{768, 256}}) {
            int R = rc.first, Cc = rc.second, M = R < Cc ? R : Cc;
            bool tall = R > Cc;
            long ne = (long)R * Cc;
            precision_t *x, *xb, *g, *gb;
            cudaMalloc(&x, ne * sizeof(precision_t)); cudaMalloc(&xb, ne * sizeof(precision_t));
            cudaMalloc(&g, (long)M * M * sizeof(precision_t));
            cudaMalloc(&gb, (long)M * M * sizeof(precision_t));
            double nrm = 0;
            std::vector<float> hx(ne);
            for (long i = 0; i < ne; i++) { float v = urand(); nrm += (double)v * v; hx[i] = v; }
            for (long i = 0; i < ne; i++) { x[i] = from_float(hx[i] / (float)sqrt(nrm)); hx[i] = to_float(x[i]); }
            std::vector<float> hxb(ne), hg((long)M * M), hgb((long)M * M);
            auto rnd = [](double v) { return to_float(from_float((float)v)); };
            Prec px = P(x, R, Cc), pxb = P(xb, R, Cc), pgm = P(g, M, M), pgb = P(gb, M, M);
            for (int it = 0; it < 5; it++) {
                Prec& src = it % 2 == 0 ? px : pxb;
                Prec& dst = it % 2 == 0 ? pxb : px;
                std::vector<float>& hs = it % 2 == 0 ? hx : hxb;
                std::vector<float>& hd = it % 2 == 0 ? hxb : hx;
                float c0 = ns_coeffs[it][0], c1 = ns_coeffs[it][1], c2 = ns_coeffs[it][2];
                if (tall) puf_mm_tn(&src, &src, &pgm, 0); else puf_mm(&src, &src, &pgm, 0);
                puf_copy(&pgb, &pgm, 0);
                puf_mm_nn(&pgm, &pgm, &pgb, 0, c2, c1);
                puf_copy(&dst, &src, 0);
                if (tall) puf_mm_nn(&src, &pgb, &dst, 0, 1.0f, c0);
                else puf_mm_nn(&pgb, &src, &dst, 0, 1.0f, c0);
                // host: gram = S S^T (wide) or S^T S (tall)
                for (int i = 0; i < M; i++) for (int j = 0; j < M; j++) {
                    double acc = 0;
                    if (tall) for (int k = 0; k < R; k++) acc += (double)hs[(long)k * Cc + i] * hs[(long)k * Cc + j];
                    else for (int k = 0; k < Cc; k++) acc += (double)hs[(long)i * Cc + k] * hs[(long)j * Cc + k];
                    hg[(long)i * M + j] = rnd(acc);
                }
                for (int i = 0; i < M; i++) for (int j = 0; j < M; j++) {
                    double acc = 0;
                    for (int k = 0; k < M; k++) acc += (double)hg[(long)i * M + k] * hg[(long)k * M + j];
                    hgb[(long)i * M + j] = rnd((float)(c2 * acc) + c1 * hg[(long)i * M + j]);
                }
                for (int i = 0; i < R; i++) for (int j = 0; j < Cc; j++) {
                    double acc = 0;
                    if (tall) for (int k = 0; k < M; k++) acc += (double)hs[(long)i * Cc + k] * hgb[(long)k * M + j];
                    else for (int k = 0; k < M; k++) acc += (double)hgb[(long)i * M + k] * hs[(long)k * Cc + j];
                    hd[(long)i * Cc + j] = rnd((float)acc + c0 * hs[(long)i * Cc + j]);
                }
            }
            cudaDeviceSynchronize();
            double num = 0, den = 0, num_g = 0, den_g = 0;
            for (long i = 0; i < ne; i++) {
                double d = to_float(xb[i]) - hxb[i];
                num += d * d; den += (double)hxb[i] * hxb[i];
            }
            for (long i = 0; i < (long)M * M; i++) {
                double d = to_float(gb[i]) - hgb[i];
                num_g += d * d; den_g += (double)hgb[i] * hgb[i];
            }
            double rel = sqrt(num / den);
            bool ok = rel < (USE_BF16 ? 5e-2 : 1e-4);
            failures += !ok;
            printf("%s muon NS %dx%d                  rel L2 %.3e (last gram_buf %.3e)\n",
                ok ? "ok  " : "FAIL", R, Cc, rel, sqrt(num_g / den_g));
            cudaFree(x); cudaFree(xb); cudaFree(g); cudaFree(gb);
        }
    }
    printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures != 0;
}
