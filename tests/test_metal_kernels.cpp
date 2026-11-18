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
struct Buf { float* p; long n; };
struct Case {
    std::vector<Buf> bufs;
    float* add(long n, float lo = -1, float hi = 1) {
        float* p;
        cudaMalloc(&p, n * sizeof(float));
        for (long i = 0; i < n; i++) p[i] = lo + (hi - lo) * 0.5f * (urand() + 1.0f);
        bufs.push_back({p, n});
        return p;
    }
    float* zeros(long n) { return add(n, 0, 0); }
};

// Copy every buffer of a case into a fresh set; returns the pointer mapping.
static std::vector<Buf> clone(const Case& c) {
    std::vector<Buf> out;
    for (const Buf& b : c.bufs) {
        float* p;
        cudaMalloc(&p, b.n * sizeof(float));
        memcpy(p, b.p, b.n * sizeof(float));
        out.push_back({p, b.n});
    }
    return out;
}

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
            double a = cpu[i].p[j], b = gpu[i].p[j];
            if (std::isnan(a) != std::isnan(b)) md = INFINITY;
            else if (!std::isnan(a)) md = fmax(md, fabs(a - b));
            scale = fmax(scale, fabs(a));
        }
        worst = fmax(worst, md / scale);
    }
    bool ok = worst <= tol;
    failures += !ok;
    printf("%s %-28s max rel diff %.3e (tol %.0e)\n", ok ? "ok  " : "FAIL", name, worst, tol);
}

static Prec P(float* p, int64_t a, int64_t b = 0, int64_t c = 0) {
    return {.data = p, .shape = {a, b, c}};
}

// Env action layout: A = total logit columns (sum of head sizes; continuous
// heads are size 1), continuous if every head is size 1.
static const int ACT_SZ[] = ACT_SIZES;
static int act_total() { int a = 0; for (int h = 0; h < NUM_ATNS; h++) a += ACT_SZ[h]; return a; }
static bool act_continuous() { for (int h = 0; h < NUM_ATNS; h++) if (ACT_SZ[h] != 1) return false; return true; }
static void fill_actions(float* act, long rows) {
    for (long i = 0; i < rows; i++)
        for (int h = 0; h < NUM_ATNS; h++)
            act[i * NUM_ATNS + h] = ACT_SZ[h] == 1 ? urand() : (float)((i * 7 + h * 3) % ACT_SZ[h]);
}

int main() {
    cublas_init_handle();
    const int B = 96, T = 32, H = 40, A = act_total();
    const bool CONT = act_continuous();
    printf("env %s: NUM_ATNS=%d A_total=%d %s\n", PUFFER_ENV_NAME, NUM_ATNS, A,
        CONT ? "continuous" : "discrete");

    {   // mingru_gate
        Case c;
        c.zeros(B * H); c.zeros(B * H);
        c.add(B * 3 * H, -4, 4); c.add(B * H); c.add(B * H);
        diff("mingru_gate", c, [&](std::vector<Buf>& b) {
            PUF_LAUNCH(mingru_gate, grid_size(B * H), BLOCK_SIZE)(
                b[0].p, b[1].p, b[2].p, b[3].p, b[4].p, H, B);
        }, 1e-5);
    }
    {   // mingru scans fwd + bwd
        Case c;
        c.add((long)B * T * 3 * H, -4, 4);  // 0 combined
        c.add(B * H);                       // 1 state
        c.add((long)B * T * H);             // 2 input
        float* term = c.zeros((long)B * T); // 3 terminals
        for (long i = 0; i < (long)B * T; i++) term[i] = (i % 7 == 3) ? 1.0f : 0.0f;
        for (int k = 0; k < 6; k++) c.zeros((long)B * T * 3 * H);  // 4..9 scan bufs
        c.add((long)B * T * H);             // 10 grad_out
        c.add(B * H);                       // 11 grad_next_state
        auto scan_of = [&](std::vector<Buf>& b) {
            PrefixScan s = {};
            s.combined_ptr = b[0].p; s.state_ptr = b[1].p; s.input_ptr = b[2].p;
            s.terminals_ptr = b[3].p; s.B = B; s.T = T; s.H = H;
            s.scan_h = P(b[4].p, B, T, H); s.out = P(b[5].p, B, T, H);
            s.next_state = P(b[6].p, B, 1, H); s.grad_combined = P(b[7].p, B, T, 3 * H);
            s.grad_state = P(b[8].p, B, 1, H); s.grad_input = P(b[9].p, B, T, H);
            return s;
        };
        diff("mingru_scan_forward+backward", c, [&](std::vector<Buf>& b) {
            PrefixScan s = scan_of(b);
            PUF_LAUNCH(mingru_scan_forward, grid_size(B * H), BLOCK_SIZE)(s);
            PUF_LAUNCH(mingru_scan_backward, grid_size(B * H), BLOCK_SIZE)(s, b[10].p, b[11].p);
        }, 1e-5);
    }
    {   // sample_logits: same Philox streams -> identical actions
        Case c;
        int N = 4099;
        c.add((long)N * (A + 1), -3, 3);   // 0 dec_out
        c.zeros((long)N * NUM_ATNS);       // 1 actions
        c.zeros((long)N * NUM_ATNS);       // 2 env actions
        c.zeros(N); c.zeros(N);            // 3 logprobs, 4 values
        c.add((long)N * A, 1, 1);          // 5 mask (all ones)
        long rng_floats = (long)N * sizeof(curandStatePhilox4_32_10_t) / 4;
        c.zeros(rng_floats);               // 6 rng states
        c.add(NUM_ATNS, -1.0f, 0.5f);      // 7 logstd (continuous only)
        int* act_sizes;
        cudaMalloc(&act_sizes, sizeof(int) * NUM_ATNS);
        int as[] = ACT_SIZES;
        memcpy(act_sizes, as, sizeof(as));
        curandStatePhilox4_32_10_t* st = (curandStatePhilox4_32_10_t*)c.bufs[6].p;
        for (int i = 0; i < N; i++) curand_init(7, i, 0, &st[i]);
        diff("sample_logits (x3 steps)", c, [&](std::vector<Buf>& b) {
            for (int step = 0; step < 3; step++) {
                Prec logstd = CONT ? P(b[7].p, 1, NUM_ATNS) : Prec{};
                PUF_LAUNCH(sample_logits, grid_size(N), BLOCK_SIZE)(
                    P(b[0].p, N, A + 1), logstd, act_sizes, b[1].p, b[2].p, b[3].p,
                    b[4].p, (curandStatePhilox4_32_10_t*)b[6].p, b[5].p, A);
            }
        }, 1e-5);
    }
    {   // puff_advantage
        Case c;
        int Nr = 100, Tr = 32;
        c.add(Nr * Tr); c.add(Nr * Tr);
        float* d = c.zeros(Nr * Tr);
        for (int i = 0; i < Nr * Tr; i++) d[i] = (i % 11 == 0) ? 1.0f : 0.0f;
        c.add(Nr * Tr, 0.5f, 1.5f); c.zeros(Nr * Tr); c.zeros(Nr * Tr);
        diff("puff_advantage", c, [&](std::vector<Buf>& b) {
            PUF_LAUNCH(puff_advantage, (Nr + 63) / 64, 64)(b[0].p, b[1].p, b[2].p, b[3].p,
                b[4].p, b[5].p, 0.97f, 0.95f, 1.0f, 1.0f, Nr, Tr);
        }, 1e-5);
    }
    {   // muon norm chain
        Case c;
        int n = 50000;
        c.add(n); c.zeros(256); c.zeros(1); c.zeros(n); float* lr = c.zeros(1); lr[0] = 0.01f;
        c.add(n);
        diff("muon sum_sq/clip/norm/update", c, [&](std::vector<Buf>& b) {
            int blocks = min(grid_size(n), 256);
            PUF_LAUNCH(muon_sum_sq_partials, blocks, 256)(b[1].p, b[0].p, n);
            PUF_LAUNCH(muon_sum_sq_reduce, 1, 256)(b[2].p, b[1].p, blocks);
            PUF_LAUNCH(muon_clip_nesterov, grid_size(n), BLOCK_SIZE)(
                b[3].p, b[0].p, b[2].p, 1.0f, 1e-6f, 0.9f, n);
            PUF_LAUNCH(muon_l2_normalize, grid_size(n), BLOCK_SIZE)(b[0].p, b[2].p, 1e-7f, n);
            PUF_LAUNCH(muon_store_update, grid_size(n), BLOCK_SIZE)(b[0].p, b[0].p, 1.5f, n);
            PUF_LAUNCH(muon_weight_update, grid_size(n), BLOCK_SIZE)(b[5].p, b[0].p, b[4].p, 0.0f, n);
        }, 1e-5);
    }
    {   // cache_imp_and_v + ppo_loss_compute + reduce
        Case c;
        int Nb = 64, Tb = 32, NT = Nb * Tb;
        c.add((long)NT * (A + 1), -3, 3);     // 0 dec
        float* act = c.zeros((long)NT * NUM_ATNS);  // 1 actions
        fill_actions(act, NT);
        c.add(NT, -1.5f, -0.5f);               // 2 old logprobs
        c.add((long)NT * A, 1, 1);             // 3 mask
        c.zeros(NT); c.zeros(NT);              // 4 imp, 5 value
        c.zeros((long)NT * A);                 // 6 logps / grad_logits
        c.zeros(NT);                           // 7 new_lp / grad_values
        c.add(NT); c.add(NT); c.add(NT);       // 8 adv, 9 values, 10 returns
        int grid = (NT + PPO_THREADS - 1) / PPO_THREADS;
        c.zeros((long)grid * LOSS_N);          // 11 partials
        c.zeros(NUM_LOSSES);                   // 12 losses
        float* ent = c.zeros(1); ent[0] = 0.01f;  // 13 ent_coef
        c.add(NUM_ATNS, -1.0f, 0.5f);          // 14 logstd (continuous)
        c.zeros((long)NT * NUM_ATNS);          // 15 grad_logstd (continuous)
        c.zeros(NUM_ATNS);                     // 16 summed logstd grad
        int* act_sizes;
        cudaMalloc(&act_sizes, sizeof(int) * NUM_ATNS);
        int as[] = ACT_SIZES;
        memcpy(act_sizes, as, sizeof(as));
        diff("cache_imp + ppo_loss", c, [&](std::vector<Buf>& b) {
            Prec dec = P(b[0].p, Nb, Tb, A + 1);
            Prec logstd = CONT ? P(b[14].p, 1, NUM_ATNS) : Prec{};
            PUF_LAUNCH(cache_imp_and_v, grid_size(NT), BLOCK_SIZE)(dec, b[1].p, b[2].p,
                b[3].p, logstd, act_sizes, b[4].p, b[5].p, b[6].p, b[7].p);
            PPOKernelArgs ka = {
                .grad_logits = b[6].p, .grad_logstd = CONT ? b[15].p : NULL,
                .grad_values_pred = b[7].p,
                .logits = b[0].p, .logstd = CONT ? b[14].p : NULL, .values_pred = b[0].p + A,
                .act_sizes = act_sizes, .action_mask = b[3].p, .num_atns = NUM_ATNS,
                .clip_coef = 0.2f, .vf_clip_coef = 0.5f, .vf_coef = 0.5f,
                .ent_coef = b[13].p, .T_seq = Tb, .A_total = A, .N = Nb,
                .is_continuous = CONT,
            };
            PPOGraphArgs ga = {.imp = b[4].p, .actions = b[1].p, .old_logprobs = b[2].p,
                .advantages = b[8].p, .values = b[9].p, .returns = b[10].p};
            PUF_LAUNCH(ppo_loss_compute, grid, PPO_THREADS)(b[11].p, ka, ga);
            PUF_LAUNCH(ppo_loss_reduce, 1, LOSS_N)(b[12].p, b[11].p, grid);
            if (CONT) {  // decoder backward: logstd grad summed over rows
                PUF_LAUNCH(sum_rows_to_precision_kernel, grid_size(NUM_ATNS), BLOCK_SIZE)(
                    b[16].p, b[15].p, NT, NUM_ATNS);
            }
        }, 1e-5);
    }
    {   // small elementwise kernels
        Case c;
        int L = 3, Na = 200, Hh = 24, n = L * Na * Hh;
        c.add(n);                                   // 0 state
        float* t = c.zeros(Na);                     // 1 terminals
        for (int i = 0; i < Na; i++) t[i] = (i % 5 == 0);
        c.zeros(2L * n);                            // 2 snapshot dst
        c.add(n); c.add(n);                         // 3,4 add dst/src
        c.zeros(n); c.add(n); c.zeros(n); c.add(n); // 5..8 cast_rew_term
        c.zeros(n); c.add(n);                       // 9,10 transpose
        diff("zero_term/snapshot/add/cast/T", c, [&](std::vector<Buf>& b) {
            Prec st = P(b[0].p, L, Na, Hh);
            Float term = {.data = b[1].p, .shape = {Na}};
            PUF_LAUNCH(zero_term_state, grid_size(n), BLOCK_SIZE)(st, term, 0, 0, Na);
            PUF_LAUNCH(snapshot_state, grid_size(n), BLOCK_SIZE)(P(b[2].p, L, 2 * Na, Hh), st, 7, Na);
            PUF_LAUNCH(add_kernel, grid_size(n), BLOCK_SIZE)(b[3].p, b[4].p, n);
            PUF_LAUNCH(cast_rew_term, grid_size(n), BLOCK_SIZE)(b[5].p, b[6].p, b[7].p, b[8].p, n);
            PUF_LAUNCH(transpose_102, grid_size(n), BLOCK_SIZE)(b[9].p, b[10].p, L, Na, Hh);
            PUF_LAUNCH(clamp_precision_kernel, grid_size(n), BLOCK_SIZE)(b[10].p, -0.5f, 0.5f, n);
        }, 1e-6);
    }
    {   // GEMM wrappers on the Metal path (gemm_tile below the size threshold,
        // MPS above) vs host cblas on the same inputs.
        int shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {64, 17, 33}, {130, 96, 258},
            {256, 256, 256}, {300, 768, 256}, {2048, 768, 256}};
        for (auto& sh : shapes) {
            int M = sh[0], N = sh[1], K = sh[2];
            for (int form = 0; form < 3; form++) {
                for (float beta : {0.0f, 0.5f}) {
                    float alpha = form == 2 ? -1.25f : 0.75f;
                    // form 0: puf_mm a(M,K) b(N,K)^T; 1: puf_mm_tn a(K,M)^T b(K,N); 2: puf_mm_nn
                    long na = (long)M * K, nb = (long)N * K, nc = (long)M * N;
                    float *A, *Bm, *C;
                    cudaMalloc(&A, na * 4); cudaMalloc(&Bm, nb * 4); cudaMalloc(&C, nc * 4);
                    for (long i = 0; i < na; i++) A[i] = urand();
                    for (long i = 0; i < nb; i++) Bm[i] = urand();
                    for (long i = 0; i < nc; i++) C[i] = urand();
                    float* ref = (float*)malloc(nc * 4);
                    memcpy(ref, C, nc * 4);
                    Prec pa, pb, pc = P(C, M, N);
                    if (form == 0) { pa = P(A, M, K); pb = P(Bm, N, K); }
                    else if (form == 1) { pa = P(A, K, M); pb = P(Bm, K, N); }
                    else { pa = P(A, M, K); pb = P(Bm, K, N); }
                    // host reference, row-major
                    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
                        double acc = 0;
                        for (int k = 0; k < K; k++) {
                            float av = form == 1 ? A[(long)k * M + i] : A[(long)i * K + k];
                            float bv = form == 0 ? Bm[(long)j * K + k] : Bm[(long)k * N + j];
                            acc += (double)av * bv;
                        }
                        ref[(long)i * N + j] = alpha * acc + beta * ref[(long)i * N + j];
                    }
                    if (form == 0) puf_mm(&pa, &pb, &pc, 0, alpha, beta);
                    else if (form == 1) puf_mm_tn(&pa, &pb, &pc, 0, alpha, beta);
                    else puf_mm_nn(&pa, &pb, &pc, 0, alpha, beta);
                    cudaDeviceSynchronize();
                    double md = 0, mx = 1e-30;
                    for (long i = 0; i < nc; i++) {
                        md = fmax(md, fabs(ref[i] - C[i]));
                        mx = fmax(mx, fabs(ref[i]));
                    }
                    bool ok = md / mx < 1e-5;
                    failures += !ok;
                    if (!ok) printf("FAIL gemm form=%d M=%d N=%d K=%d beta=%.1f rel %.3e\n",
                        form, M, N, K, beta, md / mx);
                    free(ref);
                }
            }
        }
        // Muon gram: same buffer as both operands
        float* X;
        int R = 96, Cc = 288;
        cudaMalloc(&X, (long)R * Cc * 4);
        for (long i = 0; i < (long)R * Cc; i++) X[i] = urand();
        float* G;
        cudaMalloc(&G, (long)R * R * 4);
        Prec px = P(X, R, Cc), pg = P(G, R, R);
        puf_mm(&px, &px, &pg, 0);
        cudaDeviceSynchronize();
        double md = 0;
        for (int i = 0; i < R; i++) for (int j = 0; j < R; j++) {
            double acc = 0;
            for (int k = 0; k < Cc; k++) acc += (double)X[i * Cc + k] * X[j * Cc + k];
            md = fmax(md, fabs(acc - G[i * R + j]) / fmax(1.0, fabs(acc)));
        }
        failures += md >= 1e-5;
        printf("%s gemm forms x shapes (tile + MPS), muon gram rel %.3e\n",
            md < 1e-5 ? "ok  " : "FAIL", md);
    }
    printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures != 0;
}
