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

int main() {
    cublas_init_handle();
    const int B = 96, T = 32, H = 40, A = 3;

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
        int* act_sizes;
        cudaMalloc(&act_sizes, sizeof(int) * NUM_ATNS);
        int as[] = ACT_SIZES;
        memcpy(act_sizes, as, sizeof(as));
        curandStatePhilox4_32_10_t* st = (curandStatePhilox4_32_10_t*)c.bufs[6].p;
        for (int i = 0; i < N; i++) curand_init(7, i, 0, &st[i]);
        diff("sample_logits (x3 steps)", c, [&](std::vector<Buf>& b) {
            for (int step = 0; step < 3; step++) {
                PUF_LAUNCH(sample_logits, grid_size(N), BLOCK_SIZE)(
                    P(b[0].p, N, A + 1), Prec{}, act_sizes, b[1].p, b[2].p, b[3].p,
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
        for (int i = 0; i < NT; i++) act[i] = (float)(i % A);
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
        int* act_sizes;
        cudaMalloc(&act_sizes, sizeof(int) * NUM_ATNS);
        int as[] = ACT_SIZES;
        memcpy(act_sizes, as, sizeof(as));
        diff("cache_imp + ppo_loss", c, [&](std::vector<Buf>& b) {
            Prec dec = P(b[0].p, Nb, Tb, A + 1);
            PUF_LAUNCH(cache_imp_and_v, grid_size(NT), BLOCK_SIZE)(dec, b[1].p, b[2].p,
                b[3].p, Prec{}, act_sizes, b[4].p, b[5].p, b[6].p, b[7].p);
            PPOKernelArgs ka = {
                .grad_logits = b[6].p, .grad_logstd = NULL, .grad_values_pred = b[7].p,
                .logits = b[0].p, .logstd = NULL, .values_pred = b[0].p + A,
                .act_sizes = act_sizes, .action_mask = b[3].p, .num_atns = NUM_ATNS,
                .clip_coef = 0.2f, .vf_clip_coef = 0.5f, .vf_coef = 0.5f,
                .ent_coef = b[13].p, .T_seq = Tb, .A_total = A, .N = Nb,
                .is_continuous = false,
            };
            PPOGraphArgs ga = {.imp = b[4].p, .actions = b[1].p, .old_logprobs = b[2].p,
                .advantages = b[8].p, .values = b[9].p, .returns = b[10].p};
            PUF_LAUNCH(ppo_loss_compute, grid, PPO_THREADS)(b[11].p, ka, ga);
            PUF_LAUNCH(ppo_loss_reduce, 1, LOSS_N)(b[12].p, b[11].p, grid);
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
    printf(failures ? "FAILED: %d\n" : "OK\n", failures);
    return failures != 0;
}
