// Backend parity oracle. Single source: builds with nvcc (CUDA) and with
// clang++ -DPUF_TRAIN_CPU (CPU backend). With identical host-generated weights
// and a synthetic rollout, runs one rollout forward and one train epoch, then
// dumps decoder outputs, losses and updated weights. Compare the dumps of two
// backends with tests/oracle_compare.py.
//
//   ./tests/oracle_train.sh            # builds build/oracle_train for this machine
//   ./build/oracle_train_ENV out.bin [--section.key=value ...]
//
// Precision: build both sides fp32 (CUDA: --float).
#include "pufferl.cu"

static uint32_t lcg(uint32_t* s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}
static float lcg_unit(uint32_t* s) {  // [-1, 1)
    return (float)(lcg(s) >> 8) / 8388608.0f - 1.0f;
}

template <class T> static void upload(T* dst, const float* src, long n) {
    T* tmp = (T*)malloc(n * sizeof(T));
    for (long i = 0; i < n; i++) tmp[i] = (T)src[i];
    cudaMemcpy(dst, tmp, n * sizeof(T), cudaMemcpyHostToDevice);
    free(tmp);
}
template <class T> static void download(float* dst, const T* src, long n) {
    T* tmp = (T*)malloc(n * sizeof(T));
    cudaMemcpy(tmp, src, n * sizeof(T), cudaMemcpyDeviceToHost);
    for (long i = 0; i < n; i++) dst[i] = (float)tmp[i];
    free(tmp);
}

static void fill(Prec* t, uint32_t seed, float lo, float hi) {
    long n = numel(t->shape);
    float* h = (float*)malloc(n * sizeof(float));
    for (long i = 0; i < n; i++) h[i] = lo + (hi - lo) * 0.5f * (lcg_unit(&seed) + 1.0f);
    upload(t->data, h, n);
    free(h);
}

static void write_block(FILE* fp, const char* name, const float* data, long n) {
    char tag[32] = {0};
    snprintf(tag, sizeof(tag), "%s", name);
    fwrite(tag, 1, 32, fp);
    int64_t nn = n;
    fwrite(&nn, sizeof(nn), 1, fp);
    fwrite(data, sizeof(float), n, fp);
}

int main(int argc, char** argv) {
    assert(argc >= 2 && "usage: oracle_train OUT.bin [--k=v ...]");
    const char* defaults[] = {
        "--base.async=0", "--base.cudagraphs=-1", "--base.seed=1",
        "--train.anneal_lr=0", "--train.anneal_ent_coef=0",
    };
    int n_def = sizeof(defaults) / sizeof(defaults[0]);
    char* ini_argv[64];
    int ini_argc = 0;
    for (int i = 0; i < n_def; i++) ini_argv[ini_argc++] = (char*)defaults[i];
    for (int i = 2; i < argc; i++) ini_argv[ini_argc++] = argv[i];
    Ini ini = {0};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, ini_argc, ini_argv);
    TrainContext ctx = {.world_size = 1, .artifact_owner = 1};
    PuffeRL* p = create_pufferl(&ini, &ctx);
    Hypers* hy = &p->hypers;
    Policy* pol = &p->policies[0];

    // Weights: deterministic, Kaiming-uniform-like scale per registered tensor
    long nw = numel(pol->master_weights.shape);
    float* w = (float*)malloc(nw * sizeof(float));
    long off = 0;
    uint32_t ws = 12345;
    for (int r = 0; r < pol->params_alloc.num_regs; r++) {
        int64_t* sh = pol->params_alloc.regs[r].shape;
        long ne = numel(sh);
        float fan_in = ndim(sh) >= 2 ? (float)(ne / sh[0]) : 1.0f;
        float bound = 1.0f / sqrtf(fan_in);
        for (long i = 0; i < ne; i++) w[off + i] = bound * lcg_unit(&ws);
        off += ne;
    }
    assert(off == nw);
    // ORACLE_PERTURB=eps: relative weight perturbation, to measure how much
    // the algorithm itself amplifies tiny numeric differences.
    const char* pert = getenv("ORACLE_PERTURB");
    if (pert) {
        float eps = atof(pert);
        uint32_t ps = 999;
        for (long i = 0; i < nw; i++) w[i] *= 1.0f + eps * lcg_unit(&ps);
    }
    upload(pol->master_weights.data, w, nw);

    // Synthetic rollout in the (T, B, ...) rollout layout
    RolloutBuf* rb = &p->rollouts;
    int T = rb->observations.shape[0], B = rb->observations.shape[1];
    fill(&rb->observations, 1, -1.0f, 1.0f);
    fill(&rb->values, 2, -0.5f, 0.5f);
    {
        long n = (long)T * B;
        float* lp = (float*)malloc(n * sizeof(float));
        float* rew = (float*)malloc(n * sizeof(float));
        float* term = (float*)malloc(n * sizeof(float));
        long na = numel(rb->actions.shape);
        float* act = (float*)malloc(na * sizeof(float));
        int act_sizes[] = ACT_SIZES;
        uint32_t s = 3;
        for (long i = 0; i < n; i++) {
            uint32_t r = lcg(&s);
            rew[i] = (r % 17 == 0) ? 1.0f : (r % 23 == 0 ? -1.0f : 0.0f);
            term[i] = (lcg(&s) % 50 == 0) ? 1.0f : 0.0f;
            float lpi = 0.0f;
            for (int h = 0; h < NUM_ATNS; h++) {
                act[i * NUM_ATNS + h] = (float)(lcg(&s) % (uint32_t)act_sizes[h]);
                lpi += logf(1.0f / act_sizes[h]) + 0.1f * lcg_unit(&s);
            }
            lp[i] = lpi;
        }
        upload(rb->logprobs.data, lp, n);
        upload(rb->rewards.data, rew, n);
        upload(rb->terminals.data, term, n);
        cudaMemcpy(rb->actions.data, act, na * sizeof(float), cudaMemcpyHostToDevice);
        free(lp); free(rew); free(term); free(act);
        long nm = numel(rb->action_mask.shape);
        float* ones = (float*)malloc(nm * sizeof(float));
        for (long i = 0; i < nm; i++) ones[i] = 1.0f;
        upload(rb->action_mask.data, ones, nm);
        free(ones);
    }
    if (rb->initial_states.data) {
        cudaMemset(rb->initial_states.data, 0,
            numel(rb->initial_states.shape) * sizeof(precision_t));
    }
    cudaDeviceSynchronize();

    FILE* fp = fopen(argv[1], "wb");
    assert(fp);
    write_block(fp, "weights_in", w, nw);

    // Rollout forward: first buffer's agents at t=0 with zero state
    int nfwd = (int)pol->buffer_states[0].shape[1];
    Prec obs0 = puf_slice(rb->observations, 0, 0, nfwd);
    Prec* st = &pol->buffer_states[0];
    cudaMemset(st->data, 0, numel(st->shape) * sizeof(precision_t));
    Prec dec = arch_forward(&pol->arch, pol->weights, pol->buf_acts[0],
        obs0, *st, p->default_stream);
    cudaDeviceSynchronize();
    long nd = numel(dec.shape);
    float* hd = (float*)malloc(nd * sizeof(float));
    download(hd, dec.data, nd);
    write_block(fp, "rollout_dec", hd, nd);
    long ns = numel(st->shape);
    float* hs = (float*)malloc(ns * sizeof(float));
    download(hs, st->data, ns);
    write_block(fp, "rollout_state", hs, ns);

    // One train epoch on the synthetic rollout
    cudaMemset(p->losses, 0, NUM_LOSSES * sizeof(float));
    train_impl(p, NULL);
    cudaDeviceSynchronize();
    float losses[NUM_LOSSES];
    cudaMemcpy(losses, p->losses, sizeof(losses), cudaMemcpyDeviceToHost);
    write_block(fp, "losses", losses, NUM_LOSSES);
    float* w2 = (float*)malloc(nw * sizeof(float));
    download(w2, pol->master_weights.data, nw);
    write_block(fp, "weights_out", w2, nw);
    fclose(fp);
    printf("oracle: wrote %s (T=%d B=%d params=%ld minibatches=%d)\n", argv[1], T, B,
        nw, (int)(hy->replay_ratio * hy->total_agents * hy->horizon / hy->minibatch_size));
    printf("losses:");
    for (int i = 0; i < NUM_LOSSES; i++) printf(" %.6g", losses[i]);
    printf("\n");
    fflush(stdout);
    _exit(0);  // skip env worker teardown
}
