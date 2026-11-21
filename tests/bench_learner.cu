// Learner-only throughput: repeated train epochs on a synthetic rollout (no
// envs, async=0). In async training the learner is the critical path on Metal,
// so learner SPS (= agent-steps per epoch / epoch time) bounds training SPS.
// Much less noisy than end-to-end runs. Single source (CUDA or CPU/Metal).
//
//   ./tests/oracle_train.sh-style build: see tests/bench_learner.sh
//   ./build/bench_learner_ENV [EPOCHS] [--section.key=value ...]
#include "pufferl.cu"

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    int epochs = argc > 1 ? atoi(argv[1]) : 20;
    const char* defaults[] = {"--base.async=0", "--train.anneal_lr=0"};
    char* ini_argv[64];
    int ini_argc = 0;
    for (const char* d : defaults) ini_argv[ini_argc++] = (char*)d;
    for (int i = 2; i < argc; i++) ini_argv[ini_argc++] = argv[i];
    Ini ini = {0};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, ini_argc, ini_argv);
    TrainContext ctx = {.world_size = 1, .artifact_owner = 1};
    PuffeRL* p = create_pufferl(&ini, &ctx);
    RolloutBuf* rb = &p->rollouts;
    // Plausible rollout: obs in [-1,1], valid actions, sparse rewards/terminals
    long n_obs = numel(rb->observations.shape);
    float* h = (float*)malloc(n_obs * sizeof(float));
    uint32_t s = 1;
    for (long i = 0; i < n_obs; i++) {
        s = s * 1664525u + 1013904223u;
        h[i] = (float)(s >> 8) / 8388608.0f - 1.0f;
    }
    cudaMemcpy(rb->observations.data, h, n_obs * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    long nt = numel(rb->rewards.shape);
    float* r = (float*)calloc(nt, sizeof(float));
    for (long i = 0; i < nt; i++) r[i] = (i % 97 == 0) ? 1.0f : 0.0f;
    cudaMemcpy(rb->rewards.data, r, nt * sizeof(float), cudaMemcpyHostToDevice);
    for (long i = 0; i < nt; i++) r[i] = (i % 211 == 0) ? 1.0f : 0.0f;
    cudaMemcpy(rb->terminals.data, r, nt * sizeof(float), cudaMemcpyHostToDevice);
    for (long i = 0; i < nt; i++) r[i] = -1.0f;
    cudaMemcpy(rb->logprobs.data, r, nt * sizeof(float), cudaMemcpyHostToDevice);
    free(r);
    long nm = numel(rb->action_mask.shape);
    float* ones = (float*)malloc(nm * sizeof(float));
    for (long i = 0; i < nm; i++) ones[i] = 1.0f;
    cudaMemcpy(rb->action_mask.data, ones, nm * sizeof(float), cudaMemcpyHostToDevice);
    free(ones);
    cudaDeviceSynchronize();

    long steps_per_epoch = (long)p->hypers.total_agents * p->hypers.horizon;
    train_impl(p, NULL);  // warmup
    cudaDeviceSynchronize();
    double best = 1e9, total = 0;
    for (int e = 0; e < epochs; e++) {
        double t0 = now_s();
        train_impl(p, NULL);
        cudaDeviceSynchronize();
        double dt = now_s() - t0;
        best = fmin(best, dt);
        total += dt;
    }
    double mean = total / epochs;
    printf("BENCH_LEARNER env=%s epochs=%d mean_epoch=%.4fs best_epoch=%.4fs "
        "learner_sps_mean=%.0f learner_sps_best=%.0f\n", PUFFER_ENV_NAME, epochs,
        mean, best, steps_per_epoch / mean, steps_per_epoch / best);
    fflush(stdout);
    _exit(0);
}
