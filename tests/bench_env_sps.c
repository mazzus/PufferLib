// Env-only SPS for any .h env: the trainer's rollout env loop without a model.
// Mirrors env_setup / vec_thread_main in src/pufferl.cu: one buffer, contiguous
// per-agent obs/action/reward/terminal arrays, memset rewards+terminals, then
// omp parallel for puf_step over envs. Actions are random each step.
//
// Build (from repo root, after ./build.sh ENV --cpu has fetched raylib):
//   ./tests/bench_env_sps.sh breakout
// Run:
//   ./build/bench_env_sps_breakout [--vec.total_agents=N] [--vec.num_threads=N]
//       [--bench.steps=N] [--bench.warmup=N] [other --section.key=value]

#include <assert.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include ENV_HEADER

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t xorshift(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

int main(int argc, char** argv) {
    int steps = 256;
    int warmup = 32;
    char* ini_argv[argc];
    int ini_argc = 0;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--bench.steps=", 14) == 0) {
            steps = atoi(argv[i] + 14);
        } else if (strncmp(argv[i], "--bench.warmup=", 15) == 0) {
            warmup = atoi(argv[i] + 15);
        } else {
            ini_argv[ini_argc++] = argv[i];
        }
    }
    Ini ini = {0};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, ini_argc, ini_argv);
    Dict* vk = puf_ini_section(&ini, "vec", 0);
    Dict* ek = puf_ini_section(&ini, "env", 0);
    int total_agents = dict_get(vk, "total_agents");
    int num_threads = dict_get(vk, "num_threads");
    assert(total_agents > 0 && num_threads > 0 && steps > 0);
    omp_set_dynamic(0);

    int act_sizes[] = ACT_SIZES;
    int num_acts = sizeof(act_sizes) / sizeof(act_sizes[0]);
    int mask_size = 0;
    int continuous = 1;
    for (int i = 0; i < num_acts; i++) {
        mask_size += act_sizes[i];
        if (act_sizes[i] != 1) {
            continuous = 0;
        }
    }

    double t_init = now_s();
    int num_envs = 0;
#ifdef MY_VEC_INIT
    int env_start = 0;
    int env_count = 0;
    Dict vk_one = {0};
    dict_copy(&vk_one, vk);
    dict_set(&vk_one, "num_buffers", 1);
    Env* envs = my_vec_init(&num_envs, &env_start, &env_count, &vk_one, ek);
#else
    Env* envs = (Env*)calloc(total_agents, sizeof(Env));
    int agents_created = 0;
    while (agents_created < total_agents) {
        envs[num_envs].rng = num_envs;
        puf_init(&envs[num_envs], ek);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }
#endif
    int n_agents = 0;
    for (int e = 0; e < num_envs; e++) {
        n_agents += envs[e].num_agents;
    }
    obs_t* obs = (obs_t*)calloc((size_t)n_agents * OBS_SIZE, sizeof(obs_t));
    float* actions = (float*)calloc((size_t)n_agents * NUM_ATNS, sizeof(float));
    float* rewards = (float*)calloc(n_agents, sizeof(float));
    float* terminals = (float*)calloc(n_agents, sizeof(float));
    unsigned char* mask = (unsigned char*)malloc((size_t)n_agents * mask_size + 1);
    memset(mask, 1, (size_t)n_agents * mask_size + 1);
    int phys = 0;
    for (int e = 0; e < num_envs; e++) {
        for (int s = 0; s < envs[e].num_agents; s++) {
            Agent* a = &envs[e].agents[s];
            a->observations = obs + (size_t)phys * OBS_SIZE;
            a->actions = actions + (size_t)phys * NUM_ATNS;
            a->rewards = rewards + phys;
            a->terminals = terminals + phys;
            a->action_mask = mask + (size_t)phys * mask_size;
            phys++;
        }
    }
    for (int e = 0; e < num_envs; e++) {
        puf_reset(&envs[e]);
    }
    t_init = now_s() - t_init;

    uint32_t* rngs = (uint32_t*)malloc(num_envs * sizeof(uint32_t));
    for (int e = 0; e < num_envs; e++) {
        rngs[e] = 0x9E3779B9u ^ (uint32_t)(e + 1);
    }
    double t0 = 0.0;
    for (int it = 0; it < warmup + steps; it++) {
        if (it == warmup) {
            t0 = now_s();
        }
        memset(rewards, 0, n_agents * sizeof(float));
        memset(terminals, 0, n_agents * sizeof(float));
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int e = 0; e < num_envs; e++) {
            for (int s = 0; s < envs[e].num_agents; s++) {
                float* atn = envs[e].agents[s].actions;
                for (int k = 0; k < num_acts; k++) {
                    uint32_t r = xorshift(&rngs[e]);
                    atn[k] = continuous
                        ? (float)(r % 2001) / 1000.0f - 1.0f
                        : (float)(r % (uint32_t)act_sizes[k]);
                }
            }
            puf_step(&envs[e]);
        }
    }
    double dt = now_s() - t0;
    double sps = (double)n_agents * steps / dt;
    printf("BENCH env=%s agents=%d envs=%d threads=%d steps=%d init_s=%.3f "
        "step_s=%.3f sps=%.0f\n", PUFFER_ENV_NAME, n_agents, num_envs,
        num_threads, steps, t_init, dt, sps);
    return 0;
}
