// Metal ports of the trainer's CUDA kernels (fp32). Each kernel mirrors its
// CUDA version in src/pufferl.cu / src/algo.cu line for line; the host-side
// argument struct layout is produced by PUF_METAL_PORT (src/puf_cuda_cpu.h):
// fields in declaration order with natural C alignment, pointers as device
// addresses, tensor views as {device float* data; long shape[8];}.
//
// Compiled at runtime by src/puf_metal.mm (safe math mode) with env macros
// PUF_NUM_ATNS and PUF_MAX_HEAD_A.
#include <metal_stdlib>
using namespace metal;

#ifndef PUF_NUM_ATNS
#define PUF_NUM_ATNS 1
#endif
#ifndef PUF_MAX_HEAD_A
#define PUF_MAX_HEAD_A 64
#endif

typedef float precision_t;
#define to_float(x) (x)
#define from_float(x) (x)

struct Prec { device float* data; long shape[8]; };
struct FloatT { device float* data; long shape[8]; };

#define KERNEL_IDX \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint3 threadIdx [[thread_position_in_threadgroup]], \
    uint3 blockDim [[threads_per_threadgroup]], \
    uint3 gridDim [[threadgroups_per_grid]]

static inline long numel(constant long* shape) {
    long n = 1;
    for (int i = 0; i < 8 && shape[i] != 0; i++) n *= shape[i];
    return n;
}

// Deterministic tree reduce, same order as CUDA block_reduce_sum.
static inline void block_reduce_sum(threadgroup float* smem, device float* out,
        int tid, int nthreads, int nchan) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < nchan; c++) {
                smem[c * nthreads + tid] += smem[c * nthreads + tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        for (int c = 0; c < nchan; c++) out[c] = smem[c * nthreads];
    }
}

// ---- casts / copies -------------------------------------------------------

struct CastPF { device float* dst; device float* src; int n; };
kernel void cast_p_f(constant CastPF& a [[buffer(0)]], KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) a.dst[idx] = a.src[idx];
}
struct CastPU8 { device float* dst; device uchar* src; int n; };
kernel void cast_p_u8(constant CastPU8& a [[buffer(0)]], KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) a.dst[idx] = (float)a.src[idx];
}
struct CastU8P { device uchar* dst; device float* src; int n; };
kernel void cast_u8_p(constant CastU8P& a [[buffer(0)]], KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // CUDA: unsigned char = float (C conversion, truncation)
    if (idx < a.n) a.dst[idx] = (uchar)a.src[idx];
}
struct CastRewTerm {
    device float* rew_dst; device float* rew_src;
    device float* term_dst; device float* term_src; int n;
};
kernel void cast_rew_term(constant CastRewTerm& a [[buffer(0)]], KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) {
        a.rew_dst[idx] = a.rew_src[idx];
        a.term_dst[idx] = a.term_src[idx];
    }
}

struct Transpose { device float* dst; device float* src; int A, B, C; };
kernel void transpose_102_p(constant Transpose& a [[buffer(0)]], KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = a.A * a.B * a.C;
    if (idx >= total) return;
    int ai = idx / (a.B * a.C);
    int rem = idx % (a.B * a.C);
    int b = rem / a.C;
    int c = rem % a.C;
    a.dst[b * a.A * a.C + ai * a.C + c] = a.src[idx];
}

struct Clamp { device float* dst; float lo, hi; int n; };
kernel void clamp_precision_kernel(constant Clamp& a [[buffer(0)]], KERNEL_IDX) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < a.n;
            idx += blockDim.x * gridDim.x) {
        a.dst[idx] = fmin(fmax(a.dst[idx], a.lo), a.hi);
    }
}

struct AddArgs { device float* dst; device float* src; int n; };
kernel void add_kernel(constant AddArgs& a [[buffer(0)]], KERNEL_IDX) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < a.n;
            idx += blockDim.x * gridDim.x) {
        a.dst[idx] = a.dst[idx] + a.src[idx];
    }
}

// ---- RNN state ------------------------------------------------------------

static inline long state_elem_idx(int layer, int agents_stride, int state_row, int h, int H) {
    return ((long)layer * agents_stride + state_row) * H + h;
}

struct Snapshot { Prec dst; Prec src; int dst_start, count; };
kernel void snapshot_state(constant Snapshot& a [[buffer(0)]], KERNEL_IDX) {
    int L = a.src.shape[0];
    int H = a.src.shape[2];
    int total = L * a.count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = idx % H;
    int rel = (idx / H) % a.count;
    int layer = idx / (a.count * H);
    long src_i = state_elem_idx(layer, (int)a.src.shape[1], rel, h, H);
    long dst_i = state_elem_idx(layer, (int)a.dst.shape[1], a.dst_start + rel, h, H);
    a.dst.data[dst_i] = a.src.data[src_i];
}

struct ZeroTerm { Prec state; FloatT terminals; int state_start, terminal_start, count; };
kernel void zero_term_state(constant ZeroTerm& a [[buffer(0)]], KERNEL_IDX) {
    int L = a.state.shape[0];
    int H = a.state.shape[2];
    int total = L * a.count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = idx % H;
    int rel = (idx / H) % a.count;
    int layer = idx / (a.count * H);
    if (a.terminals.data[a.terminal_start + rel] == 0.0f) return;
    long i = state_elem_idx(layer, (int)a.state.shape[1], a.state_start + rel, h, H);
    a.state.data[i] = 0.0f;
}

// ---- MinGRU -----------------------------------------------------------------

static inline float sigmoid_f(float x) {
    float z = exp(-fabs(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}
static inline float lerp_f(float a, float b, float w) {
    float diff = b - a;
    return (fabs(w) < 0.5f) ? a + w * diff : b - diff * (1.0f - w);
}

struct Gate {
    device float* out; device float* next_state; device float* combined;
    device float* state_in; device float* x_in; int H, B;
};
kernel void mingru_gate(constant Gate& a [[buffer(0)]], KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H = a.H;
    int N = a.B * H;
    if (idx >= N) return;
    int b = idx / H;
    int h = idx % H;
    int combined_base = b * 3 * H;
    float hidden = a.combined[combined_base + h];
    float gate = a.combined[combined_base + H + h];
    float proj = a.combined[combined_base + 2 * H + h];
    float state = a.state_in[idx];
    float x = a.x_in[idx];
    float z = sigmoid_f(gate);
    float h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid_f(hidden);
    float h_out = lerp_f(state, h_tilde, z);
    a.next_state[idx] = h_out;
    float s = sigmoid_f(proj);
    a.out[idx] = s * h_out + (1.0f - s) * x;
}

struct PrefixScan {
    device float* combined_ptr; device float* state_ptr;
    device float* input_ptr; device float* terminals_ptr;
    int B, T, H;
    Prec scan_h, out, next_state, grad_combined, grad_state, grad_input;
};

struct ScanFwd { PrefixScan scan; };
kernel void mingru_scan_forward(constant ScanFwd& args [[buffer(0)]], KERNEL_IDX) {
    constant PrefixScan& scan = args.scan;
    int T_seq = scan.T, H = scan.H, B = scan.B;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    int b = idx / H;
    int h = idx % H;
    int bH = b * H;
    int H3 = 3 * H;
    int bHT = bH * T_seq;
    int out_base = bHT + h;
    int cbase = 3 * bHT;
    float h_t = scan.state_ptr[bH + h];
    int h_base = b * T_seq * H + h;
    device float* ch = scan.combined_ptr + cbase + h;
    device float* cg = scan.combined_ptr + cbase + H + h;
    device float* cp = scan.combined_ptr + cbase + 2 * H + h;
    int out_curr = out_base;
    int t_offset = 0;
    for (int t = 0; t < T_seq; t++) {
        if (scan.terminals_ptr != nullptr && scan.terminals_ptr[b * T_seq + t] != 0.0f) {
            h_t = 0.0f;
        }
        scan.scan_h.data[h_base + t * H] = h_t;
        float hidden_val = ch[t_offset];
        float gate_val = cg[t_offset];
        float proj_val = cp[t_offset];
        float x_val = scan.input_ptr[out_base + t * H];
        float z = sigmoid_f(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid_f(hidden_val);
        h_t = lerp_f(h_t, h_tilde, z);
        float ps = sigmoid_f(proj_val);
        scan.out.data[out_curr] = ps * h_t + (1.0f - ps) * x_val;
        out_curr += H;
        t_offset += H3;
    }
    scan.next_state.data[bH + h] = h_t;
}

struct ScanBwd { PrefixScan scan; device float* grad_out; device float* grad_next_state; };
kernel void mingru_scan_backward(constant ScanBwd& args [[buffer(0)]], KERNEL_IDX) {
    constant PrefixScan& scan = args.scan;
    int T_seq = scan.T, H = scan.H, B = scan.B;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    int b = idx / H;
    int h = idx % H;
    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int state_idx = b * H + h;
    int out_base = bHT + h;
    int h_base = b * T_seq * H + h;
    device float* ch = scan.combined_ptr + cbase + h;
    device float* cg = scan.combined_ptr + cbase + H + h;
    device float* cp = scan.combined_ptr + cbase + 2 * H + h;
    device float* gch = scan.grad_combined.data + cbase + h;
    device float* gcg = scan.grad_combined.data + cbase + H + h;
    device float* gcp = scan.grad_combined.data + cbase + 2 * H + h;
    float dh = args.grad_next_state[state_idx];
    for (int t = T_seq; t >= 1; --t) {
        int t0 = t - 1;
        int t_offset = t0 * H3;
        int input_idx = out_base + t0 * H;
        float h_prev = scan.scan_h.data[h_base + (t - 1) * H];
        float hidden_val = ch[t_offset];
        float gate_val = cg[t_offset];
        float proj_val = cp[t_offset];
        float x_val = scan.input_ptr[input_idx];
        float grad_out_val = args.grad_out[input_idx];
        float z = sigmoid_f(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid_f(hidden_val);
        float h_t = lerp_f(h_prev, h_tilde, z);
        float ps = sigmoid_f(proj_val);
        float dh_from_out = grad_out_val * ps;
        float grad_proj = grad_out_val * (h_t - x_val) * ps * (1.0f - ps);
        scan.grad_input.data[input_idx] = grad_out_val * (1.0f - ps);
        gcp[t_offset] = grad_proj;
        float dh_total = dh + dh_from_out;
        float d_h_prev = dh_total * (1.0f - z);
        float d_h_tilde = dh_total * z;
        float d_z = dh_total * (h_tilde - h_prev);
        float d_gate = d_z * z * (1.0f - z);
        float d_hidden = (hidden_val >= 0.0f) ? d_h_tilde : d_h_tilde * h_tilde * (1.0f - h_tilde);
        gch[t_offset] = d_hidden;
        gcg[t_offset] = d_gate;
        dh = d_h_prev;
        if (scan.terminals_ptr != nullptr && scan.terminals_ptr[b * T_seq + t0] != 0.0f) {
            dh = 0.0f;
        }
    }
    scan.grad_state.data[state_idx] = dh;
}

// ---- decoder ----------------------------------------------------------------

struct SumRows { device float* dst; device float* src; int R, C; };
kernel void sum_rows_to_precision_kernel(constant SumRows& a [[buffer(0)]], KERNEL_IDX) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.C) return;
    float sum = 0.0f;
    for (int r = 0; r < a.R; r++) sum += a.src[r * a.C + col];
    a.dst[col] = sum;
}

struct AsmGrad { device float* dst; device float* grad_logits; device float* grad_value; int B_TT, od, od_plus_1; };
kernel void assemble_decoder_grad(constant AsmGrad& a [[buffer(0)]], KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.B_TT * a.od_plus_1) return;
    int row = idx / a.od_plus_1, col = idx % a.od_plus_1;
    a.dst[idx] = (col < a.od) ? a.grad_logits[row * a.od + col] : a.grad_value[row];
}

// ---- Muon -------------------------------------------------------------------

struct SumSqPartials { device float* partials; device float* src; int n; };
kernel void muon_sum_sq_partials(constant SumSqPartials& a [[buffer(0)]], KERNEL_IDX) {
    threadgroup float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < a.n; i += blockDim.x * gridDim.x) {
        float v = a.src[i];
        sum += v * v;
    }
    sdata[tid] = sum;
    block_reduce_sum(sdata, &a.partials[blockIdx.x], tid, blockDim.x, 1);
}

struct SumSqReduce { device float* out; device float* partials; int num_blocks; };
kernel void muon_sum_sq_reduce(constant SumSqReduce& a [[buffer(0)]], KERNEL_IDX) {
    threadgroup float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < a.num_blocks) ? a.partials[tid] : 0.0f;
    block_reduce_sum(sdata, a.out, tid, blockDim.x, 1);
}

struct ClipNesterov {
    device float* mb; device float* gc; device float* sum_sq_ptr;
    float max_norm, eps, mu; int n;
};
kernel void muon_clip_nesterov(constant ClipNesterov& a [[buffer(0)]], KERNEL_IDX) {
    float clip_coef = fmin(a.max_norm / (sqrt(*a.sum_sq_ptr) + a.eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) {
        float g = a.gc[idx] * clip_coef;
        float m = a.mu * a.mb[idx] + g;
        a.mb[idx] = m;
        a.gc[idx] = g + a.mu * m;
    }
}

struct L2Norm { device float* dst; device float* sum_sq_ptr; float eps; int n; };
kernel void muon_l2_normalize(constant L2Norm& a [[buffer(0)]], KERNEL_IDX) {
    float inv_norm = 1.0f / fmax(sqrt(*a.sum_sq_ptr), a.eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) a.dst[idx] = a.dst[idx] * inv_norm;
}

struct StoreUpdate { device float* dst; device float* src; float scale; int n; };
kernel void muon_store_update(constant StoreUpdate& a [[buffer(0)]], KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) a.dst[idx] = a.scale * a.src[idx];
}

struct WeightUpdate { device float* wb; device float* update; device float* lr_ptr; float wd; int n; };
kernel void muon_weight_update(constant WeightUpdate& a [[buffer(0)]], KERNEL_IDX) {
    float lr = *a.lr_ptr;
    float wd_scale = 1.0f - lr * a.wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) a.wb[idx] = a.wb[idx] * wd_scale - lr * a.update[idx];
}

// ---- policy heads (sampling, PPO) ---------------------------------------

static inline float finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) return 0.0f;
    if (isinf(x)) return x > 0.0f ? hi : lo;
    return fmin(hi, fmax(lo, x));
}
static inline float safe_continuous_mean(device const float* logits, int idx) {
    return finite_or_clamp(logits[idx], -1.0e6f, 1.0e6f);
}
static inline float safe_continuous_logstd(device const float* logstd, int idx) {
    return finite_or_clamp(logstd[idx], -20.0f, 2.0f);
}
static inline float load_logit_masked(device const float* logits, int logits_base,
        int logits_offset, int a, device const float* mask, int mask_base) {
    float l = logits[logits_base + logits_offset + a];
    float m = mask[mask_base + logits_offset + a];
    if (m == 0.0f) l = -1e4f;
    return l;
}
static inline float ppo_discrete_logsumexp(device const float* logits, int logits_base,
        int logits_offset, int A, device const float* mask, int mask_base,
        thread float* cache) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    for (int a = 0; a < A; ++a) {
        float l = load_logit_masked(logits, logits_base, logits_offset, a, mask, mask_base);
        cache[a] = l;
        if (l > max_logit) {
            sum *= exp(max_logit - l);
            max_logit = l;
        }
        sum += exp(l - max_logit);
    }
    return max_logit + log(sum);
}
static inline void ppo_continuous_head(float mean, float log_std, float action,
        thread float* out_logp, thread float* out_entropy) {
    const float HALF_LOG_2PI = 0.9189385332046727f;
    const float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float sd = exp(log_std);
    float normalized = (action - mean) / sd;
    *out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    *out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}

// Philox4x32-10, identical to src/puf_cuda_cpu.h (same struct layout).
struct PhiloxState {
    uint4 ctr; uint4 output; uint2 key; uint state; int boxmuller_flag; float boxmuller_extra;
};
static inline uint4 philox_round(uint4 c, uint2 k) {
    uint hi0 = mulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    uint hi1 = mulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    return uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
}
static inline uint4 philox10(uint4 c, uint2 k) {
    for (int i = 0; i < 10; i++) {
        c = philox_round(c, k);
        k.x += 0x9E3779B9u;
        k.y += 0xBB67AE85u;
    }
    return c;
}
static inline void philox_incr(thread uint4& c) {
    if (++c.x) return;
    if (++c.y) return;
    if (++c.z) return;
    ++c.w;
}
static inline uint philox_next(thread PhiloxState& s) {
    uint r = s.state == 0 ? s.output.x : s.state == 1 ? s.output.y
        : s.state == 2 ? s.output.z : s.output.w;
    if (++s.state == 4) {
        philox_incr(s.ctr);
        s.output = philox10(s.ctr, s.key);
        s.state = 0;
    }
    return r;
}
#define PUF_2POW32_INV 2.3283064e-10f
static inline float u32_to_uniform(uint x) {
    return x * PUF_2POW32_INV + PUF_2POW32_INV / 2.0f;
}
static inline float philox_uniform(thread PhiloxState& s) {
    return u32_to_uniform(philox_next(s));
}
static inline float philox_normal(thread PhiloxState& s) {
    if (s.boxmuller_flag) {
        s.boxmuller_flag = 0;
        return s.boxmuller_extra;
    }
    uint a = philox_next(s), b = philox_next(s);
    float u = u32_to_uniform(a);
    float v = b * PUF_2POW32_INV * 6.2831855f + PUF_2POW32_INV * 3.1415927f;
    float r = sqrt(-2.0f * log(u));
    s.boxmuller_flag = 1;
    s.boxmuller_extra = r * cos(v);
    return r * sin(v);
}

struct SampleLogits {
    Prec dec_out; Prec logstd; device int* act_sizes; device float* actions;
    device float* env_actions; device float* logprobs; device float* value_out;
    device PhiloxState* rng_states; device float* action_mask; int mask_stride;
};
kernel void sample_logits(constant SampleLogits& a [[buffer(0)]], KERNEL_IDX) {
    int B = a.dec_out.shape[0];
    int fused_cols = a.dec_out.shape[1];
    int num_atns = PUF_NUM_ATNS;
    device float* logits = a.dec_out.data;
    bool is_continuous = a.logstd.data != nullptr && numel(a.logstd.shape) > 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;
    PhiloxState state = a.rng_states[idx];
    int logits_base = idx * fused_cols;
    float total_log_prob = 0.0f;
    if (is_continuous) {
        for (int h = 0; h < num_atns; h++) {
            float mean = safe_continuous_mean(logits, logits_base + h);
            float log_std = safe_continuous_logstd(a.logstd.data, h);
            float sd = exp(log_std);
            float action = finite_or_clamp(mean + sd * philox_normal(state), -1.0e6f, 1.0e6f);
            float stored = action;
            float lp, ent;
            ppo_continuous_head(mean, log_std, stored, &lp, &ent);
            total_log_prob += lp;
            int aidx = idx * num_atns + h;
            a.actions[aidx] = stored;
            a.env_actions[aidx] = stored;
        }
    } else {
        int logits_offset = 0;
        int mask_base = idx * a.mask_stride;
        for (int h = 0; h < num_atns; h++) {
            int A = a.act_sizes[h];
            float cache[PUF_MAX_HEAD_A];
            float logsumexp = ppo_discrete_logsumexp(logits, logits_base, logits_offset,
                A, a.action_mask, mask_base, cache);
            float rand_val = philox_uniform(state);
            float cumsum = 0.0f;
            int sampled = A - 1;
            for (int k = 0; k < A; k++) {
                cumsum += exp(cache[k] - logsumexp);
                if (rand_val < cumsum) {
                    sampled = k;
                    break;
                }
            }
            if (sampled == A - 1) {
                for (int k = A - 1; k >= 0; k--) {
                    if (a.action_mask[mask_base + logits_offset + k] != 0.0f) {
                        sampled = k;
                        break;
                    }
                }
            }
            int aidx = idx * num_atns + h;
            float action = (float)sampled;
            a.actions[aidx] = action;
            a.env_actions[aidx] = action;
            total_log_prob += cache[sampled] - logsumexp;
            logits_offset += A;
        }
    }
    a.logprobs[idx] = total_log_prob;
    a.value_out[idx] = logits[logits_base + fused_cols - 1];
    a.rng_states[idx] = state;
}

struct CacheImp {
    Prec dec_out; device float* actions; device float* old_logprobs;
    device float* action_mask; Prec logstd; device int* act_sizes;
    device float* imp_out; device float* value_out; device float* logps;
    device float* new_lp_out;
};
kernel void cache_imp_and_v(constant CacheImp& a [[buffer(0)]], KERNEL_IDX) {
    int NT = (int)a.dec_out.shape[0] * (int)a.dec_out.shape[1];
    int fused_cols = (int)a.dec_out.shape[2];
    int A_total = fused_cols - 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NT) return;
    int logits_base = idx * fused_cols;
    int at_base = idx * A_total;
    device float* logits = a.dec_out.data;
    a.value_out[idx] = logits[logits_base + A_total];
    float new_lp = 0.0f;
    if (a.logstd.data != nullptr) {
        for (int h = 0; h < PUF_NUM_ATNS; ++h) {
            float lp, ent;
            ppo_continuous_head(safe_continuous_mean(logits, logits_base + h),
                safe_continuous_logstd(a.logstd.data, h),
                a.actions[idx * PUF_NUM_ATNS + h], &lp, &ent);
            new_lp += lp;
        }
    } else {
        int logits_offset = 0;
        for (int h = 0; h < PUF_NUM_ATNS; ++h) {
            int A = a.act_sizes[h];
            float cache[PUF_MAX_HEAD_A];
            float lse = ppo_discrete_logsumexp(logits, logits_base, logits_offset, A,
                a.action_mask, at_base, cache);
            for (int k = 0; k < A; ++k) {
                a.logps[at_base + logits_offset + k] = cache[k] - lse;
            }
            new_lp += cache[(int)a.actions[idx * PUF_NUM_ATNS + h]] - lse;
            logits_offset += A;
        }
    }
    a.new_lp_out[idx] = new_lp;
    a.imp_out[idx] = exp(new_lp - a.old_logprobs[idx]);
}

#define LOSS_PG 0
#define LOSS_VF 1
#define LOSS_ENT 2
#define LOSS_TOTAL 3
#define LOSS_OLD_APPROX_KL 4
#define LOSS_APPROX_KL 5
#define LOSS_CLIPFRAC 6
#define LOSS_IMP 7
#define LOSS_N 8
#define PPO_THREADS 256

struct PPOKernelArgs {
    device float* grad_logits; device float* grad_logstd; device float* grad_values_pred;
    device float* logits; device float* logstd; device float* values_pred;
    device int* act_sizes; device float* action_mask;
    int num_atns; float clip_coef, vf_clip_coef, vf_coef;
    device float* ent_coef;
    int T_seq, A_total, N;
    bool is_continuous;
};
struct PPOGraphArgs {
    device float* imp; device float* actions; device float* old_logprobs;
    device float* advantages; device float* values; device float* returns;
};
struct PPOLossArgs { device float* ppo_partials; PPOKernelArgs a; PPOGraphArgs g; };

kernel void ppo_loss_compute(constant PPOLossArgs& args [[buffer(0)]], KERNEL_IDX) {
    constant PPOKernelArgs& a = args.a;
    constant PPOGraphArgs& g = args.g;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_elements = a.N * a.T_seq;
    float inv_NT = 1.0f / float(total_elements);
    threadgroup float block_losses[LOSS_N * PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) block_losses[c * PPO_THREADS + tid] = 0.0f;
    if (idx < total_elements) {
        int nt = idx;
        int logits_base = nt * (a.A_total + 1);
        int at_base = nt * a.A_total;
        float adv_for_pg = g.advantages[nt];
        float val = g.values[nt];
        float ret = g.returns[nt];
        float val_pred = a.values_pred[logits_base];
        float ent_coef = *a.ent_coef;
        float d_entropy_term = inv_NT * (-ent_coef);
        float logratio = a.grad_values_pred[nt] - g.old_logprobs[nt];
        float ratio = exp(logratio);
        float v_error = val_pred - val;
        float v_clipped = val + fmax(-a.vf_clip_coef, fmin(a.vf_clip_coef, v_error));
        float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
        float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
        float v_loss = 0.5f * fmax(v_loss_unclipped, v_loss_clipped);
        float d_val_pred = (v_loss_clipped > v_loss_unclipped) ? 0.0f : (val_pred - ret);
        a.grad_values_pred[nt] = inv_NT * a.vf_coef * d_val_pred;
        float clip_lo = 1.0f - a.clip_coef;
        float clip_hi = 1.0f + a.clip_coef;
        float ratio_clipped = fmax(clip_lo, fmin(clip_hi, ratio));
        float wa = -adv_for_pg;
        float pg_loss1 = wa * ratio;
        float pg_loss2 = wa * ratio_clipped;
        float pg_loss = fmax(pg_loss1, pg_loss2);
        float d_ratio = wa * inv_NT;
        if (pg_loss2 > pg_loss1 && (ratio <= clip_lo || ratio >= clip_hi)) d_ratio = 0.0f;
        float d_new_logp = d_ratio * ratio;
        float total_entropy = 0.0f;
        if (a.is_continuous) {
            for (int h = 0; h < a.num_atns; ++h) {
                float mean = safe_continuous_mean(a.logits, logits_base + h);
                float c_logstd = safe_continuous_logstd(a.logstd, h);
                float c_action = finite_or_clamp(g.actions[nt * a.num_atns + h], -1.0e6f, 1.0e6f);
                float lp, ent;
                ppo_continuous_head(mean, c_logstd, c_action, &lp, &ent);
                total_entropy += ent;
                float sd = exp(c_logstd);
                float var = sd * sd;
                float diff = c_action - mean;
                a.grad_logits[at_base + h] = d_new_logp * diff / var;
                a.grad_logstd[nt * a.num_atns + h] = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
            }
        } else {
            int logits_offset = 0;
            for (int h = 0; h < a.num_atns; ++h) {
                int A = a.act_sizes[h];
                int act = (int)g.actions[nt * a.num_atns + h];
                float ent = 0.0f;
                for (int j = 0; j < A; ++j) {
                    float logp = a.grad_logits[at_base + logits_offset + j];
                    ent -= exp(logp) * logp;
                }
                total_entropy += ent;
                float d_logp = d_new_logp;
                for (int j = 0; j < A; ++j) {
                    float logp = a.grad_logits[at_base + logits_offset + j];
                    float p = exp(logp);
                    a.grad_logits[at_base + logits_offset + j] =
                        ((j == act ? 1.0f : 0.0f) - p) * d_logp + d_entropy_term * p * (-ent - logp);
                }
                logits_offset += A;
            }
        }
        float thread_loss = (pg_loss + a.vf_coef * v_loss - ent_coef * total_entropy) * inv_NT;
        block_losses[LOSS_PG * PPO_THREADS + tid] = pg_loss * inv_NT;
        block_losses[LOSS_VF * PPO_THREADS + tid] = v_loss * inv_NT;
        block_losses[LOSS_ENT * PPO_THREADS + tid] = total_entropy * inv_NT;
        block_losses[LOSS_TOTAL * PPO_THREADS + tid] = thread_loss;
        block_losses[LOSS_OLD_APPROX_KL * PPO_THREADS + tid] = (-logratio) * inv_NT;
        block_losses[LOSS_APPROX_KL * PPO_THREADS + tid] = ((ratio - 1.0f) - logratio) * inv_NT;
        block_losses[LOSS_CLIPFRAC * PPO_THREADS + tid] =
            (fabs(ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_NT;
        block_losses[LOSS_IMP * PPO_THREADS + tid] = ratio * inv_NT;
    }
    block_reduce_sum(block_losses, &args.ppo_partials[blockIdx.x * LOSS_N], tid, PPO_THREADS, LOSS_N);
}

struct LossReduce { device float* losses_acc; device float* partials; int num_blocks; };
kernel void ppo_loss_reduce(constant LossReduce& a [[buffer(0)]], KERNEL_IDX) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int b = 0; b < a.num_blocks; b++) sum += a.partials[b * LOSS_N + tid];
    a.losses_acc[tid] += sum;
    if (tid == 0) a.losses_acc[LOSS_N] += 1.0f;
}

// GAE / V-trace (fp32: ADV_VEC_WIDTH = 4)
struct Advantage {
    device float* values; device float* rewards; device float* dones; device float* importance;
    device float* advantages; device float* returns;
    float gamma, lambda, rho_clip, c_clip; int num_steps, horizon;
};
kernel void puff_advantage(constant Advantage& a [[buffer(0)]], KERNEL_IDX) {
    const int W = 4;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= a.num_steps) return;
    int horizon = a.horizon;
    int off = row * horizon;
    float lastlam = 0.f;
    float next_v = a.values[off + horizon - 1];
    float next_d = a.dones[off + horizon - 1];
    float next_r = a.rewards[off + horizon - 1];
    for (int seg = horizon / W - 1; seg >= 0; seg--) {
        int base = off + seg * W;
        float v[W], r[W], d[W], imp[W], adv[W] = {0, 0, 0, 0}, ret[W];
        for (int i = 0; i < W; i++) {
            v[i] = a.values[base + i];
            r[i] = a.rewards[base + i];
            d[i] = a.dones[base + i];
            imp[i] = a.importance != nullptr ? a.importance[base + i] : 1.f;
        }
        int i0 = (seg + 1 == horizon / W) ? W - 2 : W - 1;
        for (int i = i0; i >= 0; i--) {
            float nnt = 1.f - next_d;
            float rho_t = fmin(imp[i], a.rho_clip);
            float c_t = fmin(imp[i], a.c_clip);
            float delta = rho_t * (next_r + a.gamma * next_v * nnt - v[i]);
            lastlam = delta + a.gamma * a.lambda * c_t * lastlam * nnt;
            adv[i] = lastlam;
            next_v = v[i]; next_d = d[i]; next_r = r[i];
        }
        for (int i = 0; i < W; i++) {
            ret[i] = v[i] + adv[i];
            a.advantages[base + i] = adv[i];
            a.returns[base + i] = ret[i];
        }
    }
}
