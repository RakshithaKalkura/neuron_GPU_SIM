// snn.cuh — GPU simulator for a Brunel balanced E/I network of LIF neurons.
//
// Per timestep t (dt = 0.1 ms):
//   1. lif_update : read input I_ring[t % R], integrate LIF, threshold/reset,
//                   compact spike IDs (warp ballot), write spike bitmask.
//   2. delivery   : push  -> every spiking neuron scatters w into I_ring[(t+d) % R][post]
//                   pull  -> every neuron gathers over its inputs whose source
//                            spiked at step (t+1-d), writes I_ring[(t+1) % R][j]
//
// Delay convention (shared by all strategies and the tests):
//   a spike emitted at step t with delay d steps is integrated at step t + d.
//
// Weights are not stored per synapse: w = J for excitatory sources, -g*J for
// inhibitory ones (Brunel model). This saves 4 B/synapse.
#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cub/device/device_segmented_sort.cuh>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(e_));                            \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

typedef unsigned long long u64;
typedef curandStatePhilox4_32_10_t rng_t;

// ----------------------------------------------------------------------------
// Parameters
// ----------------------------------------------------------------------------
enum class Strategy { Naive, Push, Pull };

inline const char* strategy_name(Strategy s) {
    switch (s) {
        case Strategy::Naive: return "naive";
        case Strategy::Push:  return "push";
        case Strategy::Pull:  return "pull";
    }
    return "?";
}
inline Strategy parse_strategy(const std::string& s) {
    if (s == "naive") return Strategy::Naive;
    if (s == "push")  return Strategy::Push;
    if (s == "pull")  return Strategy::Pull;
    fprintf(stderr, "unknown strategy '%s' (naive|push|pull)\n", s.c_str());
    exit(1);
}

struct Params {
    int   N         = 12500;   // neurons
    float frac_exc  = 0.8f;    // fraction excitatory
    int   K         = 1000;    // in-degree per neuron (frac_exc from E, rest from I)
    float dt        = 0.1f;    // ms
    float tau       = 20.f;    // ms, membrane time constant
    float theta     = 20.f;    // mV, threshold
    float V_reset   = 10.f;    // mV
    float t_ref     = 2.f;     // ms, refractory period
    float J         = 0.1f;    // mV, excitatory PSP
    float g         = 5.f;     // relative inhibition
    float eta       = 2.f;     // nu_ext / nu_thr  (external drive)
    float d_min     = 1.5f;    // ms
    float d_max     = 1.5f;    // ms (d_min == d_max -> uniform delay, no delay array)
    float fixed_rate = -1.f;   // Hz; >= 0 -> benchmark mode: neurons spike as independent
                               // Poisson processes at this rate (input still integrated)
    u64   seed      = 1234;

    int N_E() const { return (int)lroundf(frac_exc * N); }
    int K_E() const { return (int)lroundf(frac_exc * K); }
    int K_I() const { return K - K_E(); }
    int dmin_steps() const { return std::max(1, (int)lroundf(d_min / dt)); }
    int dmax_steps() const { return std::max(dmin_steps(), (int)lroundf(d_max / dt)); }
    int ring() const { return dmax_steps() + 1; }
    int ref_steps() const { return (int)lroundf(t_ref / dt); }
    // Brunel: nu_thr = theta / (J C_E tau); external input = C_E Poisson sources at eta*nu_thr.
    // Expected external spikes per neuron per step = C_E * eta * nu_thr * dt.
    double ext_lambda() const {
        if (J <= 0.f || eta <= 0.f) return 0.0;
        return (double)eta * theta * dt / ((double)J * tau);
    }
};

struct NeuronConsts {
    int N;
    float decay, theta, V_reset, J_ext;
    int ref_steps, R, words;
    double lambda_ext;
    float p_fixed;  // per-step spike probability in fixed-rate mode
    int fixed;      // 1 -> fixed-rate mode
    u64 seed;
    int stateless_rng;  // 1 -> regenerate Philox state from (seed, t, j) every step
    int block_compact;  // 1 -> one spike-count atomic per block instead of per warp
};

// Optimization switches (all off = the original version), so each one can be
// measured on its own.
struct Opts {
    int  graph_steps   = 0;      // >0: capture this many steps in a CUDA graph and replay it
    bool stateless_rng = false;  // no per-neuron curand state in memory
    bool sort_rows     = false;  // sort each CSR row by target (push locality)
    bool block_compact = false;  // one atomicAdd per block for spike compaction
    static Opts all() { Opts o; o.graph_steps = 100; o.stateless_rng = o.sort_rows = o.block_compact = true; return o; }
};

struct SynConsts {
    int N, N_E, R, d_const;
    float w_exc, w_inh;
};

__device__ __forceinline__ float syn_weight(int pre, const SynConsts& c) {
    return pre < c.N_E ? c.w_exc : c.w_inh;
}

// ----------------------------------------------------------------------------
// Network construction kernels
// ----------------------------------------------------------------------------
// CSC, "interleaved" layout: synapse k of post neuron j lives at s = k*N + j.
// Fixed in-degree makes this a dense K x N matrix of source IDs, so in the pull
// kernel thread j reads pre[k*N + j] -> fully coalesced.
__global__ void k_gen_pre(int N, int N_E, int K, int K_E, int dmin, int dmax, u64 seed,
                          int* __restrict__ pre, uint8_t* __restrict__ delay) {
    const int64_t M = (int64_t)N * K;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nth = gridDim.x * blockDim.x;
    rng_t st;
    curand_init(seed, tid, 0, &st);
    for (int64_t s = tid; s < M; s += nth) {
        int k = (int)(s / N);
        int j = (int)(s - (int64_t)k * N);
        int lo = (k < K_E) ? 0 : N_E;
        int n = (k < K_E) ? N_E : N - N_E;
        int p;
        do { p = lo + (int)(curand(&st) % (unsigned)n); } while (p == j && n > 1);  // no autapses
        pre[s] = p;
        if (delay) delay[s] = (uint8_t)(dmin + (int)(curand(&st) % (unsigned)(dmax - dmin + 1)));
    }
}

__global__ void k_count_out(int64_t M, const int* __restrict__ pre, u64* __restrict__ cnt) {
    for (int64_t s = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; s < M;
         s += (int64_t)gridDim.x * blockDim.x)
        atomicAdd(&cnt[pre[s]], 1ULL);
}

// CSC -> CSR by counting sort. Order inside a row is nondeterministic.
__global__ void k_scatter_csr(int N, int64_t M, const int* __restrict__ pre,
                              const uint8_t* __restrict__ delay_in, u64* __restrict__ cursor,
                              int* __restrict__ post, uint8_t* __restrict__ delay_out) {
    for (int64_t s = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; s < M;
         s += (int64_t)gridDim.x * blockDim.x) {
        int p = pre[s];
        u64 pos = atomicAdd(&cursor[p], 1ULL);
        post[pos] = (int)(s % N);
        if (delay_in) delay_out[pos] = delay_in[s];
    }
}

// ----------------------------------------------------------------------------
// Simulation kernels
// ----------------------------------------------------------------------------
// The current step t lives on the device so that a step's kernels are identical
// every step and can be captured once in a CUDA graph and replayed:
//   tdev[0] is read by k_lif_update, whose block 0 copies it to tdev[1];
//   tdev[1] is read by the delivery kernel, whose block 0 sets tdev[0] = t + 1.
// No kernel writes a slot that its own blocks read, so there is no race.
__global__ void k_init_state(int N, u64 seed, float theta, rng_t* rng, float* V, int* ref) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    rng_t st;
    curand_init(seed ^ 0x9E3779B97F4A7C15ULL, j, 0, &st);
    V[j] = theta * curand_uniform(&st);   // V0 ~ U(0, theta)
    ref[j] = 0;
    if (rng) rng[j] = st;
}

// One thread per neuron. Block size must be a multiple of 32 (warp ballot).
// spike_count[t] starts at 0 (memset once), so no per-step reset is needed and
// the delivery kernel can read the count from device memory: no host sync.
__global__ void k_lif_update(int* __restrict__ tdev, NeuronConsts c, float* __restrict__ V,
                             int* __restrict__ ref, rng_t* __restrict__ rng,
                             float* __restrict__ I_ring, uint32_t* __restrict__ bits,
                             int* __restrict__ spikes, int* __restrict__ spike_count, int n_rec,
                             int2* __restrict__ rec, int* __restrict__ rec_n, int rec_cap) {
    const int t = tdev[0];
    if (blockIdx.x == 0 && threadIdx.x == 0) tdev[1] = t;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int slot = t % c.R;
    bool spk = false;

    if (j < c.N) {
        float* Ip = I_ring + (size_t)slot * c.N + j;
        float I = *Ip;
        *Ip = 0.f;                       // slot is reused R steps later
        float v = V[j];
        int r = ref[j];
        rng_t st;
        // Stateless: a fresh Philox stream per (t, neuron). Philox initialization
        // is just counter arithmetic, which is much cheaper than loading and
        // storing the 64-byte curand state (AoS, uncoalesced) every step.
        if (c.stateless_rng) curand_init(c.seed, (u64)t * c.N + j, 0, &st);
        else st = rng[j];
        if (c.fixed) {
            v = v * c.decay + I;         // still read/integrate input (realistic traffic)
            spk = curand_uniform(&st) < c.p_fixed;
        } else {
            int n_ext = c.lambda_ext > 0.0 ? (int)curand_poisson(&st, c.lambda_ext) : 0;
            if (r > 0) {
                r--;                     // refractory: clamp V, discard input (NEST semantics)
            } else {
                v = v * c.decay + I + c.J_ext * n_ext;
                if (v >= c.theta) { spk = true; v = c.V_reset; r = c.ref_steps; }
            }
        }
        V[j] = v; ref[j] = r;
        if (!c.stateless_rng) rng[j] = st;
    }

    // --- spike compaction ---
    unsigned mask = __ballot_sync(0xffffffffu, spk);
    int warp_global = j >> 5;
    if (bits && lane == 0 && warp_global < c.words) bits[(size_t)slot * c.words + warp_global] = mask;
    int base = 0;
    if (c.block_compact) {
        // per-block: exclusive scan of warp counts in shared memory, ONE atomic per block
        __shared__ int warp_off[32];
        __shared__ int block_base;
        const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
        if (lane == 0) warp_off[w] = __popc(mask);
        __syncthreads();
        if (threadIdx.x == 0) {
            int sum = 0;
            for (int i = 0; i < nw; ++i) { int cnt = warp_off[i]; warp_off[i] = sum; sum += cnt; }
            block_base = sum ? atomicAdd(&spike_count[t], sum) : 0;
        }
        __syncthreads();
        base = block_base + warp_off[w];
    } else {
        // per-warp: one atomic per warp that has spikes
        if (lane == 0 && mask) base = atomicAdd(&spike_count[t], __popc(mask));
        base = __shfl_sync(0xffffffffu, base, 0);
    }
    if (spk) {
        spikes[base + __popc(mask & ((1u << lane) - 1u))] = j;
        if (j < n_rec) {
            int pos = atomicAdd(rec_n, 1);
            if (pos < rec_cap) rec[pos] = make_int2(t, j);
        }
    }
}

__device__ __forceinline__ int delivery_begin(int* tdev) {
    const int t = tdev[1];
    if (blockIdx.x == 0 && threadIdx.x == 0) tdev[0] = t + 1;
    return t;
}

// Naive push: one THREAD per spike, walks its own CSR row serially.
// Uncoalesced row reads + global float atomics.
__global__ void k_push_naive(int* __restrict__ tdev, SynConsts c, const int* __restrict__ spike_count,
                             const int* __restrict__ spikes, const u64* __restrict__ row_ptr,
                             const int* __restrict__ post, const uint8_t* __restrict__ delay,
                             float* __restrict__ I_ring) {
    const int t = delivery_begin(tdev);
    const int n = spike_count[t];
    const int tslot = t % c.R;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int pre = spikes[i];
        float w = syn_weight(pre, c);
        for (u64 s = row_ptr[pre]; s < row_ptr[pre + 1]; ++s) {
            int d = delay ? delay[s] : c.d_const;
            int slot = tslot + d; if (slot >= c.R) slot -= c.R;
            atomicAdd(&I_ring[(size_t)slot * c.N + post[s]], w);
        }
    }
}

// Push: one WARP per spike; lanes stride the row -> coalesced reads of post/delay.
// TODO(opt): warp-aggregated variant. Note: within one row the 32 lanes almost
// never share a target (random targets), so aggregation needs a different
// mapping (e.g. lanes = different spikes) to find duplicates.
__global__ void k_push_warp(int* __restrict__ tdev, SynConsts c, const int* __restrict__ spike_count,
                            const int* __restrict__ spikes, const u64* __restrict__ row_ptr,
                            const int* __restrict__ post, const uint8_t* __restrict__ delay,
                            float* __restrict__ I_ring) {
    const int t = delivery_begin(tdev);
    const int n = spike_count[t];
    const int tslot = t % c.R;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    for (int i = warp; i < n; i += nwarps) {
        int pre = spikes[i];
        float w = syn_weight(pre, c);
        u64 beg = row_ptr[pre], end = row_ptr[pre + 1];
        for (u64 s = beg + lane; s < end; s += 32) {
            int d = delay ? delay[s] : c.d_const;
            int slot = tslot + d; if (slot >= c.R) slot -= c.R;
            atomicAdd(&I_ring[(size_t)slot * c.N + post[s]], w);
        }
    }
}

// Pull: one thread per POST neuron; gathers over its K inputs and checks the
// source's spike bit at step (t+1-d). Writes the input for step t+1. No atomics,
// deterministic, but work is N*K per step regardless of firing rate.
// TODO(opt): batch over d_min steps, skip empty bitmask words.
__global__ void k_pull(int* __restrict__ tdev, SynConsts c, int K, int words,
                       const int* __restrict__ pre, const uint8_t* __restrict__ delay,
                       const uint32_t* __restrict__ bits, float* __restrict__ I_ring) {
    const int t = delivery_begin(tdev);
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= c.N) return;
    const int next = (t + 1) % c.R;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        size_t s = (size_t)k * c.N + j;
        int p = pre[s];
        int d = delay ? delay[s] : c.d_const;
        int src = next - d; if (src < 0) src += c.R;   // slot of step t+1-d
        uint32_t b = __ldg(&bits[(size_t)src * words + (p >> 5)]);
        if ((b >> (p & 31)) & 1u) acc += syn_weight(p, c);
    }
    I_ring[(size_t)next * c.N + j] = acc;  // only writer of this slot in pull mode
}

// ----------------------------------------------------------------------------
// Simulator
// ----------------------------------------------------------------------------
struct Simulator {
    Params p;
    Strategy strat;
    Opts opt;
    int N, N_E, K, R, words, dmin, dmax, max_steps;
    int64_t M;
    bool uniform_delay;
    int t = 0;  // host mirror of the device step counter

    // network: CSC (pull / generator) and CSR (push)
    int* d_pre = nullptr;           uint8_t* d_delay_in = nullptr;
    u64* d_row_ptr = nullptr;       int* d_post = nullptr;   uint8_t* d_delay_out = nullptr;

    // state
    float* d_V = nullptr; int* d_ref = nullptr; rng_t* d_rng = nullptr;
    float* d_I = nullptr; uint32_t* d_bits = nullptr;
    int* d_spikes = nullptr; int* d_count = nullptr; int* d_t = nullptr;
    int n_rec = 0, rec_cap = 0; int2* d_rec = nullptr; int* d_rec_n = nullptr;

    NeuronConsts nc;
    SynConsts sc;
    int push_blocks = 0;
    static const int BS = 256;

    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    Simulator(const Params& p_, Strategy s, int max_steps_, int n_rec_ = 0, Opts o = Opts())
        : p(p_), strat(s), opt(o), max_steps(max_steps_), n_rec(n_rec_) {
        N = p.N; N_E = p.N_E(); K = p.K; M = (int64_t)N * K;
        R = p.ring(); words = (N + 31) / 32;
        dmin = p.dmin_steps(); dmax = p.dmax_steps();
        uniform_delay = (dmin == dmax);
        if (dmax > 255) { fprintf(stderr, "max delay must be <= 255 steps\n"); exit(1); }

        nc.N = N; nc.decay = expf(-p.dt / p.tau); nc.theta = p.theta; nc.V_reset = p.V_reset;
        nc.J_ext = p.J; nc.ref_steps = p.ref_steps(); nc.R = R; nc.words = words;
        nc.lambda_ext = p.ext_lambda();
        nc.fixed = p.fixed_rate >= 0.f;
        nc.p_fixed = nc.fixed ? p.fixed_rate * p.dt * 1e-3f : 0.f;
        nc.seed = p.seed;
        nc.stateless_rng = opt.stateless_rng;
        nc.block_compact = opt.block_compact;
        sc.N = N; sc.N_E = N_E; sc.R = R; sc.d_const = dmin;
        sc.w_exc = p.J; sc.w_inh = -p.g * p.J;

        int dev; cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        push_blocks = prop.multiProcessorCount * 8;
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    // Brunel fixed-in-degree random network, generated on the GPU.
    void build_random() {
        CUDA_CHECK(cudaMalloc(&d_pre, M * sizeof(int)));
        if (!uniform_delay) CUDA_CHECK(cudaMalloc(&d_delay_in, M));
        k_gen_pre<<<1024, 256>>>(N, N_E, K, p.K_E(), dmin, dmax, p.seed, d_pre, d_delay_in);
        CUDA_CHECK(cudaGetLastError());
        finalize_network();
    }

    // For tests: CSC in interleaved layout (pre[k*N + j]); delays empty if uniform.
    void build_from_host(const std::vector<int>& pre, const std::vector<uint8_t>& delay) {
        if ((int64_t)pre.size() != M) { fprintf(stderr, "bad pre size\n"); exit(1); }
        CUDA_CHECK(cudaMalloc(&d_pre, M * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_pre, pre.data(), M * sizeof(int), cudaMemcpyHostToDevice));
        if (!uniform_delay) {
            CUDA_CHECK(cudaMalloc(&d_delay_in, M));
            CUDA_CHECK(cudaMemcpy(d_delay_in, delay.data(), M, cudaMemcpyHostToDevice));
        }
        finalize_network();
    }

    void finalize_network() {
        if (strat != Strategy::Pull) {
            CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(u64)));
            CUDA_CHECK(cudaMemset(d_row_ptr, 0, (N + 1) * sizeof(u64)));
            k_count_out<<<1024, 256>>>(M, d_pre, d_row_ptr);
            thrust::device_ptr<u64> rp(d_row_ptr);
            thrust::exclusive_scan(rp, rp + N + 1, rp);
            u64* d_cursor;
            CUDA_CHECK(cudaMalloc(&d_cursor, N * sizeof(u64)));
            CUDA_CHECK(cudaMemcpy(d_cursor, d_row_ptr, N * sizeof(u64), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMalloc(&d_post, M * sizeof(int)));
            if (!uniform_delay) CUDA_CHECK(cudaMalloc(&d_delay_out, M));
            k_scatter_csr<<<1024, 256>>>(N, M, d_pre, d_delay_in, d_cursor, d_post, d_delay_out);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFree(d_cursor));
            // push doesn't need the CSC anymore -> free it to save memory
            CUDA_CHECK(cudaFree(d_pre)); d_pre = nullptr;
            if (d_delay_in) { CUDA_CHECK(cudaFree(d_delay_in)); d_delay_in = nullptr; }
            if (opt.sort_rows) sort_csr_rows();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Sort every CSR row by target ID (CUB segmented sort; delays move with targets).
    // The counting-sort scatter leaves rows in random order; sorted rows make
    // neighbouring lanes of a warp hit nearby addresses in the ring buffer.
    void sort_csr_rows() {
        if (M > INT32_MAX) { fprintf(stderr, "sort_rows: M too large for CUB int indexing\n"); exit(1); }
        int* post_alt; uint8_t* delay_alt = nullptr;
        CUDA_CHECK(cudaMalloc(&post_alt, M * sizeof(int)));
        if (d_delay_out) CUDA_CHECK(cudaMalloc(&delay_alt, M));
        void* tmp = nullptr; size_t tmp_bytes = 0;
        for (int pass = 0; pass < 2; ++pass) {   // pass 0: query temp size, pass 1: sort
            if (d_delay_out)
                CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(tmp, tmp_bytes, d_post, post_alt,
                           d_delay_out, delay_alt, (int)M, N, d_row_ptr, d_row_ptr + 1));
            else
                CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(tmp, tmp_bytes, d_post, post_alt,
                           (int)M, N, d_row_ptr, d_row_ptr + 1));
            if (pass == 0) CUDA_CHECK(cudaMalloc(&tmp, tmp_bytes));
        }
        CUDA_CHECK(cudaFree(tmp));
        CUDA_CHECK(cudaFree(d_post)); d_post = post_alt;
        if (d_delay_out) { CUDA_CHECK(cudaFree(d_delay_out)); d_delay_out = delay_alt; }
    }

    void init_state() {
        CUDA_CHECK(cudaMalloc(&d_V, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref, N * sizeof(int)));
        if (!opt.stateless_rng) CUDA_CHECK(cudaMalloc(&d_rng, N * sizeof(rng_t)));
        CUDA_CHECK(cudaMalloc(&d_I, (size_t)R * N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_I, 0, (size_t)R * N * sizeof(float)));
        if (strat == Strategy::Pull) {
            CUDA_CHECK(cudaMalloc(&d_bits, (size_t)R * words * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(d_bits, 0, (size_t)R * words * sizeof(uint32_t)));
        }
        CUDA_CHECK(cudaMalloc(&d_spikes, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_count, max_steps * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_count, 0, max_steps * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_t, 2 * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_t, 0, 2 * sizeof(int)));
        if (n_rec > 0) {
            int per_neuron = nc.fixed ? max_steps : max_steps / (nc.ref_steps + 1) + 1;
            rec_cap = (int)std::min<int64_t>((int64_t)n_rec * per_neuron, 64 << 20);
            CUDA_CHECK(cudaMalloc(&d_rec, rec_cap * sizeof(int2)));
        }
        CUDA_CHECK(cudaMalloc(&d_rec_n, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_rec_n, 0, sizeof(int)));
        k_init_state<<<(N + BS - 1) / BS, BS>>>(N, p.seed, p.theta, d_rng, d_V, d_ref);
        CUDA_CHECK(cudaDeviceSynchronize());
        t = 0;
    }

    // Two kernel launches per step on `stream`, no host synchronization.
    void launch_step() {
        k_lif_update<<<(N + BS - 1) / BS, BS, 0, stream>>>(d_t, nc, d_V, d_ref, d_rng, d_I, d_bits,
                                                           d_spikes, d_count, n_rec, d_rec, d_rec_n,
                                                           rec_cap);
        switch (strat) {
            case Strategy::Naive:
                k_push_naive<<<push_blocks, BS, 0, stream>>>(d_t, sc, d_count, d_spikes, d_row_ptr,
                                                             d_post, d_delay_out, d_I);
                break;
            case Strategy::Push:
                k_push_warp<<<push_blocks, BS, 0, stream>>>(d_t, sc, d_count, d_spikes, d_row_ptr,
                                                            d_post, d_delay_out, d_I);
                break;
            case Strategy::Pull:
                k_pull<<<(N + BS - 1) / BS, BS, 0, stream>>>(d_t, sc, K, words, d_pre, d_delay_in,
                                                             d_bits, d_I);
                break;
        }
    }

    void step() { launch_step(); ++t; }

    void run(int steps) {
        if (t + steps > max_steps) { fprintf(stderr, "run exceeds max_steps\n"); exit(1); }
        const int G = opt.graph_steps;
        if (G > 0 && steps >= G) {
            if (!graph_exec) {
                // Capture G steps once. Kernels read t from d_t, so the same graph
                // is valid for every block of G steps.
                cudaGraph_t g;
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                for (int i = 0; i < G; ++i) launch_step();
                CUDA_CHECK(cudaStreamEndCapture(stream, &g));
                CUDA_CHECK(cudaGraphInstantiate(&graph_exec, g, 0));
                CUDA_CHECK(cudaGraphDestroy(g));
            }
            for (; steps >= G; steps -= G) {
                CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
                t += G;
            }
        }
        for (int i = 0; i < steps; ++i) step();   // remainder (or graphs disabled)
        CUDA_CHECK(cudaGetLastError());
    }

    void sync() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

    size_t network_bytes() const {
        size_t b = 0;
        if (d_pre) b += M * sizeof(int);
        if (d_delay_in) b += M;
        if (d_post) b += M * sizeof(int) + (N + 1) * sizeof(u64);
        if (d_delay_out) b += M;
        return b;
    }

    std::vector<int> spike_counts() const {
        sync();
        std::vector<int> h(t);
        if (t) CUDA_CHECK(cudaMemcpy(h.data(), d_count, t * sizeof(int), cudaMemcpyDeviceToHost));
        return h;
    }
    std::vector<int2> recorded_spikes() const {
        sync();
        int n; CUDA_CHECK(cudaMemcpy(&n, d_rec_n, sizeof(int), cudaMemcpyDeviceToHost));
        if (n > rec_cap) { fprintf(stderr, "warning: spike recording overflow (%d > %d)\n", n, rec_cap); n = rec_cap; }
        std::vector<int2> h(n);
        if (n) CUDA_CHECK(cudaMemcpy(h.data(), d_rec, n * sizeof(int2), cudaMemcpyDeviceToHost));
        return h;
    }
    std::vector<float> voltages() const {
        sync();
        std::vector<float> h(N);
        CUDA_CHECK(cudaMemcpy(h.data(), d_V, N * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }

    ~Simulator() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (stream) cudaStreamDestroy(stream);
        void* ptrs[] = {d_pre, d_delay_in, d_row_ptr, d_post, d_delay_out, d_V, d_ref, d_rng,
                        d_I, d_bits, d_spikes, d_count, d_t, d_rec, d_rec_n};
        for (void* q : ptrs) if (q) cudaFree(q);
    }
};
