// Parameters, options and small helpers shared by all GPU code.
#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err_ = (call); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err_)); \
            exit(1); \
        } \
    } while (0)

typedef unsigned long long u64;
typedef curandStatePhilox4_32_10_t rng_t;

enum class Strategy { Naive, Push, Pull };

inline const char* strategy_name(Strategy s) {
    switch (s) {
        case Strategy::Naive: return "naive";
        case Strategy::Push: return "push";
        case Strategy::Pull: return "pull";
    }
    return "?";
}

inline Strategy parse_strategy(const std::string& s) {
    if (s == "naive") return Strategy::Naive;
    if (s == "push") return Strategy::Push;
    if (s == "pull") return Strategy::Pull;
    fprintf(stderr, "unknown strategy '%s' (naive|push|pull)\n", s.c_str());
    exit(1);
}

// Brunel network parameters (times in ms, voltages in mV)
struct Params {
    int N = 12500;
    float frac_exc = 0.8f;
    int K = 1000; // inputs per neuron
    float dt = 0.1f;
    float tau = 20.f;
    float theta = 20.f;
    float V_reset = 10.f;
    float t_ref = 2.f;
    float J = 0.1f; // excitatory weight; inhibitory = -g * J
    float g = 5.f;
    float eta = 2.f; // external drive strength
    float d_min = 1.5f;
    float d_max = 1.5f;
    float fixed_rate = -1.f; // >= 0: neurons spike randomly at this rate (benchmark mode)
    u64 seed = 1234;

    int N_E() const { return (int)lroundf(frac_exc * N); }
    int K_E() const { return (int)lroundf(frac_exc * K); }
    int K_I() const { return K - K_E(); }
    int dmin_steps() const { return std::max(1, (int)lroundf(d_min / dt)); }
    int dmax_steps() const { return std::max(dmin_steps(), (int)lroundf(d_max / dt)); }
    int ring() const { return dmax_steps() + 1; }
    int ref_steps() const { return (int)lroundf(t_ref / dt); }

    // mean external input spikes per neuron per step
    double ext_lambda() const {
        if (J <= 0.f || eta <= 0.f) return 0.0;
        return (double)eta * theta * dt / ((double)J * tau);
    }
};

// Optional speed-ups, all off by default so each can be measured alone
struct Opts {
    int graph_steps = 0; // record this many steps as a CUDA graph
    bool stateless_rng = false;
    bool sort_rows = false;
    bool block_compact = false;
    bool warp_agg = false; // experimental, not in all()

    static Opts all() {
        Opts o;
        o.graph_steps = 100;
        o.stateless_rng = true;
        o.sort_rows = true;
        o.block_compact = true;
        return o;
    }
};

// Constants for the neuron kernel
struct NeuronConsts {
    int N;
    float decay; // exp(-dt / tau)
    float theta, V_reset, J_ext;
    int ref_steps, R, words;
    double lambda_ext;
    float p_fixed; // spike probability per step in fixed-rate mode
    int fixed;
    u64 seed;
    int stateless_rng;
    int block_compact;
};

// Constants for the delivery kernels
struct SynConsts {
    int N, N_E, R;
    int d_const; // delay in steps when all delays are equal
    float w_exc, w_inh;
};

// Weights are not stored: they depend only on whether the source is E or I.
__device__ __forceinline__ float syn_weight(int pre, const SynConsts& c) {
    return pre < c.N_E ? c.w_exc : c.w_inh;
}
