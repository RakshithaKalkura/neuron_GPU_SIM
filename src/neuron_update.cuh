// Neuron kernel: first of the two kernels in a step. One thread per neuron.
#pragma once
#include "common.cuh"

// The step t is kept on the GPU so every step's launches are identical (needed
// for CUDA graphs). k_lif_update reads tdev[0] and copies it to tdev[1]; the
// delivery kernel reads tdev[1] and sets tdev[0] = t + 1.

__global__ void k_init_state(int N, u64 seed, float theta, rng_t* rng, float* V, int* ref) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    rng_t st;
    curand_init(seed ^ 0x9E3779B97F4A7C15ULL, j, 0, &st);
    V[j] = theta * curand_uniform(&st);
    ref[j] = 0;
    if (rng) rng[j] = st;
}

// Block size must be a multiple of 32. spike_count has one counter per step,
// zeroed once at the start, so the CPU never needs to reset or read it.
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
    bool spiked = false;

    if (j < c.N) {
        float* Ip = I_ring + (size_t)slot * c.N + j;
        float I = *Ip;
        *Ip = 0.f; // slot is reused R steps later
        float v = V[j];
        int r = ref[j];

        // stateless: recompute the generator from (t, j) instead of loading 64 bytes
        rng_t st;
        if (c.stateless_rng) curand_init(c.seed, (u64)t * c.N + j, 0, &st);
        else st = rng[j];

        if (c.fixed) {
            v = v * c.decay + I;
            spiked = curand_uniform(&st) < c.p_fixed;
        } else {
            int n_ext = c.lambda_ext > 0.0 ? (int)curand_poisson(&st, c.lambda_ext) : 0;
            if (r > 0) {
                r--; // refractory: input is discarded
            } else {
                v = v * c.decay + I + c.J_ext * n_ext;
                if (v >= c.theta) {
                    spiked = true;
                    v = c.V_reset;
                    r = c.ref_steps;
                }
            }
        }
        V[j] = v;
        ref[j] = r;
        if (!c.stateless_rng) rng[j] = st;
    }

    // bit i of mask = lane i spiked. For pull it is also the bitmask word.
    unsigned mask = __ballot_sync(0xffffffffu, spiked);
    int warp_id = j >> 5;
    if (bits && lane == 0 && warp_id < c.words) bits[(size_t)slot * c.words + warp_id] = mask;

    // Reserve room in spikes[] with one atomic, then each spiking lane
    // writes at base + (number of spiking lanes before it).
    int base = 0;
    if (c.block_compact) {
        // one atomic per block: prefix sum of the warp counts in shared memory
        __shared__ int warp_offset[32];
        __shared__ int block_base;
        const int w = threadIdx.x >> 5;
        const int nw = blockDim.x >> 5;
        if (lane == 0) warp_offset[w] = __popc(mask);
        __syncthreads();
        if (threadIdx.x == 0) {
            int sum = 0;
            for (int i = 0; i < nw; ++i) {
                int cnt = warp_offset[i];
                warp_offset[i] = sum;
                sum += cnt;
            }
            block_base = sum ? atomicAdd(&spike_count[t], sum) : 0;
        }
        __syncthreads();
        base = block_base + warp_offset[w];
    } else {
        // one atomic per warp, result shared with the other lanes
        if (lane == 0 && mask) base = atomicAdd(&spike_count[t], __popc(mask));
        base = __shfl_sync(0xffffffffu, base, 0);
    }

    if (spiked) {
        spikes[base + __popc(mask & ((1u << lane) - 1u))] = j;
        if (j < n_rec) {
            int pos = atomicAdd(rec_n, 1);
            if (pos < rec_cap) rec[pos] = make_int2(t, j);
        }
    }
}
