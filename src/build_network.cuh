// Kernels that build the random network.
// CSC: pre[k*N + j] = k-th input of neuron j (used by pull).
// CSR: post[row_ptr[i] .. row_ptr[i+1]) = targets of neuron i (used by push).
#pragma once
#include "common.cuh"

// Random inputs for every neuron: the first K_E from E neurons, the rest from I.
__global__ void k_gen_pre(int N, int N_E, int K, int K_E, int dmin, int dmax, u64 seed,
                          int* __restrict__ pre, uint8_t* __restrict__ delay) {
    const int64_t M = (int64_t)N * K;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = gridDim.x * blockDim.x;
    rng_t rng;
    curand_init(seed, tid, 0, &rng);

    for (int64_t s = tid; s < M; s += nthreads) {
        int k = (int)(s / N);
        int j = (int)(s - (int64_t)k * N);
        int first = k < K_E ? 0 : N_E;
        int count = k < K_E ? N_E : N - N_E;
        int p;
        do {
            p = first + (int)(curand(&rng) % (unsigned)count);
        } while (p == j && count > 1); // no self-connections
        pre[s] = p;
        if (delay) delay[s] = (uint8_t)(dmin + (int)(curand(&rng) % (unsigned)(dmax - dmin + 1)));
    }
}

// CSC -> CSR by counting sort. Step 1: count outgoing synapses per neuron.
__global__ void k_count_out(int64_t M, const int* __restrict__ pre, u64* __restrict__ cnt) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t s = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; s < M; s += stride)
        atomicAdd(&cnt[pre[s]], 1ULL);
}

// Step 3 (after a prefix sum): put each synapse in the next free slot of its row.
__global__ void k_scatter_csr(int N, int64_t M, const int* __restrict__ pre,
                              const uint8_t* __restrict__ delay_in, u64* __restrict__ cursor,
                              int* __restrict__ post, uint8_t* __restrict__ delay_out) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t s = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; s < M; s += stride) {
        u64 pos = atomicAdd(&cursor[pre[s]], 1ULL);
        post[pos] = (int)(s % N);
        if (delay_in) delay_out[pos] = delay_in[s];
    }
}
