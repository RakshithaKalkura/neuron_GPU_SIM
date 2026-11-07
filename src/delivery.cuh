// Delivery kernels: second kernel of a step.
// A spike at step t with delay d is added to I_ring[(t + d) % R].
#pragma once
#include "common.cuh"

__device__ __forceinline__ int delivery_begin(int* tdev) {
    const int t = tdev[1];
    if (blockIdx.x == 0 && threadIdx.x == 0) tdev[0] = t + 1;
    return t;
}

// Naive push: one thread per spike walks the whole row.
__global__ void k_push_naive(int* __restrict__ tdev, SynConsts c,
                             const int* __restrict__ spike_count, const int* __restrict__ spikes,
                             const u64* __restrict__ row_ptr, const int* __restrict__ post,
                             const uint8_t* __restrict__ delay, float* __restrict__ I_ring) {
    const int t = delivery_begin(tdev);
    const int nspk = spike_count[t];
    const int tslot = t % c.R;
    const int stride = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nspk; i += stride) {
        int src = spikes[i];
        float w = syn_weight(src, c);
        for (u64 s = row_ptr[src]; s < row_ptr[src + 1]; ++s) {
            int d = delay ? delay[s] : c.d_const;
            int slot = tslot + d;
            if (slot >= c.R) slot -= c.R;
            atomicAdd(&I_ring[(size_t)slot * c.N + post[s]], w);
        }
    }
}

// Push: one warp per spike, the 32 lanes split the row (coalesced reads).
// warp_agg: lanes with the same target are merged into one atomicAdd.
__global__ void k_push_warp(int* __restrict__ tdev, SynConsts c,
                            const int* __restrict__ spike_count, const int* __restrict__ spikes,
                            const u64* __restrict__ row_ptr, const int* __restrict__ post,
                            const uint8_t* __restrict__ delay, float* __restrict__ I_ring,
                            int warp_agg) {
    const int t = delivery_begin(tdev);
    const int nspk = spike_count[t];
    const int tslot = t % c.R;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;

    for (int i = warp; i < nspk; i += nwarps) {
        int src = spikes[i];
        float w = syn_weight(src, c);
        u64 beg = row_ptr[src];
        u64 end = row_ptr[src + 1];

        if (!warp_agg) {
            for (u64 s = beg + lane; s < end; s += 32) {
                int d = delay ? delay[s] : c.d_const;
                int slot = tslot + d;
                if (slot >= c.R) slot -= c.R;
                atomicAdd(&I_ring[(size_t)slot * c.N + post[s]], w);
            }
            continue;
        }

        // all lanes loop the same number of times (needed by the warp vote functions)
        for (u64 chunk = beg; chunk < end; chunk += 32) {
            const u64 s = chunk + lane;
            const bool valid = s < end;
            const unsigned active = __ballot_sync(0xffffffffu, valid);
            if (!valid) continue;
            int d = delay ? delay[s] : c.d_const;
            int slot = tslot + d;
            if (slot >= c.R) slot -= c.R;
            const u64 addr = (u64)slot * c.N + post[s];
            const unsigned same = __match_any_sync(active, addr); // lanes with my address
            if (lane == __ffs(same) - 1) atomicAdd(&I_ring[addr], w * __popc(same));
        }
    }
}

// Pull: one thread per neuron j computes its input for step t+1 by checking,
// for each input p with delay d, whether p spiked at step t+1-d. No atomics.
__global__ void k_pull(int* __restrict__ tdev, SynConsts c, int K, int words,
                       const int* __restrict__ pre, const uint8_t* __restrict__ delay,
                       const uint32_t* __restrict__ bits, float* __restrict__ I_ring) {
    const int t = delivery_begin(tdev);
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= c.N) return;
    const int next = (t + 1) % c.R;
    float sum = 0.f;

    for (int k = 0; k < K; ++k) {
        size_t s = (size_t)k * c.N + j;
        int p = pre[s];
        int d = delay ? delay[s] : c.d_const;
        int src_slot = next - d;
        if (src_slot < 0) src_slot += c.R;
        uint32_t word = __ldg(&bits[(size_t)src_slot * words + (p >> 5)]);
        if ((word >> (p & 31)) & 1u) sum += syn_weight(p, c);
    }
    I_ring[(size_t)next * c.N + j] = sum;
}
