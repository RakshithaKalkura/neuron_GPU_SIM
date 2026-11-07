// Simulator: owns the GPU memory and launches the kernels.
#pragma once
#include "common.cuh"
#include "build_network.cuh"
#include "neuron_update.cuh"
#include "delivery.cuh"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cub/device/device_segmented_sort.cuh>

struct Simulator {
    Params p;
    Strategy strat;
    Opts opt;
    int N, N_E, K, R, words, dmin, dmax, max_steps;
    int64_t M; // number of synapses
    bool uniform_delay;
    int t = 0; // CPU copy of the step counter

    // network
    int* d_pre = nullptr; // CSC (pull)
    uint8_t* d_delay_in = nullptr;
    u64* d_row_ptr = nullptr; // CSR (push, naive)
    int* d_post = nullptr;
    uint8_t* d_delay_out = nullptr;

    // state
    float* d_V = nullptr;
    int* d_ref = nullptr;
    rng_t* d_rng = nullptr;
    float* d_I = nullptr; // input ring buffer [R][N]
    uint32_t* d_bits = nullptr; // spike bitmask [R][words], pull only
    int* d_spikes = nullptr;
    int* d_count = nullptr; // spikes per step
    int* d_t = nullptr; // step counter on the GPU

    // spike recording
    int n_rec = 0, rec_cap = 0;
    int2* d_rec = nullptr;
    int* d_rec_n = nullptr;

    NeuronConsts nc;
    SynConsts sc;
    int push_blocks = 0;
    static const int BS = 256;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    Simulator(const Params& p_, Strategy s, int max_steps_, int n_rec_ = 0, Opts o = Opts())
        : p(p_), strat(s), opt(o), max_steps(max_steps_), n_rec(n_rec_) {
        N = p.N;
        N_E = p.N_E();
        K = p.K;
        M = (int64_t)N * K;
        R = p.ring();
        words = (N + 31) / 32;
        dmin = p.dmin_steps();
        dmax = p.dmax_steps();
        uniform_delay = (dmin == dmax);
        if (dmax > 255) {
            fprintf(stderr, "max delay must be <= 255 steps\n");
            exit(1);
        }

        nc.N = N;
        nc.decay = expf(-p.dt / p.tau);
        nc.theta = p.theta;
        nc.V_reset = p.V_reset;
        nc.J_ext = p.J;
        nc.ref_steps = p.ref_steps();
        nc.R = R;
        nc.words = words;
        nc.lambda_ext = p.ext_lambda();
        nc.fixed = p.fixed_rate >= 0.f;
        nc.p_fixed = nc.fixed ? p.fixed_rate * p.dt * 1e-3f : 0.f;
        nc.seed = p.seed;
        nc.stateless_rng = opt.stateless_rng;
        nc.block_compact = opt.block_compact;

        sc.N = N;
        sc.N_E = N_E;
        sc.R = R;
        sc.d_const = dmin;
        sc.w_exc = p.J;
        sc.w_inh = -p.g * p.J;

        int dev;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        push_blocks = prop.multiProcessorCount * 8;
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void build_random() {
        CUDA_CHECK(cudaMalloc(&d_pre, M * sizeof(int)));
        if (!uniform_delay) CUDA_CHECK(cudaMalloc(&d_delay_in, M));
        k_gen_pre<<<1024, 256>>>(N, N_E, K, p.K_E(), dmin, dmax, p.seed, d_pre, d_delay_in);
        CUDA_CHECK(cudaGetLastError());
        finalize_network();
    }

    // For tests: network given as pre[k*N + j]; delay is empty if all delays are equal.
    void build_from_host(const std::vector<int>& pre, const std::vector<uint8_t>& delay) {
        if ((int64_t)pre.size() != M) {
            fprintf(stderr, "bad pre size\n");
            exit(1);
        }
        CUDA_CHECK(cudaMalloc(&d_pre, M * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_pre, pre.data(), M * sizeof(int), cudaMemcpyHostToDevice));
        if (!uniform_delay) {
            CUDA_CHECK(cudaMalloc(&d_delay_in, M));
            CUDA_CHECK(cudaMemcpy(d_delay_in, delay.data(), M, cudaMemcpyHostToDevice));
        }
        finalize_network();
    }

    // Push and naive: build the CSR (count, prefix sum, scatter), then free the CSC.
    void finalize_network() {
        if (strat != Strategy::Pull) {
            CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(u64)));
            CUDA_CHECK(cudaMemset(d_row_ptr, 0, (N + 1) * sizeof(u64)));
            k_count_out<<<1024, 256>>>(M, d_pre, d_row_ptr);
            thrust::device_ptr<u64> rp(d_row_ptr);
            thrust::exclusive_scan(rp, rp + N + 1, rp); // counts -> row starts

            u64* d_cursor;
            CUDA_CHECK(cudaMalloc(&d_cursor, N * sizeof(u64)));
            CUDA_CHECK(cudaMemcpy(d_cursor, d_row_ptr, N * sizeof(u64), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMalloc(&d_post, M * sizeof(int)));
            if (!uniform_delay) CUDA_CHECK(cudaMalloc(&d_delay_out, M));
            k_scatter_csr<<<1024, 256>>>(N, M, d_pre, d_delay_in, d_cursor, d_post, d_delay_out);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFree(d_cursor));

            CUDA_CHECK(cudaFree(d_pre));
            d_pre = nullptr;
            if (d_delay_in) {
                CUDA_CHECK(cudaFree(d_delay_in));
                d_delay_in = nullptr;
            }
            if (opt.sort_rows) sort_csr_rows();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Sort each CSR row by target. CUB is called twice: first to get the temp size.
    void sort_csr_rows() {
        if (M > INT32_MAX) {
            fprintf(stderr, "sort_rows: M too large for CUB int indexing\n");
            exit(1);
        }
        int* post_sorted = nullptr;
        uint8_t* delay_sorted = nullptr;
        CUDA_CHECK(cudaMalloc(&post_sorted, M * sizeof(int)));
        if (d_delay_out) CUDA_CHECK(cudaMalloc(&delay_sorted, M));

        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        for (int pass = 0; pass < 2; ++pass) {
            if (d_delay_out)
                CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(tmp, tmp_bytes, d_post, post_sorted,
                           d_delay_out, delay_sorted, (int)M, N, d_row_ptr, d_row_ptr + 1));
            else
                CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(tmp, tmp_bytes, d_post, post_sorted,
                           (int)M, N, d_row_ptr, d_row_ptr + 1));
            if (pass == 0) CUDA_CHECK(cudaMalloc(&tmp, tmp_bytes));
        }
        CUDA_CHECK(cudaFree(tmp));
        CUDA_CHECK(cudaFree(d_post));
        d_post = post_sorted;
        if (d_delay_out) {
            CUDA_CHECK(cudaFree(d_delay_out));
            d_delay_out = delay_sorted;
        }
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

    // One step = two kernels in the stream, no waiting for the GPU.
    void launch_step() {
        const int neuron_blocks = (N + BS - 1) / BS;
        k_lif_update<<<neuron_blocks, BS, 0, stream>>>(d_t, nc, d_V, d_ref, d_rng, d_I, d_bits,
            d_spikes, d_count, n_rec, d_rec, d_rec_n, rec_cap);

        switch (strat) {
            case Strategy::Naive:
                k_push_naive<<<push_blocks, BS, 0, stream>>>(d_t, sc, d_count, d_spikes,
                    d_row_ptr, d_post, d_delay_out, d_I);
                break;
            case Strategy::Push:
                k_push_warp<<<push_blocks, BS, 0, stream>>>(d_t, sc, d_count, d_spikes,
                    d_row_ptr, d_post, d_delay_out, d_I, opt.warp_agg);
                break;
            case Strategy::Pull:
                k_pull<<<neuron_blocks, BS, 0, stream>>>(d_t, sc, K, words, d_pre, d_delay_in,
                    d_bits, d_I);
                break;
        }
    }

    void step() {
        launch_step();
        ++t;
    }

    // With --graph G: record G steps once as a CUDA graph, then replay it.
    void run(int steps) {
        if (t + steps > max_steps) {
            fprintf(stderr, "run exceeds max_steps\n");
            exit(1);
        }
        const int G = opt.graph_steps;
        if (G > 0 && steps >= G) {
            if (!graph_exec) {
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
        for (int i = 0; i < steps; ++i) step(); // leftover steps
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

    // Fraction of push atomics that warp aggregation would save: repeated
    // (delay, target) pairs within each chunk of 32 row entries.
    double warp_agg_saved_fraction() const {
        if (!d_post) return 0.0;
        std::vector<u64> rp(N + 1);
        std::vector<int> post(M);
        std::vector<uint8_t> del(d_delay_out ? M : 0);
        CUDA_CHECK(cudaMemcpy(rp.data(), d_row_ptr, (N + 1) * sizeof(u64), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(post.data(), d_post, M * sizeof(int), cudaMemcpyDeviceToHost));
        if (d_delay_out) CUDA_CHECK(cudaMemcpy(del.data(), d_delay_out, M, cudaMemcpyDeviceToHost));

        int64_t saved = 0;
        std::vector<u64> keys;
        for (int i = 0; i < N; ++i) {
            for (u64 chunk = rp[i]; chunk < rp[i + 1]; chunk += 32) {
                keys.clear();
                for (u64 s = chunk; s < std::min(chunk + 32, rp[i + 1]); ++s) {
                    u64 d = del.empty() ? 0 : del[s];
                    keys.push_back((d << 32) | (uint32_t)post[s]);
                }
                std::sort(keys.begin(), keys.end());
                saved += keys.size() - (std::unique(keys.begin(), keys.end()) - keys.begin());
            }
        }
        return (double)saved / (double)M;
    }

    std::vector<int> spike_counts() const {
        sync();
        std::vector<int> h(t);
        if (t) CUDA_CHECK(cudaMemcpy(h.data(), d_count, t * sizeof(int), cudaMemcpyDeviceToHost));
        return h;
    }

    // (step, neuron) pairs
    std::vector<int2> recorded_spikes() const {
        sync();
        int n;
        CUDA_CHECK(cudaMemcpy(&n, d_rec_n, sizeof(int), cudaMemcpyDeviceToHost));
        if (n > rec_cap) {
            fprintf(stderr, "warning: spike recording overflow (%d > %d)\n", n, rec_cap);
            n = rec_cap;
        }
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
        for (void* q : ptrs)
            if (q) cudaFree(q);
    }
};
