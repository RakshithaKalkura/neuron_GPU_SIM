// main.cu — command-line driver. Prints one JSON line with timing + statistics.
//
// Example:
//   ./build/snn --strategy push --N 100000 --K 1000 --T 1000
//   ./build/snn --strategy pull --fixed-rate 20 --N 100000 --K 1000 --T 200
//   ./build/snn --record 1000 --out results/spikes_push    # writes spikes_push.bin (int32 pairs t_step,id)
#include "snn.cuh"
#include <cstring>
#include <chrono>

static void usage() {
    printf(
        "usage: snn [options]\n"
        "  --strategy naive|push|pull  (default push)\n"
        "  --N int        neurons (default 12500)\n"
        "  --K int        in-degree (default 1000)\n"
        "  --T float      simulated time in ms (default 1000)\n"
        "  --dt float     ms (default 0.1)\n"
        "  --J float      mV (default 0.1)     --g float  (default 5)\n"
        "  --eta float    external drive nu_ext/nu_thr (default 2)\n"
        "  --delay float  uniform delay in ms (default 1.5)\n"
        "  --dmin/--dmax  random uniform delay range in ms\n"
        "  --fixed-rate f benchmark mode: Poisson spiking at f Hz, ignores dynamics\n"
        "  --seed int\n"
        "  --record n     record spikes of neurons [0, n)\n"
        "  --out prefix   write <prefix>.bin (recorded spikes) and <prefix>_V.bin (final V)\n"
        "  --warmup ms    simulate this long before timing starts (default 0)\n"
        "optimizations (default off):\n"
        "  --graph n          capture n steps in a CUDA graph and replay it\n"
        "  --stateless-rng    regenerate RNG from (seed, t, neuron) instead of storing curand state\n"
        "  --sort-rows        sort CSR rows by target (push/naive)\n"
        "  --block-compact    one spike-count atomic per block\n"
        "  --opt-all          all of the above (graph 100)\n");
}

int main(int argc, char** argv) {
    Params p;
    std::string strat_s = "push", out;
    float T = 1000.f, warmup = 0.f;
    int n_rec = 0;
    Opts opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(); exit(1); }
            return argv[++i];
        };
        if (a == "--strategy") strat_s = next();
        else if (a == "--N") p.N = atoi(next());
        else if (a == "--K") p.K = atoi(next());
        else if (a == "--T") T = atof(next());
        else if (a == "--dt") p.dt = atof(next());
        else if (a == "--J") p.J = atof(next());
        else if (a == "--g") p.g = atof(next());
        else if (a == "--eta") p.eta = atof(next());
        else if (a == "--delay") p.d_min = p.d_max = atof(next());
        else if (a == "--dmin") p.d_min = atof(next());
        else if (a == "--dmax") p.d_max = atof(next());
        else if (a == "--fixed-rate") p.fixed_rate = atof(next());
        else if (a == "--seed") p.seed = strtoull(next(), nullptr, 10);
        else if (a == "--record") n_rec = atoi(next());
        else if (a == "--out") out = next();
        else if (a == "--warmup") warmup = atof(next());
        else if (a == "--graph") opt.graph_steps = atoi(next());
        else if (a == "--stateless-rng") opt.stateless_rng = true;
        else if (a == "--sort-rows") opt.sort_rows = true;
        else if (a == "--block-compact") opt.block_compact = true;
        else if (a == "--opt-all") opt = Opts::all();
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    }
    Strategy strat = parse_strategy(strat_s);
    int warm_steps = (int)lroundf(warmup / p.dt);
    int steps = (int)lroundf(T / p.dt);

    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    Simulator sim(p, strat, warm_steps + steps, n_rec, opt);
    auto t0 = std::chrono::high_resolution_clock::now();
    sim.build_random();
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();
    sim.init_state();

    if (warm_steps) { sim.run(warm_steps); sim.sync(); }

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, sim.stream));
    sim.run(steps);
    CUDA_CHECK(cudaEventRecord(e1, sim.stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

    std::vector<int> counts = sim.spike_counts();
    long long spikes = 0;
    for (int i = warm_steps; i < (int)counts.size(); ++i) spikes += counts[i];
    double sim_ms = steps * p.dt;
    double rate_hz = spikes / ((double)p.N * sim_ms * 1e-3);
    // Mean out-degree == K for fixed in-degree, so events ~= spikes * K.
    double syn_events = (double)spikes * p.K;
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    printf("{\"strategy\":\"%s\",\"N\":%d,\"K\":%d,\"M\":%lld,\"dt\":%g,\"T_ms\":%g,\"steps\":%d,"
           "\"J\":%g,\"g\":%g,\"eta\":%g,\"d_min\":%g,\"d_max\":%g,\"fixed_rate\":%g,\"seed\":%llu,"
           "\"build_s\":%.4f,\"sim_s\":%.6f,\"ms_per_step\":%.6f,\"rate_hz\":%.4f,\"spikes\":%lld,"
           "\"syn_events\":%.6g,\"events_per_s\":%.6g,\"realtime_factor\":%.4f,"
           "\"net_bytes\":%zu,\"gpu_mem_used\":%zu,\"gpu\":\"%s\","
           "\"opt_graph\":%d,\"opt_stateless_rng\":%d,\"opt_sort_rows\":%d,\"opt_block_compact\":%d}\n",
           strategy_name(strat), p.N, p.K, (long long)sim.M, p.dt, sim_ms, steps, p.J, p.g, p.eta,
           p.d_min, p.d_max, p.fixed_rate, (unsigned long long)p.seed, build_s, ms * 1e-3,
           ms / steps, rate_hz, spikes, syn_events, syn_events / (ms * 1e-3), ms / sim_ms,
           sim.network_bytes(), total_b - free_b, prop.name,
           opt.graph_steps, (int)opt.stateless_rng, (int)opt.sort_rows, (int)opt.block_compact);

    if (!out.empty()) {   // <out>.bin: recorded spikes, <out>_V.bin: final V, <out>_counts.bin: spikes/step
        auto write = [&](const std::string& path, const void* data, size_t bytes) {
            FILE* f = fopen(path.c_str(), "wb");
            if (!f) { perror(path.c_str()); exit(1); }
            fwrite(data, 1, bytes, f);
            fclose(f);
        };
        auto rec = sim.recorded_spikes();
        auto V = sim.voltages();
        write(out + ".bin", rec.data(), rec.size() * sizeof(int2));
        write(out + "_V.bin", V.data(), V.size() * sizeof(float));
        write(out + "_counts.bin", counts.data(), counts.size() * sizeof(int));
    }
    return 0;
}
