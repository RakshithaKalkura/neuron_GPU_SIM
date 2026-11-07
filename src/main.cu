// Command-line program: runs one simulation and prints one JSON line.
//   ./build/snn --strategy push --N 100000 --K 1000 --T 1000
//   ./build/snn --strategy pull --fixed-rate 20 --N 100000 --K 1000 --T 200
//   ./build/snn --record 1000 --out results/spikes_push
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
        "  --warp-agg         push: warp-aggregated atomics (experimental, not in --opt-all)\n"
        "  --agg-stats        also report the fraction of atomics warp aggregation saves\n"
        "  --opt-all          graph 100 + stateless-rng + sort-rows + block-compact\n");
}

struct Args {
    Params p;
    Opts opt;
    std::string strategy = "push";
    std::string out;
    float T = 1000.f;
    float warmup = 0.f;
    int n_rec = 0;
    bool agg_stats = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                usage();
                exit(1);
            }
            return argv[++i];
        };

        if (flag == "--strategy") a.strategy = value();
        else if (flag == "--N") a.p.N = atoi(value());
        else if (flag == "--K") a.p.K = atoi(value());
        else if (flag == "--T") a.T = atof(value());
        else if (flag == "--dt") a.p.dt = atof(value());
        else if (flag == "--J") a.p.J = atof(value());
        else if (flag == "--g") a.p.g = atof(value());
        else if (flag == "--eta") a.p.eta = atof(value());
        else if (flag == "--delay") a.p.d_min = a.p.d_max = atof(value());
        else if (flag == "--dmin") a.p.d_min = atof(value());
        else if (flag == "--dmax") a.p.d_max = atof(value());
        else if (flag == "--fixed-rate") a.p.fixed_rate = atof(value());
        else if (flag == "--seed") a.p.seed = strtoull(value(), nullptr, 10);
        else if (flag == "--record") a.n_rec = atoi(value());
        else if (flag == "--out") a.out = value();
        else if (flag == "--warmup") a.warmup = atof(value());
        else if (flag == "--graph") a.opt.graph_steps = atoi(value());
        else if (flag == "--stateless-rng") a.opt.stateless_rng = true;
        else if (flag == "--sort-rows") a.opt.sort_rows = true;
        else if (flag == "--block-compact") a.opt.block_compact = true;
        else if (flag == "--warp-agg") a.opt.warp_agg = true;
        else if (flag == "--agg-stats") a.agg_stats = true;
        else if (flag == "--opt-all") {
            bool warp_agg = a.opt.warp_agg; // --opt-all keeps --warp-agg as it was
            a.opt = Opts::all();
            a.opt.warp_agg = warp_agg;
        } else if (flag == "-h" || flag == "--help") {
            usage();
            exit(0);
        } else {
            fprintf(stderr, "unknown option %s\n", flag.c_str());
            usage();
            exit(1);
        }
    }
    return a;
}

static void write_file(const std::string& path, const void* data, size_t bytes) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        perror(path.c_str());
        exit(1);
    }
    fwrite(data, 1, bytes, f);
    fclose(f);
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    const Params& p = a.p;
    Strategy strat = parse_strategy(a.strategy);
    int warm_steps = (int)lroundf(a.warmup / p.dt);
    int steps = (int)lroundf(a.T / p.dt);

    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    Simulator sim(p, strat, warm_steps + steps, a.n_rec, a.opt);
    auto t0 = std::chrono::high_resolution_clock::now();
    sim.build_random();
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();
    sim.init_state();

    if (warm_steps) {
        sim.run(warm_steps);
        sim.sync();
    }

    // CUDA events measure GPU time, not just launch time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, sim.stream));
    sim.run(steps);
    CUDA_CHECK(cudaEventRecord(stop, sim.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::vector<int> counts = sim.spike_counts();
    long long spikes = 0;
    for (int i = warm_steps; i < (int)counts.size(); ++i) spikes += counts[i];
    double sim_ms = steps * p.dt;
    double rate_hz = spikes / ((double)p.N * sim_ms * 1e-3);
    double syn_events = (double)spikes * p.K; // mean out-degree = K
    double agg_saved = a.agg_stats ? sim.warp_agg_saved_fraction() : -1.0;
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    printf("{\"strategy\":\"%s\",\"N\":%d,\"K\":%d,\"M\":%lld,\"dt\":%g,\"T_ms\":%g,\"steps\":%d,"
           "\"J\":%g,\"g\":%g,\"eta\":%g,\"d_min\":%g,\"d_max\":%g,\"fixed_rate\":%g,\"seed\":%llu,"
           "\"build_s\":%.4f,\"sim_s\":%.6f,\"ms_per_step\":%.6f,\"rate_hz\":%.4f,\"spikes\":%lld,"
           "\"syn_events\":%.6g,\"events_per_s\":%.6g,\"realtime_factor\":%.4f,"
           "\"net_bytes\":%zu,\"gpu_mem_used\":%zu,\"gpu\":\"%s\","
           "\"opt_graph\":%d,\"opt_stateless_rng\":%d,\"opt_sort_rows\":%d,\"opt_block_compact\":%d,"
           "\"opt_warp_agg\":%d,\"agg_saved_frac\":%g}\n",
           strategy_name(strat), p.N, p.K, (long long)sim.M, p.dt, sim_ms, steps,
           p.J, p.g, p.eta, p.d_min, p.d_max, p.fixed_rate, (unsigned long long)p.seed,
           build_s, ms * 1e-3, ms / steps, rate_hz, spikes,
           syn_events, syn_events / (ms * 1e-3), ms / sim_ms,
           sim.network_bytes(), total_b - free_b, prop.name,
           a.opt.graph_steps, (int)a.opt.stateless_rng, (int)a.opt.sort_rows,
           (int)a.opt.block_compact, (int)a.opt.warp_agg, agg_saved);

    // <out>.bin: (step, neuron) pairs, <out>_V.bin: final V, <out>_counts.bin: spikes per step
    if (!a.out.empty()) {
        auto rec = sim.recorded_spikes();
        auto V = sim.voltages();
        write_file(a.out + ".bin", rec.data(), rec.size() * sizeof(int2));
        write_file(a.out + "_V.bin", V.data(), V.size() * sizeof(float));
        write_file(a.out + "_counts.bin", counts.data(), counts.size() * sizeof(int));
    }
    return 0;
}
