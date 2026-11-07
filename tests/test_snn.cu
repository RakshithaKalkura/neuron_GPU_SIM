// Correctness tests. Run: make test
// 1. Delay test: two neurons excite each other with delays d1 and d2, so the
//    exact spike steps are known. Checked for every strategy.
// 2. Equivalence test: in fixed-rate mode all strategies must end with the same V.
// 3. The same checks with --warp-agg.
#include "../src/snn.cuh"
#include <map>

static int failures = 0;

#define EXPECT(cond, ...) \
    do { \
        if (!(cond)) { \
            printf("  FAIL: " __VA_ARGS__); \
            printf("\n"); \
            ++failures; \
        } \
    } while (0)

static void test_delay(Strategy s, int d1, int d2, Opts o = Opts()) {
    Params p;
    p.N = 2;
    p.K = 1;
    p.frac_exc = 1.f;
    p.J = 25.f; // one input spike crosses the threshold
    p.eta = 0.f;
    p.t_ref = 0.1f;
    p.d_min = std::min(d1, d2) * p.dt;
    p.d_max = std::max(d1, d2) * p.dt;
    const int steps = 6 * (d1 + d2) + 3;

    std::vector<int> pre = {1, 0}; // 0 <- 1, 1 <- 0
    std::vector<uint8_t> del = {(uint8_t)d2, (uint8_t)d1};
    Simulator sim(p, s, steps, 2, o);
    if (sim.uniform_delay) sim.build_from_host(pre, {});
    else sim.build_from_host(pre, del);
    sim.init_state();

    // neuron 0 starts above threshold
    std::vector<float> V0 = {p.theta + 1.f, 0.f};
    CUDA_CHECK(cudaMemcpy(sim.d_V, V0.data(), 2 * sizeof(float), cudaMemcpyHostToDevice));
    sim.run(steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::map<int, std::vector<int>> got;
    for (auto& e : sim.recorded_spikes()) got[e.y].push_back(e.x);
    for (auto& kv : got) std::sort(kv.second.begin(), kv.second.end());

    // neuron 0 spikes at 0, d1+d2, ...; neuron 1 at d1, d1+(d1+d2), ...
    std::vector<int> exp0, exp1;
    const int period = d1 + d2;
    for (int c = 0; c * period < steps; ++c) {
        exp0.push_back(c * period);
        if (c * period + d1 < steps) exp1.push_back(c * period + d1);
    }

    bool ok = got[0] == exp0 && got[1] == exp1;
    EXPECT(ok, "delay test %s%s d1=%d d2=%d: neuron0 %zu spikes (first %d), neuron1 %zu spikes (first %d)",
           strategy_name(s), o.graph_steps ? "+opt" : "", d1, d2,
           got[0].size(), got[0].empty() ? -1 : got[0][0],
           got[1].size(), got[1].empty() ? -1 : got[1][0]);
}

static std::vector<float> run_fixed(Strategy s, float dmin, float dmax, Opts o, int N, int K) {
    Params p;
    p.N = N;
    p.K = K;
    p.fixed_rate = 50.f;
    p.d_min = dmin;
    p.d_max = dmax;
    Simulator sim(p, s, 2000, 0, o);
    sim.build_random();
    sim.init_state();
    sim.run(2000);
    CUDA_CHECK(cudaDeviceSynchronize());
    return sim.voltages();
}

// pull is deterministic, so it is the reference
static void test_equivalence(float dmin, float dmax, Opts o = Opts(), int N = 4000, int K = 200) {
    auto ref = run_fixed(Strategy::Pull, dmin, dmax, o, N, K);
    for (Strategy s : {Strategy::Naive, Strategy::Push}) {
        auto v = run_fixed(s, dmin, dmax, o, N, K);
        double maxdiff = 0, maxabs = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            maxdiff = std::max(maxdiff, (double)fabsf(v[i] - ref[i]));
            maxabs = std::max(maxabs, (double)fabsf(ref[i]));
        }
        printf("  equivalence pull vs %-5s%s%s (N=%d, delay %.1f-%.1f ms): max|dV| = %.3g (max|V| = %.3g)\n",
               strategy_name(s), o.graph_steps ? " +opt" : "", o.warp_agg ? " +warp-agg" : "",
               N, dmin, dmax, maxdiff, maxabs);
        EXPECT(maxdiff < 1e-3 * std::max(1.0, maxabs), "equivalence %s", strategy_name(s));
    }
}

int main() {
    printf("[delay tests]\n");
    for (Strategy s : {Strategy::Naive, Strategy::Push, Strategy::Pull}) {
        for (int d : {1, 2, 7, 15, 40}) test_delay(s, d, d);
        test_delay(s, 1, 15);
        test_delay(s, 15, 3);
        test_delay(s, 5, 6);

        // all optimizations, graph of 7 steps so runs mix graph replays and single steps
        Opts o = Opts::all();
        o.graph_steps = 7;
        for (int d : {1, 15, 40}) test_delay(s, d, d, o);
        test_delay(s, 1, 15, o);
        test_delay(s, 15, 3, o);
    }

    printf("[equivalence tests]\n");
    test_equivalence(1.5f, 1.5f);
    test_equivalence(0.1f, 3.0f);
    test_equivalence(1.5f, 1.5f, Opts::all());
    test_equivalence(0.1f, 3.0f, Opts::all());

    printf("[warp aggregation tests]\n");
    Opts wa;
    wa.warp_agg = true;
    for (int d : {1, 15}) test_delay(Strategy::Push, d, d, wa);
    test_delay(Strategy::Push, 1, 15, wa);
    test_equivalence(1.5f, 1.5f, wa);
    test_equivalence(0.1f, 3.0f, wa);
    test_equivalence(1.5f, 1.5f, wa, 300, 200); // dense: rows repeat targets
    Opts wa_sorted = wa;
    wa_sorted.sort_rows = true;
    test_equivalence(1.5f, 1.5f, wa_sorted, 300, 200);
    test_equivalence(0.1f, 3.0f, wa_sorted, 300, 200);

    printf(failures ? "%d FAILURE(S)\n" : "ALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
