// test_snn.cu — correctness tests. Run: make test
//
// 1. Delay test: 2-neuron ping-pong network (0 -> 1 with delay d1, 1 -> 0 with
//    delay d2). Neuron 0 starts above threshold and spikes at step 0. Expected
//    spike steps: 0: {0, d1+d2, 2(d1+d2), ...}, 1: {d1, d1+(d1+d2), ...}.
//    Checked for every strategy, for uniform and per-synapse delays.
// 2. Equivalence test: in fixed-rate mode spike trains are independent of input,
//    so every strategy must produce the same final V (up to float rounding of
//    the atomic summation order).
#include "../src/snn.cuh"
#include <map>

static int failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); ++failures; } } while (0)

static void test_delay(Strategy s, int d1, int d2, Opts o = Opts()) {
    Params p;
    p.N = 2; p.K = 1; p.frac_exc = 1.f;     // both excitatory, one input each
    p.J = 25.f;                               // one input spike > threshold
    p.eta = 0.f;                              // no external drive
    p.t_ref = 0.1f;                           // 1 step refractory
    p.d_min = std::min(d1, d2) * p.dt;
    p.d_max = std::max(d1, d2) * p.dt;
    const int steps = 6 * (d1 + d2) + 3;
    Simulator sim(p, s, steps, 2, o);
    // interleaved CSC: pre[k*N + j]; K = 1 -> pre[j]
    std::vector<int> pre = {1, 0};                         // 0 <- 1, 1 <- 0
    std::vector<uint8_t> del = {(uint8_t)d2, (uint8_t)d1}; // delay of 1->0 is d2, 0->1 is d1
    if (!sim.uniform_delay) sim.build_from_host(pre, del);
    else sim.build_from_host(pre, {});
    sim.init_state();
    std::vector<float> V0 = {p.theta + 1.f, 0.f};
    CUDA_CHECK(cudaMemcpy(sim.d_V, V0.data(), 2 * sizeof(float), cudaMemcpyHostToDevice));
    sim.run(steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::map<int, std::vector<int>> got;
    for (auto& e : sim.recorded_spikes()) got[e.y].push_back(e.x);
    for (auto& kv : got) std::sort(kv.second.begin(), kv.second.end());
    std::vector<int> exp0, exp1;
    for (int c = 0; c * (d1 + d2) < steps; ++c) {
        exp0.push_back(c * (d1 + d2));
        if (c * (d1 + d2) + d1 < steps) exp1.push_back(c * (d1 + d2) + d1);
    }
    bool ok = got[0] == exp0 && got[1] == exp1;
    EXPECT(ok, "delay test %s%s d1=%d d2=%d: neuron0 %zu spikes (first %d), neuron1 %zu spikes (first %d)",
           strategy_name(s), o.graph_steps ? "+opt" : "", d1, d2, got[0].size(), got[0].empty() ? -1 : got[0][0],
           got[1].size(), got[1].empty() ? -1 : got[1][0]);
}

static std::vector<float> run_fixed(Strategy s, float dmin, float dmax, Opts o = Opts()) {
    Params p;
    p.N = 4000; p.K = 200; p.fixed_rate = 50.f; p.d_min = dmin; p.d_max = dmax;
    Simulator sim(p, s, 2000, 0, o);
    sim.build_random();
    sim.init_state();
    sim.run(2000);
    CUDA_CHECK(cudaDeviceSynchronize());
    return sim.voltages();
}

static void test_equivalence(float dmin, float dmax, Opts o = Opts()) {
    auto ref = run_fixed(Strategy::Pull, dmin, dmax, o);
    for (Strategy s : {Strategy::Naive, Strategy::Push}) {
        auto v = run_fixed(s, dmin, dmax, o);
        double maxdiff = 0, maxabs = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            maxdiff = std::max(maxdiff, (double)fabsf(v[i] - ref[i]));
            maxabs = std::max(maxabs, (double)fabsf(ref[i]));
        }
        printf("  equivalence pull vs %-5s%s (delay %.1f-%.1f ms): max|dV| = %.3g (max|V| = %.3g)\n",
               strategy_name(s), o.graph_steps ? " +opt" : "", dmin, dmax, maxdiff, maxabs);
        EXPECT(maxdiff < 1e-3 * std::max(1.0, maxabs), "equivalence %s", strategy_name(s));
    }
}

int main() {
    printf("[delay tests]\n");
    for (Strategy s : {Strategy::Naive, Strategy::Push, Strategy::Pull}) {
        for (int d : {1, 2, 7, 15, 40}) test_delay(s, d, d);   // uniform delays
        test_delay(s, 1, 15);                                    // per-synapse delays
        test_delay(s, 15, 3);
        test_delay(s, 5, 6);
        // all optimizations on; graph of 7 steps so runs mix graph replays and a remainder
        Opts o = Opts::all(); o.graph_steps = 7;
        for (int d : {1, 15, 40}) test_delay(s, d, d, o);
        test_delay(s, 1, 15, o);
        test_delay(s, 15, 3, o);
    }
    printf("[equivalence tests]\n");
    test_equivalence(1.5f, 1.5f);
    test_equivalence(0.1f, 3.0f);
    test_equivalence(1.5f, 1.5f, Opts::all());
    test_equivalence(0.1f, 3.0f, Opts::all());
    printf(failures ? "%d FAILURE(S)\n" : "ALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
