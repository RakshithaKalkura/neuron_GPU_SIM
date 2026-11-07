"""Brian2 (CPU) reference for the same Brunel network. Compare statistics, not spike times.

Brian2 applies an input one step after it arrives, so we use delay - dt to match
the GPU simulator.

    .venv/bin/python python/brian2_ref.py --N 12500 --K 1000 --T 1000 --out results/brian2
"""
import argparse
import json
import time

import numpy as np
from brian2 import (NeuronGroup, PoissonInput, SpikeMonitor, Synapses, defaultclock, ms, mV,
                    Network, prefs, seed as b2seed)

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=12500)
ap.add_argument("--K", type=int, default=1000)
ap.add_argument("--T", type=float, default=1000.0)
ap.add_argument("--dt", type=float, default=0.1)
ap.add_argument("--J", type=float, default=0.1)
ap.add_argument("--g", type=float, default=5.0)
ap.add_argument("--eta", type=float, default=2.0)
ap.add_argument("--delay", type=float, default=1.5)
ap.add_argument("--record", type=int, default=1000)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--target", default="cython", choices=["cython", "numpy"])
ap.add_argument("--out", default=None, help="prefix for <out>.bin spike file")
a = ap.parse_args()

prefs.codegen.target = a.target
# gcc-11 crashes on some Cython modules with -march=native, so leave it out
prefs.codegen.cpp.extra_compile_args_gcc = ["-w", "-O3", "-ffast-math", "-fno-finite-math-only"]
b2seed(a.seed)
rng = np.random.default_rng(a.seed)
defaultclock.dt = a.dt * ms

N, K = a.N, a.K
N_E = int(round(0.8 * N))
K_E = int(round(0.8 * K))
K_I = K - K_E
tau, theta, V_r, t_ref = 20.0, 20.0, 10.0, 2.0
J = a.J
nu_thr = theta / (J * K_E * tau)  # kHz
nu_ext = a.eta * nu_thr

eqs = "dv/dt = -v / (20*ms) : volt (unless refractory)"
G = NeuronGroup(N, eqs, threshold="v >= 20*mV", reset="v = 10*mV",
                refractory=t_ref * ms, method="exact")
G.v = "20*mV * rand()"


def fixed_indegree(src_first, src_count, k):
    """Every neuron gets k random inputs from [src_first, src_first + src_count)."""
    post = np.repeat(np.arange(N), k)
    pre = src_first + rng.integers(0, src_count, size=N * k)
    bad = pre == post  # redraw self-connections
    while bad.any():
        pre[bad] = src_first + rng.integers(0, src_count, size=bad.sum())
        bad = pre == post
    return pre, post


delay = (a.delay - a.dt) * ms
on_pre = "v_post += J_syn * int(not_refractory_post)"  # ignored while refractory

SE = Synapses(G, G, on_pre=on_pre, namespace={"J_syn": J * mV}, delay=delay)
pre, post = fixed_indegree(0, N_E, K_E)
SE.connect(i=pre, j=post)

SI = Synapses(G, G, on_pre=on_pre, namespace={"J_syn": -a.g * J * mV}, delay=delay)
pre, post = fixed_indegree(N_E, N - N_E, K_I)
SI.connect(i=pre, j=post)

P = PoissonInput(G, "v", K_E, nu_ext / ms, weight="J_ext * int(not_refractory)")
M = SpikeMonitor(G[:a.record])

net = Network(G, SE, SI, P, M)
t0 = time.time()
net.run(a.T * ms, namespace={"J_ext": J * mV})
wall = time.time() - t0

t_steps = np.round(np.asarray(M.t / ms) / a.dt).astype(np.int32)
ids = np.asarray(M.i, dtype=np.int32)
if a.out:
    np.stack([t_steps, ids], axis=1).astype(np.int32).tofile(a.out + ".bin")
print(json.dumps({"strategy": "brian2-" + a.target, "N": N, "K": K, "T_ms": a.T,
                  "sim_s": wall, "realtime_factor": wall / (a.T * 1e-3),
                  "rate_hz": len(ids) / (a.record * a.T * 1e-3)}))
