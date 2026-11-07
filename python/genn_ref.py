"""GeNN (PyGeNN 5) version of the same Brunel network.

--span presynaptic ~ our naive (thread per spike); --span postsynaptic ~ our push.
Differences from our model (negligible): inputs drawn with replacement, and the
external drive is Binomial(K_E, p) because GeNN has no Poisson generator.
GeNN's axonal_delay_steps = 0 already means one step, so we pass D - 1.

    .venv/bin/python python/genn_ref.py --N 12500 --T 1000 --record 1000 --out results/genn
    .venv/bin/python python/genn_ref.py --N 100000 --fixed-rate 20 --T 200 --record 0
"""
import argparse
import json
import math
import pathlib
import time

import numpy as np
from pygenn import (GeNNModel, ParallelismHint, create_neuron_model, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update)

ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=12500)
ap.add_argument("--K", type=int, default=1000)
ap.add_argument("--T", type=float, default=1000.0)
ap.add_argument("--dt", type=float, default=0.1)
ap.add_argument("--J", type=float, default=0.1)
ap.add_argument("--g", type=float, default=5.0)
ap.add_argument("--eta", type=float, default=2.0)
ap.add_argument("--delay", type=float, default=1.5)
ap.add_argument("--fixed-rate", type=float, default=-1.0)
ap.add_argument("--span", default="postsynaptic", choices=["postsynaptic", "presynaptic"])
ap.add_argument("--record", type=int, default=0, help="record spikes of first n E neurons")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", default=None)
a = ap.parse_args()

N, K, dt = a.N, a.K, a.dt
N_E, K_E = round(0.8 * N), round(0.8 * K)
N_I, K_I = N - N_E, K - K_E
tau, theta, V_r, t_ref = 20.0, 20.0, 10.0, 2.0
D = max(1, round(a.delay / dt))
steps = round(a.T / dt)
nu_thr = theta / (a.J * K_E * tau)  # kHz
p_ext = a.eta * nu_thr * dt
fixed = a.fixed_rate >= 0

lif = create_neuron_model(
    "lif_brunel",
    params=["decay", "theta", "Vr", "ref_steps", "n_ext", "p_ext", "Jext", "p_fixed", "fixed"],
    vars=[("V", "scalar"), ("RefSteps", "int"), ("Spk", "int")],
    sim_code="""
    if (fixed != 0.0) {
        V = V * decay + Isyn;
        Spk = (gennrand_uniform() < p_fixed) ? 1 : 0;
    }
    else {
        const unsigned int next = (n_ext > 0.0) ? gennrand_binomial((unsigned int)n_ext, p_ext) : 0;
        if (RefSteps > 0) {
            RefSteps--;
            Spk = 0;
        }
        else {
            V = V * decay + Isyn + Jext * (scalar)next;
            Spk = (V >= theta) ? 1 : 0;
        }
    }
    """,
    threshold_condition_code="Spk != 0",
    reset_code="""
    if (fixed == 0.0) {
        V = Vr;
        RefSteps = (int)ref_steps;
    }
    """)

params = {"decay": math.exp(-dt / tau), "theta": theta, "Vr": V_r,
          "ref_steps": float(round(t_ref / dt)), "n_ext": float(K_E if a.eta > 0 else 0),
          "p_ext": p_ext, "Jext": a.J, "p_fixed": max(a.fixed_rate, 0) * dt * 1e-3,
          "fixed": 1.0 if fixed else 0.0}

model = GeNNModel("float", "brunel_genn", backend="cuda")
model.dt = dt
model.seed = a.seed
rng = np.random.default_rng(a.seed)


def init_vars(n):
    return {"V": rng.uniform(0.0, theta, n).astype(np.float32), "RefSteps": 0, "Spk": 0}


pop_e = model.add_neuron_population("E", N_E, lif, params, init_vars(N_E))
pop_i = model.add_neuron_population("I", N_I, lif, params, init_vars(N_I))
pop_e.spike_recording_enabled = a.record > 0
# dynamic parameters: a rate sweep reuses one compiled model
for pop in (pop_e, pop_i):
    for prm in ("p_fixed", "p_ext", "fixed"):
        pop.set_param_dynamic(prm)

hint = ParallelismHint.PRESYNAPTIC if a.span == "presynaptic" else ParallelismHint.POSTSYNAPTIC
for src, w, k in [(pop_e, a.J, K_E), (pop_i, -a.g * a.J, K_I)]:
    for tgt in [pop_e, pop_i]:
        sg = model.add_synapse_population(
            f"{src.name}{tgt.name}", "SPARSE", src, tgt,
            init_weight_update("StaticPulseConstantWeight", {"g": w}),
            init_postsynaptic("DeltaCurr"),
            init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": k}))
        sg.axonal_delay_steps = D - 1
        sg.parallelism_hint = hint

build_dir = ROOT / "build" / f"genn_{a.span}"
build_dir.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
model.build(str(build_dir))
model.load(num_recording_timesteps=steps if a.record > 0 else None)
build_s = time.perf_counter() - t0

# copying V back makes the CPU wait for the GPU
pop_e.vars["V"].pull_from_device()
t0 = time.perf_counter()
for _ in range(steps):
    model.step_time()
pop_e.vars["V"].pull_from_device()
wall = time.perf_counter() - t0

out = {"strategy": f"genn-{a.span}", "N": N, "K": K, "M": N * K, "dt": dt, "T_ms": steps * dt,
       "steps": steps, "J": a.J, "g": a.g, "eta": a.eta, "d_min": a.delay, "d_max": a.delay,
       "fixed_rate": a.fixed_rate, "seed": a.seed, "build_s": build_s, "sim_s": wall,
       "ms_per_step": wall * 1e3 / steps, "realtime_factor": wall * 1e3 / (steps * dt),
       "gpu": "cuda"}
if a.record > 0:
    model.pull_recording_buffers_from_device()
    times, ids = pop_e.spike_recording_data[0]
    m = ids < a.record
    t_steps = np.round(times[m] / dt).astype(np.int32)
    if a.out:
        np.stack([t_steps, ids[m].astype(np.int32)], axis=1).tofile(a.out + ".bin")
    out["rate_hz"] = len(times) / (N_E * steps * dt * 1e-3)  # E population
else:
    out["rate_hz"] = None
print(json.dumps(out))
