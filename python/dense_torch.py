"""Dense PyTorch baseline: input = spikes @ W with a full N x N weight matrix.
Every step costs N^2, no matter how few neurons spike. One fixed delay only.

    python3 python/dense_torch.py --N 12500 --K 1000 --T 1000
    python3 python/dense_torch.py --N 12500 --fixed-rate 20 --T 200
"""
import argparse
import json
import math
import time

import torch

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
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)
dtype = getattr(torch, a.dtype)

N, K, dt = a.N, a.K, a.dt
N_E, K_E = round(0.8 * N), round(0.8 * K)
K_I = K - K_E
tau, theta, V_r = 20.0, 20.0, 10.0
ref_steps = round(2.0 / dt)
D = max(1, round(a.delay / dt))
R = D + 1
lam = a.eta * theta * dt / (a.J * tau) if a.eta > 0 else 0.0
decay = math.exp(-dt / tau)
steps = round(a.T / dt)
p_fixed = a.fixed_rate * dt * 1e-3

# W[pre, post] with K random inputs per neuron
t0 = time.time()
W = torch.zeros(N, N, device=dev, dtype=torch.float32)
post = torch.arange(N, device=dev)
pre_e = torch.randint(0, N_E, (K_E, N), device=dev)
pre_i = torch.randint(N_E, N, (K_I, N), device=dev)
W.index_put_((pre_e.flatten(), post.repeat(K_E)),
             torch.full((K_E * N,), a.J, device=dev), accumulate=True)
W.index_put_((pre_i.flatten(), post.repeat(K_I)),
             torch.full((K_I * N,), -a.g * a.J, device=dev), accumulate=True)
W.fill_diagonal_(0.0)  # no self-connections
W = W.to(dtype)
torch.cuda.synchronize()
build_s = time.time() - t0

V = theta * torch.rand(N, device=dev)
ref = torch.zeros(N, dtype=torch.int32, device=dev)
hist = torch.zeros(R, N, device=dev, dtype=dtype)  # spikes of the last R steps
lam_t = torch.full((N,), lam, device=dev)
counts = torch.zeros(steps, dtype=torch.int32, device=dev)

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    for t in range(steps):
        I = (hist[(t - D) % R] @ W).float()  # spikes from step t-D arrive now
        if a.fixed_rate >= 0:
            V = V * decay + I
            spk = torch.rand(N, device=dev) < p_fixed
        else:
            n_ext = torch.poisson(lam_t) if lam > 0 else 0.0
            active = ref == 0
            V = torch.where(active, V * decay + I + a.J * n_ext, V)
            ref = torch.where(active, ref, ref - 1)
            spk = active & (V >= theta)
            V = torch.where(spk, torch.full_like(V, V_r), V)
            ref = torch.where(spk, torch.full_like(ref, ref_steps), ref)
        hist[t % R] = spk.to(dtype)
        counts[t] = spk.sum()
torch.cuda.synchronize()
wall = time.time() - t0

spikes = int(counts.sum())
sim_ms = steps * dt
events = spikes * K
print(json.dumps({
    "strategy": f"dense-torch-{a.dtype}", "N": N, "K": K, "M": N * K, "dt": dt, "T_ms": sim_ms,
    "steps": steps, "J": a.J, "g": a.g, "eta": a.eta, "d_min": a.delay, "d_max": a.delay,
    "fixed_rate": a.fixed_rate, "seed": a.seed, "build_s": build_s, "sim_s": wall,
    "ms_per_step": wall * 1e3 / steps, "rate_hz": spikes / (N * sim_ms * 1e-3), "spikes": spikes,
    "syn_events": events, "events_per_s": events / wall, "realtime_factor": wall * 1e3 / sim_ms,
    "net_bytes": W.numel() * W.element_size(), "gpu": torch.cuda.get_device_name()}))
