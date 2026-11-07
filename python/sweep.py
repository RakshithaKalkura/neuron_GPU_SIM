"""Benchmark sweeps. Appends one JSON line per run to results/<name>.jsonl.

    python3 python/sweep.py rate  # time/step vs firing rate, N=100k
    python3 python/sweep.py indegree  # vary K at 5 and 50 Hz, N=50k
    python3 python/sweep.py size  # vary N up to 1M (Brunel dynamics, ~43 Hz)
    python3 python/sweep.py dense  # dense PyTorch vs sparse at small N
    python3 python/sweep.py genn  # GeNN rows added to rate/size (slow: recompiles)
    python3 python/sweep.py all  # everything except genn

Each setting runs REPS times and the fastest run is kept (others were slowed by
other jobs on the GPU).
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNN = str(ROOT / "build" / "snn")
VENV_PY = ROOT / ".venv" / "bin" / "python"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
OPT = ["--stateless-rng", "--graph", "100"]  # the optimizations that helped

# (label, strategy, extra flags)
VARIANTS = [
    ("naive", "naive", []),
    ("push", "push", []),
    ("pull", "pull", []),
    ("push-opt", "push", OPT),
    ("pull-opt", "pull", OPT),
]

ENV = dict(os.environ, CUDA_PATH=os.environ.get("CUDA_PATH", "/usr/local/cuda"))  # for GeNN


def run(cmd, outfile, label, reps=3, timeout=900):
    """Run a command `reps` times, save and return the fastest result."""
    best = None
    for _ in range(reps):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)
        except subprocess.TimeoutExpired:
            print("TIMEOUT:", " ".join(cmd), flush=True)
            continue
        if r.returncode != 0:
            print("FAILED:", " ".join(cmd), r.stderr.strip()[-300:], flush=True)
            return None
        d = json.loads(r.stdout.strip().splitlines()[-1])
        if best is None or d["sim_s"] < best["sim_s"]:
            best = d
    if best is None:
        return None

    best["label"] = label
    with open(outfile, "a") as f:
        f.write(json.dumps(best) + "\n")
    rate = best.get("rate_hz")  # None for GeNN without recording
    rate_s = "-" if rate is None else str(round(rate, 2))
    print(f"{label:>12s} N={best['N']:>8} K={best['K']:>5} rate={rate_s:>7}Hz "
          f"ms/step={best['ms_per_step']:.4f}", flush=True)
    return best


def flags(**kw):
    """fixed_rate=20 -> ["--fixed-rate", "20"]"""
    out = []
    for k, v in kw.items():
        out += ["--" + k.replace("_", "-"), str(v)]
    return out


def snn_cmd(strategy, extra, **kw):
    return [SNN, "--strategy", strategy] + extra + flags(**kw)


def genn_cmd(span, **kw):
    return [PY, str(ROOT / "python" / "genn_ref.py"), "--span", span] + flags(**kw)


def all_variants(outfile, reps=3, **kw):
    for label, strategy, extra in VARIANTS:
        run(snn_cmd(strategy, extra, **kw), outfile, label, reps=reps)


def fresh(name):
    f = OUT / f"{name}.jsonl"
    f.unlink(missing_ok=True)
    return f


def sweep_rate():
    f = fresh("rate")
    for rate in [0.5, 1, 2, 5, 10, 20, 50, 100, 200]:
        all_variants(f, N=100_000, K=1000, T=200, fixed_rate=rate)


def sweep_indegree():
    f = fresh("indegree")
    for K in [100, 250, 500, 1000, 2000, 5000]:
        for rate in [5, 50]:
            all_variants(f, N=50_000, K=K, T=200, fixed_rate=rate)


def sweep_size():
    # N = 1M needs ~10 GB of GPU memory; on smaller GPUs that point fails and the sweep goes on
    f = fresh("size")
    for N in [12_500, 25_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]:
        all_variants(f, reps=1, N=N, K=1000, T=500, eta=2.0)


def sweep_genn():
    """GeNN rows, added to rate.jsonl and size.jsonl. GeNN recompiles whenever the
    model changes (minutes each), so it gets fewer points."""
    f = OUT / "rate.jsonl"
    for rate in [1, 5, 20, 50, 200]:
        run(genn_cmd("postsynaptic", N=100_000, K=1000, T=200, fixed_rate=rate), f, "genn-post")
    run(genn_cmd("presynaptic", N=100_000, K=1000, T=200, fixed_rate=20), f, "genn-pre")

    f = OUT / "size.jsonl"
    for N in [12_500, 50_000, 100_000, 200_000]:
        run(genn_cmd("postsynaptic", N=N, K=1000, T=500, eta=2.0), f, "genn-post", reps=1)


def sweep_dense():
    f = fresh("dense")
    for N in [2_000, 5_000, 10_000, 12_500, 20_000]:
        dense = [PY, str(ROOT / "python" / "dense_torch.py"), "--N", str(N), "--T", "200"]
        run(dense, f, "dense-torch", reps=1)
        for label, strategy, extra in VARIANTS[1:]:
            run(snn_cmd(strategy, extra, N=N, K=1000, T=200), f, label, reps=1)


if __name__ == "__main__":
    SWEEPS = {"rate": sweep_rate, "indegree": sweep_indegree, "size": sweep_size,
              "dense": sweep_dense, "genn": sweep_genn}
    which = sys.argv[1] if len(sys.argv) > 1 else "rate"
    names = ["rate", "indegree", "size", "dense"] if which == "all" else [which]
    for name in names:
        print(f"===== {name}", flush=True)
        SWEEPS[name]()
