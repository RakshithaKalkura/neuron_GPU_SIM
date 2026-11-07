"""Benchmark sweeps. Appends one JSON line per configuration to results/<name>.jsonl.

    python3 python/sweep.py rate      # time/step vs firing rate (fixed-rate mode), N=100k
    python3 python/sweep.py indegree  # vary K at 5 and 50 Hz, N=50k
    python3 python/sweep.py size      # vary N up to 1M (Brunel dynamics, ~43 Hz)
    python3 python/sweep.py dense     # dense PyTorch vs sparse at small N
    python3 python/sweep.py genn      # GeNN rows appended to rate/size (slow: recompiles)
    python3 python/sweep.py all       # everything except genn

Each configuration is run REPS times and the fastest run is kept. Every row gets a
"label" (naive, push, pull, push-opt, pull-opt, genn-post, genn-pre, dense-torch).
Run on an otherwise idle GPU.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNN = str(ROOT / "build" / "snn")
PY = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv").exists() else sys.executable
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
OPT = ["--stateless-rng", "--graph", "100"]   # the optimizations that helped (see ablation)
VARIANTS = [("naive", "naive", []), ("push", "push", []), ("pull", "pull", []),
            ("push-opt", "push", OPT), ("pull-opt", "pull", OPT)]
GENN = [("genn-post", "postsynaptic"), ("genn-pre", "presynaptic")]
ENV = dict(os.environ, CUDA_PATH=os.environ.get("CUDA_PATH", "/usr/local/cuda"))  # GeNN needs it


def run(cmd, outfile, label, reps=3, timeout=900):
    """Run `reps` times, keep the fastest (least disturbed) measurement."""
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
    rate = best.get("rate_hz")   # None for GeNN runs without spike recording
    print(f"{label:>12s} N={best['N']:>8} K={best['K']:>5} "
          f"rate={'-' if rate is None else round(rate, 2)!s:>7}Hz "
          f"ms/step={best['ms_per_step']:.4f}", flush=True)
    return best


def snn_cmd(strategy, extra, **kw):
    cmd = [SNN, "--strategy", strategy] + extra
    for k, v in kw.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    return cmd


def genn_cmd(span, **kw):
    cmd = [PY, str(ROOT / "python" / "genn_ref.py"), "--span", span]
    for k, v in kw.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    return cmd


def all_variants(f, reps=3, **kw):
    for label, strat, extra in VARIANTS:
        run(snn_cmd(strat, extra, **kw), f, label, reps=reps)


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
    f = fresh("size")
    # N = 1M needs ~10 GB of GPU memory; on a smaller GPU that point fails ("FAILED")
    # and the sweep continues with the rest.
    for N in [12_500, 25_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]:
        # Brunel dynamics, fixed in-degree -> same ~43 Hz regime at every N
        all_variants(f, reps=1, N=N, K=1000, T=500, eta=2.0)


def sweep_genn():
    """GeNN rows, appended to rate.jsonl and size.jsonl (run after the other sweeps).
    GeNN recompiles its generated CUDA code whenever the model changes (several
    minutes each), so this uses fewer points than our own sweeps."""
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
        run([PY, str(ROOT / "python" / "dense_torch.py"), "--N", str(N), "--T", "200"], f,
            "dense-torch", reps=1)
        for label, strat, extra in VARIANTS[1:]:
            run(snn_cmd(strat, extra, N=N, K=1000, T=200), f, label, reps=1)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "rate"
    sweeps = {"rate": sweep_rate, "indegree": sweep_indegree, "size": sweep_size,
              "dense": sweep_dense, "genn": sweep_genn}
    for name in (["rate", "indegree", "size", "dense"] if which == "all" else [which]):
        print(f"===== {name}", flush=True)
        sweeps[name]()
