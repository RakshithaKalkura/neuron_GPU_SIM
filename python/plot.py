"""Plots for the report.

    python3 python/plot.py             # all plots
    python3 python/plot.py rate        # results/rate.jsonl -> results/rate.png  (main result)
    python3 python/plot.py validate    # GPU vs Brian2 raster + stats -> results/validate.png
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from analysis import load_spikes, raster, stats  # noqa: E402

RES = pathlib.Path(__file__).resolve().parent.parent / "results"
# One hue per implementation (validated reference palette, fixed order); optimized
# variants share the hue with a dashed line. Naive is the grey baseline.
STYLE = {  # label -> (color, linestyle, marker, legend name)
    "naive":     ("#8a8984", "-",  "o", "naive (thread/spike)"),
    "push":      ("#2a78d6", "-",  "o", "push (warp/spike)"),
    "push-opt":  ("#2a78d6", "--", "s", "push + opt"),
    "pull":      ("#eb6834", "-",  "o", "pull"),
    "pull-opt":  ("#eb6834", "--", "s", "pull + opt"),
    "genn-post": ("#1baf7a", "-",  "D", "GeNN (postsynaptic)"),
    "genn-pre":  ("#1baf7a", "--", "D", "GeNN (presynaptic)"),
    "dense-torch": ("#52514e", ":", "^", "dense PyTorch"),
}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "lines.linewidth": 2,
                     "lines.markersize": 6, "legend.frameon": False, "font.size": 10})


def load(name):
    p = RES / f"{name}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def label(d):
    return d.get("label", d["strategy"])


def events_per_s(d):
    """Synaptic events per wall-clock second (also for GeNN rows, which lack it)."""
    rate = d.get("rate_hz") or d.get("fixed_rate")
    return d["N"] * d["K"] * rate * d["dt"] * 1e-3 / (d["ms_per_step"] * 1e-3)


def series(ax, rows, x_key, y_fn, labels=None):
    for lab in labels or STYLE:
        r = sorted([d for d in rows if label(d) == lab], key=lambda d: d[x_key])
        if not r:
            continue
        c, ls, m, name = STYLE[lab]
        ax.loglog([d[x_key] for d in r], [y_fn(d) for d in r], ls=ls, marker=m, color=c,
                  label=name)


def finish(fig, axes, name):
    for ax in np.atleast_1d(axes):
        ax.grid(True, which="major", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / f"{name}.png", dpi=150)
    print("wrote", RES / f"{name}.png")


def crossover(ax, rows, a="push", b="pull"):
    A = {d["fixed_rate"]: d["ms_per_step"] for d in rows if label(d) == a}
    B = {d["fixed_rate"]: d["ms_per_step"] for d in rows if label(d) == b}
    xs = sorted(set(A) & set(B))
    for x0, x1 in zip(xs, xs[1:]):
        if (A[x0] - B[x0]) * (A[x1] - B[x1]) < 0:   # sign change: log-linear interpolation
            f0, f1 = np.log(A[x0] / B[x0]), np.log(A[x1] / B[x1])
            xc = np.exp(np.log(x0) + (np.log(x1) - np.log(x0)) * f0 / (f0 - f1))
            ax.axvline(xc, ls=":", color="#52514e", lw=1)
            ax.annotate(f"{a} = {b} at {xc:.0f} Hz", (xc, ax.get_ylim()[1]), fontsize=8,
                        va="top", ha="left", color="#52514e")


def plot_rate():
    rows = load("rate")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    series(axes[0], rows, "fixed_rate", lambda d: d["ms_per_step"])
    series(axes[1], rows, "fixed_rate", events_per_s)
    crossover(axes[0], rows, "push", "pull")
    crossover(axes[0], rows, "push-opt", "pull-opt")
    N, K = rows[0]["N"], rows[0]["K"]
    axes[0].set(xlabel="firing rate (Hz)", ylabel="time per step (ms)",
                title=f"Time per step vs firing rate (N={N:,}, K={K})")
    axes[1].set(xlabel="firing rate (Hz)", ylabel="synaptic events / s",
                title="Throughput")
    finish(fig, axes, "rate")


def plot_size():
    rows = load("size")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    series(axes[0], rows, "N", lambda d: d["ms_per_step"])
    series(axes[1], rows, "N", lambda d: d["realtime_factor"])
    axes[0].set(xlabel="neurons N", ylabel="time per step (ms)",
                title="Time per step vs network size (K=1000, ~43 Hz)")
    axes[1].axhline(1.0, color="#52514e", lw=1, ls=":")
    axes[1].set(xlabel="neurons N", ylabel="wall time / simulated time",
                title="Real-time factor (below 1 = faster than real time)")
    finish(fig, axes, "size")


def plot_indegree():
    rows = load("indegree")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, rate in zip(axes, sorted({d["fixed_rate"] for d in rows})):
        series(ax, [d for d in rows if d["fixed_rate"] == rate], "K", lambda d: d["ms_per_step"])
        ax.set(xlabel="in-degree K", title=f"{rate:g} Hz (N={rows[0]['N']:,})")
    axes[0].set_ylabel("time per step (ms)")
    finish(fig, axes, "indegree")


def plot_dense():
    rows = load("dense")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    series(ax, rows, "N", lambda d: d["ms_per_step"])
    ax.set(xlabel="neurons N", ylabel="time per step (ms)",
           title="Dense PyTorch vs sparse kernels (K=1000)")
    finish(fig, ax, "dense")


def plot_validate(T=1000.0, n_rec=1000):
    files = {"GPU push": RES / "gpu_push.bin", "GPU pull": RES / "gpu_pull.bin",
             "GeNN": RES / "genn.bin", "Brian2": RES / "brian2.bin"}
    files = {k: v for k, v in files.items() if v.exists()}
    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, path) in zip(axes, files.items()):
        t, i = load_spikes(path)
        s = stats(t, i, n_rec, T)
        print(f"{name:10s} rate={s['rate_hz']:.2f} Hz  CV(ISI)={s['cv_isi']:.3f}  Fano={s['fano_pop']:.2f}")
        raster(ax, t, i, t_window=(200, 400),
               title=f"{name}\n{s['rate_hz']:.1f} Hz, CV={s['cv_isi']:.2f}")
    fig.tight_layout()
    fig.savefig(RES / "validate.png", dpi=150)
    print("wrote", RES / "validate.png")


if __name__ == "__main__":
    plots = {"rate": plot_rate, "size": plot_size, "indegree": plot_indegree,
             "dense": plot_dense, "validate": plot_validate}
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (plots if which == "all" else [which]):
        plots[name]()
