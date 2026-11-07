"""Spike-train statistics. Spike files are int32 pairs (step, neuron_id)."""
import numpy as np


def load_spikes(path):
    a = np.fromfile(path, dtype=np.int32).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def stats(t_steps, ids, n_rec, T_ms, dt=0.1, t_skip_ms=100.0):
    """Mean rate (Hz), mean CV(ISI) and Fano factor of the population count (5 ms bins).
    The first t_skip_ms are dropped (start-up transient)."""
    keep = t_steps * dt >= t_skip_ms
    t = t_steps[keep] * dt
    ids = ids[keep]
    duration_s = (T_ms - t_skip_ms) * 1e-3
    rate = len(t) / (n_rec * duration_s)

    # CV(ISI) per neuron
    order = np.lexsort((t, ids))
    t, ids = t[order], ids[order]
    bounds = np.searchsorted(ids, np.arange(n_rec + 1))
    cvs = []
    for i in range(n_rec):
        ti = t[bounds[i]:bounds[i + 1]]
        if len(ti) >= 3:
            isi = np.diff(ti)
            if isi.mean() > 0:
                cvs.append(isi.std() / isi.mean())

    counts, _ = np.histogram(t, bins=np.arange(t_skip_ms, T_ms + 1e-9, 5.0))
    fano = counts.var() / counts.mean() if counts.mean() > 0 else float("nan")
    return {"rate_hz": rate,
            "cv_isi": float(np.mean(cvs)) if cvs else float("nan"),
            "fano_pop": float(fano),
            "n_spikes": int(len(t))}


def raster(ax, t_steps, ids, dt=0.1, max_ids=200, t_window=(0, 500), title=""):
    t_ms = t_steps * dt
    m = (ids < max_ids) & (t_ms >= t_window[0]) & (t_ms < t_window[1])
    ax.plot(t_ms[m], ids[m], "|", ms=2, color="k")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron")
    ax.set_title(title)
