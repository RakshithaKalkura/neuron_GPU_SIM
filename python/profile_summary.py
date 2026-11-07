"""Summarize the Nsight reports from scripts/profile.sh.

    python3 python/profile_summary.py     -> prints markdown, writes results/profile/summary.md

Nsight Systems: median per-kernel time (medians are robust to other GPU tenants).
Nsight Compute: memory/compute throughput, DRAM traffic, cache hit rates, coalescing,
                atomic (RED) requests, averaged over the profiled launches.
"""
import csv
import io
import os
import pathlib
import shutil
import re
import statistics
import subprocess

P = pathlib.Path(__file__).resolve().parent.parent / "results" / "profile"
NCU = os.environ.get("NCU") or shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
STRATS = ["naive", "push", "pull", "push-opt"]

M = {  # metric -> header
    "gpu__time_duration.sum": "time (us)",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem SOL %",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM SOL %",
    "dram__bytes_read.sum": "DRAM rd MB",
    "dram__bytes_write.sum": "DRAM wr MB",
    "lts__t_sector_hit_rate.pct": "L2 hit %",
    "l1tex__t_sector_hit_rate.pct": "L1 hit %",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy %",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "_ld_sectors",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "_ld_requests",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum": "atomic reqs",
}
SCALE = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3,               # -> us
         "byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}     # -> MB


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def ncu_table(strat):
    rep = P / f"ncu_{strat}.ncu-rep"
    if not rep.exists():
        return {}
    out = subprocess.run([NCU, "--import", str(rep), "--csv", "--page", "raw", "--metrics",
                          ",".join(M)], capture_output=True, text=True, timeout=600).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr, units, data = rows[0], rows[1], rows[2:]
    idx = {h: i for i, h in enumerate(hdr)}
    per_kernel = {}
    for r in data:
        k = r[idx["Kernel Name"]].split("(")[0]
        d = per_kernel.setdefault(k, {m: [] for m in M})
        for m in M:
            if m in idx and (v := num(r[idx[m]])) is not None:
                d[m].append(v * SCALE.get(units[idx[m]], 1.0))
    res = {}
    for k, d in per_kernel.items():
        avg = {m: (sum(v) / len(v) if v else None) for m, v in d.items()}
        ld_s, ld_r = avg.pop("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"), \
            avg.pop("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
        avg["sectors/load req"] = ld_s / ld_r if ld_s and ld_r else None
        res[k] = avg
    return res


def nsys_medians(strat):
    f = P / f"nsys_{strat}.txt"
    res = {}
    if not f.exists():
        return res
    for line in f.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 9 and cells[-1].startswith("k_"):
            name = cells[-1].split("(")[0]
            res[name] = (num(cells[4]) / 1e3, int(num(cells[2])))   # median us, instances
    return res


def fmt(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if abs(v) < 1e5 else f"{v:.3g}")


lines = []
lines.append("### Nsight Systems: median kernel time per step (us)\n")
lines.append("| strategy | k_lif_update | delivery kernel | delivery | total/step |")
lines.append("|---|---|---|---|---|")
for s in STRATS:
    med = nsys_medians(s)
    lif = med.get("k_lif_update", (None,))[0]
    dk = [k for k in med if k.startswith(("k_push", "k_pull"))]
    dv = med[dk[0]][0] if dk else None
    tot = (lif or 0) + (dv or 0)
    lines.append(f"| {s} | {fmt(lif)} | {dk[0] if dk else '-'} | {fmt(dv)} | {fmt(tot)} |")

cols = list(M.values())
cols = [c for c in cols if not c.startswith("_")] + ["sectors/load req"]
keys = [m for m in M if not M[m].startswith("_")] + ["sectors/load req"]
lines.append("\n### Nsight Compute: per-kernel metrics (avg over profiled launches)\n")
lines.append("| strategy | kernel | " + " | ".join(cols) + " |")
lines.append("|---|---|" + "---|" * len(cols))
for s in STRATS:
    for k, avg in ncu_table(s).items():
        lines.append(f"| {s} | {k} | " + " | ".join(fmt(avg.get(m), 2 if 'MB' in M.get(m, '') else 1)
                                                    for m in keys) + " |")
out = "\n".join(lines)
print(out)
(P / "summary.md").write_text(out + "\n")
