"""Turn the Nsight reports from scripts/profile.sh into results/profile/summary.md.

Nsight Systems: median time of each kernel.
Nsight Compute: bandwidth, DRAM traffic, cache hits, coalescing, atomics (avg over launches).
"""
import csv
import io
import os
import pathlib
import shutil
import subprocess

P = pathlib.Path(__file__).resolve().parent.parent / "results" / "profile"
NCU = os.environ.get("NCU") or shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
STRATS = ["naive", "push", "pull", "push-opt"]

# metric -> column name ("_" = only used to compute sectors/load req)
METRICS = {
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
LD_SECTORS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
LD_REQUESTS = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"

# convert to us and MB
SCALE = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3,
         "byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def ncu_table(strat):
    """{kernel: {metric: average}} from ncu_<strat>.ncu-rep."""
    rep = P / f"ncu_{strat}.ncu-rep"
    if not rep.exists():
        return {}
    # CSV: row 0 = names, row 1 = units, then one row per launch
    out = subprocess.run([NCU, "--import", str(rep), "--csv", "--page", "raw",
                          "--metrics", ",".join(METRICS)],
                         capture_output=True, text=True, timeout=600).stdout
    rows = list(csv.reader(io.StringIO(out)))
    header, units, launches = rows[0], rows[1], rows[2:]
    col = {h: i for i, h in enumerate(header)}

    per_kernel = {}
    for r in launches:
        kernel = r[col["Kernel Name"]].split("(")[0]
        values = per_kernel.setdefault(kernel, {m: [] for m in METRICS})
        for m in METRICS:
            if m in col and (v := num(r[col[m]])) is not None:
                values[m].append(v * SCALE.get(units[col[m]], 1.0))

    result = {}
    for kernel, values in per_kernel.items():
        avg = {m: (sum(v) / len(v) if v else None) for m, v in values.items()}
        sectors, requests = avg.pop(LD_SECTORS), avg.pop(LD_REQUESTS)
        avg["sectors/load req"] = sectors / requests if sectors and requests else None  # 4 = coalesced
        result[kernel] = avg
    return result


def nsys_medians(strat):
    """{kernel: (median us, launches)} from nsys_<strat>.txt."""
    f = P / f"nsys_{strat}.txt"
    result = {}
    if not f.exists():
        return result
    for line in f.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 9 and cells[-1].startswith("k_"):
            name = cells[-1].split("(")[0]
            result[name] = (num(cells[4]) / 1e3, int(num(cells[2])))
    return result


def fmt(v, nd=1):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if abs(v) < 1e5 else f"{v:.3g}"


lines = ["### Nsight Systems: median kernel time per step (us)\n",
         "| strategy | k_lif_update | delivery kernel | delivery | total/step |",
         "|---|---|---|---|---|"]
for s in STRATS:
    med = nsys_medians(s)
    lif = med.get("k_lif_update", (None,))[0]
    delivery = [k for k in med if k.startswith(("k_push", "k_pull"))]
    dv = med[delivery[0]][0] if delivery else None
    total = (lif or 0) + (dv or 0)
    lines.append(f"| {s} | {fmt(lif)} | {delivery[0] if delivery else '-'} | "
                 f"{fmt(dv)} | {fmt(total)} |")

keys = [m for m in METRICS if not METRICS[m].startswith("_")] + ["sectors/load req"]
names = [METRICS.get(m, m) for m in keys]
lines += ["\n### Nsight Compute: per-kernel metrics (avg over profiled launches)\n",
          "| strategy | kernel | " + " | ".join(names) + " |",
          "|---|---|" + "---|" * len(names)]
for s in STRATS:
    for kernel, avg in ncu_table(s).items():
        cells = [fmt(avg.get(m), 2 if "MB" in METRICS.get(m, "") else 1) for m in keys]
        lines.append(f"| {s} | {kernel} | " + " | ".join(cells) + " |")

out = "\n".join(lines)
print(out)
(P / "summary.md").write_text(out + "\n")
