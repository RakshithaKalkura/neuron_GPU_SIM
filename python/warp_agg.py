"""Warp-aggregated atomics experiment (push delivery).

    python3 python/warp_agg.py  -> results/warp_agg.jsonl, results/warp_agg.md

Push is timed with and without --warp-agg, on unsorted and sorted rows, all with
--stateless-rng --graph 100. "saved" = fraction of atomics aggregation removes.
Median of REPS runs per setting.
"""
import json
import pathlib
import statistics
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNN = str(ROOT / "build" / "snn")
REPS = 5
BASE = ["--strategy", "push", "--K", "1000", "--T", "200", "--stateless-rng", "--graph", "100"]

CONFIGS = [
    ("push", []),
    ("push + warp-agg", ["--warp-agg"]),
    ("push, sorted rows", ["--sort-rows"]),
    ("push, sorted rows + warp-agg", ["--sort-rows", "--warp-agg"]),
]

WORKLOADS = [(f"N={n:,}, 20 Hz", ["--N", str(n), "--fixed-rate", "20"])
             for n in [1250, 2500, 5000, 12500, 100_000, 1_000_000]]
WORKLOADS += [(f"N=100,000, {r} Hz", ["--N", "100000", "--fixed-rate", str(r)])
              for r in [5, 200]]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    r.check_returncode()
    return json.loads(r.stdout.strip().splitlines()[-1])


out = ROOT / "results" / "warp_agg.jsonl"
out.unlink(missing_ok=True)
rows = []

for wname, wflags in WORKLOADS:
    big = "1,000,000" in wname  # --agg-stats would copy 4 GB to the CPU
    for cname, cflags in CONFIGS:
        cmd = [SNN, *BASE, *wflags, *cflags]
        results = []
        for i in range(REPS):
            stats_flag = ["--agg-stats"] if i == 0 and not big else []
            results.append(run(cmd + stats_flag))
        med = statistics.median(d["ms_per_step"] for d in results)

        row = results[0] | {"config": cname, "workload": wname, "ms_per_step_median": med}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        saved = row["agg_saved_frac"]  # -1 = not measured
        saved_s = "-" if saved < 0 else f"{saved:.2%}"
        print(f"{wname:20s} {cname:30s} {med * 1e3:8.1f} us/step  saved {saved_s}", flush=True)

# markdown table: one row per workload
lines = ["| workload | atomics saved (unsorted / sorted) | push | + warp-agg | sorted rows | sorted + warp-agg |",
         "|---|---|---|---|---|---|"]
for wname, _ in WORKLOADS:
    r = {d["config"]: d for d in rows if d["workload"] == wname}
    us = {name: d["ms_per_step_median"] * 1e3 for name, d in r.items()}
    s_unsorted = r["push"]["agg_saved_frac"]
    s_sorted = r["push, sorted rows"]["agg_saved_frac"]
    saved = "–" if s_unsorted < 0 else f"{s_unsorted:.2%} / {s_sorted:.1%}"
    plain, agg = us["push"], us["push + warp-agg"]
    plain_s, agg_s = us["push, sorted rows"], us["push, sorted rows + warp-agg"]
    lines.append(f"| {wname} | {saved} | {plain:.1f} | {agg:.1f} ({plain / agg:.2f}x) | "
                 f"{plain_s:.1f} | {agg_s:.1f} ({plain_s / agg_s:.2f}x) |")

md = "\n".join(lines)
print(md)
(ROOT / "results" / "warp_agg.md").write_text(md + "\n")
