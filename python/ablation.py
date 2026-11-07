"""Ablation: each optimization alone and all together.

    python3 python/ablation.py  -> results/ablation.jsonl, results/ablation.md

Median of REPS runs per setting.
"""
import json
import pathlib
import statistics
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNN = str(ROOT / "build" / "snn")
REPS = 5
STRATS = ["naive", "push", "pull"]

CONFIGS = [
    ("baseline", []),
    ("+stateless-rng", ["--stateless-rng"]),
    ("+block-compact", ["--block-compact"]),
    ("+sort-rows", ["--sort-rows"]),
    ("+graph100", ["--graph", "100"]),
    ("all", ["--opt-all"]),
]

WORKLOADS = [
    ("N=12.5k Brunel (43 Hz)", ["--N", "12500", "--K", "1000", "--T", "500"]),
    ("N=100k fixed 20 Hz", ["--N", "100000", "--K", "1000", "--T", "200", "--fixed-rate", "20"]),
]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    r.check_returncode()
    return json.loads(r.stdout.strip().splitlines()[-1])


out = ROOT / "results" / "ablation.jsonl"
out.unlink(missing_ok=True)
median_ms = {}  # (workload, strategy, config) -> median ms/step

for wname, wflags in WORKLOADS:
    for strat in STRATS:
        for cname, cflags in CONFIGS:
            results = [run([SNN, "--strategy", strat, *wflags, *cflags]) for _ in range(REPS)]
            med = statistics.median(d["ms_per_step"] for d in results)
            row = results[0] | {"config": cname, "workload": wname, "ms_per_step_median": med}
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            median_ms[(wname, strat, cname)] = med
            print(f"{wname:24s} {strat:5s} {cname:16s} {med * 1e3:8.1f} us/step  "
                  f"(rate {row['rate_hz']:.1f} Hz)", flush=True)

# markdown table: us/step and speedup vs baseline
lines = []
for wname, _ in WORKLOADS:
    lines += [f"\n### {wname}: median us/step (speedup vs baseline)\n",
              "| config | " + " | ".join(STRATS) + " |",
              "|---|" + "---|" * len(STRATS)]
    for cname, _ in CONFIGS:
        cells = []
        for strat in STRATS:
            t = median_ms[(wname, strat, cname)]
            base = median_ms[(wname, strat, "baseline")]
            cells.append(f"{t * 1e3:.1f} ({base / t:.2f}x)")
        lines.append(f"| {cname} | " + " | ".join(cells) + " |")

md = "\n".join(lines)
print(md)
(ROOT / "results" / "ablation.md").write_text(md + "\n")
