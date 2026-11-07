"""Ablation: effect of each optimization alone and all together.

    python3 python/ablation.py      -> results/ablation.jsonl, results/ablation.md

Median of REPS runs per configuration (robust to other GPU tenants).
"""
import json
import pathlib
import statistics
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNN = str(ROOT / "build" / "snn")
REPS = 5
CONFIGS = [  # name, extra flags
    ("baseline", []),
    ("+stateless-rng", ["--stateless-rng"]),
    ("+block-compact", ["--block-compact"]),
    ("+sort-rows", ["--sort-rows"]),
    ("+graph100", ["--graph", "100"]),
    ("all", ["--opt-all"]),
]
WORKLOADS = [  # name, flags
    ("N=12.5k Brunel (43 Hz)", ["--N", "12500", "--K", "1000", "--T", "500"]),
    ("N=100k fixed 20 Hz", ["--N", "100000", "--K", "1000", "--T", "200", "--fixed-rate", "20"]),
]
STRATS = ["naive", "push", "pull"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    r.check_returncode()
    return json.loads(r.stdout.strip().splitlines()[-1])


out = ROOT / "results" / "ablation.jsonl"
out.unlink(missing_ok=True)
table = {}
for wname, wflags in WORKLOADS:
    for s in STRATS:
        for cname, cflags in CONFIGS:
            ds = [run([SNN, "--strategy", s, *wflags, *cflags]) for _ in range(REPS)]
            med = statistics.median(d["ms_per_step"] for d in ds)
            d = ds[0] | {"config": cname, "workload": wname, "ms_per_step_median": med}
            with open(out, "a") as f:
                f.write(json.dumps(d) + "\n")
            table[(wname, s, cname)] = med
            print(f"{wname:24s} {s:5s} {cname:16s} {med * 1e3:8.1f} us/step  "
                  f"(rate {d['rate_hz']:.1f} Hz)", flush=True)

lines = []
for wname, _ in WORKLOADS:
    lines += [f"\n### {wname}: median us/step (speedup vs baseline)\n",
              "| config | " + " | ".join(STRATS) + " |", "|---|" + "---|" * len(STRATS)]
    for cname, _ in CONFIGS:
        cells = []
        for s in STRATS:
            v, b = table[(wname, s, cname)], table[(wname, s, "baseline")]
            cells.append(f"{v * 1e3:.1f} ({b / v:.2f}x)")
        lines.append(f"| {cname} | " + " | ".join(cells) + " |")
md = "\n".join(lines)
print(md)
(ROOT / "results" / "ablation.md").write_text(md + "\n")
