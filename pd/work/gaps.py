"""List the largest idle gaps between consecutive GPU kernels in one torch-profiler trace,
with the kernels on both sides.  usage: gaps.py trace.json.gz [min_gap_us=500] [topn=25]"""
import collections
import gzip
import json
import sys

ev = sorted((e for e in json.load(gzip.open(sys.argv[1]))["traceEvents"]
             if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 500
topn = int(sys.argv[3]) if len(sys.argv) > 3 else 25
gaps = []
end = ev[0]["ts"] + ev[0]["dur"]
for a, b in zip(ev, ev[1:]):
    end = max(end, a["ts"] + a["dur"])
    g = b["ts"] - end
    if g >= thr:
        gaps.append((g, a["name"][:60], b["name"][:60]))
tot = sum(g for g, _, _ in gaps)
print(f"{len(gaps)} gaps >= {thr} us, total {tot / 1e3:.1f} ms")
by = collections.Counter()
for g, a, b in gaps:
    by[(a, b)] += g
for (a, b), g in by.most_common(topn):
    print(f"{g / 1e3:8.1f} ms  {a}  ->  {b}")
