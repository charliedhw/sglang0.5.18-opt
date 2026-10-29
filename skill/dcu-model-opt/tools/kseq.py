"""Kernel sequence of one torch-profiler GPU trace between two occurrences of an anchor
kernel (e.g. one decoder layer between consecutive attention kernels), in start order,
with start offset, duration and stream - to tell which op a graph-replayed kernel belongs to
and to build the per-layer time table.
usage: python kseq.py trace.json.gz anchor_regex [occurrence=10] [count=1]"""
import gzip
import json
import re
import sys

ev = json.load(gzip.open(sys.argv[1]))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
anchor = re.compile(sys.argv[2])
occ = int(sys.argv[3]) if len(sys.argv) > 3 else 10
cnt = int(sys.argv[4]) if len(sys.argv) > 4 else 1
ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
idx = [i for i, e in enumerate(ks) if anchor.search(e["name"])]
a, b = idx[occ], idx[min(occ + cnt, len(idx) - 1)]
t0 = ks[a]["ts"]
tot = 0.0
for e in ks[a:b]:
    n = re.sub(r"^void ", "", re.sub(r"<.*", "", e["name"]))[:90]
    tot += e["dur"]
    print(f"{e['ts'] - t0:9.1f} {e['dur']:7.1f} s{e.get('args', {}).get('stream', '?'):<3} {n}")
print(f"{b - a} kernels, kernel sum {tot:.1f} us, span {ks[b]['ts'] - t0:.1f} us")
