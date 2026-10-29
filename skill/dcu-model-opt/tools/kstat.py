"""Per-kernel stats of one torch-profiler GPU trace: count, total / mean / min / max us,
and the distinct grid / block launch shapes (tells call sites of the same kernel apart),
for kernels whose name matches a regex.
usage: python kstat.py trace.json.gz [name_regex] [topn]"""
import collections
import gzip
import json
import re
import sys

ev = json.load(gzip.open(sys.argv[1]))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
topn = int(sys.argv[3]) if len(sys.argv) > 3 else 30
agg = collections.defaultdict(list)
for e in ev:
    if e.get("ph") != "X" or e.get("cat") != "kernel":
        continue
    if pat and not pat.search(e["name"]):
        continue
    a = e.get("args", {})
    agg[re.sub(r"<.*", "", e["name"])[:80]].append((e["dur"], str(a.get("grid")), str(a.get("block"))))
for n, v in sorted(agg.items(), key=lambda kv: -sum(d for d, _, _ in kv[1]))[:topn]:
    d = [x[0] for x in v]
    print(f"{sum(d) / 1e3:8.2f} ms {len(d):5d}x mean {sum(d) / len(d):7.1f} min {min(d):7.1f} max {max(d):7.1f}  {n}")
    for (g, b), c in collections.Counter((g, b) for _, g, b in v).most_common(4):
        sub = [x[0] for x in v if x[1] == g and x[2] == b]
        print(f"      grid {g} block {b}: {c}x mean {sum(sub) / len(sub):.1f} us")
