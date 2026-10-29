"""Per-kernel totals of a torch-profiler GPU trace, plus busy / idle of the GPU timeline.
usage: python kagg.py trace.json.gz [topn]"""
import collections
import gzip
import json
import re
import sys

ev = json.load(gzip.open(sys.argv[1]))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
topn = int(sys.argv[2]) if len(sys.argv) > 2 else 45
ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "Kernel")),
            key=lambda e: e["ts"])
if not ks:
    print("no kernel events; categories:", collections.Counter(e.get("cat") for e in ev).most_common(20))
    sys.exit()
busy, cs, ce, gaps = 0.0, None, None, []
for e in ks:
    s, en = e["ts"], e["ts"] + e["dur"]
    if cs is None:
        cs, ce = s, en
    elif s > ce:
        busy += ce - cs
        gaps.append(s - ce)
        cs, ce = s, en
    else:
        ce = max(ce, en)
busy += ce - cs
c, t = collections.Counter(), collections.Counter()
for e in ks:
    n = re.sub(r"<.*", "", e["name"])[:100]
    c[n] += 1
    t[n] += e["dur"]
tot = sum(t.values())
big = sorted(g for g in gaps if g > 1000)
print(f"span {(ce - ks[0]['ts']) / 1e3:.1f} ms  kernel sum {tot / 1e3:.1f} ms  busy {busy / 1e3:.1f} ms  "
      f"{len(ks)} kernels  idle gaps>1ms: {len(big)} ({sum(big) / 1e3:.1f} ms)")
for k, v in sorted(t.items(), key=lambda kv: -kv[1])[:topn]:
    print(f"{v / 1e3:9.2f} ms {100 * v / tot:5.1f}% {c[k]:6d}x {v / c[k]:8.1f} us  {k}")
