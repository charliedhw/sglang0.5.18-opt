"""Compare two torch-profiler GPU traces (same number of profiled steps): busy / span / idle
gaps of each, then per-kernel total time A vs B sorted by |difference|.
usage: python tracediff.py a.trace.json.gz b.trace.json.gz [topn]"""
import collections
import gzip
import json
import re
import sys


def load(path):
    ev = json.load(gzip.open(path))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                key=lambda e: e["ts"])
    busy, gaps, cs, ce = 0.0, [], None, None
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
    span = ce - ks[0]["ts"]
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in ks:
        a = agg[re.sub(r"<.*", "", e["name"])[:90]]
        a[0] += e["dur"]
        a[1] += 1
    big = sorted(g for g in gaps if g > 200)
    print(f"{path.replace(chr(92), '/').split('/')[-1][:60]}: span {span / 1e3:.1f} ms  busy {busy / 1e3:.1f} ms  "
          f"idle {(span - busy) / 1e3:.1f} ms  (gaps>200us: {len(big)}, sum {sum(big) / 1e3:.1f} ms, "
          f"largest {[round(g / 1e3, 2) for g in big[-6:]]})")
    return agg


a, b = load(sys.argv[1]), load(sys.argv[2])
topn = int(sys.argv[3]) if len(sys.argv) > 3 else 40
rows = [(n, a.get(n, [0, 0]), b.get(n, [0, 0])) for n in set(a) | set(b)]
rows.sort(key=lambda r: -abs(r[1][0] - r[2][0]))
print(f"{'A ms':>9} {'nA':>6} {'B ms':>9} {'nB':>6} {'B-A':>8}  kernel")
for n, (ta, ca), (tb, cb) in rows[:topn]:
    print(f"{ta / 1e3:9.2f} {ca:6d} {tb / 1e3:9.2f} {cb:6d} {(tb - ta) / 1e3:8.2f}  {n}")
print(f"total kernel time A {sum(v[0] for v in a.values()) / 1e3:.1f} ms  B {sum(v[0] for v in b.values()) / 1e3:.1f} ms")
