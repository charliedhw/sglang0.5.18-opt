"""Clean decode step time per load pattern from an SGLang dp-attention server log (needs the
"Decode batch ... accept len ... gen throughput" lines, i.e. speculative decoding on).
Splits the log into 10 s windows, drops every window that contains a prefill on any rank,
and classifies the rest by the per-DP #running-req pattern:
  1rank_r1   : exactly one DP rank decoding 1 request (the c1 scenario)
  allN_rK    : all DP ranks decoding K requests each (balanced cN*K)
Step ms per decode line = #running-req * accept_len / gen_throughput.
Much less noisy than end-to-end TPOT when accept length / prefill interference differ by arm.
usage: [DP=8] [MIN_TOK=50000] python3 ana_clean.py srv_a.out [srv_b.out ...]"""
import os
import re
import statistics as st
import sys

DP = int(os.environ.get("DP", 8))
MIN_TOK = int(os.environ.get("MIN_TOK", 50000))   # only long-context steps (#token >= MIN_TOK)
TS = re.compile(r"^\[\S+ (\d\d):(\d\d):(\d\d) DP(\d+) ")
DEC = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+).*?accept len: ([\d.]+)"
                 r".*?gen throughput \(token/s\): ([\d.]+)")

for path in sys.argv[1:]:
    win = {}
    for line in open(path, errors="replace"):
        m = TS.search(line)
        if not m:
            continue
        w = win.setdefault((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])) // 10,
                           {"r": {}, "ms": [], "acc": [], "prefill": False})
        if "Prefill batch" in line:
            w["prefill"] = True
            continue
        d = DEC.search(line)
        if not d:
            continue
        r, tok, acc, tput = int(d[1]), int(d[2]), float(d[3]), float(d[4])
        w["r"].setdefault(int(m[4]), set()).add(r)
        if tput > 0 and tok >= MIN_TOK:
            w["ms"].append(r * acc / tput * 1e3)
            w["acc"].append(acc)
    cls = {}
    for w in win.values():
        if w["prefill"] or not w["ms"] or any(len(s) > 1 for s in w["r"].values()):
            continue
        rs = sorted(next(iter(s)) for s in w["r"].values())
        if rs == [1]:
            key = "1rank_r1"
        elif len(rs) == DP and rs[0] == rs[-1]:
            key = f"all{DP}_r{rs[0]}"
        else:
            continue
        cls.setdefault(key, ([], []))
        cls[key][0].extend(w["ms"])
        cls[key][1].extend(w["acc"])
    print(f"== {path}")
    for key in sorted(cls):
        ms, acc = sorted(cls[key][0]), cls[key][1]
        q = st.quantiles(ms, n=4) if len(ms) > 1 else [ms[0]] * 3
        print(f"  {key:9s} n={len(ms):4d}  step ms median {st.median(ms):6.1f}  "
              f"p25 {q[0]:6.1f}  p75 {q[2]:6.1f}  accept {st.median(acc):.2f}")
