"""Per-rank GPU busy / span and a few kernel-group totals for a directory of torch-profiler traces.
usage: rankspan.py trace_dir"""
import glob
import gzip
import json
import re
import sys

groups = {"notify_combine": "cached_notify_combine", "combine": r"intranode::combine",
          "dispatch": r"intranode::dispatch", "moe_gemm": "m_grouped_fp8", "attn": "sparse_attn_fwd"}
for f in sorted(glob.glob(sys.argv[1] + "/*.trace.json.gz")):
    ev = [e for e in json.load(gzip.open(f))["traceEvents"]
          if e.get("ph") == "X" and e.get("cat") == "kernel"]
    t0 = min(e["ts"] for e in ev)
    t1 = max(e["ts"] + e["dur"] for e in ev)
    busy = sum(e["dur"] for e in ev)
    g = {k: sum(e["dur"] for e in ev if re.search(p, e["name"])) / 1e3 for k, p in groups.items()}
    rank = re.search(r"TP-(\d+)", f).group(1)
    print(f"rank{rank} span {(t1 - t0) / 1e3:8.1f} busy {busy / 1e3:8.1f} ms  "
          + "  ".join(f"{k} {v:6.1f}" for k, v in g.items()))
