"""skinny_tune.json (tune_skinny.py) -> hcu_ops/skinny_cfg.json for the sglang hook.
A tuned M uses skinny_gemm when it beats the default GEMM by >= MIN_GAIN (relative)
and >= MIN_US; rows cover (previous tuned M, M]; equal neighbours are merged.
usage: make_cfg.py [skinny_tune.json ...]  (later files override earlier shapes)"""
import json
import sys

MIN_GAIN, MIN_US = 0.05, 1.0

tune = {}
for path in sys.argv[1:] or ["skinny_tune.json"]:
    tune.update(json.load(open(path)))
cfg = {"fp8": {}, "bf16": {}}
for name, t in tune.items():
    rows = []
    for m, r in sorted(t["M"].items(), key=lambda kv: int(kv[0])):
        us, ver, nw, nt = r["best"]
        win = us <= r["base"] * (1 - MIN_GAIN) and r["base"] - us >= MIN_US
        row = [int(m)] + ([ver, nw, nt] if win else [0, 0, 0])
        if rows and rows[-1][1:] == row[1:]:
            rows[-1][0] = row[0]
        else:
            rows.append(row)
    while rows and rows[-1][1] == 0:  # trailing fallback rows are implicit
        rows.pop()
    print(f"{name:10s} {t['N']},{t['K']}: {rows}")
    if rows:
        cfg["fp8" if t["fp8"] else "bf16"][f"{t['N']},{t['K']}"] = rows
json.dump(cfg, open("hcu_ops/skinny_cfg.json", "w"), indent=1)
