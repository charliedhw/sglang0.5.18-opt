"""Add the -1015 (w8a8 FP8_DYNAMIC) skinny rows tuned in /tmp/tune_w1015_fp8 (tune.log:
idx_wq_b / sh_dn8 / idx_wk, ~1-1.8 us per call for M <= 8) to the installed
hcu_ops/skinny_cfg.json (backup *.bak_cfg2) and to the source tree copy."""
import json
import os
import shutil

import hcu_ops

ADD = {
    "4096,2048": [[1, 2, 2, 1], [4, 4, 2, 1], [6, 2, 2, 1], [8, 4, 2, 1]],  # indexer wq_b
    "6144,2048": [[2, 2, 1, 1], [8, 2, 2, 2]],                              # shared expert down
    "128,6144": [[3, 2, 8, 1], [4, 4, 8, 1], [5, 2, 8, 1]],                 # indexer wk
}
SRC = "/data/home/daihw/scripts/glm5.3/opt2/hcu_ops/hcu_ops/skinny_cfg.json"
dst = os.path.join(os.path.dirname(hcu_ops.__file__), "skinny_cfg.json")
for path in (dst, SRC):
    if not os.path.exists(path):
        print("skip (missing)", path)
        continue
    bak = path + ".bak_cfg2"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    cfg = json.load(open(bak))
    for key, rows in ADD.items():
        cfg["fp8"][key] = rows
    json.dump(cfg, open(path, "w"), indent=1)
    print("wrote", path, sorted(cfg["fp8"]))
