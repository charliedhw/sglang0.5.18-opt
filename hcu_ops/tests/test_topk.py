import sys
import torch
from topk_lib import make, bench, same_values, lt_topk
from hcu_ops import _C

torch.manual_seed(0)
print("CUs", torch.cuda.get_device_properties(0).multi_processor_count, flush=True)

quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
impls = {
    "single": lambda x, m, l, p, c: _C.sparse_mask_topk_transform(x, m, l, p, c, 2048),
    "s4": lambda x, m, l, p, c: _C.sparse_mask_topk_transform_split(x, m, l, p, c, 2048, 4),
    "s8": lambda x, m, l, p, c: _C.sparse_mask_topk_transform_split(x, m, l, p, c, 2048, 8),
    "s16": lambda x, m, l, p, c: _C.sparse_mask_topk_transform_split(x, m, l, p, c, 2048, 16),
}
allok = True
for dist in ["pos", "sparse30", "sparse2", "relu_sum", "mixed", "ties", "zeros", "neg",
             "narrow", "const"]:
    for B in ([6] if quick else [6, 24]):
        for L in [1000, 2048, 2049, 4096, 16384, 65536, 131072]:
            for ragged in [False, True]:
                if dist in ("zeros", "neg", "ties") and L in (2048,):
                    continue
                # ragged rows also emulate a graph-padded width (W >> len)
                W = 2 * L + 64 if ragged and L >= 16384 else L + 64
                x, mask, lengths, pt, cu = make(B, L, dist, W=W, ragged=ragged)
                ref = lt_topk(x, mask, lengths, pt, cu, 2048)
                line = f"{dist:9s} B={B:3d} L={L:7d} rag={int(ragged)}"
                timed = L >= 4096 and (not ragged or L >= 65536)
                if timed:
                    t_lt = bench(lambda: lt_topk(x, mask, lengths, pt, cu, 2048))
                    line += f" lightop={t_lt:6.1f}"
                for name, fn in impls.items():
                    out = fn(x, mask, lengths, pt, cu)
                    torch.cuda.synchronize()
                    sval = same_values(x, lengths, ref, out)
                    allok &= sval
                    line += f" {name}:v{int(sval)}"
                    if timed:
                        line += f"={bench(lambda: fn(x, mask, lengths, pt, cu)):6.1f}"
                print(line, flush=True)
# production layout: score width = page-table capacity (--context-length 1M)
for dist in ["relu_sum", "narrow"]:
    for L in [16384, 65536, 131072]:
        x, mask, lengths, pt, cu = make(6, L, dist, W=1 << 20, ragged=True)
        ref = lt_topk(x, mask, lengths, pt, cu, 2048)
        line = f"W1M {dist:9s} B=  6 L={L:7d} lightop={bench(lambda: lt_topk(x, mask, lengths, pt, cu, 2048)):6.1f}"
        for name, fn in impls.items():
            out = fn(x, mask, lengths, pt, cu)
            torch.cuda.synchronize()
            sval = same_values(x, lengths, ref, out)
            allok &= sval
            line += f" {name}:v{int(sval)}={bench(lambda: fn(x, mask, lengths, pt, cu)):6.1f}"
        print(line, flush=True)
        del x, mask, pt
print("ALL_VALID", allok)
