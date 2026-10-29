import torch
from topk_lib import make, bench, same_values, lt_topk
from hcu_ops import _C
import hcu_ops

torch.manual_seed(0)
allok = True
# rows = bs * 6 (MTP verify); production score width = 1M page-table capacity
for B in [6, 12, 24, 36, 48, 192]:
    for L in [16384, 65536, 131072]:
        x, mask, lengths, pt, cu = make(B, L, "relu_sum", W=1 << 20, ragged=True)
        ref = lt_topk(x, mask, lengths, pt, cu, 2048)
        line = f"relu_sum rows={B:4d} L={L:6d} lightop={bench(lambda: lt_topk(x, mask, lengths, pt, cu, 2048), 10):8.1f}"
        fns = {"single": lambda: _C.sparse_mask_topk_transform(x, mask, lengths, pt, cu, 2048),
               "split": lambda: hcu_ops.sparse_mask_topk_transform(x, mask, lengths, pt, cu, 2048)}
        for name, fn in fns.items():
            out = fn()
            torch.cuda.synchronize()
            ok = same_values(x, lengths, ref, out)
            allok &= ok
            line += f" {name}:v{int(ok)}={bench(fn, 10):8.1f}"
        print(line, flush=True)
        del x, mask, pt, ref
        torch.cuda.empty_cache()
# eager large batch (bs 256 * 6 rows)
x, mask, lengths, pt, cu = make(1536, 16384, "relu_sum", W=16448, ragged=True)
ok = same_values(x, lengths, lt_topk(x, mask, lengths, pt, cu, 2048),
                 hcu_ops.sparse_mask_topk_transform(x, mask, lengths, pt, cu, 2048))
allok &= ok
print("rows=1536 valid", ok)
# idle / empty batch
e = torch.empty(0, 4096, device="cuda")
out = hcu_ops.sparse_mask_topk_transform(e, torch.empty(0, 256, dtype=torch.int16, device="cuda"),
                                         torch.empty(0, dtype=torch.int32, device="cuda"),
                                         torch.empty(0, 4096, dtype=torch.int32, device="cuda"),
                                         torch.zeros(1, dtype=torch.int32, device="cuda"))
print("empty", tuple(out.shape))
print("ALL_VALID", allok)
