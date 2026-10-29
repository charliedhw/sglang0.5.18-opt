"""hcu_ops.skinny_nn (bf16 skinny GEMM on an n-contiguous [K, N] weight, the NN lm_head
layout) vs torch.matmul on the same weight: error vs an fp32 reference and GPU time
(graph replay over weight copies that don't fit in L2).
usage: test_nn.py [M list] [N list] [sweep]"""
import sys

import torch
import hcu_ops
from test_skinny import bench, rel

dev = "cuda"
torch.manual_seed(0)
K = 6144
MS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 4, 6, 8, 12, 16, 24, 32]
NS = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [154880, 19360]
SWEEP = len(sys.argv) > 3 and sys.argv[3] == "sweep"
HP = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
CFGS = [dict(nw=nw, h=h, pf=pf, stream_w=st) for nw in (1, 2, 4, 8) for h, pf in HP for st in (False, True)]


def tag(c):
    return f"nw{c['nw']}h{c['h']}p{c['pf']}{'s' if c['stream_w'] else ''}"


for n in NS:
    nc = max(2, -(-(1 << 30) // (2 * n * K)))
    wl = [(torch.randn(K, n, device=dev) * 0.02).to(torch.bfloat16) for _ in range(nc)]
    gb = 2 * n * K / 1e9
    print(f"N={n} K={K}: {nc} weight copies, {gb:.3f} GB each", flush=True)
    for m in MS:
        x = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
        ref = x.float() @ wl[0].float()
        out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        e_mm = rel(torch.matmul(x, wl[0]), ref)
        hcu_ops.skinny_nn(x, wl[0], out=out)
        torch.cuda.synchronize()
        e_nn = rel(out, ref)
        t_mm = bench(lambda i: torch.matmul(x, wl[i % nc]))
        t_nn = bench(lambda i: hcu_ops.skinny_nn(x, wl[i % nc], out=out))
        line = (f"M={m:3d} matmul {t_mm:7.1f} us ({gb / t_mm * 1e3:4.2f} TB/s) err {e_mm:.1e} | "
                f"nn-default {t_nn:7.1f} us ({gb / t_nn * 1e3:4.2f} TB/s) err {e_nn:.1e}")
        if SWEEP:
            res, worst = [], 0.0
            for c in CFGS:
                out.zero_()
                hcu_ops.skinny_nn(x, wl[0], out=out, **c)
                torch.cuda.synchronize()
                worst = max(worst, rel(out, ref))
                res.append((bench(lambda i: hcu_ops.skinny_nn(x, wl[i % nc], out=out, **c)), tag(c)))
            res.sort()
            line += " | " + " ".join(f"[{t:.1f} {s}]" for t, s in res[:4]) + f" maxerr {worst:.1e}"
        print(line, flush=True)
    del wl, ref
    torch.cuda.empty_cache()
