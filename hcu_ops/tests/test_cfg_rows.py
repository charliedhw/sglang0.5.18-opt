"""Every fp8 row of the installed skinny_cfg.json: skinny_gemm with the configured
(ver, nw, nt) vs an fp32 reference and vs hipBLASLt (deepgemm.fp8_gemm), for each M the
row covers.  Flags rows whose error is clearly worse than hipBLASLt's (a wrong kernel
gives O(1) relative error, a different summation order ~1e-3).
usage: test_cfg_rows.py [N,K filter ...]"""
import sys

import torch
import deepgemm
import hcu_ops

dev = "cuda"
torch.manual_seed(0)
FP8 = torch.float8_e4m3fn
ONLY = set(sys.argv[1:])


def quant_rows(x):
    s = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 448.0
    return (x.float() / s).to(FP8), s


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


hcu_ops.skinny_config(True, 1, 1, 1)
bad = 0
for (n, k), rows in sorted(hcu_ops._SKINNY["fp8"].items()):
    key = f"{n},{k}"
    if ONLY and key not in ONLY:
        continue
    w32 = torch.randn(n, k, device=dev) * 0.02
    wq, ws = quant_rows(w32)
    ws = ws.view(-1).contiguous()
    wd = wq.float() * ws[:, None]
    lo = 0
    for max_m, ver, nw, nt in rows:
        for m in range(lo + 1, max_m + 1):
            if not ver:
                continue
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            x[:, ::7] *= 8  # some outliers, like real activations
            xq, xs = quant_rows(x)
            ref = (xq.float() * xs) @ wd.t()
            out_h = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            out_s = torch.full_like(out_h, float("nan"))
            deepgemm.fp8_gemm((xq, xs), (wq.t(), ws), out_h)
            hcu_ops.skinny_gemm(xq, wq, xs, ws, out_s, nw=nw, nt=nt, ver=ver)
            torch.cuda.synchronize()
            es, eh = rel(out_s, ref), rel(out_h, ref)
            same = torch.equal(out_s, out_h)
            flag = "" if es <= max(2 * eh, 4e-3) and not out_s.isnan().any() else "  <-- BAD"
            bad += bool(flag)
            print(f"N={n:6d} K={k:6d} M={m:3d} v{ver} nw{nw} nt{nt}  skinny {es:.2e} hipblaslt {eh:.2e}"
                  f" bitexact_vs_hipblaslt={same}{flag}", flush=True)
        lo = max_m
print("BAD rows:", bad)
