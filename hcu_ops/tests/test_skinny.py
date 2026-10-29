"""hcu_ops.skinny_gemm: correctness vs fp32 reference and speed vs the current paths
(deepgemm.fp8_gemm = hipBLASLt for fp8 w8a8, F.linear for bf16).
usage: test_skinny.py [M list] [sweep] [shape filter]"""
import itertools
import sys

import torch
import torch.nn.functional as F
import deepgemm
import hcu_ops

dev = "cuda"
torch.manual_seed(0)
FP8 = torch.float8_e4m3fn
FP8_SHAPES = {"qkv_a": (2624, 6144), "q_b": (16384, 2048), "o_proj": (6144, 16384),
              "idx_wq_b": (4096, 2048), "dense_gu": (24576, 6144), "dense_dn": (6144, 12288),
              # GLM-5.3-Channel-FP8-w8a8-1015: shared experts + indexer wk are fp8 too
              "sh_gu8": (4096, 6144), "sh_dn8": (6144, 2048), "idx_wk": (128, 6144)}
BF16_SHAPES = {"sh_gate_up": (4096, 6144), "sh_down": (6144, 2048), "eh_proj": (6144, 12288),
               # lm_head: full vocab (dp-lm-head) and draft vocab-parallel 8
               "lm_full": (154880, 6144), "lm_vp8": (19360, 6144)}
MS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 6, 12, 24, 48]
SWEEP = len(sys.argv) > 2 and sys.argv[2] == "sweep"
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
VERS = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [2, 4, 5]


def bench(fn, iters=30, reps=5):
    fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(iters):
            fn(i)
    best = 1e9
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / iters)
    return best


def quant_rows(x):
    s = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 448.0
    return (x.float() / s).to(FP8), s


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def sweep(call, n, m, ref, out):
    """call(i, **cfg) -> run; returns top-3 configs and the worst error over them."""
    res, worst = [], 0.0
    for ver, nw, nt in itertools.product(VERS, [1, 2, 4, 8], [1, 2, 4]):
        if nw * nt > (16 if ver >= 4 else 8) or n % (16 * nt):
            continue
        cfg = dict(ver=ver, nw=nw, nt=nt)
        out.zero_()
        try:
            call(0, **cfg)
        except RuntimeError:  # LDS limit at this M
            continue
        torch.cuda.synchronize()
        worst = max(worst, rel(out, ref))
        res.append((bench(lambda i: call(i, **cfg)), ver, nw, nt))
    res.sort()
    return " ".join(f"[{t:.1f} v{v} nw{a} nt{b}]" for t, v, a, b in res[:3]) + f" maxerr {worst:.1e}"


def run_fp8():
    print("FP8 w8a8: hipBLASLt(us) skinny-v2-default(us) [TB/s] err_skinny err_hipblaslt | sweep top3")
    for name, (n, k) in FP8_SHAPES.items():
        if ONLY and name not in ONLY:
            continue
        nc = max(2, -(-(512 << 20) // (n * k)))
        w32 = torch.randn(n, k, device=dev) * 0.02
        w32[0, :64] = 1e-4  # some fp8 subnormals
        wq, ws = quant_rows(w32)
        del w32
        ws = ws.view(-1).contiguous()
        wl = [wq] + [wq.clone() for _ in range(nc - 1)]
        for m in MS:
            xq, xs = quant_rows(torch.randn(m, k, device=dev, dtype=torch.bfloat16))
            ref = (xq.float() * xs) @ (wq.float() * ws[:, None]).t()
            out_h = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            out_s = torch.empty_like(out_h)
            deepgemm.fp8_gemm((xq, xs), (wq.t(), ws), out_h)
            hcu_ops.skinny_gemm(xq, wq, xs, ws, out_s)
            torch.cuda.synchronize()
            eh, es = rel(out_h, ref), rel(out_s, ref)
            th = bench(lambda i: deepgemm.fp8_gemm((xq, xs), (wl[i % nc].t(), ws), out_h))
            ts = bench(lambda i: hcu_ops.skinny_gemm(xq, wl[i % nc], xs, ws, out_s))
            best = sweep(lambda i, **c: hcu_ops.skinny_gemm(xq, wl[i % nc], xs, ws, out_s, **c),
                         n, m, ref, out_s) if SWEEP else ""
            print(f"{name:9s} N={n:6d} K={k:6d} M={m:3d}  {th:7.1f} {ts:7.1f}  [{n * k / th / 1e6:4.2f} {n * k / ts / 1e6:4.2f}]"
                  f"  {es:.1e} {eh:.1e} | {best}", flush=True)
        del wl


def run_bf16():
    # sglang stores large bf16 linear weights [K, N]-contiguous (SGLANG_HCU_LINEAR_NN),
    # so the production baseline is F.linear on that NN layout
    print("BF16: F.linear-NT(us) F.linear-NN(us) skinny-default(us) [TB/s NN skinny] err_skinny err_flinear | sweep top3")
    for name, (n, k) in BF16_SHAPES.items():
        if ONLY and name not in ONLY:
            continue
        nc = max(2, -(-(512 << 20) // (2 * n * k)))
        wl = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(nc)]
        wnn = [w.t().contiguous().t() for w in wl]
        for m in MS:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            ref = x.float() @ wl[0].float().t()
            out_s = hcu_ops.skinny_gemm(x, wl[0])
            ef, es = rel(F.linear(x, wnn[0]), ref), rel(out_s, ref)
            tf = bench(lambda i: F.linear(x, wl[i % nc]))
            tn = bench(lambda i: F.linear(x, wnn[i % nc]))
            ts = bench(lambda i: hcu_ops.skinny_gemm(x, wl[i % nc], out=out_s))
            best = sweep(lambda i, **c: hcu_ops.skinny_gemm(x, wl[i % nc], out=out_s, **c),
                         n, m, ref, out_s) if SWEEP else ""
            print(f"{name:10s} N={n:6d} K={k:6d} M={m:3d}  {tf:7.1f} {tn:7.1f} {ts:7.1f}  [{2 * n * k / tn / 1e6:4.2f} {2 * n * k / ts / 1e6:4.2f}]"
                  f"  {es:.1e} {ef:.1e} | {best}", flush=True)
        del wl, wnn


if __name__ == "__main__":
    run_fp8()
    run_bf16()
