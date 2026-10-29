"""hcu_ops.fused_add_rms_quant / silu_mul_quant vs the unfused lightop ops they replace:
op.fused_add_rms_norm_opt + per_token_quant_fp8 and fuse_silu_and_mul + per_token_quant_fp8.
All outputs must be bit-identical (torch.equal on x, residual, q bytes, s).  Also times both
inside a CUDA graph (100 calls per replay) at decode sizes.
usage: test_norm_quant.py [--bench-only]"""
import sys

import torch
from lightop import fuse_silu_and_mul, op
from lightop.quant import per_token_quant_fp8

import hcu_ops
from hcu_ops import _C

dev = "cuda"
FP8 = torch.float8_e4m3fn
torch.manual_seed(0)


def ref_quant(y):
    q = torch.empty_like(y, dtype=FP8)
    s = torch.empty(y.shape[0], 1, device=dev, dtype=torch.float32)
    q, s = per_token_quant_fp8(y, dtype=FP8, out_q=q, out_scale=s)
    return q, s


def same(a, b):
    if a.dtype == FP8:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return torch.equal(a, b)


def diff(a, b):
    if a.dtype == FP8:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return int((a != b).sum())


def make_rows(m, h, scale):
    x = (torch.randn(m, h, device=dev) * scale).bfloat16()
    r = (torch.randn(m, h, device=dev) * scale * 4).bfloat16()
    x[:, ::37] *= 16  # outlier channels
    if m > 2:
        x[1] = -r[1]  # residual sum exactly 0 -> normed 0, amax 0
    if m > 3:
        x[2, 5] = 3e4  # one huge element
    return x, r


def check_norm(m, h, eps, mode):
    x, r = make_rows(m, h, 0.5)
    w = (torch.rand(h, device=dev) * 2).bfloat16()
    x1, r1, x2, r2 = x.clone(), r.clone(), x.clone(), r.clone()
    op.fused_add_rms_norm_opt(x1, r1, w, eps)
    q1, s1 = ref_quant(x1)
    q2 = torch.empty_like(x2, dtype=FP8)
    s2 = torch.empty(m, 1, device=dev, dtype=torch.float32)
    _C.fused_add_rms_quant(x2, r2, w, eps, q2, s2, mode)
    torch.cuda.synchronize()
    return [diff(x1, x2), diff(r1, r2), diff(q1, q2), diff(s1, s2)]


def check_silu(m, d):
    g = (torch.randn(m, 2 * d, device=dev) * 3).bfloat16()
    g[:, 3] = 120.0  # exp underflow side
    g[:, 5] = -120.0  # exp overflow side
    if m > 1:
        g[1] = 0.0  # zero row: amax 0
    y = torch.empty(m, d, device=dev, dtype=torch.bfloat16)
    fuse_silu_and_mul(g, y)
    q1, s1 = ref_quant(y)
    q2, s2 = hcu_ops.silu_mul_quant(g)
    torch.cuda.synchronize()
    return [diff(q1, q2), diff(s1, s2)]


def graph_us(fn, reps=100):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (10 * reps)


def bench():
    h = 6144
    w = (torch.rand(h, device=dev) * 2).bfloat16()
    for m in (8, 48, 96, 192):
        x, r = make_rows(m, h, 0.5)

        def unfused():
            op.fused_add_rms_norm_opt(x, r, w, 1e-5)
            ref_quant(x)

        def fused():
            hcu_ops.fused_add_rms_quant(x, r, w, 1e-5)

        print(f"norm+quant H={h} M={m:4d}: lightop {graph_us(unfused):6.2f} us  fused {graph_us(fused):6.2f} us",
              flush=True)
    for d in (2048, 12288):
        for m in (8, 48, 96, 192):
            g = (torch.randn(m, 2 * d, device=dev) * 3).bfloat16()
            y = torch.empty(m, d, device=dev, dtype=torch.bfloat16)

            def unfused():
                fuse_silu_and_mul(g, y)
                ref_quant(y)

            def fused():
                hcu_ops.silu_mul_quant(g)

            print(f"silu+quant d={d:5d} M={m:4d}: lightop {graph_us(unfused):6.2f} us  fused {graph_us(fused):6.2f} us",
                  flush=True)


if "--bench-only" not in sys.argv:
    MS = list(range(1, 17)) + [24, 32, 48, 64, 96, 128, 192, 288, 1000, 3000]
    for mode in (0, 1, 2):
        bad = 0
        for h in (6144, 3072, 5120):
            for eps in (1e-5, 1e-6):
                for m in MS:
                    d = check_norm(m, h, eps, mode)
                    if any(d):
                        bad += 1
                        if bad <= 5:
                            print(f"  norm mode {mode} H={h} eps={eps} M={m}: mismatches x/res/q/s = {d}")
        print(f"fused_add_rms_quant rsqrt_mode {mode}: {'BIT-EXACT' if not bad else f'{bad} cases differ'}",
              flush=True)
    bad = 0
    for d in (2048, 12288, 1536, 256, 7168):
        for m in MS:
            r = check_silu(m, d)
            if any(r):
                bad += 1
                if bad <= 5:
                    print(f"  silu d={d} M={m}: mismatches q/s = {r}")
    print(f"silu_mul_quant: {'BIT-EXACT' if not bad else f'{bad} cases differ'}", flush=True)
bench()
