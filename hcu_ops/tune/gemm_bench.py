"""Decode-shape dense GEMM microbenchmark for GLM-5.3 (per DP-attention rank).

FP8 w8a8 per-channel linears go through deepgemm.fp8_gemm (hipBLASLt heuristic),
exactly as sglang's apply_fp8_linear does on HCU; bf16 linears through F.linear.
Weights rotate over several copies so every call streams from HBM like in the
real 78-layer forward.
"""
import sys

import torch
import torch.nn.functional as F
import deepgemm

dev = "cuda"
torch.manual_seed(0)
FP8 = torch.float8_e4m3fn

FP8_SHAPES = {  # name: (N, K)
    "qkv_a": (2624, 6144),
    "q_b": (16384, 2048),
    "o_proj": (6144, 16384),
    "idx_wq_b": (4096, 2048),
    "dense_gu": (24576, 6144),
    "dense_dn": (6144, 12288),
}
BF16_SHAPES = {
    "sh_gate_up": (4096, 6144),
    "sh_down": (6144, 2048),
    "router": (256, 6144),
}
MS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 4, 6, 12, 24, 48, 96, 192]


def bench(fn, iters=30, reps=5):
    fn()
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


def copies(nbytes):
    return max(2, -(-(512 << 20) // nbytes))


def quant_rows(x):
    s = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 448.0
    return (x.float() / s).to(FP8), s


def hbm_bw():
    a = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    b = torch.empty_like(a)
    t = bench(lambda i=0: b.copy_(a), iters=10)
    print(f"HBM copy 1GiB: {t:.0f} us -> {2 * a.numel() / t / 1e6:.2f} TB/s (r+w)")
    del a, b


def main():
    hbm_bw()
    print("\nFP8 w8a8 per-channel (deepgemm.fp8_gemm / hipBLASLt): us, TB/s of weight, rel-err vs fp32")
    for name, (n, k) in FP8_SHAPES.items():
        nc = copies(n * k)
        w32 = torch.randn(n, k, device=dev) * 0.02
        wq, ws = quant_rows(w32)
        ws = ws.view(-1).contiguous()
        ws_list = [wq.t()] + [wq.clone().t() for _ in range(nc - 1)]
        for m in MS:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            xq, xs = quant_rows(x)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            fn = lambda i=0: deepgemm.fp8_gemm((xq, xs), (ws_list[i % nc], ws), out)
            t = bench(fn)
            ref = (xq.float() * xs) @ (wq.float() * ws[:, None]).t()
            fn()
            err = ((out.float() - ref).norm() / ref.norm()).item()
            print(f"{name:9s} N={n:6d} K={k:6d} M={m:4d}  {t:8.1f} us  {n * k / t / 1e6:5.2f} TB/s  err={err:.1e}")
        del ws_list
    print("\nBF16 F.linear")
    for name, (n, k) in BF16_SHAPES.items():
        nc = copies(n * k * 2)
        wl = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(nc)]
        for m in MS:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            t = bench(lambda i=0: F.linear(x, wl[i % nc]))
            print(f"{name:10s} N={n:6d} K={k:6d} M={m:4d}  {t:8.1f} us  {2 * n * k / t / 1e6:5.2f} TB/s")
        del wl


if __name__ == "__main__":
    main()
