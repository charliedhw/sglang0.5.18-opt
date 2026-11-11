"""Measure the real MoE grouped fp8 GEMM at GLM-5.3's prefill shapes.

It is 13.7% of the 0%-cache prefill profile (1824 calls x 2647 us).  Deriving its efficiency from
the profile alone gave ~228 GB/s of weight traffic against a 1361 GB/s copy roof, which would be a
6x gap -- large enough that it is more likely my assumption about how many experts each call
touches is wrong than that the vendor kernel is that far off.  So call the same deepgemm entry
point sglang calls, at the model's own shapes, and read the number off directly.

GLM-5.3: hidden 6144, moe_inter 2048, 256 experts, topk 8, EP8 -> 32 local experts.
CP8 prefill chunk 32768 -> 4096 tokens/rank -> 4096 expert-rows/rank (topk 8 / EP 8).
"""
import time

import torch
import deepgemm

DEV = "cuda:0"
H, I, E_LOCAL = 6144, 2048, 32
FP8 = torch.float8_e4m3fn


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def run(label, rows, k, n, e_local):
    # a: (fp8 [rows,k], scale [rows, k/128]) ; b: (fp8 [E,n,k], scale [E, n/128, k/128])
    a_q = torch.randn(rows, k, device=DEV).to(FP8)
    a_s = torch.randn(rows, k // 128, device=DEV, dtype=torch.float32).abs() + 0.1
    b_q = torch.randn(e_local, n, k, device=DEV).to(FP8)
    b_s = torch.randn(e_local, n // 128, k // 128, device=DEV, dtype=torch.float32).abs() + 0.1
    out = torch.empty(rows, n, device=DEV, dtype=torch.bfloat16)
    # contiguous layout: rows are already grouped by expert
    per = rows // e_local
    m_idx = torch.arange(e_local, device=DEV, dtype=torch.int32).repeat_interleave(per)

    try:
        s = timed(lambda: deepgemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_q, a_s), (b_q, b_s), out, m_idx))
    except Exception as exc:
        print(f"  {label}: FAILED {type(exc).__name__}: {str(exc)[:160]}")
        return

    w_bytes = e_local * n * k                 # every local expert's weights are read
    io_bytes = rows * k + rows * n * 2
    flops = 2 * rows * k * n
    print(f"  {label:10s} rows={rows} k={k} n={n} experts={e_local} ({per} rows/expert)")
    print(f"             {s*1e6:9.1f} us   weights {w_bytes/1e6:7.1f} MB"
          f"   {(w_bytes+io_bytes)/s/1e9:7.0f} GB/s   {flops/s/1e12:6.1f} TFLOPS")


def main():
    print(f"deepgemm {getattr(deepgemm,'__version__','?')}  roof: 1361 GB/s copy")
    print("\n--- prefill shapes (CP8, chunk 32768 -> 4096 rows/rank) ---")
    run("gate_up", 4096, H, 2 * I, E_LOCAL)
    run("down", 4096, I, H, E_LOCAL)
    print("\n--- half and double, to see how it scales with M ---")
    run("gate_up", 2048, H, 2 * I, E_LOCAL)
    run("gate_up", 8192, H, 2 * I, E_LOCAL)
    print("\nProfile observed: 2647 us mean over 1824 calls (gate_up and down mixed).")


main()