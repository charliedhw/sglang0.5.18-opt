"""Measure the achievable HBM bandwidth on this HCU, then compare the two prefill hot kernels
against it.

The 90%-cache prefill profile is dominated by _mqa_logits (18.3%) and _dequantize_k_cache_paged
(12.2%).  Neither has an alternative backend -- the indexer only has a lightop implementation on
HCU, and the attention backends that skip the dequant were measured slower overall.  So the
question is whether these two are already at the memory-bandwidth roof, which decides if a kernel
rewrite is worth attempting at all.

Run inside the glm5.3 container on an idle card.
"""
import time

import torch

DEV = "cuda:0"


def bw(nbytes, secs):
    return nbytes / secs / 1e9


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


print("=== roof: plain device-to-device copy")
for mb in (256, 1024):
    n = mb * 1024 * 1024
    src = torch.empty(n, dtype=torch.uint8, device=DEV)
    dst = torch.empty_like(src)
    s = timed(lambda: dst.copy_(src))
    print(f"  copy {mb:5d} MB   {s*1e6:9.1f} us   {bw(2*n, s):7.0f} GB/s (read+write)")

print("\n=== fp8 -> bf16 widening, the shape the K-cache dequant does")
# DIM_NOPE + DIM_ROPE for GLM-5.3 is 576 per token
DIM = 576
for ntok in (131072, 392832):
    src = torch.empty((ntok, DIM), dtype=torch.uint8, device=DEV)
    out = torch.empty((ntok, DIM), dtype=torch.bfloat16, device=DEV)
    def go():
        out.copy_(src.view(torch.uint8).to(torch.bfloat16))
    s = timed(go, warmup=3, iters=10)
    nbytes = ntok * DIM * (1 + 2)
    print(f"  {ntok:7d} tok  {s*1e6:9.1f} us   {bw(nbytes, s):7.0f} GB/s"
          f"   ({nbytes/1e6:.0f} MB touched)")

print("\nFor reference, the profile's dequant kernel at grid[392832,5]: 7292 us")
print("and at grid[131072,5]: 2435 us.")