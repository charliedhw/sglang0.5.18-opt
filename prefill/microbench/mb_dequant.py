"""Benchmark a vectorised rewrite of _dequantize_k_cache_paged_kernel.

The shipped kernel launches grid (num_tokens, 5): one program per 128-element group, so every
program moves 128 bytes and the 90%-cache prefill profile shows 1.96M workgroups running at
~93 GB/s against a measured 1361 GB/s copy roof.

The same file already contains the fix for its sibling kernel:
    _GATHER_TOKENS_PER_PROG = 4   # 4 tokens x 512 fp8 = 16 B per thread, one vectorised access
This applies that shape to the paged dequant: grid (cdiv(num_tokens, T),), each program handling
T whole tokens (512 nope + 4 scales + 64 rope) instead of one 128-element slice.

Per-token traffic is 656 B read + 1152 B written = 1808 B.
Run inside the glm5.3 container on an idle card.
"""

import time

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache_paged

DEV = "cuda:0"
DIM_NOPE, DIM_ROPE, GROUP = 512, 64, 128
NUM_GROUPS = DIM_NOPE // GROUP       # 4
DIM_QUANT = 656                      # 512 fp8 + 16 scale bytes + 128 rope bytes


@triton.jit
def _dequant_paged_v2(
    out_ptr, q_ptr, s_ptr, rope_ptr, page_ptr,
    num_tokens,
    out_s0, q_s0, s_s0, rope_s0,
    T: tl.constexpr, DN: tl.constexpr, DR: tl.constexpr,
    NG: tl.constexpr, G: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * T + tl.arange(0, T)
    m_t = offs_t < num_tokens
    src = tl.load(page_ptr + offs_t, mask=m_t, other=0).to(tl.int32)

    # nope: one [T, 512] tile per program -> 16 contiguous bytes per thread
    offs_n = tl.arange(0, DN)
    q = tl.load(q_ptr + src[:, None] * q_s0 + offs_n[None, :],
                mask=m_t[:, None], other=0.0).to(tl.float32)
    s = tl.load(s_ptr + src[:, None] * s_s0 + tl.arange(0, NG)[None, :],
                mask=m_t[:, None], other=0.0)
    y = tl.reshape(q, (T, NG, G)) * s[:, :, None]
    tl.store(out_ptr + offs_t[:, None] * out_s0 + offs_n[None, :],
             tl.reshape(y, (T, DN)).to(out_ptr.dtype.element_ty), mask=m_t[:, None])

    offs_r = tl.arange(0, DR)
    r = tl.load(rope_ptr + src[:, None] * rope_s0 + offs_r[None, :],
                mask=m_t[:, None], other=0.0)
    tl.store(out_ptr + offs_t[:, None] * out_s0 + DN + offs_r[None, :],
             r.to(out_ptr.dtype.element_ty), mask=m_t[:, None])


def dequant_v2(quant_k_cache, page_table, tokens_per_prog=4, num_warps=4):
    qk = quant_k_cache.view((-1, DIM_QUANT))
    n = page_table.shape[0]
    out = torch.empty((n, 1, DIM_NOPE + DIM_ROPE), dtype=torch.bfloat16, device=qk.device)
    q = qk[:, :DIM_NOPE]
    s = qk[:, DIM_NOPE:DIM_NOPE + NUM_GROUPS * 4].view(torch.float32)
    rope = qk[:, DIM_NOPE + NUM_GROUPS * 4:].view(torch.bfloat16)
    _dequant_paged_v2[(triton.cdiv(n, tokens_per_prog),)](
        out, q, s, rope, page_table, n,
        out.stride(0), q.stride(0), s.stride(0), rope.stride(0),
        T=tokens_per_prog, DN=DIM_NOPE, DR=DIM_ROPE, NG=NUM_GROUPS, G=GROUP,
        num_warps=num_warps,
    )
    return out


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    torch.manual_seed(0)
    for n_tok in (131072, 392832):
        pool = 1 << 19
        cache = (torch.randn(pool, DIM_QUANT, device=DEV) * 0.1).to(torch.float8_e4m3fn)
        cache = cache.view(pool, 1, DIM_QUANT)
        page = torch.randint(0, pool, (n_tok,), device=DEV, dtype=torch.int32)

        ref = dequantize_k_cache_paged(cache, page)
        got = dequant_v2(cache, page)
        same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
        bad = int((ref != got).sum()) if not same else 0

        t_ref = timed(lambda: dequantize_k_cache_paged(cache, page))
        bytes_moved = n_tok * (DIM_QUANT + (DIM_NOPE + DIM_ROPE) * 2)
        print(f"\n--- {n_tok} tokens ({bytes_moved/1e6:.0f} MB)")
        print(f"  shipped        {t_ref*1e6:9.1f} us   {bytes_moved/t_ref/1e9:7.0f} GB/s")
        for T in (2, 4, 8):
            for w in (4, 8):
                try:
                    t = timed(lambda: dequant_v2(cache, page, T, w))
                except Exception as e:
                    print(f"  v2 T={T} warps={w}: {type(e).__name__}")
                    continue
                print(f"  v2 T={T} warps={w}  {t*1e6:9.1f} us   "
                      f"{bytes_moved/t/1e9:7.0f} GB/s   speedup {t_ref/t:5.2f}x")
        print(f"  bit-exact vs shipped: {same}" + ("" if same else f"  mismatching elems={bad}"))


main()