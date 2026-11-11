#!/usr/bin/env python3
"""Replace the paged K-cache dequant Triton kernel with a vectorised one.

The shipped `_dequantize_k_cache_paged_kernel` launches grid (num_tokens, 5): one program per
128-element group, moving 128 bytes each.  On the 90%-cache prefill profile that is 1.96M
workgroups running at 97 GB/s against a measured 1361 GB/s copy roof, and it costs 12.2% of
prefill time -- the single largest cache-specific cost.

This installs `dequantize_k_cache_paged_v2`, which follows the shape the same file already uses
for `gather_dequant_requant_fp8_paged` (_GATHER_TOKENS_PER_PROG = 4): one program per T whole
tokens, so each thread issues one 16-byte vectorised access instead of a 1-byte one.

Microbenchmark on an idle card, 392832 tokens (710 MB moved):
    shipped          7309 us    97 GB/s
    v2 T=8 warps=4    592 us  1200 GB/s   12.3x, torch.equal vs shipped == True

Off by default.  Enable with SGLANG_DSA_FAST_DEQUANT_PAGED=1.
Idempotent: backs the file up once and always re-patches from that backup.
Revert with:  python3 patch_fast_dequant.py --revert
"""
import os
import py_compile
import shutil
import sys

SRC = ("/usr/local/lib/python3.10/dist-packages/sglang/kernels/ops/attention/"
       "dsa/dequant_k_cache.py")
BAK = SRC + ".bak_fastdequant"

ANCHOR = "def dequantize_k_cache_paged(\n"

ADDITION = '''

# ---- added by patch_fast_dequant.py ------------------------------------------------
# Vectorised replacement for the (num_tokens, 5) launch above.  Same numerics: the per-group
# fp32 scale is applied to the same 128-element groups, rope is the same bf16 copy.  The only
# change is how the work is partitioned, so the result is bit-identical.
import os as _fd_os


@triton.jit
def _dequantize_k_cache_paged_v2_kernel(
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


_FD_ON = _fd_os.environ.get("SGLANG_DSA_FAST_DEQUANT_PAGED", "0") == "1"
_FD_T = int(_fd_os.environ.get("SGLANG_DSA_FAST_DEQUANT_TOKENS", "8"))
_FD_WARPS = int(_fd_os.environ.get("SGLANG_DSA_FAST_DEQUANT_WARPS", "4"))
_dequantize_k_cache_paged_orig = dequantize_k_cache_paged


def dequantize_k_cache_paged(quant_k_cache, page_table_1_flattened, group_size: int = 128):
    if not _FD_ON or group_size != 128 or quant_k_cache.shape[-1] != 656:
        return _dequantize_k_cache_paged_orig(
            quant_k_cache, page_table_1_flattened, group_size
        )
    dim_nope, dim_rope, num_groups = 512, 64, 4
    qk = quant_k_cache.view((-1, 656))
    n = page_table_1_flattened.shape[0]
    out = torch.empty((n, 1, dim_nope + dim_rope), dtype=torch.bfloat16, device=qk.device)
    q = qk[:, :dim_nope]
    s = qk[:, dim_nope:dim_nope + num_groups * 4].view(torch.float32)
    rope = qk[:, dim_nope + num_groups * 4:].view(torch.bfloat16)
    _dequantize_k_cache_paged_v2_kernel[(triton.cdiv(n, _FD_T),)](
        out, q, s, rope, page_table_1_flattened, n,
        out.stride(0), q.stride(0), s.stride(0), rope.stride(0),
        T=_FD_T, DN=dim_nope, DR=dim_rope, NG=num_groups, G=group_size,
        num_warps=_FD_WARPS,
    )
    return out
'''


def main():
    if "--revert" in sys.argv:
        if os.path.exists(BAK):
            shutil.copy2(BAK, SRC)
            print("reverted from", BAK)
        else:
            print("no backup, nothing to do")
        return
    if not os.path.exists(BAK):
        shutil.copy2(SRC, BAK)
        print("backed up ->", BAK)
    src = open(BAK, encoding="utf-8").read()
    assert src.count(ANCHOR) == 1, "anchor not unique: %d" % src.count(ANCHOR)
    assert "_dequantize_k_cache_paged_v2_kernel" not in src, "backup already patched"
    open(SRC, "w", encoding="utf-8").write(src + ADDITION)
    py_compile.compile(SRC, doraise=True)
    print("patched", SRC)


main()