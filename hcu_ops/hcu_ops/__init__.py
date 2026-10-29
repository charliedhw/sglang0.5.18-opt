"""HCU (gfx938) kernels for GLM-5.3 DSA decode, used via env switches in sglang."""
import json
import os

import torch  # noqa: F401  (loads libtorch before the extension)

from . import _C

# split: few rows are spread over up to HCU_TOPK_SPLITS CTAs each (default;
#        one CTA per row once rows >= CUs / 2); single: always 1 CTA per row
_TOPK_IMPL = os.environ.get("HCU_TOPK_IMPL", "split")
_TOPK_SPLITS = int(os.environ.get("HCU_TOPK_SPLITS", "8"))


def sparse_mask_topk_transform(score, nonzero_mask, lengths, page_table_size_1,
                               cu_seqlens_q, topk=2048):
    """Drop-in for lightop.attention.fast_topk_transform_sparse_mask_fused."""
    global _MQA_HIST
    st = _MQA_HIST
    if st is not None:
        _MQA_HIST = None
        if st[0] is score and st[1].data_ptr() == lengths.data_ptr() and st[1].shape == lengths.shape:
            # pass 1 (histogram) was built by mqa_logits(..., hist_lens=lengths)
            return _C.sparse_mask_topk_transform_split_pre(score, nonzero_mask, lengths,
                                                           page_table_size_1, cu_seqlens_q, topk,
                                                           _TOPK_SPLITS, st[2])
    if _TOPK_IMPL == "single":
        return _C.sparse_mask_topk_transform(score, nonzero_mask, lengths,
                                             page_table_size_1, cu_seqlens_q, topk)
    return _C.sparse_mask_topk_transform_split(score, nonzero_mask, lengths,
                                               page_table_size_1, cu_seqlens_q, topk,
                                               _TOPK_SPLITS)


# Indexer logits (lightop page_mqa_logits_sparse_mask[_grouped] native fp8 drop-in).
# HCU_MQA_PCTAS[_G1]: CTAs per request for MTP groups of 6 [single rows]
# (0: lightop default min(chunks, 32); -1 (default): 64 while there are <= 12 [4] rows, else 0
# -- extra CTAs only pay off while the requests cannot fill the 64 CUs);
# HCU_MQA_RPC: rows per CTA for groups of 6; HCU_MQA_HIST_MAX_ROWS[_G6]: build the topk
# histogram in the logits kernel only up to this many single [grouped] rows.  For G=1 it costs
# 3-5 us and saves ~10 us of topk pass 1 (64K, 1-8 rows); the G=6 R 2 / 3 kernels run at
# occupancy 1 and their LDS atomics cost 15-100 us, more than they save, so G=6 is off.
_MQA_PCTAS = int(os.environ.get("HCU_MQA_PCTAS", "-1"))
_MQA_PCTAS_G1 = int(os.environ.get("HCU_MQA_PCTAS_G1", str(_MQA_PCTAS)))
_MQA_RPC = int(os.environ.get("HCU_MQA_RPC", "3"))
_MQA_HIST_MAX_ROWS = int(os.environ.get("HCU_MQA_HIST_MAX_ROWS", "8"))
_MQA_HIST_MAX_ROWS_G6 = int(os.environ.get("HCU_MQA_HIST_MAX_ROWS_G6", "0"))
_MQA_HIST = None  # (logits, hist_lens, zws) of the last mqa_logits with a fused histogram
_N_CU = None


def _topk_splits(rows, width):
    """Splits sparse_mask_topk_transform_split uses for [rows, width] scores."""
    global _N_CU
    if _N_CU is None:
        _N_CU = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return min(max((width + 8191) // 8192, 1), _N_CU // max(rows, 1), min(_TOPK_SPLITS, 64))


def mqa_logits_supported(q, kv_cache, weights, group_size, context_lens=None, block_table=None,
                         max_context_len=None):
    """Whether mqa_logits takes these operands (else keep lightop)."""
    b = q.shape[0] if q.dim() == 4 else 0
    ok = (q.dtype == torch.float8_e4m3fn and q.dim() == 4 and tuple(q.shape[1:]) == (1, 32, 128)
          and b > 0 and q.is_contiguous() and weights.dtype == torch.float32
          and tuple(weights.shape) == (b, 32) and weights.is_contiguous()
          and kv_cache.dtype == torch.uint8 and kv_cache.dim() == 4
          and tuple(kv_cache.shape[1:]) == (64, 1, 132) and kv_cache.stride(1) == 132
          and kv_cache.stride(3) == 1 and kv_cache.stride(0) % 16 == 0
          and (group_size == 1 or (group_size == 6 and _MQA_RPC in (1, 2, 3)))
          and b % group_size == 0)
    if ok and context_lens is not None:
        ok = (context_lens.dtype == torch.int32 and context_lens.numel() == b
              and context_lens.is_contiguous())
    if ok and block_table is not None:
        ok = (block_table.dtype == torch.int32 and block_table.dim() == 2
              and block_table.shape[0] == b and block_table.stride(1) == 1
              and (max_context_len is None or max_context_len == block_table.shape[1] * 64))
    return ok


def mqa_logits(q, kv_cache, weights, context_lens, block_table, max_context_len, group_size,
               hist_lens=None, persistent_ctas=None, rows_per_cta=None):
    """(logits, mask) bit-identical to lightop page_mqa_logits_sparse_mask[_grouped].

    hist_lens: lengths of the paired sparse_mask_topk_transform; when that call will
    split rows, its pass-1 histogram is built here and consumed by it."""
    global _MQA_HIST
    rows = q.shape[0]
    if persistent_ctas is None:
        persistent_ctas = _MQA_PCTAS_G1 if group_size == 1 else _MQA_PCTAS
    if persistent_ctas < 0:
        persistent_ctas = 64 if rows <= (4 if group_size == 1 else 12) else 0
    if rows_per_cta is None:
        rows_per_cta = 1 if group_size == 1 else _MQA_RPC
    hist_max_rows = _MQA_HIST_MAX_ROWS if group_size == 1 else _MQA_HIST_MAX_ROWS_G6
    use_hist = (hist_lens is not None and _TOPK_IMPL == "split" and rows <= hist_max_rows
                and hist_lens.dtype == torch.int32 and hist_lens.dim() == 1
                and hist_lens.shape[0] == rows and hist_lens.is_contiguous()
                and _topk_splits(rows, max_context_len) > 1)
    logits, mask, zws = _C.mqa_logits_hist(q, kv_cache, weights, context_lens.view(-1), block_table,
                                           max_context_len, group_size, persistent_ctas,
                                           rows_per_cta, hist_lens if use_hist else None)
    _MQA_HIST = (logits, hist_lens, zws) if use_hist else None
    return logits, mask


def skinny_gemm(x, w, x_scale=None, w_scale=None, out=None, nw=0, nt=0, stream_w=False, ver=4):
    """out[M, N] (bf16) = x[M, K] @ w[N, K]^T for decode-sized M (<= 64).

    fp8 e4m3fn x/w with fp32 per-token x_scale [M] and per-channel w_scale [N]
    (w8a8 dynamic), or bf16 x/w without scales.  ver 1: W loaded straight in mmac
    layout; ver 2/3: W staged through LDS with 256 B row segments, prefetching
    1/2 chunks ahead; ver 4/5: X staged through LDS as well (row-contiguous loads).
    """
    if out is None:
        out = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
    if x_scale is None:
        x_scale = w_scale = _EMPTY.get(x.device)
        if x_scale is None:
            x_scale = w_scale = _EMPTY[x.device] = torch.empty(0, device=x.device)
    _C.skinny_gemm(x, x_scale, w, w_scale, out, nw, nt, stream_w, ver)
    return out


_EMPTY = {}


def rope_inplace(positions, query, key, head_size, cos_sin_cache, is_neox):
    """Drop-in for sgl_kernel.rotary_embedding (rotates query / key in place, same
    numerics) as a single launch: query / key are [T, n * head_size] or
    [T, n, head_size] with a unit last stride, cos_sin_cache [max_pos, rot_dim] in the
    query dtype.  sglang hook: SGLANG_HCU_FUSED_ROPE=1."""
    _C.rope_inplace(positions.reshape(-1), query, key, head_size, cos_sin_cache, is_neox)


def rope_qout(positions, query, q_out, key, head_size, cos_sin_cache, is_neox):
    """rope_inplace with the rotated query written to q_out (query's shape, unit last stride:
    the rope tail of MLA's concatenated [T, H, kv_lora_rank + rope] q) instead of in place;
    key is still rotated in place.  Same numerics.  sglang hook: SGLANG_HCU_MLA_QCAT=1."""
    _C.rope_qout(positions.reshape(-1), query, q_out, key, head_size, cos_sin_cache, is_neox)


def adjacent_q_all(q_nope, q_rope):
    """The [T, H, Dn + Dr] tensor whose rows are q_nope [T, H, Dn] followed by q_rope
    [T, H, Dr] when both already sit that way in one row-contiguous buffer (as
    SGLANG_HCU_MLA_QCAT writes them): a zero-copy stand-in for concat_mla_absorb_q.
    None otherwise."""
    if (q_nope.dim() != 3 or q_rope.dim() != 3 or q_nope.shape[:2] != q_rope.shape[:2]
            or q_nope.dtype != q_rope.dtype or q_nope.device != q_rope.device):
        return None
    t, h, dn = q_nope.shape
    d = dn + q_rope.shape[2]
    for x in (q_nope, q_rope):
        if x.stride(2) != 1 or (h > 1 and x.stride(1) != d) or (t > 1 and x.stride(0) != h * d):
            return None
    if (q_rope.data_ptr() != q_nope.data_ptr() + dn * q_nope.element_size()
            or q_rope.untyped_storage().data_ptr() != q_nope.untyped_storage().data_ptr()):
        return None
    return q_nope.as_strided((t, h, d), (h * d, d, 1))


def moe_gate_supported(logits, bias, topk):
    """Shapes / dtypes moe_gate_topk covers (otherwise keep lightop moe_fused_gate); topk 8
    only: the kernel also takes 6, but lightop (the bit-exact reference) does not."""
    ok_dt = (torch.float32, torch.bfloat16)
    return (logits.dim() == 2 and logits.shape[1] in (128, 256, 384, 512) and topk == 8
            and logits.dtype in ok_dt and bias.dtype in ok_dt and bias.dim() == 1
            and bias.numel() == logits.shape[1] and logits.is_contiguous() and bias.is_contiguous()
            and _aligned(logits, bias))


def moe_gate_topk(logits, bias, topk, routed_scaling_factor, apply_on_output, num_token_non_padded=None):
    """lightop moe_fused_gate(logits, bias, 1, 1, topk, 0, rsf, apply_on_output) (sigmoid
    scores, correction-bias selection, renormalized weights; same numerics) plus sglang's
    HCU DeepEP id postprocess in the same kernel: returns fp32 weights [T, topk] and int64
    ids [T, topk] with rows >= num_token_non_padded set to -1.  sglang hook:
    SGLANG_HCU_FUSED_GATE=1."""
    return _C.moe_gate_topk(logits, bias, topk, float(routed_scaling_factor or 0.0),
                            bool(apply_on_output), num_token_non_padded)


# (CTAs, waves per row; 0 = by row count) of silu_mul_fp8_quant_masked, tuned on BW1100
# (test_moe_ops.py)
_ACT_LAUNCH = (512, 0)


def silu_mul_fp8_quant_masked(x, masked_m, ctas=0, waves_per_row=None):
    """lightop fuse_silu_mul_fp8_quant_ep(x, 0, masked_m, limit=None) drop-in: x [E, cap, 2d]
    contiguous bf16 (d % 512 == 0, d <= 4096, E <= 256), masked_m int32 [E].  Returns fp8
    e4m3fn [E, cap, d] and fp32 scales [E, cap, 1]; rows t >= masked_m[e] are left
    uninitialised (as in lightop), the others match bit for bit."""
    c0, w0 = _ACT_LAUNCH
    return _C.silu_mul_fp8_quant_masked(x, masked_m, ctas or c0, w0 if waves_per_row is None else waves_per_row)


def try_silu_mul_fp8_quant_masked(x, masked_m):
    """sglang DeepEP-LL masked MoE hook: silu_mul_fp8_quant_masked when the operands fit,
    else None (keep lightop).  sglang hook: SGLANG_HCU_FUSED_ACT=1."""
    if (x.dim() != 3 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.shape[2] % 1024
            or not 1024 <= x.shape[2] <= 8192 or x.shape[0] > 256 or masked_m is None
            or masked_m.dtype != torch.int32 or masked_m.dim() != 1 or masked_m.numel() != x.shape[0]
            or not masked_m.is_contiguous() or masked_m.device != x.device or not _aligned(x)):
        return None
    return silu_mul_fp8_quant_masked(x, masked_m)


def bmm_skinny(x, w, out, cw=0, nwv=4, lds=True):
    """out[b] = x[b] @ w[b]^T in bf16 for decode-sized M (<= 32): x [B, M, K], w [B, N, K],
    out [B, M, N], each with a unit last stride (other strides free, 16 B aligned rows for
    x / w).  cw: K bytes per pipelined chunk (0 = auto), nwv: waves (16-row W tiles) per
    CTA, lds: stage W through LDS (coalesced) instead of loading it in mmac layout."""
    _C.bmm_skinny(x, w, out, cw, nwv, lds)
    return out


# (B, N, K) -> rows (max_m, cw, nwv, lds) in increasing max_m, tuned on BW1100 (test_bmm.py,
# bmm_tune.txt) for GLM-5.3's absorb bmm's: B = 64 local heads (DP8 attention) or 8 (TP8),
# kc N 512 / K 192, vc N 256 / K 512.  A row covers M in (previous max_m, max_m]; M past the
# last row and untuned shapes keep torch.bmm.
_BMM_CFG = {
    (64, 512, 192): [(4, 128, 2, True), (12, 128, 1, True)],
    (64, 256, 512): [(12, 256, 1, True)],
    (8, 512, 192): [(4, 384, 4, True), (16, 128, 2, True)],
    (8, 256, 512): [(32, 512, 1, True)],
}


def try_bmm_bf16(x, w_kn, out=None):
    """sglang MLA absorb hook for torch.bmm(x, w_kn[, out=out]): x [B, M, K] bf16 and
    w_kn the [B, K, N] view of per-batch [N, K] row-major bf16 weights (w_kc / w_vc).
    Returns the [B, M, N] result (out when given; any strides with a unit last one), or
    None to keep torch.bmm (M without a tuned config in _BMM_CFG or operands that don't
    fit).  sglang hook: SGLANG_HCU_SKINNY_BMM=1."""
    if x.dim() != 3 or w_kn.dim() != 3 or x.dtype != torch.bfloat16 or w_kn.dtype != torch.bfloat16:
        return None
    b, m, k = x.shape
    n = w_kn.shape[2]
    cfg = next((r[1:] for r in _BMM_CFG.get((b, n, k), ()) if m <= r[0]), None) if m else None
    if (cfg is None or n % 16 or w_kn.shape[0] != b or w_kn.shape[1] != k or (2 * k) % 128
            or x.stride(2) != 1 or w_kn.stride(1) != 1
            or (x.stride(0) | x.stride(1) | w_kn.stride(0) | w_kn.stride(2)) % 8
            or not _aligned(x, w_kn)):
        return None
    if out is None:
        out = torch.empty(b, m, n, device=x.device, dtype=torch.bfloat16)
    elif out.shape != (b, m, n) or out.dtype != torch.bfloat16 or out.stride(2) != 1:
        return None
    cw, nwv, lds = cfg
    _C.bmm_skinny(x, w_kn.transpose(1, 2), out, cw, nwv, lds)
    return out


def skinny_nn(x, w_kn, out=None, nw=1, h=2, pf=2, stream_w=False):
    """out[M, N] (bf16) = x[M, K] @ w_kn[K, N] for M <= 64 with an n-contiguous weight (the
    NN lm_head layout).  nw: waves splitting K per 128-column CTA, h: 16-row k slices per
    lane per chunk (1 / 2), pf: chunks prefetched (1..3 for h 1, 1..2 for h 2)."""
    if out is None:
        out = torch.empty(x.shape[0], w_kn.shape[1], device=x.device, dtype=torch.bfloat16)
    _C.skinny_nn(x, w_kn, out, nw, h, pf, stream_w)
    return out


# (N, K) -> rows (max_m, nw, h, pf) in increasing max_m, tuned on BW1100 (test_nn.py sweep,
# nn_sweep.log / nn_sweep4.log): GLM-5.3's full-vocab lm_head (1110-1216 us for M <= 48,
# 1274-1326 for M 49-64 vs matmul 1583-1595; h 2 / pf 2 spills at M > 48) and its 8-way
# vocab-parallel draft shard (156-201 us for M <= 48 vs 208-392; M 49-64 is no faster than
# rocBLAS).  Other shapes / M keep matmul.
_NN_CFG = {
    (154880, 6144): [(48, 1, 2, 2), (64, 1, 2, 1)],
    (19360, 6144): [(48, 1, 2, 2)],
}


def try_skinny_nn_bf16(x, w_kn, out=None):
    """sglang lm_head hook for torch.matmul(x, w_kn[, out=out]): x [M, K] bf16 and w_kn an
    n-contiguous [K, N] bf16 weight (lm_head.weight.T after SGLANG_HCU_LM_HEAD_NN's relayout,
    or the draft VP weight_shard_t).  Returns the [M, N] result (out when given), or None to
    keep matmul.  sglang hook: SGLANG_HCU_SKINNY_LM_HEAD=1."""
    if x.dim() != 2 or w_kn.dim() != 2 or x.dtype != torch.bfloat16 or w_kn.dtype != torch.bfloat16:
        return None
    m, k = x.shape
    n = w_kn.shape[1]
    cfg = next((r[1:] for r in _NN_CFG.get((n, k), ()) if m <= r[0]), None) if m else None
    if (cfg is None or w_kn.shape[0] != k or x.stride(1) != 1 or w_kn.stride(1) != 1
            or (x.stride(0) | w_kn.stride(0)) % 8 or not _aligned(x, w_kn)):
        return None
    if out is None:
        out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    elif (out.shape != (m, n) or out.dtype != torch.bfloat16 or out.stride(1) != 1
          or out.stride(0) % 8 or not _aligned(out)):
        return None
    nw, h, pf = cfg
    _C.skinny_nn(x, w_kn, out, nw, h, pf, False)
    return out


# --- sglang hooks (SGLANG_HCU_SKINNY_GEMM=1) --------------------------------------
# skinny_cfg.json (made by make_cfg.py from tune_skinny.py on BW1100) lists, per
# dtype and "N,K", rows [max_m, ver, nw, nt] in increasing max_m; a row covers M in
# (previous max_m, max_m], ver 0 = keep the default GEMM (hipBLASLt / rocBLAS) there,
# and M past the last row always does.  HCU_SKINNY_CFG overrides the file.
_SKINNY = None


def _load_cfg():
    path = os.environ.get("HCU_SKINNY_CFG") or os.path.join(os.path.dirname(__file__), "skinny_cfg.json")
    with open(path) as f:
        raw = json.load(f)
    return {dt: {tuple(int(v) for v in key.split(",")): [tuple(r) for r in rows] for key, rows in tab.items()}
            for dt, tab in raw.items()}


def skinny_config(fp8, n, k, m):
    """(ver, nw, nt) for an [M, K] x [N, K]^T GEMM, or None to keep the default GEMM."""
    global _SKINNY
    if _SKINNY is None:
        _SKINNY = _load_cfg()
    rows = _SKINNY["fp8" if fp8 else "bf16"].get((n, k))
    if rows and m > 0:
        for max_m, ver, nw, nt in rows:
            if m <= max_m:
                return (ver, nw, nt) if ver else None
    return None


def _aligned(*ts):
    return all(t.data_ptr() % 16 == 0 for t in ts)


def try_skinny_fp8(x, x_scale, w_kn, w_scale, out):
    """sglang apply_fp8_linear hook for deepgemm.fp8_gemm((x, x_scale), (w_kn, w_scale), out):
    x [M, K] fp8 e4m3fn with per-token fp32 x_scale, w_kn the [K, N] view of an
    [N, K]-contiguous fp8 weight with per-channel fp32 w_scale, out bf16 [M, N].
    Runs skinny_gemm and returns True when M has a tuned config for (N, K) and the
    operands fit; otherwise does nothing and returns False."""
    k, n = w_kn.shape
    if x.dim() != 2:
        return False
    m = x.shape[0]
    cfg = skinny_config(True, n, k, m)
    if (cfg is None or x.dtype != torch.float8_e4m3fn or w_kn.dtype != x.dtype
            or w_kn.stride() != (1, k) or x.stride(1) != 1 or x.stride(0) % 16
            or x_scale.dtype != torch.float32 or x_scale.numel() != m or not x_scale.is_contiguous()
            or w_scale.dtype != torch.float32 or w_scale.numel() != n or not w_scale.is_contiguous()
            or out.dtype != torch.bfloat16 or not out.is_contiguous() or out.numel() != m * n
            or not _aligned(x, w_kn)):
        return False
    ver, nw, nt = cfg
    _C.skinny_gemm(x, x_scale, w_kn.t(), w_scale, out.view(m, n), nw, nt, False, ver)
    return True


def try_skinny_bf16(x, w):
    """sglang UnquantizedLinearMethod hook for F.linear(x, w) without bias: x [..., K]
    bf16, w [N, K] row-major bf16.  Returns the bf16 [..., N] output, or None when M
    has no tuned config for (N, K) or the operands don't fit."""
    n, k = w.shape
    m = x.numel() // k if x.shape[-1] == k else 0
    cfg = skinny_config(False, n, k, m)
    if (cfg is None or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16
            or w.stride() != (k, 1) or not x.is_contiguous() or not _aligned(x, w)):
        return None
    ver, nw, nt = cfg
    out = torch.empty(*x.shape[:-1], n, device=x.device, dtype=torch.bfloat16)
    empty = _EMPTY.get(x.device)
    if empty is None:
        empty = _EMPTY[x.device] = torch.empty(0, device=x.device)
    _C.skinny_gemm(x.view(m, k), empty, w, empty, out.view(m, n), nw, nt, False, ver)
    return out


# --- producer + per-token fp8 quant fusions (SGLANG_HCU_NORM_QUANT / SGLANG_HCU_SILU_QUANT) ----
# Both give the bits of the unfused lightop ops followed by scaled_fp8_quant(...,
# use_per_token_if_dynamic=True) (lightop per_token_quant_fp8), checked by test_norm_quant.py.
# HCU_NORM_RSQRT_MODE picks the rsqrt form matching lightop's build (0: c10 compat rsqrt).
_RSQRT_MODE = int(os.environ.get("HCU_NORM_RSQRT_MODE", "0"))


def norm_quant_supported(x, residual, weight):
    """Shapes fused_add_rms_quant covers: those lightop fused_add_rms_norm_opt runs with
    fused_add_rms_kernel_opt<8, 512 | 1024> (GLM-5.3 hidden 6144)."""
    if x.dim() != 2 or x.dtype != torch.bfloat16 or residual.dtype != x.dtype or weight.dtype != x.dtype:
        return False
    h = x.shape[1]
    return (h % 16 == 0 and 2048 < h < 8192 and h != 4096 and x.is_contiguous()
            and residual.is_contiguous() and residual.shape == x.shape and weight.is_contiguous()
            and weight.numel() == h and _aligned(x, residual, weight))


def fused_add_rms_quant(x, residual, weight, eps, q_dtype=torch.float8_e4m3fn):
    """lightop op.fused_add_rms_norm_opt(x, residual, weight, eps) (in place: residual += x,
    x = rms_norm(residual) * weight) followed by per_token_quant_fp8(x), in one kernel.
    Returns (q [M, H] q_dtype, s [M, 1] fp32)."""
    q = torch.empty(x.shape, device=x.device, dtype=q_dtype)
    s = torch.empty(x.shape[0], 1, device=x.device, dtype=torch.float32)
    _C.fused_add_rms_quant(x, residual, weight, float(eps), q, s, _RSQRT_MODE)
    return q, s


def silu_mul_quant_supported(gate_up):
    if gate_up.dim() != 2 or gate_up.dtype != torch.bfloat16 or not gate_up.is_contiguous():
        return False
    d = gate_up.shape[1] // 2
    return gate_up.shape[1] == 2 * d and d % 8 == 0 and 0 < d <= 16384 and _aligned(gate_up)


def silu_mul_quant(gate_up, q_dtype=torch.float8_e4m3fn):
    """lightop fuse_silu_and_mul(gate_up) followed by per_token_quant_fp8 of the bf16 product,
    in one kernel (the product never goes to memory).  Returns (q [M, d], s [M, 1] fp32)."""
    m, d = gate_up.shape[0], gate_up.shape[1] // 2
    q = torch.empty(m, d, device=gate_up.device, dtype=q_dtype)
    s = torch.empty(m, 1, device=gate_up.device, dtype=torch.float32)
    _C.silu_mul_quant(gate_up, q, s)
    return q, s


# --- MLA decode front end: q rope + kv_a norm + k rope + fp8 KV store (SGLANG_HCU_MLA_KV_FUSED) --


def _rows(x, width):
    """x is [T, width] / [T, 1, width] with a unit last stride."""
    return (x.shape[-1] == width and x.stride(-1) == 1
            and (x.dim() == 2 or (x.dim() == 3 and x.shape[1] == 1)))


def kv_fused_supported(positions, q_pe, q_out, kv_c, kv_weight, k_pe, cos_sin_cache, kv_buffer, loc):
    """Operands rope_qout_kv_store covers (GLM-5.3 / DeepSeek MLA: kv_lora_rank 512, rope 64,
    bf16, DSA fp8 656-byte slots)."""
    t = positions.numel()
    bf = torch.bfloat16
    return (q_pe.dtype == bf and q_out.dtype == bf and kv_c.dtype == bf and kv_weight.dtype == bf
            and k_pe.dtype == bf and cos_sin_cache.dtype == bf
            and positions.dtype in (torch.int32, torch.int64) and positions.is_contiguous()
            and q_pe.dim() == 3 and q_pe.shape[0] == t and q_pe.shape[2] == 64 and q_pe.stride(2) == 1
            and q_out.shape == q_pe.shape and q_out.stride(2) == 1
            and cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64 and cos_sin_cache.is_contiguous()
            and _rows(kv_c, 512) and kv_c.shape[0] == t and kv_c.stride(0) % 8 == 0
            and _rows(k_pe, 64) and k_pe.shape[0] == t
            and kv_weight.is_contiguous() and kv_weight.numel() == 512
            and kv_buffer.dtype == torch.uint8 and kv_buffer.dim() >= 2
            and kv_buffer.numel() // max(kv_buffer.shape[0], 1) >= 656 and kv_buffer.stride(0) % 16 == 0
            and loc.dtype == torch.int64 and loc.is_contiguous() and loc.numel() <= t
            and _aligned(kv_c, kv_weight, kv_buffer))


def rope_qout_kv_store(positions, q_pe, q_out, kv_c, kv_weight, norm_eps, k_pe, cos_sin_cache, is_neox,
                       kv_buffer, loc, store_eps=1e-6):
    """One launch for what sglang's DSA decode runs as
        kv_c = rms_norm_opt(kv_c, kv_weight, norm_eps)          (here: in place)
        rope_qout(positions, q_pe, q_out, k_pe, 64, cos_sin_cache, is_neox)
        lightop fused_quantize_and_store_mla_kv_cache(kv_c, k_pe, kv_buffer, loc, "fp8_e4m3", store_eps)
    with the same bits in q_out, kv_c, k_pe and kv_buffer[loc].  q_out may be q_pe itself."""
    _C.rope_qout_kv_store(positions.reshape(-1), q_pe, q_out, kv_c, kv_weight, float(norm_eps), k_pe,
                          cos_sin_cache, bool(is_neox), kv_buffer, loc, float(store_eps), _RSQRT_MODE)


# The last RMSNorm's fused quant: (x, x._version, q, s).  RMSNorm.forward_hip replaces it on
# every call; a CHANNEL-fp8 linear fed that very tensor, unmodified since, takes (q, s)
# instead of quantizing x again.  The strong reference keeps x's memory from being recycled.
_NQ_STASH = None


def norm_quant_stash_set(x, q, s):
    global _NQ_STASH
    _NQ_STASH = (x, x._version, q, s)


def norm_quant_stash_clear():
    global _NQ_STASH
    _NQ_STASH = None


def norm_quant_stash_get(x):
    """(q, s) of the stashed fused norm quant if x is its output, else None."""
    st = _NQ_STASH
    if st is not None and st[0] is x and x._version == st[1]:
        return st[2], st[3]
    return None
