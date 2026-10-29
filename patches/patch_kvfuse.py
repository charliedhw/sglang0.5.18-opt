"""GLM-5.3 MLA decode front-end fusion behind SGLANG_HCU_MLA_KV_FUSED=1 (default off):

On DSA decode-like batches (decode / target_verify / draft_extend_v2 with dsa_decode_impl
flashmla_kv and lightop's fp8 KV-cache store) kv_a_layernorm (lightop rms_norm_opt), the q / k
rope and MLATokenToKVPool's fused_quantize_and_store_mla_kv_cache run as one
hcu_ops.rope_qout_kv_store launch with the same bits (test_kv_fused.py): the latent is normed in
place in the packed qkv buffer, k_pe rotated in place, q_pe rotated into q_cat's rope tail (or
in place), and this layer's slots written; attn_mqa then gets k=None and save_kv_cache=False.
3 launches -> 1 per layer.  Anything _hcu_kv_fused_buffer / kv_fused_supported rejects keeps
the unfused ops.

Run after patch_qcat.py (and patch_fuse2.py).  The file is backed up once as *.bak_kvfuse and
patched from that backup, so re-running is idempotent."""
import os
import shutil

D = os.environ.get("SGL_SRT_DIR", "/usr/local/lib/python3.10/dist-packages/sglang/srt/")
F_ROCM = D + "models/deepseek_common/attention_forward_methods/forward_mla_rocm.py"

ROCM_EDITS = [
    ("""    from sglang.srt.layers.rotary_embedding.base import (
        RotaryEmbedding as _RotaryEmbedding,
    )


if TYPE_CHECKING:
""",
     """    from sglang.srt.layers.rotary_embedding.base import (
        RotaryEmbedding as _RotaryEmbedding,
    )
# SGLANG_HCU_MLA_KV_FUSED=1: on DSA decode-like batches kv_a_layernorm, the q / k rope and the
# fp8 KV-cache store run as one hcu_ops.rope_qout_kv_store launch with the same bits: the
# latent is normed in place in the packed qkv buffer and attn_mqa gets k=None with
# save_kv_cache=False (_hcu_kv_fused_buffer lists the conditions).
_hcu_kv = None
_hcu_kv_logged = set()
if _is_hcu and get_bool_env_var("SGLANG_HCU_MLA_KV_FUSED", "false"):
    import hcu_ops as _hcu_kv
    from sglang.srt.layers.rotary_embedding.base import (
        RotaryEmbedding as _RotaryEmbedding,
    )
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend as _hcu_get_attn_backend,
    )


if TYPE_CHECKING:
"""),
    ("""

def rocm_absorb_v_bmm(
""",
     """

def _hcu_kv_fused_buffer(attn, k_nope, forward_batch, q_replicate_active):
    \"\"\"SGLANG_HCU_MLA_KV_FUSED: this layer's fp8 KV-cache buffer when its kv_a_layernorm can
    be deferred into the rope + store launch (_hcu_kv_rope_norm_store), else None.  Only
    decode-like batches on dsa_backend's flashmla_kv with MLATokenToKVPool's lightop fp8 store
    (not overridden), the RotaryEmbedding fallback rope and a plain-tensor out_cache_loc.\"\"\"
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    mode = forward_batch.forward_mode
    rope = attn.rotary_emb
    be = _hcu_get_attn_backend()  # ForwardBatch has no attn_backend in this sglang
    pool = getattr(be, "token_to_kv_pool", None)
    if (
        q_replicate_active
        or not attn.use_dsa
        or get_parallel().dcp_enabled
        or not (
            mode.is_decode_or_idle()
            or mode.is_target_verify()
            or mode.is_draft_extend_v2()
        )
        or attn.current_attention_backend not in ("dsa", "nsa")
        or attn.current_attention_backend not in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
        or attn._fuse_rope_for_trtllm_mla(forward_batch)
        or attn._skip_rope_for_dsa_tilelang_fused()
        or attn._skip_rope_for_aiter_fused_mla()
        or attn.attn_mqa.is_cross_attention
        or type(rope) is not _RotaryEmbedding
        or not getattr(rope, "use_fallback_kernel", False)
        or rope.head_size != rope.rotary_dim
        or k_nope.dtype != torch.bfloat16
        or getattr(be, "dsa_decode_impl", None) != "flashmla_kv"
        or getattr(be, "use_mha", True)
        or getattr(be, "hisparse_coordinator", None) is not None
        or not isinstance(pool, MLATokenToKVPool)
        or not getattr(pool, "dsa_kv_cache_store_fp8", False)
        or type(pool).set_mla_kv_buffer is not MLATokenToKVPool.set_mla_kv_buffer
        or type(pool)._write_mla_kv_buffer
        is not MLATokenToKVPool._write_mla_kv_buffer
        or not isinstance(forward_batch.out_cache_loc, torch.Tensor)
    ):
        return None
    if rope.cos_sin_cache.dtype != k_nope.dtype or rope.cos_sin_cache.device != k_nope.device:
        # what RotaryEmbedding.forward_cuda's fallback branch does on its first call
        rope.cos_sin_cache = rope.cos_sin_cache.to(k_nope.device, dtype=k_nope.dtype)
    return pool.kv_buffer[attn.attn_mqa.layer_id - pool.start_layer]


def _hcu_kv_rope_norm_store(
    attn, positions, q_pe, q_cat, k_nope, k_pe, kv_buf, forward_batch
):
    \"\"\"SGLANG_HCU_MLA_KV_FUSED: the rope step for a layer whose kv_a_layernorm was deferred.
    Returns (q_pe, k_pe, k_nope); k_nope is None when hcu_ops.rope_qout_kv_store normed the
    latent in place and stored this layer's slots, else the normed latent (unfused ops).\"\"\"
    rope = attn.rotary_emb
    q_out = q_pe if q_cat is None else q_cat[..., attn.kv_lora_rank :]
    loc = forward_batch.out_cache_loc
    norm = attn.kv_a_layernorm
    if _hcu_kv.kv_fused_supported(
        positions, q_pe, q_out, k_nope, norm.weight, k_pe, rope.cos_sin_cache, kv_buf, loc
    ):
        _hcu_kv.rope_qout_kv_store(
            positions,
            q_pe,
            q_out,
            k_nope,
            norm.weight,
            norm.variance_epsilon,
            k_pe,
            rope.cos_sin_cache,
            rope.is_neox_style,
            kv_buf,
            loc,
        )
        if "fused" not in _hcu_kv_logged:
            _hcu_kv_logged.add("fused")
            logger.info("SGLANG_HCU_MLA_KV_FUSED: kv norm + rope + fp8 store in one launch")
        return q_out, k_pe, None
    if "unfused" not in _hcu_kv_logged:
        _hcu_kv_logged.add("unfused")
        logger.info(
            "SGLANG_HCU_MLA_KV_FUSED: operands not covered (T %d, loc %d), unfused ops",
            positions.numel(),
            loc.numel(),
        )
    k_nope = norm(k_nope.squeeze(1)).unsqueeze(1)
    if q_cat is not None:
        _hcu_kv.rope_qout(
            positions, q_pe, q_out, k_pe, rope.head_size, rope.cos_sin_cache, rope.is_neox_style
        )
        return q_out, k_pe, k_nope
    q_pe, k_pe = rope(positions, q_pe, k_pe)
    return q_pe, k_pe, k_nope


def rocm_absorb_v_bmm(
"""),
    ("""        q_lora = None
        q_input_quant_args = None
        topk_indices = None
""",
     """        q_lora = None
        q_input_quant_args = None
        topk_indices = None
        hcu_kv_buf = None
"""),
    ("""            else:
                q = self.q_a_layernorm(q)
                k_nope = self.kv_a_layernorm(k_nope)
""",
     """            else:
                q = self.q_a_layernorm(q)
                if _hcu_kv is not None:
                    hcu_kv_buf = _hcu_kv_fused_buffer(
                        self, k_nope, forward_batch, q_replicate_active
                    )
                if hcu_kv_buf is None:
                    k_nope = self.kv_a_layernorm(k_nope)
"""),
    ("""            if q_cat is not None:
                # rotated q_pe straight into q_cat's rope tail, k_pe in place
""",
     """            if hcu_kv_buf is not None:
                # SGLANG_HCU_MLA_KV_FUSED: kv_a_layernorm + rope + fp8 KV store
                q_pe, k_pe, k_nope = _hcu_kv_rope_norm_store(
                    self, positions, q_pe, q_cat, k_nope, k_pe, hcu_kv_buf, forward_batch
                )
                hcu_kv_buf = None
            elif q_cat is not None:
                # rotated q_pe straight into q_cat's rope tail, k_pe in place
"""),
    ("""
        dsa_prefill_cp = dsa_use_prefill_cp(forward_batch)
""",
     """
        if hcu_kv_buf is not None:  # deferred kv_a_layernorm, rope step skipped
            k_nope = self.kv_a_layernorm(k_nope.squeeze(1)).unsqueeze(1)
        dsa_prefill_cp = dsa_use_prefill_cp(forward_batch)
"""),
    ("""                    attn_output = self.attn_mqa(
                        q_nope_out,
                        k_nope,
                        k_nope,
                        forward_batch,
                        q_rope=q_pe,
                        k_rope=k_pe,
                        **extra_args,
                        **sink_args,
""",
     """                    attn_output = self.attn_mqa(
                        q_nope_out,
                        k_nope,
                        k_nope,
                        forward_batch,
                        q_rope=q_pe,
                        k_rope=k_pe,
                        # k_nope None: SGLANG_HCU_MLA_KV_FUSED stored the KV already
                        save_kv_cache=k_nope is not None,
                        **extra_args,
                        **sink_args,
"""),
]


def patch(path, edits, need):
    bak = path + ".bak_kvfuse"
    if not os.path.exists(bak):
        assert need in open(path, encoding="utf-8").read(), (path, "run patch_qcat.py first")
        shutil.copy2(path, bak)
    src = open(bak, encoding="utf-8").read()
    for old, new in edits:
        assert src.count(old) == 1, (path, old[:70], src.count(old))
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print("patched", path)


patch(F_ROCM, ROCM_EDITS, "SGLANG_HCU_MLA_QCAT")
