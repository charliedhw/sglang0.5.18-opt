"""SGLANG_HCU_MLA_QCAT=1 (default off): drop concat_mla_absorb_q from GLM-5.3's MLA
decode / target-verify / draft-extend (flashmla_kv, ~12.5 us per layer on BW1100).

forward_mla_rocm.py: on those batches the absorb bmm (q_nope @ w_kc) writes q_cat[..., :512]
  and the q rope writes q_cat[..., 512:] (hcu_ops.rope_qout: same kernel math as the
  SGLANG_HCU_FUSED_ROPE in-place rope, k_pe still rotated in place) of one [T, H, 576] buffer;
  q_nope_out / q_pe are passed on as the two views of it.
dsa_backend.py: flashmla_kv (forward_extend + forward_decode) takes such a pair as q_all via
  hcu_ops.adjacent_q_all (a view, no kernel) and keeps concat_mla_absorb_q_general otherwise.
Backs each file up once as *.bak_qcat (the state with the earlier hcu patches) and patches
from that backup."""
import os
import shutil

D = "/usr/local/lib/python3.10/dist-packages/sglang/srt/"
F_MLA = D + "models/deepseek_common/attention_forward_methods/forward_mla_rocm.py"
F_DSA = D + "layers/attention/dsa_backend.py"

MLA_EDITS = [
    ("""if _is_hcu and get_bool_env_var("SGLANG_HCU_SKINNY_BMM", "false"):
    import hcu_ops as _hcu_bmm
""",
     """if _is_hcu and get_bool_env_var("SGLANG_HCU_SKINNY_BMM", "false"):
    import hcu_ops as _hcu_bmm
# SGLANG_HCU_MLA_QCAT=1: on decode-like batches the absorb bmm and the q rope write into
# one [T, H, kv_lora_rank + rope] buffer, which dsa_backend's flashmla_kv path then takes
# as q_all without concat_mla_absorb_q (_hcu_qcat_buffer lists the conditions).
_hcu_qcat = None
if _is_hcu and get_bool_env_var("SGLANG_HCU_MLA_QCAT", "false"):
    import hcu_ops as _hcu_qcat
    from sglang.srt.layers.rotary_embedding.base import (
        RotaryEmbedding as _RotaryEmbedding,
    )
"""),
    ("""def rocm_absorb_q_bmm(
    attn: DeepseekV2AttentionMLA,
    q_nope: torch.Tensor,
    *,
    is_capture_mode: bool,
) -> torch.Tensor:
    \"\"\"Absorb ``q_nope @ w_kc`` on HIP/AITER (pre-transpose layout).\"\"\"
""",
     """def rocm_absorb_q_bmm(
    attn: DeepseekV2AttentionMLA,
    q_nope: torch.Tensor,
    *,
    is_capture_mode: bool,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    \"\"\"Absorb ``q_nope @ w_kc`` on HIP/AITER (pre-transpose layout).

    ``out``: optional [H, T, kv_lora_rank] destination (SGLANG_HCU_MLA_QCAT).\"\"\"
"""),
    ("""            if _hcu_bmm is not None and q_nope.dtype == torch.bfloat16:
                q_nope_out = _hcu_bmm.try_bmm_bf16(q_nope.transpose(0, 1), w_kc)
            if q_nope_out is None:
                q_nope_out = torch.bmm(q_nope.to(torch.bfloat16).transpose(0, 1), w_kc)
    return q_nope_out
""",
     """            if _hcu_bmm is not None and q_nope.dtype == torch.bfloat16:
                q_nope_out = _hcu_bmm.try_bmm_bf16(q_nope.transpose(0, 1), w_kc, out)
            if q_nope_out is None:
                x = q_nope.to(torch.bfloat16).transpose(0, 1)
                if out is None:
                    q_nope_out = torch.bmm(x, w_kc)
                else:
                    q_nope_out = torch.bmm(x, w_kc, out=out)
    if out is not None and q_nope_out.data_ptr() != out.data_ptr():
        out.copy_(q_nope_out)
        q_nope_out = out
    return q_nope_out


def _hcu_qcat_buffer(attn, q_pe, forward_batch, q_replicate_active):
    \"\"\"SGLANG_HCU_MLA_QCAT: the [T, H, kv_lora_rank + rope] buffer the absorb bmm and the
    q rope write into, or None to keep the split q_nope_out / q_pe.  Only decode-like
    batches (flashmla_kv in dsa_backend) and the plain bf16 bmm + RotaryEmbedding fallback
    rope (the SGLANG_HCU_FUSED_ROPE kernel) path.\"\"\"
    mode = forward_batch.forward_mode
    rope = attn.rotary_emb
    if (
        q_replicate_active
        or attn.use_deep_gemm_bmm
        or _SGLANG_EXPERIMENTAL_LORA_OPTI
        or is_kv_b_lora_active(attn)
        or get_parallel().dcp_enabled
        or not (
            mode.is_decode_or_idle()
            or mode.is_target_verify()
            or mode.is_draft_extend_v2()
        )
        or attn.current_attention_backend not in ("dsa", "nsa")
        or attn._fuse_rope_for_trtllm_mla(forward_batch)
        or type(rope) is not _RotaryEmbedding
        or not getattr(rope, "use_fallback_kernel", False)
        or rope.head_size != rope.rotary_dim
        or q_pe.dtype != torch.bfloat16
        or q_pe.dim() != 3
        or q_pe.shape[2] != rope.head_size
    ):
        return None
    if rope.cos_sin_cache.dtype != q_pe.dtype or rope.cos_sin_cache.device != q_pe.device:
        # what RotaryEmbedding.forward_cuda's fallback branch does on its first call
        rope.cos_sin_cache = rope.cos_sin_cache.to(q_pe.device, dtype=q_pe.dtype)
    return q_pe.new_empty(
        q_pe.shape[0], q_pe.shape[1], attn.kv_lora_rank + q_pe.shape[2]
    )
"""),
    ("""        q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)

        if q_replicate_active:
""",
     """        q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)
        q_cat = None
        if _hcu_qcat is not None:
            q_cat = _hcu_qcat_buffer(self, q_pe, forward_batch, q_replicate_active)

        if q_replicate_active:
"""),
    ("""                q_nope_out = rocm_absorb_q_bmm(
                    self, q_nope, is_capture_mode=get_is_capture_mode()
                )
""",
     """                q_nope_out = rocm_absorb_q_bmm(
                    self,
                    q_nope,
                    is_capture_mode=get_is_capture_mode(),
                    out=(
                        None
                        if q_cat is None
                        else q_cat[..., : self.kv_lora_rank].transpose(0, 1)
                    ),
                )
"""),
    ("""        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
""",
     """        ):
            if q_cat is not None:
                # rotated q_pe straight into q_cat's rope tail, k_pe in place
                _hcu_qcat.rope_qout(
                    positions,
                    q_pe,
                    q_cat[..., self.kv_lora_rank :],
                    k_pe,
                    self.rotary_emb.head_size,
                    self.rotary_emb.cos_sin_cache,
                    self.rotary_emb.is_neox_style,
                )
                q_pe = q_cat[..., self.kv_lora_rank :]
            else:
                q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
"""),
]

DSA_EDITS = [
    ("""_is_hcu = is_hcu()
""",
     """_is_hcu = is_hcu()

# SGLANG_HCU_MLA_QCAT=1: flashmla_kv takes q_nope / q_rope that forward_mla_rocm already
# wrote side by side into one [T, H, 576] buffer as q_all (a view) instead of concatenating.
_hcu_qcat = None
if _is_hcu and get_bool_env_var("SGLANG_HCU_MLA_QCAT"):
    import hcu_ops as _hcu_qcat
"""),
    ("""        elif dsa_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
""",
     """        elif dsa_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = None
                if _hcu_qcat is not None:
                    q_all = _hcu_qcat.adjacent_q_all(q_nope, q_rope)
                if q_all is None:
                    q_all = concat_mla_absorb_q_general(q_nope, q_rope)
"""),
    ("""        elif self.dsa_decode_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
""",
     """        elif self.dsa_decode_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = None
                if _hcu_qcat is not None:
                    q_all = _hcu_qcat.adjacent_q_all(q_nope, q_rope)
                if q_all is None:
                    q_all = concat_mla_absorb_q_general(q_nope, q_rope)
"""),
]

for path, edits in ((F_MLA, MLA_EDITS), (F_DSA, DSA_EDITS)):
    bak = path + ".bak_qcat"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    src = open(bak).read()
    for old, new in edits:
        assert src.count(old) == 1, (path, old[:70], src.count(old))
        src = src.replace(old, new)
    open(path, "w").write(src)
    print("patched", path)
