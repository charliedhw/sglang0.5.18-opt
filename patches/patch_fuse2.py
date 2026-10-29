"""Two more GLM-5.3 decode trims, each behind its own switch (default off), bit-exact by
construction (and checked by test_fuse2.py):

SGLANG_HCU_STRIDED_RMS=1   layernorm.py: RMSNorm.forward_hip runs lightop rms_norm_opt on
  the row-strided q_a / kv_a slices of the packed qkv latent (bf16, width 2048 / 512) instead
  of first copying them with .contiguous() -- 2 copy kernels less per layer.
SGLANG_HCU_QUANT_REUSE=1   the indexer's wq_b reads q_lora and its wk reads hidden_states,
  the same bf16 inputs q_b_proj and fused_qkv_a_proj_with_mqa quantize (per-token fp8,
  FP8_DYNAMIC); each input is quantized once and both GEMMs take the pair -- 2 quant kernels
  less per layer.  compressed_tensors_w8a8_fp8.py (HcuQuantArgs / hcu_quant_input /
  hcu_reusable_fp8_input, apply_weights), deepseek_v2.py (prepare_qkv_latent keeps the
  hidden_states pair on decode / target-verify), forward_mla_rocm.py (q_lora pair, hands both
  to Indexer._hcu_quant_reuse around the indexer call), dsa_indexer.py (_get_q_k_bf16 HCU
  branch uses a pair only if its source tensor is the GEMM's input).

Run after patch_qcat.py (forward_mla_rocm.py's backup here is the QCAT-patched file).  Each
file is backed up once as *.bak_fuse2 and patched from that backup."""
import os
import shutil

D = "/usr/local/lib/python3.10/dist-packages/sglang/srt/"
F_LN = D + "layers/layernorm.py"
F_CT = D + "layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py"
F_DS = D + "models/deepseek_v2.py"
F_MLA = D + "models/deepseek_common/attention_forward_methods/forward_mla_rocm.py"
F_IDX = D + "layers/attention/dsa/dsa_indexer.py"

CT_IMPORT = (
    "    from sglang.srt.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w8a8_fp8 import (  # noqa: E501\n"
    "        hcu_quant_input,\n"
    "        hcu_reusable_fp8_input,\n"
    "    )\n"
)

LN_EDITS = [
    ("""_use_hcu_lightop_gemma_rmsnorm = _is_hcu and get_bool_env_var(
    "SGLANG_USE_LIGHTOP_GEMMA_RMSNORM"
)
""",
     """_use_hcu_lightop_gemma_rmsnorm = _is_hcu and get_bool_env_var(
    "SGLANG_USE_LIGHTOP_GEMMA_RMSNORM"
)
# SGLANG_HCU_STRIDED_RMS=1: RMSNorm.forward_hip normalizes a row-strided 2-D bf16 x (GLM-5.3's
# q_a / kv_a slices of the packed qkv latent, width 2048 / 512) without the .contiguous()
# copy; lightop rms_norm_opt honours the row stride and gives the same bits.
_hcu_strided_rms = _is_hcu and get_bool_env_var("SGLANG_HCU_STRIDED_RMS")
"""),
    ("""        if not x.is_contiguous():
            x = x.contiguous()

        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            if _is_hcu:
                op.fused_add_rms_norm_opt(
""",
     """        if (
            _hcu_strided_rms
            and residual is None
            and x.dim() == 2
            and x.stride(-1) == 1
            and x.shape[-1] in (512, 2048)
            and x.dtype == torch.bfloat16
        ):
            out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            op.rms_norm_opt(out, x, self.weight.data, self.variance_epsilon)
            return out

        if not x.is_contiguous():
            x = x.contiguous()

        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            if _is_hcu:
                op.fused_add_rms_norm_opt(
"""),
]

CT_EDITS = [
    ("""_use_lightop_channel_fp8 = (
    get_bool_env_var("SGLANG_USE_LIGHTOP_CHANNEL_FP8") and _is_hip
)
""",
     """_use_lightop_channel_fp8 = (
    get_bool_env_var("SGLANG_USE_LIGHTOP_CHANNEL_FP8") and _is_hip
)
# SGLANG_HCU_QUANT_REUSE=1: a caller feeding one bf16 activation to two CHANNEL-fp8 linears
# (GLM-5.3: q_lora -> q_b_proj + indexer wq_b, hidden_states -> fused_qkv_a_proj + indexer
# wk) quantizes it once with hcu_quant_input and passes the HcuQuantArgs pair as
# input_quant_args.  It is the scaled_fp8_quant call apply_fp8_linear makes for such a layer,
# so the GEMM sees the same operands.
_hcu_quant_reuse = False
if get_bool_env_var("SGLANG_HCU_QUANT_REUSE"):
    from sglang.srt.utils import is_hcu

    _hcu_quant_reuse = is_hcu()
if _hcu_quant_reuse:
    from sglang.kernels.ops.quantization.fp8_kernel import (
        scaled_fp8_quant as _scaled_fp8_quant,
    )
"""),
    ("""                bias=bias,
            )

        if isinstance(x, tuple):
""",
     """                bias=bias,
            )

        if isinstance(input_quant_args, HcuQuantArgs):
            # SGLANG_HCU_QUANT_REUSE: x was already quantized by hcu_quant_input
            return apply_fp8_linear(
                input=(input_quant_args[0], input_quant_args[1]),
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                bias=bias,
            )

        if isinstance(x, tuple):
"""),
]

CT_APPEND = '''

class HcuQuantArgs(tuple):
    """(fp8 activation, per-token fp32 scale) made by hcu_quant_input."""


def hcu_reusable_fp8_input(layer, x) -> bool:
    """SGLANG_HCU_QUANT_REUSE: whether ``layer`` would quantize ``x`` exactly like
    hcu_quant_input (CHANNEL fp8 weight, dynamic per-token activation, plain
    apply_fp8_linear path), i.e. may take hcu_quant_input(x) as input_quant_args."""
    scheme = getattr(layer, "scheme", None)
    return (
        _hcu_quant_reuse
        and isinstance(scheme, CompressedTensorsW8A8Fp8)
        and scheme.strategy == QuantizationStrategy.CHANNEL
        and scheme.weight_block_size is None
        and not _use_aiter
        and not _use_lightop_channel_fp8
        and getattr(layer, "input_scale", None) is None
        and isinstance(x, torch.Tensor)
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[0] > 0
    )


def hcu_quant_input(x: torch.Tensor) -> HcuQuantArgs:
    return HcuQuantArgs(_scaled_fp8_quant(x, None, use_per_token_if_dynamic=True))
'''

DS_EDITS = [
    ("""logger = logging.getLogger(__name__)

# One-time SGLANG_OPT_MOE_QUANT_ONCE""",
     """logger = logging.getLogger(__name__)

# SGLANG_HCU_QUANT_REUSE=1: on decode / target-verify batches prepare_qkv_latent quantizes
# hidden_states once for fused_qkv_a_proj_with_mqa and keeps the pair as self._hcu_x_quant;
# forward_mla_rocm passes it on to the indexer's wk, which reads the same hidden_states.
_hcu_quant_reuse = False
if _is_hcu:
    from sglang.srt.utils import get_bool_env_var as _get_bool_env_var

    _hcu_quant_reuse = _get_bool_env_var("SGLANG_HCU_QUANT_REUSE")
if _hcu_quant_reuse:
""" + CT_IMPORT + """
# One-time SGLANG_OPT_MOE_QUANT_ONCE"""),
    ("""        assert self.q_lora_rank is not None
        if self._use_min_latency_fused_a_gemm is None:
""",
     """        assert self.q_lora_rank is not None
        if _hcu_quant_reuse:
            self._hcu_x_quant = None
        if self._use_min_latency_fused_a_gemm is None:
"""),
    ("""                backend=self.fused_a_gemm_backend,
            )
        return self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
""",
     """                backend=self.fused_a_gemm_backend,
            )
        if (
            _hcu_quant_reuse
            and self.use_dsa
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            and hcu_reusable_fp8_input(self.fused_qkv_a_proj_with_mqa, hidden_states)
            and hcu_reusable_fp8_input(getattr(self.indexer, "wk", None), hidden_states)
        ):
            qx = hcu_quant_input(hidden_states)
            self._hcu_x_quant = (hidden_states, qx)
            return self.fused_qkv_a_proj_with_mqa(hidden_states, quant_args=qx)[0]
        return self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
"""),
]

MLA_EDITS = [
    ("""_K3_FUSED_MLA_POST = get_bool_env_var("SGLANG_K3_FUSED_MLA_POST", "false")
""",
     """_K3_FUSED_MLA_POST = get_bool_env_var("SGLANG_K3_FUSED_MLA_POST", "false")
# SGLANG_HCU_QUANT_REUSE=1: q_b_proj and the indexer's wq_b share one per-token fp8 quant of
# q_lora, and the indexer's wk takes the hidden_states quant prepare_qkv_latent made for
# fused_qkv_a_proj_with_mqa (both handed over as Indexer._hcu_quant_reuse around the call).
_hcu_quant_reuse = _is_hcu and get_bool_env_var("SGLANG_HCU_QUANT_REUSE", "false")
if _hcu_quant_reuse:
""" + CT_IMPORT),
    ("""            packed_qkv = get_attn_tp_context().fetch_qkv_latent()
""",
     """            packed_qkv = get_attn_tp_context().fetch_qkv_latent()
            hcu_x_quant = hcu_q_quant = None
            if _hcu_quant_reuse:
                hcu_x_quant = getattr(self, "_hcu_x_quant", None)
                self._hcu_x_quant = None
"""),
    ("""                else:
                    if q_input_quant_args is None:
                        q = self.q_b_proj_forward(q)
                    else:
                        q = self.q_b_proj_forward(
                            q, input_quant_args=q_input_quant_args
                        )

                if q_lora is not None:
                    if self.should_run_indexer(prev_topk_indices):
                        topk_indices = self.indexer(
                            x=hidden_states,
                            q_lora=q_lora,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=self.layer_id,
                        )
                    else:
""",
     """                else:
                    if (
                        _hcu_quant_reuse
                        and q_input_quant_args is None
                        and q_lora is q
                        and hcu_reusable_fp8_input(self.q_b_proj, q)
                        and hcu_reusable_fp8_input(getattr(self.indexer, "wq_b", None), q)
                    ):
                        q_input_quant_args = hcu_quant_input(q)
                        hcu_q_quant = (q, q_input_quant_args)
                    if q_input_quant_args is None:
                        q = self.q_b_proj_forward(q)
                    else:
                        q = self.q_b_proj_forward(
                            q, input_quant_args=q_input_quant_args
                        )

                if q_lora is not None:
                    if self.should_run_indexer(prev_topk_indices):
                        if _hcu_quant_reuse:
                            self.indexer._hcu_quant_reuse = (hcu_q_quant, hcu_x_quant)
                        try:
                            topk_indices = self.indexer(
                                x=hidden_states,
                                q_lora=q_lora,
                                positions=positions,
                                forward_batch=forward_batch,
                                layer_id=self.layer_id,
                            )
                        finally:
                            if _hcu_quant_reuse:
                                self.indexer._hcu_quant_reuse = None
                    else:
"""),
]

IDX_EDITS = [
    ("""_is_hcu = is_hcu()
""",
     """_is_hcu = is_hcu()
# SGLANG_HCU_QUANT_REUSE=1: forward_mla_rocm sets Indexer._hcu_quant_reuse to
# ((q_lora, its fp8 quant) | None, (hidden_states, its fp8 quant) | None) around the indexer
# call; the HCU branch of _get_q_k_bf16 feeds wq_b / wk a pair whose source is its input.
_hcu_quant_reuse = _is_hcu and get_bool_env_var("SGLANG_HCU_QUANT_REUSE")
_hcu_reuse_logged = set()


def _hcu_reused_quant(pair, t):
    \"\"\"pair[1] when pair = (source, quant args) has ``t`` as its source, else None.\"\"\"
    if pair is None:
        return None
    src, args = pair
    if src is t or (
        isinstance(t, torch.Tensor)
        and src.data_ptr() == t.data_ptr()
        and src.shape == t.shape
        and src.stride() == t.stride()
        and src.dtype == t.dtype
    ):
        return args
    return None
"""),
    ("""        if _is_hcu:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            key, _ = self.wk(x)
""",
     """        if _is_hcu:
            qa = xa = None
            reuse = getattr(self, "_hcu_quant_reuse", None) if _hcu_quant_reuse else None
            if reuse is not None:
                qa = _hcu_reused_quant(reuse[0], q_lora)
                xa = _hcu_reused_quant(reuse[1], x)
                hit = (qa is not None, xa is not None)
                if hit not in _hcu_reuse_logged:
                    _hcu_reuse_logged.add(hit)
                    logger.info("SGLANG_HCU_QUANT_REUSE: wq_b reuse=%s wk reuse=%s", *hit)
            if qa is None:
                query, _ = self.wq_b(q_lora)
            else:
                query, _ = self.wq_b(q_lora, quant_args=qa)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            if xa is None:
                key, _ = self.wk(x)
            else:
                key, _ = self.wk(x, quant_args=xa)
"""),
]


def patch(path, edits, append="", need=None):
    bak = path + ".bak_fuse2"
    if not os.path.exists(bak):
        if need is not None:
            assert need in open(path, encoding="utf-8").read(), (path, "run patch_qcat.py first")
        shutil.copy2(path, bak)
    src = open(bak, encoding="utf-8").read()
    for old, new in edits:
        assert src.count(old) == 1, (path, old[:70], src.count(old))
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src + append)
    print("patched", path)


patch(F_LN, LN_EDITS)
patch(F_CT, CT_EDITS, CT_APPEND)
patch(F_DS, DS_EDITS)
patch(F_MLA, MLA_EDITS, need="SGLANG_HCU_MLA_QCAT")
patch(F_IDX, IDX_EDITS)
