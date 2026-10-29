"""GLM-5.3 producer + fp8-quant fusions and an MQA launch knob, each behind its own switch
(default off).  The fusions give the bits of the unfused ops (test_norm_quant.py):

SGLANG_HCU_NORM_QUANT=1   the decoder layers' input / post-attention RMSNorms (residual path,
  flagged _hcu_norm_quant in DeepseekV2DecoderLayer) run hcu_ops.fused_add_rms_quant: lightop
  fused_add_rms_norm_opt plus the per-token fp8 quant of the normed x in one kernel, stashed
  as (x, x._version, q, s).  A CHANNEL-fp8 linear (compressed_tensors_w8a8_fp8.py apply_weights,
  hcu_quant_input) fed that very x, unmodified since, takes (q, s) instead of running
  scaled_fp8_quant: fused_qkv_a_proj_with_mqa + indexer wk, shared-expert / dense gate_up --
  2 quant kernels less per layer.  SGLANG_HCU_NORM_QUANT_MAX_M (default 4096) caps the rows.
SGLANG_HCU_SILU_QUANT=1   DeepseekV2MLP (shared experts, dense layers): SiluAndMul and the
  down_proj input quant run as one hcu_ops.silu_mul_quant kernel (the bf16 product never goes
  to memory) and down_proj takes the pair -- 1 kernel less per layer.
SGLANG_HCU_MQA_PCTAS=<n>  persistent CTAs of lightop page_mqa_logits_sparse_mask[_grouped]
  (DSA indexer logits; 0 / unset = lightop default min(chunks, 32)).

Run after patch_fuse2.py.  Each file is backed up once as *.bak_fuse3 and patched from that
backup, so re-running is idempotent."""
import os
import shutil

D = os.environ.get("SGL_SRT_DIR", "/usr/local/lib/python3.10/dist-packages/sglang/srt/")
F_LN = D + "layers/layernorm.py"
F_CT = D + "layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py"
F_DS = D + "models/deepseek_v2.py"
F_IDX = D + "layers/attention/dsa/dsa_indexer.py"

LN_EDITS = [
    ("""_hcu_strided_rms = _is_hcu and get_bool_env_var("SGLANG_HCU_STRIDED_RMS")
""",
     """_hcu_strided_rms = _is_hcu and get_bool_env_var("SGLANG_HCU_STRIDED_RMS")
# SGLANG_HCU_NORM_QUANT=1: an RMSNorm flagged _hcu_norm_quant (GLM-5.3 decoder input /
# post-attention norms) runs its residual path as hcu_ops.fused_add_rms_quant -- the x /
# residual bits of lightop fused_add_rms_norm_opt plus the per-token fp8 quant of x -- and
# stashes (x, q, s) for the CHANNEL-fp8 linears reading x (compressed_tensors_w8a8_fp8.py).
_hcu_norm_quant = _is_hcu and get_bool_env_var("SGLANG_HCU_NORM_QUANT")
_hcu_norm_quant_max_m = 0
if _hcu_norm_quant:
    import os

    import hcu_ops as _hcu_ops

    _hcu_norm_quant_max_m = int(os.environ.get("SGLANG_HCU_NORM_QUANT_MAX_M", "4096"))
"""),
    ("""            if _is_hcu:
                op.fused_add_rms_norm_opt(
                    x,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return x, residual
            return _call_vllm_fused_add_rms_norm(
""",
     """            if _is_hcu:
                if (
                    _hcu_norm_quant
                    and getattr(self, "_hcu_norm_quant", False)
                    and 0 < x.shape[0] <= _hcu_norm_quant_max_m
                    and not torch.compiler.is_compiling()
                    and _hcu_ops.norm_quant_supported(x, residual, self.weight.data)
                ):
                    q, s = _hcu_ops.fused_add_rms_quant(
                        x, residual, self.weight.data, self.variance_epsilon
                    )
                    _hcu_ops.norm_quant_stash_set(x, q, s)
                    return x, residual
                op.fused_add_rms_norm_opt(
                    x,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return x, residual
            return _call_vllm_fused_add_rms_norm(
"""),
]

CT_EDITS = [
    ("""    _hcu_quant_reuse = is_hcu()
if _hcu_quant_reuse:
    from sglang.kernels.ops.quantization.fp8_kernel import (
        scaled_fp8_quant as _scaled_fp8_quant,
    )
""",
     """    _hcu_quant_reuse = is_hcu()
# SGLANG_HCU_NORM_QUANT=1 / SGLANG_HCU_SILU_QUANT=1: the per-token fp8 quant of a CHANNEL-fp8
# linear's bf16 input comes from the kernel producing that input (hcu_ops.fused_add_rms_quant
# in RMSNorm.forward_hip, stashed per tensor; hcu_ops.silu_mul_quant in DeepseekV2MLP) with
# the bits of the scaled_fp8_quant below.  apply_weights / hcu_quant_input take a stashed pair
# whose source tensor is x itself.
_hcu_norm_quant = _hcu_silu_quant = False
if get_bool_env_var("SGLANG_HCU_NORM_QUANT") or get_bool_env_var("SGLANG_HCU_SILU_QUANT"):
    from sglang.srt.utils import is_hcu as _is_hcu_fn

    if _is_hcu_fn():
        _hcu_norm_quant = get_bool_env_var("SGLANG_HCU_NORM_QUANT")
        _hcu_silu_quant = get_bool_env_var("SGLANG_HCU_SILU_QUANT")
_hcu_prequant = _hcu_quant_reuse or _hcu_norm_quant or _hcu_silu_quant
if _hcu_prequant:
    from sglang.kernels.ops.quantization.fp8_kernel import (
        scaled_fp8_quant as _scaled_fp8_quant,
    )
if _hcu_norm_quant:
    import logging

    import hcu_ops as _hcu_ops

    _hcu_nq_logger = logging.getLogger(__name__)
    _hcu_nq_logged = set()
"""),
    ("""                bias=bias,
            )

        if isinstance(x, tuple):
""",
     """                bias=bias,
            )

        if _hcu_norm_quant and input_quant_args is None and isinstance(x, torch.Tensor):
            qs = _hcu_ops.norm_quant_stash_get(x)
            if qs is not None and hcu_reusable_fp8_input(layer, x):
                # SGLANG_HCU_NORM_QUANT: x is an RMSNorm output its kernel already quantized
                key = tuple(layer.weight.shape)
                if key not in _hcu_nq_logged:
                    _hcu_nq_logged.add(key)
                    _hcu_nq_logger.info(
                        "SGLANG_HCU_NORM_QUANT: fp8 linear weight %s takes the norm's quant",
                        key,
                    )
                return apply_fp8_linear(
                    input=qs,
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    bias=bias,
                )

        if isinstance(x, tuple):
"""),
    ("""    return (
        _hcu_quant_reuse
        and isinstance(scheme, CompressedTensorsW8A8Fp8)
""",
     """    return (
        _hcu_prequant
        and isinstance(scheme, CompressedTensorsW8A8Fp8)
"""),
    ("""def hcu_quant_input(x: torch.Tensor) -> HcuQuantArgs:
    return HcuQuantArgs(_scaled_fp8_quant(x, None, use_per_token_if_dynamic=True))
""",
     """def hcu_quant_input(x: torch.Tensor) -> HcuQuantArgs:
    if _hcu_norm_quant:
        qs = _hcu_ops.norm_quant_stash_get(x)
        if qs is not None:
            return HcuQuantArgs(qs)
    return HcuQuantArgs(_scaled_fp8_quant(x, None, use_per_token_if_dynamic=True))
"""),
]

DS_EDITS = [
    ("""    )

# One-time SGLANG_OPT_MOE_QUANT_ONCE""",
     """    )

# SGLANG_HCU_NORM_QUANT=1: the decoder layers' input / post-attention RMSNorms (flagged
# _hcu_norm_quant below) also emit the per-token fp8 quant of their output for the
# CHANNEL-fp8 linears reading it (fused_qkv_a_proj_with_mqa + indexer wk, shared-expert /
# dense gate_up).  SGLANG_HCU_SILU_QUANT=1: DeepseekV2MLP's SiluAndMul and down_proj's input
# quant run as one hcu_ops.silu_mul_quant kernel.  Same bits as the unfused ops.
_hcu_norm_quant = _hcu_silu_quant = False
if _is_hcu:
    _hcu_norm_quant = _get_bool_env_var("SGLANG_HCU_NORM_QUANT")
    _hcu_silu_quant = _get_bool_env_var("SGLANG_HCU_SILU_QUANT")
if _hcu_silu_quant:
    import hcu_ops as _hcu_ops

    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (  # noqa: E501
        HcuQuantArgs,
        hcu_reusable_fp8_input,
    )

# One-time SGLANG_OPT_MOE_QUANT_ONCE"""),
    ("""        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
""",
     """        else:
            if (
                _hcu_silu_quant
                and self.down_proj.input_is_parallel
                and hcu_reusable_fp8_input(self.down_proj, gate_up)
                and _hcu_ops.silu_mul_quant_supported(gate_up)
                and not torch.compiler.is_compiling()
            ):
                # SGLANG_HCU_SILU_QUANT: down_proj takes the fused kernel's fp8 pair
                x, _ = self.down_proj(
                    gate_up,
                    input_quant_args=HcuQuantArgs(_hcu_ops.silu_mul_quant(gate_up)),
                )
                return x
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
"""),
    ("""        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self._gfx95_quant_format = self._detect_gfx95_quant_format()
""",
     """        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if _hcu_norm_quant:
            self.input_layernorm._hcu_norm_quant = True
            self.post_attention_layernorm._hcu_norm_quant = True

        self._gfx95_quant_format = self._detect_gfx95_quant_format()
"""),
]

IDX_EDITS = [
    ("""_hcu_reuse_logged = set()
""",
     """_hcu_reuse_logged = set()
# SGLANG_HCU_MQA_PCTAS=<n>: persistent CTAs of lightop page_mqa_logits_sparse_mask[_grouped]
# (0 = lightop default, min(chunks, 32)); test_mqa_pctas.py checks the bits do not change.
_hcu_mqa_pctas = 0
if _is_hcu:
    import os as _os

    _hcu_mqa_pctas = int(_os.environ.get("SGLANG_HCU_MQA_PCTAS", "0"))
"""),
    ("""                    clean_logits=True,
                    num_warps=4,
                )
            else:
                from lightop.gemmopt import page_mqa_logits_sparse_mask_grouped
""",
     """                    clean_logits=True,
                    num_warps=4,
                    persistent_ctas=_hcu_mqa_pctas,
                )
            else:
                from lightop.gemmopt import page_mqa_logits_sparse_mask_grouped
"""),
    ("""                    num_warps=4,
                    group_size=sparse_route.group_size,
""",
     """                    num_warps=4,
                    persistent_ctas=_hcu_mqa_pctas,
                    group_size=sparse_route.group_size,
"""),
]


def patch(path, edits, need):
    bak = path + ".bak_fuse3"
    if not os.path.exists(bak):
        assert need in open(path, encoding="utf-8").read(), (path, "run patch_fuse2.py first")
        shutil.copy2(path, bak)
    src = open(bak, encoding="utf-8").read()
    for old, new in edits:
        assert src.count(old) == 1, (path, old[:70], src.count(old))
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print("patched", path)


patch(F_LN, LN_EDITS, "SGLANG_HCU_STRIDED_RMS")
patch(F_CT, CT_EDITS, "SGLANG_HCU_QUANT_REUSE")
patch(F_DS, DS_EDITS, "SGLANG_HCU_QUANT_REUSE")
patch(F_IDX, IDX_EDITS, "SGLANG_HCU_QUANT_REUSE")
