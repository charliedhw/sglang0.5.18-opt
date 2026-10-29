"""MoE decode switches for sglang on HCU (default off, each backs its file up once as
*.bak_20261023 and patches from the backup):

SGLANG_HCU_FUSED_ACT=1  ep_moe/layer.py masked DeepGEMM path (DeepEP low latency): without a
    swiglu limit, hcu_ops.silu_mul_fp8_quant_masked replaces lightop
    fuse_silu_mul_fp8_quant_ep (bit-exact; it only walks the masked_m rows, spread over all
    CUs, where lightop's kernel strides over the whole [E, cap] buffer); shapes it doesn't
    cover keep lightop.
SGLANG_HCU_FUSED_GATE=1 topk.py select_experts: on the HCU DeepEP + DeepGEMM path with
    one expert group, no fused shared experts and no EPLB remap, hcu_ops.moe_gate_topk
    replaces lightop moe_fused_gate plus the compiled padded-row mask / int64 postprocess.
"""
import os
import shutil

SRT = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/moe"

LAYER_EDITS = [
    ("""_use_lightop_ep_gather = get_bool_env_var("SGLANG_USE_LIGHTOP_EP_GATHER", "true")
""",
     """_use_lightop_ep_gather = get_bool_env_var("SGLANG_USE_LIGHTOP_EP_GATHER", "true")
# SGLANG_HCU_FUSED_ACT=1: masked SwiGLU + fp8 quant over the masked_m rows only
# (hcu_ops.silu_mul_fp8_quant_masked, same numerics as lightop's limit=None kernel)
_hcu_act = None
if get_bool_env_var("SGLANG_HCU_FUSED_ACT"):
    import hcu_ops as _hcu_act
"""),
    ("""        q_a2_all, q_a2_scale = fuse_silu_mul_fp8_quant_ep(
            input=gateup_output,
            fp8type=0,
            tokens_per_expert=masked_m,
            limit=self.moe_runner_config.swiglu_limit,
        )
""",
     """        q_a2 = None
        if _hcu_act is not None and self.moe_runner_config.swiglu_limit is None:
            q_a2 = _hcu_act.try_silu_mul_fp8_quant_masked(gateup_output, masked_m)
        if q_a2 is None:
            q_a2 = fuse_silu_mul_fp8_quant_ep(
                input=gateup_output,
                fp8type=0,
                tokens_per_expert=masked_m,
                limit=self.moe_runner_config.swiglu_limit,
            )
        q_a2_all, q_a2_scale = q_a2
"""),
]

TOPK_EDITS = [
    ("""_skip_hip_pad_mask = get_bool_env_var("SGLANG_MORI_NO_PAD_MASK", "False")
""",
     """_skip_hip_pad_mask = get_bool_env_var("SGLANG_MORI_NO_PAD_MASK", "False")

# SGLANG_HCU_FUSED_GATE=1: hcu_ops.moe_gate_topk = lightop moe_fused_gate (one expert
# group, same numerics) + the HCU DeepEP id postprocess (padded rows -> -1, int64 ids)
# in one kernel, for the configurations where that postprocess is all that follows.
_hcu_gate = None
if get_bool_env_var("SGLANG_HCU_FUSED_GATE"):
    import hcu_ops as _hcu_gate
_hcu_gate_path_ok = None


def _hcu_gate_usable(
    router_logits, correction_bias, top_k, renormalize, num_expert_group, topk_group,
    num_fused_shared_experts,
):
    global _hcu_gate_path_ok
    if (
        _hcu_gate is None
        or correction_bias is None
        or num_expert_group != 1
        or topk_group != 1
        or num_fused_shared_experts != 0
        or not renormalize
    ):
        return False
    if _hcu_gate_path_ok is None:
        # static server configuration: the branch _post_process_topk_ids takes on HIP
        _hcu_gate_path_ok = bool(
            _is_hcu
            and _use_lightop
            and get_moe_a2a_backend().is_deepep()
            and (get_moe_runner_backend().is_deep_gemm() or _use_deepgemm_moe)
            and not _eplb_remap_enabled()
            and not _simulated_expert_balance
            and not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
            and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
        )
    return _hcu_gate_path_ok and _hcu_gate.moe_gate_supported(
        router_logits, correction_bias, top_k
    )
"""),
    ("""        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
""",
     """        elif _hcu_gate is not None and _hcu_gate_usable(
            router_logits, correction_bias, num_routed_topk, renormalize,
            num_expert_group, topk_group, num_fused_shared_experts,
        ):
            topk_weights, topk_ids = _hcu_gate.moe_gate_topk(
                router_logits,
                correction_bias,
                num_routed_topk,
                routed_scaling_factor,
                apply_routed_scaling_factor_on_output,
                num_token_non_padded,
            )
            # ids are already final (int64, padded rows -1), so the HIP DeepEP
            # postprocess in _post_process_topk_ids has nothing left to do
            num_token_non_padded = None
        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
"""),
]


def patch(path, edits, checks=()):
    bak = path + ".bak_20261023"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    src = open(bak).read()
    for c in checks:
        assert c in src, c
    for old, new in edits:
        assert src.count(old) == 1, (path, old[:70], src.count(old))
        src = src.replace(old, new)
    open(path, "w").write(src)
    print("patched", path)


patch(SRT + "/ep_moe/layer.py", LAYER_EDITS, ["get_bool_env_var,"])
patch(SRT + "/topk.py", TOPK_EDITS,
      ["def _eplb_remap_enabled", "_simulated_expert_balance = ", "_use_deepgemm_moe = ",
       "from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend"])
