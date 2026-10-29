"""Patch forward_mla_rocm.py: route HCU decode-sized bf16 MLA absorb bmm's through
hcu_ops.try_bmm_bf16 under SGLANG_HCU_SKINNY_BMM=1 (default off)."""
import sys

F = sys.argv[1]
s = open(F).read()
if "SGLANG_HCU_SKINNY_BMM" in s:
    print("already patched")
    sys.exit(0)


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)


sub('''_K3_FUSED_MLA_POST = get_bool_env_var("SGLANG_K3_FUSED_MLA_POST", "false")
''', '''_K3_FUSED_MLA_POST = get_bool_env_var("SGLANG_K3_FUSED_MLA_POST", "false")
# SGLANG_HCU_SKINNY_BMM=1: decode-sized bf16 absorb bmm's (q_nope @ w_kc, attn @ w_vc)
# run hcu_ops' batched skinny GEMM; it returns None (-> torch.bmm) when it doesn't apply.
_hcu_bmm = None
if _is_hcu and get_bool_env_var("SGLANG_HCU_SKINNY_BMM", "false"):
    import hcu_ops as _hcu_bmm
''')

sub('''        else:
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                _scaled_bmm_weight(attn.w_kc, attn.w_scale),
            )
    return q_nope_out
''', '''        else:
            w_kc = _scaled_bmm_weight(attn.w_kc, attn.w_scale)
            q_nope_out = None
            if _hcu_bmm is not None and q_nope.dtype == torch.bfloat16:
                q_nope_out = _hcu_bmm.try_bmm_bf16(q_nope.transpose(0, 1), w_kc)
            if q_nope_out is None:
                q_nope_out = torch.bmm(q_nope.to(torch.bfloat16).transpose(0, 1), w_kc)
    return q_nope_out
''')

sub('''                torch.bmm(
                    attn_output.transpose(0, 1),
                    _scaled_bmm_weight(attn.w_vc, attn.w_scale),
                    out=output.transpose(0, 1),
                )
                return output.flatten(1, 2)
''', '''                w_vc = _scaled_bmm_weight(attn.w_vc, attn.w_scale)
                if (
                    _hcu_bmm is None
                    or _hcu_bmm.try_bmm_bf16(
                        attn_output.transpose(0, 1), w_vc, output.transpose(0, 1)
                    )
                    is None
                ):
                    torch.bmm(
                        attn_output.transpose(0, 1), w_vc, out=output.transpose(0, 1)
                    )
                return output.flatten(1, 2)
''')
open(F, "w").write(s)
print("patched", F)
