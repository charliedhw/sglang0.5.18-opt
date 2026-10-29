"""SGLANG_HCU_SKINNY_BMM=1 (default off): the bf16 MLA absorb bmm's in
models/deepseek_common/attention_forward_methods/forward_mla.py (q_nope @ w_kc in
forward_absorb_prepare, attn_output @ w_vc in forward_absorb_core) go through
hcu_ops.try_bmm_bf16 for the decode-sized M it has a tuned config for (hcu_ops._BMM_CFG) and
keep torch.bmm otherwise.
Backs the file up once as *.bak_20261023 and patches from the backup."""
import os
import shutil

F = ("/usr/local/lib/python3.10/dist-packages/sglang/srt/models/deepseek_common/"
     "attention_forward_methods/forward_mla.py")

EDITS = [
    ("""_ENABLE_DSA_Q8KV8_QPREP_OVERLAP = envs.SGLANG_ENABLE_DSA_Q8KV8_QPREP_OVERLAP.get()
""",
     """_ENABLE_DSA_Q8KV8_QPREP_OVERLAP = envs.SGLANG_ENABLE_DSA_Q8KV8_QPREP_OVERLAP.get()

# SGLANG_HCU_SKINNY_BMM=1: decode-sized bf16 absorb bmm's (q_nope @ w_kc, attn @ w_vc)
# run hcu_ops' batched skinny GEMM; it returns None (-> torch.bmm) when it doesn't apply.
_hcu_bmm = None
if __import__("os").environ.get("SGLANG_HCU_SKINNY_BMM", "0").lower() in ("1", "true"):
    import hcu_ops as _hcu_bmm
"""),
    ("""            else:
                q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
""",
     """            else:
                q_nope_out = None
                if _hcu_bmm is not None:
                    q_nope_out = _hcu_bmm.try_bmm_bf16(q_nope.transpose(0, 1), self.w_kc)
                if q_nope_out is None:
                    q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
"""),
    ("""                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
""",
     """                _vc_out = attn_bmm_output.view(
                    -1, self.num_local_heads, self.v_head_dim
                ).transpose(0, 1)
                if (
                    _hcu_bmm is None
                    or _hcu_bmm.try_bmm_bf16(
                        attn_output.transpose(0, 1), self.w_vc, _vc_out
                    )
                    is None
                ):
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc, out=_vc_out)
"""),
]

bak = F + ".bak_20261023"
if not os.path.exists(bak):
    shutil.copy2(F, bak)
src = open(bak).read()
for old, new in EDITS:
    assert src.count(old) == 1, (old[:70], src.count(old))
    src = src.replace(old, new)
open(F, "w").write(src)
print("patched", F)
