"""Wire hcu_ops.skinny_gemm into sglang's HCU fp8 GEMM call sites behind
SGLANG_HCU_SKINNY_GEMM=1 (default off).  Backs up each file once as *.bak_20261023."""
import os
import shutil

S = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers"
FP8 = f"{S}/quantization/fp8_utils.py"
LIN = f"{S}/linear.py"

HELPER = '''if _is_hcu:
    import deepgemm

    # SGLANG_HCU_SKINNY_GEMM=1: decode-sized M with a tuned config in hcu_ops
    # (skinny_cfg.json) runs hcu_ops.skinny_gemm, which streams the fp8 weight at
    # ~1.5-1.6 TB/s where hipBLASLt's decode tiles reach ~1 TB/s; every other
    # shape / M stays on deepgemm.fp8_gemm (hipBLASLt).
    _hcu_try_skinny_fp8 = None
    if get_bool_env_var("SGLANG_HCU_SKINNY_GEMM"):
        from hcu_ops import try_skinny_fp8 as _hcu_try_skinny_fp8

    def _hcu_fp8_gemm(qinput, x_scale, weight, weight_scale, output):
        if _hcu_try_skinny_fp8 is None or not _hcu_try_skinny_fp8(
            qinput, x_scale, weight, weight_scale, output
        ):
            deepgemm.fp8_gemm((qinput, x_scale), (weight, weight_scale), output)
'''

EDITS = {
    FP8: [
        ("deepgemm.fp8_gemm((input[0], input[1]), (weight, weight_scale), output)",
         "_hcu_fp8_gemm(input[0], input[1], weight, weight_scale, output)"),
        ("deepgemm.fp8_gemm((qinput, x_scale), (weight, weight_scale), output)",
         "_hcu_fp8_gemm(qinput, x_scale, weight, weight_scale, output)"),
        ("if _is_hcu:\n    import deepgemm\n", HELPER),  # last: HELPER calls fp8_gemm itself
    ],
    LIN: [
        ("""                deepgemm.fp8_gemm(
                    (input_x, x_scale),
                    (self.weight, self.weight_scale),
                    output,
                )""",
         """                from sglang.srt.layers.quantization.fp8_utils import _hcu_fp8_gemm

                _hcu_fp8_gemm(
                    input_x, x_scale, self.weight, self.weight_scale, output
                )"""),
    ],
}

for path, edits in EDITS.items():
    bak = path + ".bak_20261023"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    src = open(bak).read()
    for old, new in edits:
        assert src.count(old) == 1, (path, old[:60], src.count(old))
        src = src.replace(old, new)
    open(path, "w").write(src)
    print("patched", path)
