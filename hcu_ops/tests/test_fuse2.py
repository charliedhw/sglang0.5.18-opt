"""patch_fuse2.py's two switches vs the unpatched path, bit for bit (GPU, run with
SGLANG_HCU_QUANT_REUSE=1 SGLANG_HCU_STRIDED_RMS=1 SGLANG_HCU_SKINNY_GEMM=1).

  R  CompressedTensorsW8A8Fp8.apply_weights(layer, x, input_quant_args=hcu_quant_input(x))
     vs apply_weights(layer, x) for GLM-5.3's qkv_a / q_b / indexer wq_b / wk shapes
  S  RMSNorm.forward_hip on the q_a / kv_a row views of a packed [T, 2624] latent vs on
     their .contiguous() copies"""
import os

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8 as ct,
)

assert ct._hcu_quant_reuse, "set SGLANG_HCU_QUANT_REUSE=1"
dev = "cuda"
torch.manual_seed(0)
TS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 18, 24, 32, 48, 64, 96, 128, 192, 256)


def make_layer(n, k):
    scheme = ct.CompressedTensorsW8A8Fp8(
        QuantizationArgs(num_bits=8, type="float", strategy=QuantizationStrategy.CHANNEL,
                         symmetric=True, dynamic=False),
        is_static_input_scheme=False,
    )
    layer = torch.nn.Module()
    w = (torch.randn(n, k, device=dev) * 0.05).to(torch.bfloat16)
    s = w.float().abs().amax(dim=1, keepdim=True) / 448.0
    layer.weight = torch.nn.Parameter((w.float() / s).to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
    layer.logical_widths = [n]
    layer.scheme = scheme
    scheme.process_weights_after_loading(layer)
    return layer


ok = True
print("== R: quant reuse ==")
for name, n, k in (("qkv_a", 2624, 6144), ("q_b", 16384, 2048), ("wq_b", 4096, 2048), ("wk", 128, 6144)):
    layer = make_layer(n, k)
    bad = []
    for t in TS:
        x = (torch.randn(t, k, device=dev) * 2).to(torch.bfloat16)
        assert ct.hcu_reusable_fp8_input(layer, x)
        ref = layer.scheme.apply_weights(layer, x)
        qa = ct.hcu_quant_input(x)
        out = layer.scheme.apply_weights(layer, x, input_quant_args=qa)
        out2 = layer.scheme.apply_weights(layer, x, input_quant_args=qa)  # second reader
        if not (torch.equal(out, ref) and torch.equal(out2, ref) and out.shape == ref.shape):
            bad.append((t, (out.float() - ref.float()).abs().max().item()))
    ok &= not bad
    print(f"{name:6s} N={n:5d} K={k}: {'bit-exact' if not bad else bad}")

print("== S: strided rms ==")
from sglang.srt.layers import layernorm as ln  # noqa: E402

assert ln._hcu_strided_rms, "set SGLANG_HCU_STRIDED_RMS=1"
for n, off in ((2048, 0), (512, 2048)):
    norm = RMSNorm(n, eps=1e-6).to(dev).to(torch.bfloat16)
    norm.weight.data = (torch.rand(n, device=dev) + 0.5).to(torch.bfloat16)
    bad = []
    for t in TS:
        packed = (torch.randn(t, 2624, device=dev) * 3).to(torch.bfloat16)
        view = packed[:, off:off + n]
        assert not view.is_contiguous() or t == 1
        out = norm.forward_hip(view)
        ref = norm.forward_hip(view.contiguous())
        if not (torch.equal(out, ref) and out.is_contiguous()):
            bad.append((t, (out.float() - ref.float()).abs().max().item()))
    ok &= not bad
    print(f"width {n} off {off}: {'bit-exact' if not bad else bad}")
print("ALL OK" if ok else "MISMATCH")
