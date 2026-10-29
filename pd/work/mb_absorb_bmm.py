"""MLA absorb BMM timing at the P-node prefill shape (T tokens per CP rank, 64 heads):
  q absorb  : q_nope [T,64,192] (strided view of q) -> bmm with w_kc [64,192,512] -> [64,T,512]
  out absorb: attn   [T,64,512]                    -> bmm with w_vc [64,512,256] -> [64,T,256]
Variants: current strided torch.bmm, contiguous-A bmm, hcu_ops.try_bmm_bf16 (if it accepts the shape).
usage: python mb_absorb_bmm.py [T=8192]"""
import sys

import torch

T = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
H, DN, DR, L, DV = 64, 192, 64, 512, 256
dev = "cuda"
torch.manual_seed(0)
q = torch.randn(T, H, DN + DR, device=dev, dtype=torch.bfloat16)
q_nope = q[..., :DN]
w_kc = torch.randn(H, DN, L, device=dev, dtype=torch.bfloat16) * 0.05
attn = torch.randn(T, H, L, device=dev, dtype=torch.bfloat16)
w_vc = torch.randn(H, L, DV, device=dev, dtype=torch.bfloat16) * 0.05
try:
    import hcu_ops
except ImportError:
    hcu_ops = None


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1e3


cases = {
    "q   strided bmm": (lambda: torch.bmm(q_nope.transpose(0, 1), w_kc), T * H * DN * L * 2),
    "q   contig  bmm": (lambda: torch.bmm(q_nope.transpose(0, 1).contiguous(), w_kc), T * H * DN * L * 2),
    "out strided bmm": (lambda: torch.bmm(attn.transpose(0, 1), w_vc), T * H * L * DV * 2),
    "out contig  bmm": (lambda: torch.bmm(attn.transpose(0, 1).contiguous(), w_vc), T * H * L * DV * 2),
}
if hcu_ops is not None and hasattr(hcu_ops, "try_bmm_bf16"):
    cases["q   hcu_ops    "] = (lambda: hcu_ops.try_bmm_bf16(q_nope.transpose(0, 1), w_kc), T * H * DN * L * 2)
ref = {"q": torch.bmm(q_nope.transpose(0, 1), w_kc).float(), "out": torch.bmm(attn.transpose(0, 1), w_vc).float()}
for name, (fn, fl) in cases.items():
    r = fn()
    if r is None:
        print(f"{name}: not handled")
        continue
    err = (r.float() - ref[name.split()[0]]).abs().max().item()
    us = bench(fn)
    print(f"T={T} {name}: {us:8.1f} us  {fl / us / 1e6:6.1f} TFLOPS  maxerr {err:.3g}")
