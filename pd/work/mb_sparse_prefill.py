"""flash_mla sparse_prefill_fwd timing at the P-node CP8 prefill shape (s_q tokens per rank,
64 heads, d_qk 576, d_v 512, topk 2048).  Env knobs are read by the .so, so run one process per
variant: e.g. FLASH_MLA_FORCE_DSA_MLS_PREFILL=1 python mb_sparse_prefill.py
usage: python mb_sparse_prefill.py [s_q] [ctx] [pattern: rand|local]"""
import sys

import torch
from flash_mla import flash_mla_interface as fmi

s_q = int(sys.argv[1]) if len(sys.argv) > 1 else 3752
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
pat = sys.argv[3] if len(sys.argv) > 3 else "rand"
H, DQK, DV, TOPK = 64, 576, 512, 2048
torch.manual_seed(0)
dev = "cuda"
q = torch.randn(s_q, H, DQK, device=dev, dtype=torch.bfloat16)
kv = torch.randn(ctx, 1, DQK, device=dev, dtype=torch.bfloat16)
# CP interleave: rank 0 owns positions start, start+8, ... of the last chunk
pos = ctx - 8 * s_q + 8 * torch.arange(s_q, device=dev)
r = torch.rand(s_q, ctx, device="cpu")
idx = torch.empty(s_q, 1, TOPK, dtype=torch.int32)
for i in range(s_q):
    p = int(pos[i]) + 1
    if pat == "local":
        loc = torch.arange(max(0, p - TOPK // 2), p)
        rest = torch.randperm(max(1, p - len(loc)))[: TOPK - len(loc)]
        sel = torch.cat([rest, loc])
    else:
        sel = torch.randperm(p)[:TOPK]
    sel, _ = torch.sort(sel)
    row = torch.full((TOPK,), -1, dtype=torch.int32)
    row[: len(sel)] = sel.to(torch.int32)
    idx[i, 0] = row
idx = idx.to(dev)
scale = DQK ** -0.5
fn = fmi.flash_mla_cuda.sparse_prefill_fwd
o = fn(q, kv, idx, scale, DV, None, None)[0]
torch.cuda.synchronize()
ref = o.float().clone()
for _ in range(3):
    fn(q, kv, idx, scale, DV, None, None)
torch.cuda.synchronize()
st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
n = 10
st.record()
for _ in range(n):
    o = fn(q, kv, idx, scale, DV, None, None)[0]
en.record()
torch.cuda.synchronize()
ms = st.elapsed_time(en) / n
fl = s_q * H * TOPK * (DQK + DV) * 2
print(f"s_q={s_q} ctx={ctx} pat={pat}: {ms:.3f} ms  {fl / ms / 1e9:.1f} TFLOPS  "
      f"out_sum={ref.abs().sum().item():.4e}")
