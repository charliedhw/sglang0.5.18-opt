"""hcu_ops MoE routing / activation checks for the GLM-5.3 DP8 DeepEP-LL decode path.

gate: hcu_ops.moe_gate_topk vs lightop moe_fused_gate + sglang's HCU postprocess (padded
      rows -> -1, int64), bit-exact on weights and ids, and timing.
act : hcu_ops.silu_mul_fp8_quant_masked vs lightop fuse_silu_mul_fp8_quant_ep(limit=None):
      bit-exact fp8 bytes and scales on the valid rows for every launch shape, and timing.
"""
import torch
from lightop import moe as lo_moe
from lightop.activation import fuse_silu_mul_fp8_quant_ep
import hcu_ops

torch.manual_seed(0)
dev = "cuda"


def bench(fn, it=200):
    for _ in range(3):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it):
            fn()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record(); g.replay(); en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3 / it


def ref_gate(x, bias, topk, rsf, apply, npad):
    w, ids = lo_moe.moe_fused_gate(x, bias, 1, 1, topk, 0, rsf, apply)
    ids = ids.clone()
    if npad is not None:
        ids[torch.arange(ids.shape[0], device=dev) >= npad, :] = -1
    return w, ids.long()


print("== gate ==")
E, K, RSF = 256, 8, 2.5
for xdt, bdt in [(torch.float32, torch.float32), (torch.bfloat16, torch.float32),
                 (torch.bfloat16, torch.bfloat16)]:
    bias = (torch.randn(E, device=dev) * 0.05).to(bdt)
    bad = 0
    for T in [1, 3, 6, 24, 48, 192, 1000, 8192]:
        for mode in ("randn", "ties"):
            x = torch.randn(T, E, device=dev) * 2
            if mode == "ties":  # few distinct logits -> many equal scores
                x = x.round()
                bias_t = torch.zeros_like(bias)
            else:
                bias_t = bias
            x = x.to(xdt)
            for npv in (None, max(T - 2, 0)):
                npad = None if npv is None else torch.tensor([npv], device=dev, dtype=torch.int32)
                for apply in (True, False):
                    w0, i0 = ref_gate(x, bias_t, K, RSF, apply, npad)
                    w1, i1 = hcu_ops.moe_gate_topk(x, bias_t, K, RSF, apply, npad)
                    if not (torch.equal(w0, w1) and torch.equal(i0, i1)):
                        bad += 1
                        if bad <= 3:
                            d = (i0 != i1).any(1).nonzero().flatten()[:2].tolist()
                            print(f"  MISMATCH x={xdt} b={bdt} T={T} {mode} np={npv} apply={apply} "
                                  f"wdiff={(w0 - w1).abs().max().item():.3g} rows={d}")
                            for r in d:
                                print("   ref", i0[r].tolist(), w0[r].tolist())
                                print("   hcu", i1[r].tolist(), w1[r].tolist())
    print(f"x={xdt} bias={bdt}: {'bit-exact' if bad == 0 else f'{bad} mismatching cases'}")

for E2, K2 in [(128, 8), (384, 8), (512, 8)]:  # lightop has no topk 6 for one group
    x = torch.randn(64, E2, device=dev)
    b = torch.randn(E2, device=dev) * 0.05
    w0, i0 = ref_gate(x, b, K2, RSF, True, None)
    w1, i1 = hcu_ops.moe_gate_topk(x, b, K2, RSF, True, None)
    print(f"E={E2} topk={K2}: bit-exact={torch.equal(w0, w1) and torch.equal(i0, i1)}")

bias = torch.randn(E, device=dev) * 0.05
for T in [6, 24, 48, 192]:
    x = torch.randn(T, E, device=dev)
    npad = torch.tensor([T], device=dev, dtype=torch.int32)
    t0 = bench(lambda: lo_moe.moe_fused_gate(x, bias, 1, 1, K, 0, RSF, True))
    t1 = bench(lambda: hcu_ops.moe_gate_topk(x, bias, K, RSF, True, npad))
    print(f"T={T:4d} lightop gate {t0:6.1f} us (+ separate mask/int64 kernel)  hcu fused {t1:6.1f} us")

print("== act (masked SwiGLU + fp8 quant) ==")
nb, first = hcu_ops._C.bf16_round_check(torch.empty(0, device=dev)).tolist()
print(f"v_cvt_bf16_f32 vs software RNE, all 2^32 floats: {nb} mismatches"
      + (f" (first 0x{first & 0xffffffff:08x})" if nb else ""))

NE, CAP, D = 32, 256, 2048
x = (torch.randn(NE, CAP, 2 * D, device=dev) * 2).to(torch.bfloat16)
x[3, 0, :7] = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 88.0, -88.0, 3e4])  # tiny / huge gate inputs
x[3, 0, 7:13] = torch.tensor([-87.5, -88.5, -89.0, -300.0, 5.9375, -5.9375])  # rcp-silu misses
x[3, 1, :] = 0  # all-zero row -> min scale
x[5, 0, 10:13] = torch.tensor([float("nan"), float("-inf"), float("inf")])  # NaN paths
x[5, 0, D + 20] = float("nan")


def check(x, m, c, w):
    o0, s0 = fuse_silu_mul_fp8_quant_ep(x, 0, m, limit=None)
    o1, s1 = hcu_ops.silu_mul_fp8_quant_masked(x, m, c, w)
    same = o1.dtype == o0.dtype and s1.shape == s0.shape
    for e in range(x.shape[0]):
        k = int(m[e])
        same &= torch.equal(o0[e, :k].view(torch.uint8), o1[e, :k].view(torch.uint8))
        same &= torch.equal(s0[e, :k], s1[e, :k])
    return same


LAUNCH = [(c, w) for w in (0, 1, 2, 4) for c in (256, 512)]
for name, m in [("all1", torch.ones(NE)), ("rand0-3", torch.randint(0, 4, (NE,))),
                ("rand0-6", torch.randint(0, 7, (NE,))), ("skew", torch.tensor([40] + [2] * (NE - 1))),
                ("all4", torch.full((NE,), 4)), ("rand0-9", torch.randint(0, 10, (NE,))),
                ("all5", torch.full((NE,), 5)), ("all6", torch.full((NE,), 6)),
                ("rand0-14", torch.randint(0, 15, (NE,))), ("all8", torch.full((NE,), 8)),
                ("all12", torch.full((NE,), 12)),
                ("all48", torch.full((NE,), 48)), ("full", torch.full((NE,), CAP))]:
    m = m.to(device=dev, dtype=torch.int32)
    m[3] = max(int(m[3]), 2)
    m[5] = max(int(m[5]), 1)
    res = {}
    bad = [cw for cw in LAUNCH if not check(x, m, *cw)]
    for c, w in LAUNCH:
        res[(c, w)] = bench(lambda: hcu_ops.silu_mul_fp8_quant_masked(x, m, c, w))
    t0 = bench(lambda: fuse_silu_mul_fp8_quant_ep(x, 0, m, limit=None))
    best = min(res, key=res.get)
    print(f"m={name:8s} rows={int(m.sum()):5d} bit-exact={'ALL' if not bad else f'BAD {bad}'}  "
          f"lightop {t0:7.1f} us  hcu best {best} {res[best]:6.1f} us  default {res[hcu_ops._ACT_LAUNCH]:6.1f} us")
    print("   " + "  ".join(f"w{w}c{c}:{t:.1f}" for (c, w), t in res.items()))
for d2 in (512, 1024, 1536, 3072, 4096):  # other widths, auto split
    x2 = (torch.randn(8, 64, 2 * d2, device=dev) * 2).to(torch.bfloat16)
    ok = all(check(x2, torch.tensor(mm, device=dev, dtype=torch.int32), 512, 0)
             for mm in ([1] * 8, [5, 0, 9, 3, 7, 2, 0, 1], [40] * 8, [64] * 8))
    print(f"d={d2}: bit-exact={ok}")
o0, s0 = fuse_silu_mul_fp8_quant_ep(x, 0, m, limit=None)
o1, s1 = hcu_ops.try_silu_mul_fp8_quant_masked(x, m)
print("hook ok:", o1.shape == o0.shape and torch.equal(o1.view(torch.uint8)[:, :8], o0.view(torch.uint8)[:, :8]))
