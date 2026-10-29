"""hcu_ops.rope_qout_kv_store vs the ops it replaces in sglang's DSA decode front end:
lightop op.rms_norm_opt(kv_a) + hcu_ops.rope_qout (q -> q_out, k_pe in place) + lightop
kvcache.fused_quantize_and_store_mla_kv_cache(fp8_e4m3, eps 1e-6).  kv_c / k_pe / q_out and the
whole kv_buffer must be bit-identical, and the q half of the packed latent untouched.  Operands
are laid out as in GLM-5.3 (packed [T, 2048 + 576] latent, q [T, 64, 256] with the rope tail at
192, q_cat [T, 64, 576], 656-byte slots).  Also times both inside a CUDA graph.
usage: test_kv_fused.py [--bench-only]"""
import sys

import torch
from lightop import kvcache as kvop
from lightop import op

import hcu_ops
from hcu_ops import _C

dev = "cuda"
torch.manual_seed(0)
H, QL, KVL, ROT, SLOT = 64, 2048, 512, 64, 656
MAX_POS, SLOTS = 70000, 4096


def make(t, scale=1.0):
    packed = (torch.randn(t, QL + KVL + ROT, device=dev) * scale).bfloat16()
    packed[:, QL : QL + KVL : 37] *= 16  # outlier channels
    if t > 2:
        packed[1, QL : QL + KVL] = 0  # zero row: rstd = rsqrt(eps), amax 0 -> eps
    if t > 3:
        packed[2, QL + 5] = 3e4  # one huge element
    q = torch.randn(t, H, 256, device=dev).bfloat16()
    pos = torch.randint(0, MAX_POS, (t,), device=dev)
    return packed, q, pos


def views(packed, q):
    kv_c = packed[:, QL : QL + KVL]
    k_pe = packed[:, QL + KVL :].unsqueeze(1)
    q_pe = q[..., 192:]
    return kv_c, k_pe, q_pe


def diff(a, b):
    return int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) if a.dtype == torch.uint8 else int((a != b).sum())


def run_ref(packed, q, pos, cache, w, eps, neox, buf, loc, inplace):
    kv_c, k_pe, q_pe = views(packed, q)
    kvn = torch.empty(kv_c.shape, device=dev, dtype=torch.bfloat16)
    op.rms_norm_opt(kvn, kv_c, w, eps)
    q_cat = q_pe if inplace else torch.zeros(q.shape[0], H, KVL + ROT, device=dev, dtype=torch.bfloat16)
    q_out = q_pe if inplace else q_cat[..., KVL:]
    if inplace:
        hcu_ops.rope_inplace(pos, q_pe, k_pe, ROT, cache, neox)
    else:
        hcu_ops.rope_qout(pos, q_pe, q_out, k_pe, ROT, cache, neox)
    kvop.fused_quantize_and_store_mla_kv_cache(kvn.unsqueeze(1), k_pe, buf, loc, "fp8_e4m3", 1e-6)
    return kvn, k_pe, q_out, q_cat


def run_fused(packed, q, pos, cache, w, eps, neox, buf, loc, inplace, mode):
    kv_c, k_pe, q_pe = views(packed, q)
    q_cat = q_pe if inplace else torch.zeros(q.shape[0], H, KVL + ROT, device=dev, dtype=torch.bfloat16)
    q_out = q_pe if inplace else q_cat[..., KVL:]
    assert hcu_ops.kv_fused_supported(pos, q_pe, q_out, kv_c, w, k_pe, cache, buf, loc)
    _C.rope_qout_kv_store(pos, q_pe, q_out, kv_c, w, eps, k_pe, cache, neox, buf, loc, 1e-6, mode)
    return kv_c, k_pe, q_out, q_cat


def check(t, eps, neox, inplace, mode, pos32, short_loc, cache, w):
    packed, q, pos = make(t)
    if pos32:
        pos = pos.int()
    n_loc = t - 1 if short_loc and t > 1 else t
    loc = torch.randperm(SLOTS, device=dev)[:n_loc].contiguous()
    buf1 = torch.randint(0, 256, (SLOTS, 1, SLOT), device=dev, dtype=torch.uint8)
    buf2 = buf1.clone()
    p1, q1, p2, q2 = packed.clone(), q.clone(), packed.clone(), q.clone()
    kvn, k1, o1, c1 = run_ref(p1, q1, pos, cache, w, eps, neox, buf1, loc, inplace)
    kv2, k2, o2, c2 = run_fused(p2, q2, pos, cache, w, eps, neox, buf2, loc, inplace, mode)
    torch.cuda.synchronize()
    return [diff(kvn, kv2), diff(k1, k2), diff(o1, o2), diff(c1, c2), diff(buf1, buf2),
            diff(p1[:, :QL], p2[:, :QL]), diff(q1, q2)]


def graph_us(fn, reps=100):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (10 * reps)


def bench(cache, w):
    for t in (1, 4, 8, 16, 32, 48, 96, 192):
        packed, q, pos = make(t)
        kv_c, k_pe, q_pe = views(packed, q)
        loc = torch.randperm(SLOTS, device=dev)[:t].contiguous()
        buf = torch.zeros(SLOTS, 1, SLOT, device=dev, dtype=torch.uint8)
        q_cat = torch.empty(t, H, KVL + ROT, device=dev, dtype=torch.bfloat16)
        q_out = q_cat[..., KVL:]
        kvn = torch.empty(kv_c.shape, device=dev, dtype=torch.bfloat16)

        def unfused():
            op.rms_norm_opt(kvn, kv_c, w, 1e-6)
            hcu_ops.rope_qout(pos, q_pe, q_out, k_pe, ROT, cache, False)
            kvop.fused_quantize_and_store_mla_kv_cache(kvn.unsqueeze(1), k_pe, buf, loc, "fp8_e4m3", 1e-6)

        def fused():
            hcu_ops.rope_qout_kv_store(pos, q_pe, q_out, kv_c, w, 1e-6, k_pe, cache, False, buf, loc)

        print(f"T={t:4d}: unfused {graph_us(unfused):6.2f} us  fused {graph_us(fused):6.2f} us", flush=True)


ang = torch.arange(MAX_POS, device=dev)[:, None] * (10000.0 ** (-torch.arange(ROT // 2, device=dev) / (ROT // 2)))
cache = torch.cat([ang.cos(), ang.sin()], dim=-1).bfloat16().contiguous()
w = (torch.rand(KVL, device=dev) * 2).bfloat16()

if "--bench-only" not in sys.argv:
    TS = list(range(1, 17)) + [24, 32, 48, 64, 96, 128, 192, 256, 1000]
    for mode in (0, 1, 2):
        bad = 0
        for eps in (1e-6, 1e-5):
            for neox in (False, True):
                for inplace in (False, True):
                    for t in TS:
                        d = check(t, eps, neox, inplace, mode, pos32=t % 3 == 0, short_loc=t % 4 == 1,
                                  cache=cache, w=w)
                        if any(d):
                            bad += 1
                            if bad <= 5:
                                print(f"  mode {mode} eps={eps} neox={neox} inplace={inplace} T={t}: mismatches "
                                      f"kv_c/k_pe/q_out/q_cat/kv_buffer/q_lora/q = {d}")
        print(f"rope_qout_kv_store rsqrt_mode {mode}: {'BIT-EXACT' if not bad else f'{bad} cases differ'}",
              flush=True)
bench(cache, w)
