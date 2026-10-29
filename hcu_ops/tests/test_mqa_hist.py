"""hcu_ops mqa_logits_hist vs lightop page_mqa_logits_sparse_mask[_grouped] (bit-exact mask /
logits at set bits), its fused TopK pass-1 workspace vs a torch reference, and
sparse_mask_topk_transform_split_pre (on our logits) vs sparse_mask_topk_transform_split (on
lightop's).  Timings are CUDA-graph replays (us per call).
usage: python3 test_mqa_hist.py [ctx=65536] [reqs=1,2,4] [pctas=0,64,128] [G=6] [rpc=1,2,3] [cap_pages=3200]
"""
import os
import sys
import time

import torch
from lightop.gemmopt import page_mqa_logits_sparse_mask, page_mqa_logits_sparse_mask_grouped
from hcu_ops import _C

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
REQS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4").split(",")]
PCTAS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,64,128").split(",")]
G = int(sys.argv[4]) if len(sys.argv) > 4 else 6
RPCS = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else ("1" if G == 1 else "1,2,3")).split(",")]
CAP = int(sys.argv[6]) if len(sys.argv) > 6 else 3200
SPLITS = 8
dev = "cuda"
torch.manual_seed(0)

MAXP = max((CTX + 6 + 63) // 64 + 8, CAP)  # block-table width = page-table capacity (CUDA graphs)
WIDTH = MAXP * 64
USED = (CTX + 6 + 63) // 64 + 8
NBLK = 4 * USED + 16


def make_kv(mode):
    v = torch.randn(NBLK, 64, 128, device=dev) * 0.5
    if mode == "sparse":  # ~1500 positive tokens per row, the rest exact zeros (TopK fast path)
        pos = torch.rand(NBLK, 64, 1, device=dev) < 1500.0 / CTX
        v = v.abs() * torch.where(pos, 1.0, -1.0)
    fp8 = v.to(torch.float8_e4m3fn).view(torch.uint8).view(NBLK, 64 * 128)
    sc = (torch.rand(NBLK, 64, device=dev) * 0.02 + 0.001).float().view(torch.uint8).view(NBLK, 256)
    return torch.cat([fp8, sc], 1).contiguous().view(NBLK, 64, 1, 132)


def make_inputs(nreq, mode):
    B = nreq * G
    q = torch.randn(B, 1, 32, 128, device=dev)
    w = torch.randn(B, 32, device=dev) * 0.1
    if mode == "sparse":
        q, w = q.abs(), w.abs()
    elif mode == "neg":
        w = -w.abs()
    q = q.to(torch.float8_e4m3fn)
    lens = torch.empty(B, dtype=torch.int32, device=dev)
    bt = torch.zeros(B, MAXP, dtype=torch.int32, device=dev)
    for r in range(nreq):
        L = CTX - 977 * r
        lens[r * G:(r + 1) * G] = torch.arange(L + 1, L + 1 + G, dtype=torch.int32, device=dev)
        bt[r * G:(r + 1) * G, :USED] = torch.randperm(NBLK, device=dev)[:USED].int()
    if G == 1 and B > 1:
        lens[-1] = 1500  # a row below the TopK split threshold
    pt = (torch.arange(WIDTH, dtype=torch.int32, device=dev).unsqueeze(0).repeat(B, 1)
          + 7 + 1000000 * torch.arange(B, dtype=torch.int32, device=dev).unsqueeze(1)).contiguous()
    cu = torch.arange(B + 1, dtype=torch.int32, device=dev)
    return q, w.float().contiguous(), lens, bt, pt, cu


# LT_PCTAS >= 0: lightop baseline with persistent_ctas (as SGLANG_HCU_MQA_PCTAS does in sglang);
# set LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA=3 as the start script does for a fair comparison
LT_PCTAS = int(os.environ.get("LT_PCTAS", "-1"))
LT_KW = {"persistent_ctas": LT_PCTAS} if LT_PCTAS >= 0 else {}


def lt_logits(kv, q, w, lens, bt):
    if G == 1:
        return page_mqa_logits_sparse_mask(q, kv, w, lens, bt, None, WIDTH, clean_logits=True, num_warps=4,
                                           **LT_KW)
    return page_mqa_logits_sparse_mask_grouped(q, kv, w, lens, bt, None, WIDTH, clean_logits=True,
                                               num_warps=4, group_size=G, **LT_KW)


def bench(fn, it=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(it):
                fn()
        g.replay()
        torch.cuda.synchronize()
        t = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / it * 1e6
    except Exception:  # noqa: BLE001
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
        return -(time.perf_counter() - t) / it * 1e6  # negative: eager timing


bits16 = torch.arange(16, device=dev, dtype=torch.int32)


def mask_bits(m, width):
    return ((m.int().unsqueeze(-1) >> bits16) & 1).flatten(1)[:, :width].bool()


def ref_hist(lg, mm, lens):
    """split pass 1 for rows with min(len, width) > 2048: ghist1 of key >> 20 over set bits below
    the length, zeros (len - set) added to bin 0x800, cnt = (positives, zeros)."""
    B = lg.shape[0]
    h = lens.long().clamp(max=WIDTH)
    act = h > 2048
    sel = mm & (torch.arange(WIDTH, device=dev) < h[:, None]) & act[:, None]
    b = lg.contiguous().view(torch.int32).long() & 0xffffffff
    key = torch.where(b >= 0x80000000, (~b) & 0xffffffff, b | 0x80000000)
    g1 = torch.zeros(B, 4096, dtype=torch.long, device=dev)
    g1.scatter_add_(1, torch.where(sel, key >> 20, 0), sel.long())
    nz = sel.sum(1)
    zeros = torch.where(act, h - nz, 0)
    g1[:, 0x800] += zeros
    pos = (sel & (key > 0x80000000)).sum(1)
    return g1, torch.stack([pos, zeros], 1)


def sorted_rows(o):
    return o.sort(-1).values


def topk_values(o, lg, mm, lens):
    """sorted selected values (unset bits count as 0, pads as -inf) and whether the indices are
    unique and below the row length"""
    B = o.shape[0]
    idx = (o - 7 - 1000000 * torch.arange(B, dtype=torch.int32, device=dev).unsqueeze(1)).long()
    valid = o >= 0
    ic = idx.clamp(0, WIDTH - 1)
    v = torch.where(mm.gather(1, ic), lg.gather(1, ic), torch.zeros((), device=dev))
    v = torch.where(valid, v, torch.full_like(v, -float("inf"))).sort(-1).values
    ok = True
    for b in range(B):
        sel = idx[b][valid[b]]
        ok &= torch.unique(sel).numel() == sel.numel() and not bool((sel >= lens[b]).any())
    return v, ok


allok = True
kvs = {m: make_kv("sparse" if m == "sparse" else "rand") for m in ("rand", "sparse", "neg")}
for mode in ("rand", "sparse", "neg"):
    kv = kvs[mode]
    for nreq in REQS:
        q, w, lens, bt, pt, cu = make_inputs(nreq, mode)
        B = q.shape[0]
        l0, m0 = lt_logits(kv, q, w, lens, bt)
        mm0 = mask_bits(m0, WIDTH)
        li0 = l0.contiguous().view(torch.int32)[mm0]
        t_lt = bench(lambda: lt_logits(kv, q, w, lens, bt))
        topk0 = _C.sparse_mask_topk_transform_split(l0, m0, lens, pt, cu, 2048, SPLITS)
        t_lt_topk = bench(lambda: _C.sparse_mask_topk_transform_split(l0, m0, lens, pt, cu, 2048, SPLITS))
        print(f"[{mode}] G={G} ctx={CTX} reqs={nreq} rows={B} width={WIDTH} nnz={int(mm0.sum())}: "
              f"lightop logits {t_lt:6.1f} us, split{SPLITS} topk {t_lt_topk:6.1f} us", flush=True)
        for rpc in RPCS:
            for p in PCTAS:
                l1, m1, z1 = _C.mqa_logits_hist(q, kv, w, lens, bt, WIDTH, G, p, rpc, None)
                ok_m = torch.equal(m1, m0)
                ok_l = ok_m and torch.equal(l1.contiguous().view(torch.int32)[mask_bits(m1, WIDTH)], li0)
                l2, m2, z2 = _C.mqa_logits_hist(q, kv, w, lens, bt, WIDTH, G, p, rpc, lens)
                ok_m2 = torch.equal(m2, m0)
                ok_l2 = ok_m2 and torch.equal(l2.contiguous().view(torch.int32)[mask_bits(m2, WIDTH)], li0)
                g1r, cr = ref_hist(l0, mm0, lens)
                g1 = z2[:B * 4096].view(B, 4096).long()
                g2 = z2[B * 4096:2 * B * 4096]
                cnt = z2[2 * B * 4096:].view(B, 2).long()
                ok_h = torch.equal(g1, g1r) and torch.equal(cnt, cr) and not bool(g2.any())
                topk2 = _C.sparse_mask_topk_transform_split_pre(l2, m2, lens, pt, cu, 2048, SPLITS, z2)
                ok_t = torch.equal(sorted_rows(topk2), sorted_rows(topk0))
                if not ok_t:  # ties (e.g. zeros) may pick other indices: compare values
                    v0, _ = topk_values(topk0, l0, mm0, lens)
                    v2, valid2 = topk_values(topk2, l2, mask_bits(m2, WIDTH), lens)
                    ok_t = "values" if valid2 and torch.equal(v0, v2) else False
                t1 = bench(lambda: _C.mqa_logits_hist(q, kv, w, lens, bt, WIDTH, G, p, rpc, None))
                t2 = bench(lambda: _C.mqa_logits_hist(q, kv, w, lens, bt, WIDTH, G, p, rpc, lens))

                def pipe():
                    a, b_, z = _C.mqa_logits_hist(q, kv, w, lens, bt, WIDTH, G, p, rpc, lens)
                    return _C.sparse_mask_topk_transform_split_pre(a, b_, lens, pt, cu, 2048, SPLITS, z)
                t3 = bench(pipe)
                ok = ok_m and ok_l and ok_m2 and ok_l2 and ok_h and bool(ok_t)
                allok &= ok
                print(f"  rpc={rpc} pctas={p:4d}: logits {t1:6.1f} us, +hist {t2:6.1f} us, hist+topk_pre "
                      f"{t3:6.1f} us (lightop+split {t_lt + t_lt_topk:6.1f})  mask_eq={ok_m}/{ok_m2} "
                      f"logits_bitexact={ok_l}/{ok_l2} hist_eq={ok_h} topk_eq={ok_t}", flush=True)
        del l0, m0, mm0, li0
print("ALL_OK", allok)
