"""SGLANG_HCU_MLA_QCAT building blocks vs the path they replace, on GLM-5.3's MLA decode
shapes (DP8 attention: 64 local heads, q [T, H, 192 nope + 64 rope], kv_lora_rank 512):

  old: q_nope_out = bmm(q_nope, w_kc) (hcu skinny bmm / torch.bmm) -> rope in place on q_pe /
       k_pe -> q_all = concat_mla_absorb_q_general(q_nope_out, q_pe)   (hcu concat kernel)
  new: bmm writes q_cat[..., :512], rope_qout writes q_cat[..., 512:] (k_pe in place),
       q_all = adjacent_q_all(...) (a view of q_cat, no kernel)

Checks q_all / k_pe bit-exact, adjacent_q_all's accept / reject cases, then times both
paths under a CUDA graph."""
import os

os.environ.setdefault("SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q", "1")
import torch
import hcu_ops
from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general

torch.manual_seed(0)
dev = "cuda"
H, NOPE, ROT, KVL = 64, 192, 64, 512
cache = torch.randn(1 << 20, ROT, device=dev).to(torch.bfloat16)
w_kc = (torch.randn(H, KVL, NOPE, device=dev) * 0.05).to(torch.bfloat16).transpose(1, 2)


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


def bmm(x, out=None):
    y = hcu_ops.try_bmm_bf16(x, w_kc, out)
    if y is None:
        y = torch.bmm(x, w_kc) if out is None else torch.bmm(x, w_kc, out=out)
    return y


def old(pos, q, lat):
    qn = bmm(q[..., :NOPE].transpose(0, 1)).transpose(0, 1)
    q_pe, k_pe = q[..., NOPE:], lat[:, None, KVL:]
    hcu_ops.rope_inplace(pos, q_pe, k_pe, ROT, cache, False)
    return concat_mla_absorb_q_general(qn, q_pe)


def new(pos, q, lat):
    q_cat = q.new_empty(q.shape[0], H, KVL + ROT)
    bmm(q[..., :NOPE].transpose(0, 1), q_cat[..., :KVL].transpose(0, 1))
    hcu_ops.rope_qout(pos, q[..., NOPE:], q_cat[..., KVL:], lat[:, None, KVL:], ROT, cache, False)
    q_all = hcu_ops.adjacent_q_all(q_cat[..., :KVL], q_cat[..., KVL:])
    assert q_all is not None and q_all.data_ptr() == q_cat.data_ptr()
    return q_all


# adjacent_q_all must refuse anything that is not one row-contiguous [T, H, 576] buffer
a = torch.empty(6, H, KVL + ROT, device=dev, dtype=torch.bfloat16)
b = torch.empty(6, H, KVL + ROT, device=dev, dtype=torch.bfloat16)
c = torch.empty(6, H, KVL + ROT + 8, device=dev, dtype=torch.bfloat16)
assert hcu_ops.adjacent_q_all(a[..., :KVL], a[..., KVL:]) is not None
assert hcu_ops.adjacent_q_all(a[..., :KVL], b[..., KVL:]) is None              # other buffer
assert hcu_ops.adjacent_q_all(a[..., :KVL], a[..., KVL - 8:-8]) is None       # not adjacent
assert hcu_ops.adjacent_q_all(c[..., :KVL], c[..., KVL:KVL + ROT]) is None    # padded rows
assert hcu_ops.adjacent_q_all(a[:3, :, :KVL], a[3:, :, KVL:]) is None         # other tokens
assert hcu_ops.adjacent_q_all(a[::2, :, :KVL], a[::2, :, KVL:]) is None       # token stride
assert hcu_ops.adjacent_q_all(a[:1, :, :KVL], a[:1, :, KVL:]) is not None
print("adjacent_q_all accept / reject cases OK")

for neox in (False, True):
    pos = torch.randint(0, 1 << 20, (6,), device=dev)
    q = torch.randn(6, H, NOPE + ROT, device=dev).to(torch.bfloat16)
    k = torch.randn(6, KVL + ROT, device=dev).to(torch.bfloat16)
    qa, ka, qb, kb = q.clone(), k.clone(), q.clone(), k.clone()
    hcu_ops.rope_inplace(pos, qa[..., NOPE:], ka[:, None, KVL:], ROT, cache, neox)
    qo = torch.empty(6, H, ROT, device=dev, dtype=torch.bfloat16)
    hcu_ops.rope_qout(pos, qb[..., NOPE:], qo, kb[:, None, KVL:], ROT, cache, neox)
    print(f"rope_qout neox={neox} bit-exact={torch.equal(qa[..., NOPE:], qo) and torch.equal(ka, kb)} "
          f"query untouched={torch.equal(qb, q)}")

for T in (1, 2, 4, 6, 8, 12, 16, 18, 24, 32, 48, 96):
    pos = torch.randint(0, 1 << 20, (T,), device=dev)
    q = torch.randn(T, H, NOPE + ROT, device=dev).to(torch.bfloat16)
    lat = torch.randn(T, KVL + ROT, device=dev).to(torch.bfloat16)
    qa, la, qb, lb = q.clone(), lat.clone(), q.clone(), lat.clone()
    ref = old(pos, qa, la)
    out = new(pos, qb, lb)
    same = torch.equal(ref, out) and torch.equal(la, lb)
    hook = hcu_ops.try_bmm_bf16(q[..., :NOPE].transpose(0, 1), w_kc) is not None
    t0 = bench(lambda: old(pos, qa, la))
    t1 = bench(lambda: new(pos, qb, lb))
    print(f"T={T:3d} bit-exact={same} bmm={'hcu' if hook else 'torch'}  old {t0:6.1f} us  new {t1:6.1f} us  "
          f"saved {t0 - t1:5.1f} us")
