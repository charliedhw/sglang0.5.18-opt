"""hcu_ops.rope_inplace vs sgl_kernel.rotary_embedding (as sglang launches it on HCU)
for the GLM-5.3 MLA decode shapes: bit-exactness and timing."""
import torch
from sgl_kernel import rotary_embedding
import hcu_ops

torch.manual_seed(0)
dev = "cuda"
H, NOPE, ROT, KVL = 64, 192, 64, 512
cache = torch.randn(1 << 20, ROT, device=dev).to(torch.bfloat16)


def split_launch(pos, q, k):
    # sglang _launch_rotary_embedding: 8 heads per launch, key with the first one
    rotary_embedding(pos, q[:, :8], k, ROT, cache, False)
    for s in range(8, H, 8):
        rotary_embedding(pos, q[:, s:s + 8], None, ROT, cache, False)


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


for T in [1, 6, 12, 24, 48, 192, 2048]:
    pos = torch.randint(0, 1 << 20, (T,), device=dev)
    qf = torch.randn(T, H, NOPE + ROT, device=dev).to(torch.bfloat16)
    lat = torch.randn(T, KVL + ROT, device=dev).to(torch.bfloat16)
    qa, qb = qf.clone(), qf.clone()
    la, lb = lat.clone(), lat.clone()
    # q_pe [T, H, 64] strided view of q; k_pe [T, 1, 64] view of the latent cache
    split_launch(pos, qa[..., NOPE:], la[:, None, KVL:])
    hcu_ops.rope_inplace(pos, qb[..., NOPE:], lb[:, None, KVL:], ROT, cache, False)
    same = torch.equal(qa, qb) and torch.equal(la, lb)
    changed = not torch.equal(qa, qf)
    t0 = bench(lambda: split_launch(pos, qa[..., NOPE:], la[:, None, KVL:]))
    t1 = bench(lambda: hcu_ops.rope_inplace(pos, qb[..., NOPE:], lb[:, None, KVL:], ROT, cache, False))
    print(f"T={T:5d} bit-exact={same} rotated={changed}  8-launch {t0:6.1f} us  fused {t1:6.1f} us")

# neox style + 2-D contiguous q / k, int32 positions
for neox in (True, False):
    T = 6
    pos = torch.randint(0, 1 << 20, (T,), device=dev, dtype=torch.int32)
    q = torch.randn(T, 4 * ROT, device=dev).to(torch.bfloat16)
    k = torch.randn(T, ROT, device=dev).to(torch.bfloat16)
    qa, ka, qb, kb = q.clone(), k.clone(), q.clone(), k.clone()
    rotary_embedding(pos.long(), qa, ka, ROT, cache, neox)
    hcu_ops.rope_inplace(pos, qb, kb, ROT, cache, neox)
    print(f"2-D neox={neox} bit-exact={torch.equal(qa, qb) and torch.equal(ka, kb)}")
