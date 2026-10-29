"""Can lightop's fused norm / act + fp8-quant ops replace GLM-5.3's unfused decode chains
bit for bit (GLM-5.3 compressed-tensors FP8_DYNAMIC: per-token fp8 activations)?

  A  op.rms_norm_opt on a strided row view (q_a / kv_a slices of packed qkv) vs on
     .contiguous() (RMSNorm.forward_hip copies first)
  B  rms_norm_per_token_fp8_quant(x, w, residual=r, update_input=True) vs
     op.fused_add_rms_norm_opt(x, r, w) + per_token_quant_fp8(x)      (input / post-attn norm)
  C  rms_norm_per_token_fp8_quant(x, w, update_input=?) vs rms_norm_opt + per_token_quant_fp8
     (q_a norm -> q_b_proj)
  D  fuse_silu_mul_fp8_quant(gu) vs fuse_silu_and_mul + per_token_quant_fp8 (shared expert)
  E  per_token_quant_fp8's formula vs torch references
Timings: CUDA graph, 200 calls."""
import torch
from lightop import op, fuse_silu_and_mul
from lightop.quant import per_token_quant_fp8
from lightop.rmsnorm import rms_norm_per_token_fp8_quant
from lightop.fuse_silu_mul_quant import fuse_silu_mul_fp8_quant

torch.manual_seed(0)
dev = "cuda"
EPS = 1e-5
TS = (1, 6, 12, 24, 48, 96, 192)


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


def act(t, n, scale=1.0):
    """bf16 activations with per-row scale spread and a few outliers."""
    x = torch.randn(t, n, device=dev) * scale * torch.logspace(-1, 1, t, device=dev)[:, None]
    x[:, torch.randint(0, n, (4,))] *= 20
    return x.to(torch.bfloat16)


def u8(q):
    return q.view(torch.uint8)


def diff(a, b):
    if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        a, b = a.float(), b.float()
    d = (a.float() - b.float()).abs()
    return f"maxdiff {d.max().item():.3g} n_ne {(a != b).sum().item()}"


print("== A: rms_norm_opt on strided rows ==")
for n, off, width in ((2048, 0, 2624), (512, 2048, 2624)):
    for t in TS:
        packed = act(t, width)
        w = (torch.rand(n, device=dev) + 0.5).to(torch.bfloat16)
        view = packed[:, off:off + n]
        ref = torch.empty(t, n, device=dev, dtype=torch.bfloat16)
        op.rms_norm_opt(ref, view.contiguous(), w, EPS)
        out = torch.empty_like(ref)
        try:
            op.rms_norm_opt(out, view, w, EPS)
            res = f"bit-exact={torch.equal(out, ref)} {diff(out, ref)}"
        except Exception as e:  # noqa: BLE001
            res = f"error {str(e).splitlines()[0][:80]}"
        print(f"n={n} T={t:3d} {res}")

print("== B: fused add rms + quant (hidden 6144) ==")
for t in TS:
    x0, r0 = act(t, 6144), act(t, 6144, 4.0)
    w = (torch.rand(6144, device=dev) + 0.5).to(torch.bfloat16)
    xa, ra = x0.clone(), r0.clone()
    op.fused_add_rms_norm_opt(xa, ra, w, EPS)
    qa, sa = per_token_quant_fp8(xa)
    for upd in (True, False):
        xb, rb = x0.clone(), r0.clone()
        qb, sb = rms_norm_per_token_fp8_quant(xb, w, EPS, 0, residual=rb, update_input=upd)
        print(f"T={t:3d} update_input={upd}: resid={torch.equal(ra, rb)} "
              f"x={'normed ' + str(torch.equal(xa, xb)) if upd else 'kept ' + str(torch.equal(xb, x0))} "
              f"q={torch.equal(u8(qa), u8(qb))} ({diff(qa, qb)}) s={torch.equal(sa, sb)} ({diff(sa, sb)})")
    xc, rc = x0.clone(), r0.clone()
    xd, rd = x0.clone(), r0.clone()
    t0 = bench(lambda: (op.fused_add_rms_norm_opt(xc, rc, w, EPS), per_token_quant_fp8(xc)))
    t1 = bench(lambda: rms_norm_per_token_fp8_quant(xd, w, EPS, 0, residual=rd, update_input=True))
    print(f"T={t:3d} time old {t0:6.1f} us  fused {t1:6.1f} us")

print("== C: rms + quant without residual (q_a 2048) ==")
for t in TS:
    x0 = act(t, 2048)
    w = (torch.rand(2048, device=dev) + 0.5).to(torch.bfloat16)
    ya = torch.empty_like(x0)
    op.rms_norm_opt(ya, x0, w, EPS)
    qa, sa = per_token_quant_fp8(ya)
    xb = x0.clone()
    qb, sb = rms_norm_per_token_fp8_quant(xb, w, EPS, 0, update_input=True)
    print(f"T={t:3d} x->normed={torch.equal(xb, ya)} q={torch.equal(u8(qa), u8(qb))} ({diff(qa, qb)}) "
          f"s={torch.equal(sa, sb)} ({diff(sa, sb)})")

print("== D: silu_mul + quant (shared expert 2 x 2048) ==")
for t in TS:
    gu = act(t, 4096, 3.0)
    y = torch.empty(t, 2048, device=dev, dtype=torch.bfloat16)
    fuse_silu_and_mul(gu, y)
    qa, sa = per_token_quant_fp8(y)
    qb, sb = fuse_silu_mul_fp8_quant(gu, 0)
    sb = sb.view(sa.shape) if sb.numel() == sa.numel() else sb
    print(f"T={t:3d} q={torch.equal(u8(qa), u8(qb))} ({diff(qa, qb)}) s={torch.equal(sa, sb)} ({diff(sa, sb)}) "
          f"q shape {tuple(qb.shape)} s shape {tuple(sb.shape)}")
    t0 = bench(lambda: (fuse_silu_and_mul(gu, y), per_token_quant_fp8(y)))
    t1 = bench(lambda: fuse_silu_mul_fp8_quant(gu, 0))
    print(f"T={t:3d} time old {t0:6.1f} us  fused {t1:6.1f} us")

print("== E: per_token_quant_fp8 formula ==")
x = act(192, 6144)
q, s = per_token_quant_fp8(x)
amax = x.float().abs().amax(-1, keepdim=True)
for name, sc in (("amax/448", amax / 448.0),
                 ("max(amax/448, 1/(448*512))", torch.clamp(amax / 448.0, min=1.0 / (448 * 512))),
                 ("max(amax,1e-10)/448", torch.clamp(amax, min=1e-10) / 448.0)):
    q_mul = (x.float() * (1.0 / sc)).clamp(-448, 448).to(torch.float8_e4m3fn)
    q_div = (x.float() / sc).clamp(-448, 448).to(torch.float8_e4m3fn)
    print(f"{name:28s} scale={torch.equal(s, sc)} q(x*(1/s))={torch.equal(u8(q), u8(q_mul))} "
          f"q(x/s)={torch.equal(u8(q), u8(q_div))}")
