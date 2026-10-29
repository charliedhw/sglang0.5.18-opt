import time
import torch
from lightop.fuse_topk_transform import fast_topk_transform_sparse_mask_fused as lt_topk  # noqa: F401

dev = "cuda"


def make(B, L, dist, W=None, ragged=False):
    W = W or L
    if dist == "pos":
        x = torch.rand(B, W, device=dev) * 10
    elif dist == "sparse30":
        x = torch.rand(B, W, device=dev) * 10
        x[torch.rand(B, W, device=dev) > 0.3] = 0
    elif dist == "sparse2":  # fast-path regime: positives < 2048 < positives + zeros
        x = torch.rand(B, W, device=dev) * 10
        x[torch.rand(B, W, device=dev) > 1500.0 / W] = 0
    elif dist == "relu_sum":
        n = torch.randn(B, W, 8, device=dev)
        x = (torch.relu(n) * (torch.rand(8, device=dev) - 0.3)).sum(-1)
    elif dist == "mixed":
        x = torch.randn(B, W, device=dev)
    elif dist == "ties":  # heavy exact ties
        x = torch.randint(0, 50, (B, W), device=dev).float() / 4
    elif dist == "zeros":
        x = torch.zeros(B, W, device=dev)
    elif dist == "neg":
        x = -torch.rand(B, W, device=dev)
        x[torch.rand(B, W, device=dev) > 0.5] = 0
    elif dist == "narrow":  # one 12-bit key bin: candidate lists overflow
        x = 1.0 + torch.rand(B, W, device=dev) * 1e-3
    elif dist == "const":  # all ties
        x = torch.full((B, W), 0.5, device=dev)
    x = x.float().contiguous()
    nz = (x != 0).view(B, -1)
    pad = (-W) % 16
    if pad:
        nz = torch.cat([nz, torch.zeros(B, pad, dtype=torch.bool, device=dev)], 1)
    w16 = torch.tensor([1 << i for i in range(16)], dtype=torch.int32, device=dev)
    bits = (nz.view(B, -1, 16).to(torch.int32) * w16).sum(-1, dtype=torch.int32)
    mask = torch.where(bits > 32767, bits - 65536, bits).to(torch.int16).contiguous()
    # unwritten storage is garbage in production: poison masked-out positions
    x = torch.where(x == 0, torch.full_like(x, float("nan")), x)
    if ragged:
        lengths = (L - torch.arange(B, device=dev) * 37 - 3).clamp(min=1).to(torch.int32)
    else:
        lengths = torch.full((B,), L, dtype=torch.int32, device=dev)
    pt = (torch.arange(W, dtype=torch.int32, device=dev).unsqueeze(0).repeat(B, 1)
          + 7 + 1000000 * torch.arange(B, dtype=torch.int32, device=dev).unsqueeze(1)).contiguous()
    cu = torch.arange(B + 1, dtype=torch.int32, device=dev)
    return x, mask, lengths, pt, cu


def bench(fn, it=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
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


def same_values(x, lengths, ref, out):
    """Both outputs are valid top-k: equal sorted value multisets, unique, -1 pads equal."""
    B = x.shape[0]
    xv = torch.nan_to_num(x, nan=0.0)
    rowoff = 7 + 1000000 * torch.arange(B, dtype=torch.int32, device=dev).unsqueeze(1)

    def vals(o):
        idx = (o - rowoff).long()
        valid = o >= 0
        v = torch.gather(xv, 1, idx.clamp(min=0))
        return torch.where(valid, v, torch.full_like(v, -float("inf"))).sort(-1).values, idx, valid
    vr, _, _ = vals(ref)
    vo, io, valid = vals(out)
    if not bool((vr == vo).all()):
        return False
    for b in range(B):
        sel = io[b][valid[b]]
        if torch.unique(sel).numel() != sel.numel() or (sel >= lengths[b]).any():
            return False
    return True
