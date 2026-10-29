"""FlashMLA sparse FP8 decode (GLM-5.3 flashmla_kv path): time vs FLASH_MLA_FP8_FIXED_SPLITS.

Shapes follow dsa_backend._forward_flashmla_kv: q (rows, 1, 64, 576) bf16,
kv (blocks, 64, 1, 656) fp8 V32 layout, indices (rows, 1, 2048).
"""
import os
import sys

import torch
import flash_mla

H, D, DV, TOPK, PAGE = 64, 576, 512, 2048, 64
torch.manual_seed(0)
dev = "cuda"


def make_kv(num_tokens):
    nb = (num_tokens + PAGE - 1) // PAGE
    kv = torch.empty(nb, PAGE, 1, 656, dtype=torch.uint8, device=dev)
    nope = (torch.randn(nb, PAGE, 512, device=dev) * 0.5).to(torch.float8_e4m3fn)
    kv[..., 0, :512] = nope.view(torch.uint8)
    scale = torch.full((nb, PAGE, 4), 0.05, device=dev) * (1 + torch.rand(nb, PAGE, 4, device=dev))
    kv[..., 0, 512:528] = scale.view(torch.uint8)
    rope = torch.randn(nb, PAGE, 64, device=dev, dtype=torch.bfloat16)
    kv[..., 0, 528:656] = rope.view(torch.uint8)
    return kv.view(torch.float8_e4m3fn)


def run(q, kv, idx, seqlens, sink):
    meta, _ = flash_mla.get_mla_metadata()
    o, _ = flash_mla.flash_mla_with_kvcache(
        q=q, k_cache=kv, block_table=torch.empty((q.shape[0], 0), dtype=torch.int32, device=dev),
        cache_seqlens=seqlens, head_dim_v=DV, tile_scheduler_metadata=meta,
        softmax_scale=D ** -0.5, is_fp8_kvcache=True, indices=idx, attn_sink=sink)
    return o


def bench(fn, iters=20, reps=5):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / iters)
    return best


def main():
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    rows_list = [6, 12, 24, 48, 96, 192]
    splits_list = ["default", "1", "2", "4", "8", "16", "32"]
    kv = make_kv(max(rows_list) // 6 * ctx + PAGE)
    sink = torch.randn(H, device=dev, dtype=torch.float32)
    print(f"ctx={ctx}  us/call   (rows x FLASH_MLA_FP8_FIXED_SPLITS)")
    print("rows " + " ".join(f"{s:>8}" for s in splits_list) + "   maxdiff")
    for rows in rows_list:
        q = torch.randn(rows, 1, H, D, device=dev, dtype=torch.bfloat16)
        # each request (6 rows) owns its own ctx-token region
        base = (torch.arange(rows, device=dev) // 6) * ctx
        idx = torch.stack([torch.randperm(ctx, device=dev)[:TOPK] for _ in range(rows)]).int()
        idx = (idx + base[:, None].int()).unsqueeze(1).contiguous()
        seqlens = torch.full((rows,), ctx, dtype=torch.int32, device=dev)
        times, ref, diff = [], None, 0.0
        for s in splits_list:
            if s == "default":
                os.environ.pop("FLASH_MLA_FP8_FIXED_SPLITS", None)
            else:
                os.environ["FLASH_MLA_FP8_FIXED_SPLITS"] = s
            fn = lambda: run(q, kv, idx, seqlens, sink)
            o = fn().float()
            if ref is None:
                ref = o
            else:
                diff = max(diff, (o - ref).abs().max().item())
            times.append(bench(fn))
        os.environ.pop("FLASH_MLA_FP8_FIXED_SPLITS", None)
        print(f"{rows:4d} " + " ".join(f"{t:8.1f}" for t in times) + f"   {diff:.2e}")


if __name__ == "__main__":
    main()
