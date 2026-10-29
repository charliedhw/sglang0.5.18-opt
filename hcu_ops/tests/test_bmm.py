"""hcu_ops.bmm_skinny vs torch.bmm on the GLM-5.3 MLA absorb shapes (H local heads: 64 with DP8
attention, 8 with TP8).

kc: q_nope [T, H, 192] (slice of q [T, H, 256]) x w_kc [H, 192, 512] (= [H, 512, 192] row-major)
    -> [H, T, 512]
vc: attn_output [T, H, 512] x w_vc [H, 512, 256] (= [H, 256, 512] row-major)
    -> view of [T, H * 256]
Accuracy against an fp32 reference (torch.bmm's own error as the yardstick), then timing of
torch.bmm and each (cw, nwv, lds) config under a CUDA graph.
"""
import torch
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


def case_kc(T, H):
    q = torch.randn(T, H, 256, device=dev).to(torch.bfloat16)
    w = (torch.randn(H, 512, 192, device=dev) * 0.05).to(torch.bfloat16).transpose(1, 2)
    x = q[..., :192].transpose(0, 1)
    return x, w, lambda: torch.empty(H, T, 512, device=dev, dtype=torch.bfloat16)


def case_vc(T, H):
    a = torch.randn(T, H, 512, device=dev).to(torch.bfloat16)
    w = (torch.randn(H, 256, 512, device=dev) * 0.05).to(torch.bfloat16).transpose(1, 2)
    x = a.transpose(0, 1)
    return x, w, lambda: torch.empty(T, H * 256, device=dev, dtype=torch.bfloat16).view(T, H, 256).transpose(0, 1)


CFGS = {"kc": [(cw, nwv, lds) for cw in (384, 128) for nwv in (1, 2, 4) for lds in (True, False)],
        "vc": [(cw, nwv, lds) for cw in (256, 512, 128) for nwv in (1, 2, 4) for lds in (True, False)]}

for H, name, mk in [(h, n, f) for h in (64, 8) for n, f in (("kc", case_kc), ("vc", case_vc))]:
    print(f"== {name} H={H} ==")
    for T in (1, 2, 4, 6, 8, 12, 16, 18, 24, 32):
        x, w, new_out = mk(T, H)
        ref32 = torch.bmm(x.float(), w.float())
        tb = torch.bmm(x, w)
        err_t = (tb.float() - ref32).abs().max().item()
        res = []
        for cw, nwv, lds in CFGS[name]:
            out = new_out()
            hcu_ops.bmm_skinny(x, w.transpose(1, 2), out, cw, nwv, lds)
            err = (out.float() - ref32).abs().max().item()
            ok = err <= 1.25 * err_t + 1e-5
            t = bench(lambda: hcu_ops.bmm_skinny(x, w.transpose(1, 2), out, cw, nwv, lds))
            res.append((t, cw, nwv, lds, err, ok))
        out = new_out()
        hook = hcu_ops.try_bmm_bf16(x, w, out)  # None: the hook keeps torch.bmm for this M
        hook_ok = hook is None or (hook.data_ptr() == out.data_ptr()
                                   and (out.float() - ref32).abs().max().item() <= 1.25 * err_t + 1e-5)
        th = bench(lambda: hcu_ops.try_bmm_bf16(x, w, out) is None and torch.bmm(x, w, out=out))
        o2 = new_out()
        tt = bench(lambda: torch.bmm(x, w, out=o2))
        res.sort()
        best = res[0]
        bad = [r for r in res if not r[5]]
        mb = w.numel() * 2 / 1e6
        print(f"T={T:3d} torch.bmm {tt:6.1f} us ({mb / tt:.2f} TB/s, err {err_t:.3g})  best "
              f"cw={best[1]} nwv={best[2]} lds={best[3]} {best[0]:6.1f} us ({mb / best[0]:.2f} TB/s, "
              f"err {best[4]:.3g})  hook {'torch' if hook is None else 'hcu'} {th:6.1f} us ok={hook_ok}  "
              f"{'ALL OK' if not bad else f'BAD {bad[:2]}'}")
        print("        " + "  ".join(f"{cw}/{nwv}/{'L' if lds else 'D'}:{t:.1f}" for t, cw, nwv, lds, _, _ in
                                    sorted(res, key=lambda r: (r[1], r[2], r[3]))))
