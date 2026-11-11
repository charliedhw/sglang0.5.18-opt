"""Is the MoE grouped fp8 GEMM already at a hardware roof?

It is 13.7% of the 0%-cache prefill profile (1824 calls, 2647 us each) and is the largest
remaining item after sparse_attn_fwd that is not pure communication.  Before considering any
kernel work on it, establish whether it is memory-bound and already near the measured copy roof
(1361 GB/s on this card), or compute-bound and near fp8 peak.  If either, there is nothing to win.

Measures the same deepgemm entry point sglang calls, at the model's own expert shapes and at the
per-rank token count a CP8 prefill chunk produces.
"""
import json
import time

import torch

DEV = "cuda:0"
CFG = "/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015/config.json"


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    c = json.load(open(CFG))
    t = c.get("text_config", c)
    H = t["hidden_size"]
    I = t.get("moe_intermediate_size") or t["intermediate_size"]
    E = t.get("n_routed_experts")
    K = t.get("num_experts_per_tok")
    L = t.get("num_hidden_layers")
    print(f"GLM-5.3: hidden={H} moe_inter={I} experts={E} topk_experts={K} layers={L}")

    EP = 8
    e_local = E // EP
    # CP8 prefill: chunked-prefill-size 32768 split over 8 ranks
    for tokens in (4096,):
        rows = tokens * K // EP          # expert-token pairs landing on this rank
        print(f"\nper-rank tokens={tokens}  expert-rows={rows}  local experts={e_local}")

        # gate_up: [rows, H] x [e_local, H, 2I]   down: [rows, I] x [e_local, I, H]
        for name, (n, k, outn) in (
            ("gate_up", (rows, H, 2 * I)),
            ("down", (rows, I, H)),
        ):
            a = torch.randn(n, k, device=DEV).to(torch.float8_e4m3fn)
            b = torch.randn(e_local, k, outn, device=DEV).to(torch.float8_e4m3fn)
            w_bytes = e_local * k * outn          # fp8 weights, 1 B each
            a_bytes = n * k
            o_bytes = n * outn * 2                # bf16 out
            flops = 2 * n * k * outn
            try:
                import deep_gemm
                fn_name = [x for x in dir(deep_gemm)
                           if "grouped" in x and "contiguous" in x]
                print(f"  deepgemm candidates: {fn_name[:4]}")
            except Exception as e:
                print(f"  deep_gemm import failed: {e}")

            # torch reference at the same shape, as a floor
            a16 = a.to(torch.bfloat16)
            b16 = b[0].to(torch.bfloat16)
            s = timed(lambda: torch.matmul(a16, b16))
            print(f"  {name:8s} [{n}x{k}]x[{k}x{outn}]  weights={w_bytes/1e6:7.1f} MB"
                  f"  flops={flops/1e9:8.1f} GF")
            print(f"           torch bf16 single-expert: {s*1e6:9.1f} us"
                  f"   {flops/s/1e12:6.1f} TFLOPS   {(a_bytes+w_bytes/e_local+o_bytes)/s/1e9:6.0f} GB/s")

    print("\nObserved in the prefill profile: 2647 us per call, 1824 calls, 13.7% of GPU time.")
    print("Copy roof measured on this card: 1361 GB/s.")


main()