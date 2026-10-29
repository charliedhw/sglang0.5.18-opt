"""Does a HIP graph on this DTK run kernels captured on two streams concurrently?
(SGLANG_ROCM_USE_MULTI_STREAM=1 gives sglang an alt_stream on HCU: q_a / kv_a norm and the
shared experts of the DeepEP MoE then run on it inside the CUDA graph.)

Captures N pairs of independent ops (a: 2048x6144 skinny-ish bf16 matmul, b: small
elementwise) either on one stream or forked onto an alt stream, and compares replay time."""
import torch

dev = "cuda"
x = torch.randn(48, 6144, device=dev, dtype=torch.bfloat16)
w1 = torch.randn(4096, 6144, device=dev, dtype=torch.bfloat16)
w2 = torch.randn(2048, 6144, device=dev, dtype=torch.bfloat16)
y = torch.randn(48, 2048, device=dev, dtype=torch.bfloat16)
alt = torch.cuda.Stream()


def a():
    return x @ w1.t()


def b():
    return x @ w2.t()


def c():
    return (y * 2 + 1).sum()


def run(kind, it=50):
    def body():
        if kind == "serial":
            a(); b(); c()
        else:
            cur = torch.cuda.current_stream()
            alt.wait_stream(cur)
            with torch.cuda.stream(alt):
                b(); c()
            a()
            cur.wait_stream(alt)

    for _ in range(3):
        body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it):
            body()
    for _ in range(3):
        g.replay()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record(); g.replay(); en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3 / it


def single(fn, it=50):
    g = torch.cuda.CUDAGraph()
    fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(it):
            fn()
    g.replay(); torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record(); g.replay(); en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3 / it


print(f"a alone {single(a):.1f} us, b alone {single(b):.1f} us, c alone {single(c):.1f} us")
for k in ("serial", "fork", "serial", "fork"):
    print(f"{k:6s} {run(k):.1f} us per iteration")
