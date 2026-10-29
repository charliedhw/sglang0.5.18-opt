"""Unit-test helpers for custom DCU kernels (import from a test_*.py next to the extension).

    from ut_bench import graph_us, same, ndiff, report
    ok = same(ref_out, new_out)                    # bit-exact (fp8 compared as bytes, NaN == NaN)
    print(f"ref {graph_us(lambda: ref(x)):.2f} us  new {graph_us(lambda: new(x)):.2f} us")

graph_us: time per call inside a CUDA/HIP graph (reps calls captured back to back, replayed 10x),
i.e. what the op costs in a graph-replayed decode step (launch overhead included, no host gaps).
Warm up on a side stream first (lazy code-object load, allocator).  Run on an idle card only.

Test-case checklist for bit-exact claims: M = 1..16, 24, 32, 48, 64, 96, 128, 192, 288, 1000+;
non-multiple-of-tile hidden sizes; zero rows (amax 0 -> scale floor); NaN / +-inf / -0.0;
huge outliers (one element 3e4); exp under/overflow (+-120 for silu); ties (top-k: equal
scores -> lower index); ragged lengths and padding rows (-1 indices, seqlen 0); strided inputs."""
import torch

FP8_TYPES = tuple(t for t in (getattr(torch, n, None) for n in
                              ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")) if t)


def _bits(a):
    if a.dtype in FP8_TYPES:
        return a.contiguous().view(torch.uint8)
    if a.dtype in (torch.bfloat16, torch.float16):
        return a.contiguous().view(torch.int16)
    if a.dtype == torch.float32:
        return a.contiguous().view(torch.int32)
    return a


def same(a, b):
    """bit-identical (shape, dtype and every bit; NaN payloads included)"""
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def ndiff(a, b):
    """number of elements whose bits differ"""
    return int((_bits(a) != _bits(b)).sum())


def maxdiff(a, b):
    """max |a - b| over finite elements (for not-bit-exact kernels: compare with 1 ulp of the dtype)"""
    a, b = a.float(), b.float()
    m = torch.isfinite(a) & torch.isfinite(b)
    return float((a[m] - b[m]).abs().max()) if m.any() else 0.0


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


def eager_us(fn, reps=100):
    """per-call time without graphs (includes host launch cost)"""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / reps


def report(name, cases):
    """cases: iterable of (label, ref_tensors, new_tensors); prints BIT-EXACT or the first mismatches"""
    bad = 0
    for label, ref, new in cases:
        d = [ndiff(r, n) if r.shape == n.shape else -1 for r, n in zip(ref, new)]
        if any(d):
            bad += 1
            if bad <= 5:
                print(f"  {name} {label}: mismatching elements per output = {d}")
    print(f"{name}: {'BIT-EXACT' if not bad else f'{bad} cases differ'}", flush=True)
    return bad == 0
