import time
import torch
from hcu_ops import _C

dev = "cuda"


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


out = torch.zeros(4096, device=dev)
for rows in [6, 24, 64]:
    for L in [65536, 131072]:
        x = torch.rand(rows, L, device=dev)
        mb = rows * L * 4 / 1e6
        for T in [1024, 256]:
            for U in [1, 4, 8]:
                res = []
                for S in [1, 2, 4, 8, 16]:
                    t = bench(lambda: _C.calib_stream(x, S, U, T, out))
                    res.append(f"S{S}:{t:6.1f}us({mb / t * 1e3:5.0f}GB/s)")
                print(f"rows={rows:3d} L={L:6d} T={T:4d} U={U}  " + " ".join(res), flush=True)
