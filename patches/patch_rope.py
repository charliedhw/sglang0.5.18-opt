"""Route sglang's HCU rotary fallback (_launch_rotary_embedding, which splits MLA's 64
query heads into 8 sgl_kernel launches) to hcu_ops.rope_inplace (one launch, same
numerics) behind SGLANG_HCU_FUSED_ROPE=1 (default off).
Backs up the file once as *.bak_20261023 and patches from the backup."""
import os
import shutil

PATH = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/rotary_embedding/base.py"

EDITS = [
    ("_HCU_ROPE_MAX_THREADS = 256\n",
     """_HCU_ROPE_MAX_THREADS = 256

# SGLANG_HCU_FUSED_ROPE=1: hcu_ops.rope_inplace tiles heads over the grid, so all
# query heads and the key go in one launch instead of nq * rot_dim / 512 launches.
_hcu_rope = None
if get_bool_env_var("SGLANG_HCU_FUSED_ROPE"):
    from hcu_ops import rope_inplace as _hcu_rope
"""),
    ("""    num_tokens = int(positions.numel())
    if num_tokens == 0:
        return
    rot_dim = int(cos_sin_cache.shape[-1])
""",
     """    num_tokens = int(positions.numel())
    if num_tokens == 0:
        return
    if (
        _hcu_rope is not None
        and query.dtype in (torch.bfloat16, torch.float16)
        and cos_sin_cache.dtype == query.dtype
        and query.dim() in (2, 3)
        and query.stride(-1) == 1
        and (key is None or (key.dim() in (2, 3) and key.stride(-1) == 1))
    ):
        _hcu_rope(positions, query, key, head_size, cos_sin_cache, is_neox)
        return
    rot_dim = int(cos_sin_cache.shape[-1])
"""),
]

bak = PATH + ".bak_20261023"
if not os.path.exists(bak):
    shutil.copy2(PATH, bak)
src = open(bak).read()
assert "get_bool_env_var" in src.split("_HCU_ROPE_MAX_THREADS")[0], "get_bool_env_var not imported"
for old, new in EDITS:
    assert src.count(old) == 1, (old[:60], src.count(old))
    src = src.replace(old, new)
open(PATH, "w").write(src)
print("patched", PATH)
