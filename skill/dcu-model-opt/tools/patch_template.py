"""Template of a framework patch script (copy, rename, fill EDITS).

Conventions (they make the patch chain reproducible and reversible):
  * every target file is backed up ONCE as <file>.bak_<TAG>, and every run re-patches from that
    backup -> re-running is idempotent, and the chain can be replayed on a mirror (verify_chain.sh)
  * `need`: a string a previous patch of the chain must have put into the file before this one
    may take its backup (enforces the chain order); None for the first patch of a file
  * every anchor must occur exactly once in the backup (assert count == 1) -> a framework update
    makes the patch fail loudly instead of patching the wrong place
  * the new code path is behind an env switch read at import, default OFF; at call time a
    *_supported(...) check falls back to the original code for any operand it does not cover;
    the first hit logs one line (grep the server log in A/B to prove the hook is live)
  * SGL_SRT_DIR overrides the sglang/srt directory (mirror replay / other installs)
usage: python3 patch_xxx.py"""
import os
import shutil
import sysconfig

TAG = "xxx"   # backup suffix: <file>.bak_xxx
D = os.environ.get("SGL_SRT_DIR", os.path.join(sysconfig.get_paths()["purelib"], "sglang/srt/"))
F = D + "layers/layernorm.py"   # target file(s)

EDITS = [
    # (old, new): `old` copied verbatim from the backup (unique), `new` keeps it plus the hook
    ("""import torch
""",
     """import torch

# SGLANG_HCU_XXX=1: <what is replaced, by what, numerics (bit-exact? vs which op)>
_hcu_xxx = None
if os.environ.get("SGLANG_HCU_XXX", "0") == "1":
    import hcu_ops as _hcu_xxx
_hcu_xxx_logged = False
"""),
    ("""        out = original_op(x, weight, eps)
""",
     """        if _hcu_xxx is not None and _hcu_xxx.xxx_supported(x, weight):
            out = _hcu_xxx.xxx(x, weight, eps)
            global _hcu_xxx_logged
            if not _hcu_xxx_logged:
                _hcu_xxx_logged = True
                logger.info("SGLANG_HCU_XXX: hcu_ops.xxx shape=%s", tuple(x.shape))
        else:
            out = original_op(x, weight, eps)
"""),
]


def patch(path, edits, need=None):
    bak = path + ".bak_" + TAG
    if not os.path.exists(bak):
        src = open(path, encoding="utf-8").read()
        assert need is None or need in src, (path, "run the previous patch of the chain first")
        shutil.copy2(path, bak)
    src = open(bak, encoding="utf-8").read()
    for old, new in edits:
        assert src.count(old) == 1, (path, old[:70], src.count(old))
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print("patched", path)


patch(F, EDITS, need=None)
