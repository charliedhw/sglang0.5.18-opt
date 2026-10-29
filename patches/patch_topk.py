"""SGLANG_HCU_FAST_MASK_TOPK (default 1): the DSA indexer's sparse-mask TopK-2048 + page-table
transform (dsa_indexer_metadata.py) runs hcu_ops.sparse_mask_topk_transform instead of lightop
fast_topk_transform_sparse_mask_fused; 0 keeps lightop.
Backs the file up once as *.bak_20261022 and patches from the backup."""
import os
import shutil

PATH = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/dsa/dsa_indexer_metadata.py"

EDITS = [
    ("""from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
""",
     """from typing import TYPE_CHECKING, List, Optional, Tuple

import os

import torch
"""),
    ("""

class BaseIndexerMetadata(ABC):
""",
     """

# Exact multi-CTA TopK-2048 kernel for HCU (hcu_ops); 0 falls back to LightOp.
_HCU_FAST_MASK_TOPK = os.environ.get("SGLANG_HCU_FAST_MASK_TOPK", "1") == "1"


class BaseIndexerMetadata(ABC):
"""),
    ("""        from lightop.attention import fast_topk_transform_sparse_mask_fused

        return fast_topk_transform_sparse_mask_fused(
""",
     """        if _HCU_FAST_MASK_TOPK:
            from hcu_ops import sparse_mask_topk_transform

            return sparse_mask_topk_transform(
                logits, nonzero_mask, lengths, page_table_size_1, cu_seqlens_q, topk
            )

        from lightop.attention import fast_topk_transform_sparse_mask_fused

        return fast_topk_transform_sparse_mask_fused(
"""),
]

bak = PATH + ".bak_20261022"
if not os.path.exists(bak):
    shutil.copy2(PATH, bak)
src = open(bak).read()
for old, new in EDITS:
    assert src.count(old) == 1, (old[:60], src.count(old))
    src = src.replace(old, new)
open(PATH, "w").write(src)
print("patched", PATH)
