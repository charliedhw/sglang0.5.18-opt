"""GLM-5.3 DSA indexer logits on hcu_ops, behind switches (default off), bit-exact
(test_mqa_hist.py):

SGLANG_HCU_MQA_LOGITS=1   dsa_indexer.py: the sparse-mask indexer logits (lightop
  page_mqa_logits_sparse_mask / _grouped, decode G 1 and MTP verify G 6) run
  hcu_ops.mqa_logits -- the same numerics with 3 MTP rows per CTA (one K tile load feeds 3
  rows) and 64 CTAs per request while the requests cannot fill the 64 CUs (HCU_MQA_RPC /
  HCU_MQA_PCTAS[_G1] override).  Operands it does not cover keep lightop.
SGLANG_HCU_MQA_HIST=1     (with SGLANG_HCU_MQA_LOGITS=1 and the default
  SGLANG_HCU_FAST_MASK_TOPK=1) the logits kernel also builds pass 1 (the 12-bit histogram) of
  the paired hcu_ops sparse_mask_topk_transform for rows it will split, and replaces its two
  workspace memsets; the top-k then runs passes 2-4 only.  Up to HCU_MQA_HIST_MAX_ROWS rows
  (default 8; groups of 6: HCU_MQA_HIST_MAX_ROWS_G6, default 0; a fused histogram loses once several requests fill the GPU).

Run after patch_fuse3.py.  The file is backed up once as *.bak_mqa and patched from that
backup, so re-running is idempotent."""
import os
import shutil

D = os.environ.get("SGL_SRT_DIR", "/usr/local/lib/python3.10/dist-packages/sglang/srt/")
F_IDX = D + "layers/attention/dsa/dsa_indexer.py"

IDX_EDITS = [
    ("""    _hcu_mqa_pctas = int(_os.environ.get("SGLANG_HCU_MQA_PCTAS", "0"))
""",
     """    _hcu_mqa_pctas = int(_os.environ.get("SGLANG_HCU_MQA_PCTAS", "0"))
# SGLANG_HCU_MQA_LOGITS=1: the sparse-mask indexer logits run hcu_ops.mqa_logits (same bits as
# lightop page_mqa_logits_sparse_mask[_grouped], tuned launch) when it covers the operands.
# SGLANG_HCU_MQA_HIST=1: that kernel also builds the first histogram pass of the paired
# hcu_ops sparse_mask_topk_transform (dsa_indexer_metadata's SGLANG_HCU_FAST_MASK_TOPK path).
_hcu_mqa = None
_hcu_mqa_hist = False
_hcu_mqa_logged = set()
if _is_hcu and _os.environ.get("SGLANG_HCU_MQA_LOGITS", "0") == "1":
    import hcu_ops as _hcu_mqa

    _hcu_mqa_hist = (
        _os.environ.get("SGLANG_HCU_MQA_HIST", "0") == "1"
        and _os.environ.get("SGLANG_HCU_FAST_MASK_TOPK", "1") == "1"
    )
"""),
    ("""            elif sparse_route.group_size == 1:
                from lightop.gemmopt import page_mqa_logits_sparse_mask
""",
     """            elif _hcu_mqa is not None and _hcu_mqa.mqa_logits_supported(
                active_q,
                kv_cache,
                active_weights,
                sparse_route.group_size,
                seqlens_32,
                block_tables,
                max_seq_len,
            ):
                # SGLANG_HCU_MQA_LOGITS: lightop's logits / mask bits, tuned launch
                logits, sparse_mask = _hcu_mqa.mqa_logits(
                    active_q,
                    kv_cache,
                    active_weights,
                    seqlens_32,
                    block_tables,
                    max_seq_len,
                    sparse_route.group_size,
                    hist_lens=(
                        metadata.get_seqlens_expanded() if _hcu_mqa_hist else None
                    ),
                )
                if sparse_route.group_size not in _hcu_mqa_logged:
                    _hcu_mqa_logged.add(sparse_route.group_size)
                    logger.info(
                        "SGLANG_HCU_MQA_LOGITS: hcu_ops.mqa_logits group_size=%d hist=%s",
                        sparse_route.group_size,
                        _hcu_mqa_hist,
                    )
            elif sparse_route.group_size == 1:
                from lightop.gemmopt import page_mqa_logits_sparse_mask
"""),
]


def patch(path, edits, need):
    bak = path + ".bak_mqa"
    if not os.path.exists(bak):
        assert need in open(path, encoding="utf-8").read(), (path, "run patch_fuse3.py first")
        shutil.copy2(path, bak)
    src = open(bak, encoding="utf-8").read()
    for old, new in edits:
        assert src.count(old) == 1, (path, old[:70], src.count(old))
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print("patched", path)


patch(F_IDX, IDX_EDITS, "SGLANG_HCU_MQA_PCTAS")
