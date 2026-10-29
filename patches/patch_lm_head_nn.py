"""Apply the SGLANG_HCU_SKINNY_LM_HEAD patch (default off) to sglang's lm_head paths
(re-runnable: backs each pristine file up to <file>.bak_20261023 once and always patches that
copy).  usage (inside glm5.3-pd): python3 patch_lm_head_nn.py

  layers/logits_processor.py   _compute_lm_head: decode-sized (M <= 64) bf16 GEMMs on the
                               NN-relaid lm_head weight (SGLANG_HCU_LM_HEAD_NN) run
                               hcu_ops.try_skinny_nn_bf16 (target verify + draft extend).
  speculative/draft_lm_head_vp.py  (separate switch SGLANG_HCU_SKINNY_DRAFT_VP, default off,
                               NOT safe with DP attention): project_top1 under CUDA graph
                               capture gathers / projects / exchanges only the captured rows
                               of each VP rank instead of the fixed max_rows_per_rank buffer.
                               DP ranks can run different forward modes at once (one replays
                               the decode graph while another runs an eager prefill / draft
                               extend with the fixed buffer), so the all_gather sizes differ
                               and NCCL hangs (seen at c >= 8 on DP8; c1 passes)."""
import os
import shutil

SRT = "/usr/local/lib/python3.10/dist-packages/sglang/srt"


def load(rel):
    p = os.path.join(SRT, rel)
    bak = p + ".bak_20261023"
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
    src = open(bak).read()  # always patch the pristine copy, so re-running updates the patch
    assert "SGLANG_HCU_SKINNY_LM_HEAD" not in src, "backup is not pristine: " + bak
    return p, src


def sub(src, old, new):
    assert src.count(old) == 1, (src.count(old), old)
    return src.replace(old, new)


# --- logits_processor.py ------------------------------------------------------------------
p, s = load("layers/logits_processor.py")
s = sub(s, '''_RELAYOUT_LM_HEAD = os.environ.get("SGLANG_HCU_LM_HEAD_NN", "1") == "1"
''', '''_RELAYOUT_LM_HEAD = os.environ.get("SGLANG_HCU_LM_HEAD_NN", "1") == "1"
# SGLANG_HCU_SKINNY_LM_HEAD=1: decode-sized bf16 lm_head GEMMs on the relaid NN weight run
# hcu_ops' skinny NN GEMM (~1.11-1.33 vs 1.58-1.59 ms for GLM-5.3's 154880-row head at
# M <= 64 on BW1100); it returns None (-> torch.matmul) for M > 64, untuned shapes or the NT
# layout.
_hcu_lm_head = None
if os.environ.get("SGLANG_HCU_SKINNY_LM_HEAD", "0") == "1":
    import hcu_ops as _hcu_lm_head
''')
s = sub(s, '''                _maybe_relayout_lm_head_weight(lm_head)
                logits = torch.matmul(
                    hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                )
''', '''                _maybe_relayout_lm_head_weight(lm_head)
                logits = None
                if (
                    _hcu_lm_head is not None
                    and hidden_states.dtype == lm_head.weight.dtype
                ):
                    logits = _hcu_lm_head.try_skinny_nn_bf16(
                        hidden_states, lm_head.weight.T
                    )
                if logits is None:
                    logits = torch.matmul(
                        hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                    )
''')
open(p, "w").write(s)
print("patched", p)

# --- draft_lm_head_vp.py ------------------------------------------------------------------
p, s = load("speculative/draft_lm_head_vp.py")
s = sub(s, '''import logging
from typing''', '''import logging
import os
from typing''')
s = sub(s, '''logger = logging.getLogger(__name__)
''', '''logger = logging.getLogger(__name__)

# SGLANG_HCU_SKINNY_DRAFT_VP=1 (experimental, default off): under CUDA graph capture assume
# every rank of the VP group runs the same padded batch, so project_top1 only gathers /
# projects / exchanges those rows instead of the fixed max_rows_per_rank buffer, and the
# shard GEMM runs hcu_ops' skinny NN GEMM when vp_size * rows <= 48.  Eager calls are
# unchanged.  Unsafe with DP attention: a rank replaying the decode graph exchanges
# vp_size * rows while a rank in an eager prefill / draft extend exchanges the fixed buffer,
# and the mismatched all_gather hangs NCCL.
_hcu_uniform_rows = os.environ.get("SGLANG_HCU_SKINNY_DRAFT_VP", "0") == "1"
_hcu_lm_head = None
if _hcu_uniform_rows:
    try:
        import hcu_ops as _hcu_lm_head
    except ImportError:
        pass


def _hcu_gemm_rows(rows: int, cap: int) -> int:
    """Row count to run the shard GEMM at: rocBLAS picks slow kernels for some M of GLM-5.3's
    [M, 6144] x [6144, 19360] shard GEMM on BW1100 (M 40 / 72 / 136-152 / 200: 370-500 us vs
    216 / 238 / 346 at M 64 / 128 / 256), so round up to those."""
    for bucket in (64, 128, 256):
        if rows <= bucket:
            return min(bucket, cap)
    return rows
''')
s = sub(s, '''        if local_rows:
            self.local_hidden[:local_rows].copy_(hidden_states)
''', '''        if (
            _hcu_uniform_rows
            and local_rows
            and torch.cuda.is_current_stream_capturing()
        ):
            return self._project_top1_uniform_rows(
                hidden_states, logit_scale, final_logit_softcapping
            )

        if local_rows:
            self.local_hidden[:local_rows].copy_(hidden_states)
''')
s = s.rstrip("\n") + '''

    def _project_top1_uniform_rows(
        self,
        hidden_states: torch.Tensor,
        logit_scale: float | None,
        final_logit_softcapping: float | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """project_top1 when every VP rank has the same row count (graph capture)."""
        rows = hidden_states.shape[0]
        total = self.vp_size * rows
        local_hidden = self.local_hidden[:rows]
        local_hidden.copy_(hidden_states)
        gathered_hidden = self.gathered_hidden[:total]
        torch.distributed.all_gather_into_tensor(
            gathered_hidden,
            local_hidden,
            group=self.process_group,
        )

        local_logits = self.local_logits[:total]
        if (
            _hcu_lm_head is None
            or _hcu_lm_head.try_skinny_nn_bf16(
                gathered_hidden, self.weight_shard_t, local_logits
            )
            is None
        ):
            # padding rows of the gathered buffer are stale; their logits are never read
            mm_rows = _hcu_gemm_rows(total, self.total_rows)
            torch.mm(
                self.gathered_hidden[:mm_rows],
                self.weight_shard_t,
                out=self.local_logits[:mm_rows],
            )
        if logit_scale is not None:
            local_logits.mul_(logit_scale)
        if final_logit_softcapping:
            local_logits.div_(final_logit_softcapping)
            local_logits.tanh_()
            local_logits.mul_(final_logit_softcapping)

        local_scores, local_token_ids = torch.max(local_logits, dim=-1)
        local_candidates = self.local_candidates[:total]
        local_candidates[:, 0].copy_(local_scores)
        local_candidates[:, 1].copy_(local_token_ids)
        gathered_candidates = self.gathered_candidates[: self.vp_size * total]
        torch.distributed.all_gather_into_tensor(
            gathered_candidates,
            local_candidates,
            group=self.process_group,
        )

        global_scores, global_token_ids = select_global_top1_from_candidates(
            gathered_candidates.view(self.vp_size, total, 2),
            self.shard_size,
        )
        local_start = self.vp_rank * rows
        local_end = local_start + rows
        return (
            global_scores[local_start:local_end],
            global_token_ids[local_start:local_end],
        )
'''
open(p, "w").write(s)
print("patched", p)
