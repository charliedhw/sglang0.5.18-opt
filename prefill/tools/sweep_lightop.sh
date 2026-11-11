#!/bin/bash
# Sweep the two lightop knobs that reach the indexer, on top of the fast dequant.
#
# LIGHTOP_MQA_LOGITS_BETA_FP8 selects a newer fp8 mqa_logits implementation whose own error text
# says it "requires gfx938/wave64" -- exactly this GPU.  _mqa_logits is 8.3% of the 0%-cache
# prefill profile and 18.3% of the 90% one, the largest indexer cost.
# SGL_USE_LIGHTOP_TOPK_BACKAND (0/1/2, default 0 = exact) reaches adaptive_topk, 2.3%.
#
# Baseline for all of these is fastdq-on, which is already a measured +13.9% at 90% hit.
set -u
P=/data/home/daihw/scripts/glm5.3/pd
W=$P/work
FD="SGLANG_DSA_FAST_DEQUANT_PAGED=1"

while pgrep -f "prof_cache_ab\.sh" > /dev/null; do sleep 30; done
echo "=== profile finished, starting knob sweep $(date)"

bash $W/ab_one.sh fdq_only   "$FD"
bash $W/ab_one.sh beta_fp8   "$FD LIGHTOP_MQA_LOGITS_BETA_FP8=1"
bash $W/ab_one.sh topk1      "$FD SGL_USE_LIGHTOP_TOPK_BACKAND=1"
bash $W/ab_one.sh topk2      "$FD SGL_USE_LIGHTOP_TOPK_BACKAND=2"
bash $W/ab_one.sh beta_topk1 "$FD LIGHTOP_MQA_LOGITS_BETA_FP8=1 SGL_USE_LIGHTOP_TOPK_BACKAND=1"
echo "=== KNOB SWEEP DONE $(date)"