#!/bin/bash
# after the f3allb A/B (ab_wait_f3.sh): 1) the round-1 state on the -1015 weights through the final
# start script with every hcu_ops switch off (w15r1, for the overall gain), 2) gsm8k with the final
# start script (eval/drive_gsm8k.sh, tag final1015).  touch opt2/ab/HOLD_GSM8K to stop before 2).
G=/data/home/daihw/scripts/glm5.3; W=$G/opt2
while pgrep -f "bash $W/ab_wait_f3.sh" > /dev/null; do sleep 30; done
OFF=""
for v in FAST_MASK_TOPK SKINNY_GEMM FUSED_ROPE FUSED_GATE FUSED_ACT SKINNY_BMM MLA_QCAT STRIDED_RMS QUANT_REUSE \
         NORM_QUANT SILU_QUANT MLA_KV_FUSED MQA_LOGITS MQA_HIST SKINNY_LM_HEAD; do
  OFF="$OFF${OFF:+,}SGLANG_HCU_$v=0"
done
START=sglang_start_glm53_opt.sh REF0=$W/ab/w15base_greedy.json bash $W/ab_env2.sh "w15r1:$OFF"
[ -f $W/ab/HOLD_GSM8K ] && { echo "HOLD_GSM8K: gsm8k not started"; exit 0; }
echo "gsm8k start $(date '+%F %T')"
bash $G/eval/drive_gsm8k.sh final1015 sglang_start_glm53_opt.sh > $G/eval/drive_gsm8k_final1015.log 2>&1
echo "gsm8k end $(date '+%F %T')"
