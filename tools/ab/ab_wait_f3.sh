#!/bin/bash
# wait for the running ab_env.sh (f2allb arm) to finish, then re-run the f3all arm with the fixed
# patch_kvfuse (greedy reference: f2allb of the same run)
W=/data/home/daihw/scripts/glm5.3/opt2
while pgrep -f "bash $W/ab_env.sh" > /dev/null; do sleep 30; done
F2="SGLANG_HCU_MLA_QCAT=1,SGLANG_HCU_STRIDED_RMS=1,SGLANG_HCU_QUANT_REUSE=1"
F3="$F2,SGLANG_HCU_NORM_QUANT=1,SGLANG_HCU_SILU_QUANT=1,SGLANG_HCU_MLA_KV_FUSED=1,SGLANG_HCU_MQA_LOGITS=1,SGLANG_HCU_MQA_HIST=1,SGLANG_HCU_SKINNY_LM_HEAD=1"
REF0=$W/ab/f2allb_greedy.json bash $W/ab_env.sh "f3allb:$F3"
