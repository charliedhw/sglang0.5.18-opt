#!/bin/bash
# GLM-5.3 accuracy check, gsm8k only (full test set, 1319), against the sglang server on :30000
# (served name glm5.3-fp8); the gsm8k half of run_eval.sh (aime26 is left to the user).
# Run in the glm5.2_evalscope container (evalscope 1.10.0, host network):
#   docker exec glm5.2_evalscope bash /data/home/daihw/scripts/glm5.3/eval/run_gsm8k.sh <tag> [limit]
# Results: eval/<tag>/gsm8k/<timestamp>/reports, log eval/<tag>/gsm8k.log.  Sampling as in
# run_eval.sh: generation_config (temperature 1.0, top_p 0.95), thinking on, reasoning_effort max.
TAG=${1:-final}
GSM_LIMIT=${2:-0}       # 0 = whole test set
EFFORT=${EFFORT:-max}
D=/data/home/daihw/scripts/glm5.3/eval
API=http://127.0.0.1:30000/v1/chat/completions
mkdir -p $D/$TAG && cd $D/$TAG
LIM=""
[ "$GSM_LIMIT" != 0 ] && LIM="--limit $GSM_LIMIT"
echo "=== gsm8k $(date) effort=$EFFORT"
evalscope eval --model glm5.3-fp8 --api-url $API --api-key EMPTY --eval-type openai_api \
  --datasets gsm8k $LIM \
  --generation-config '{"max_tokens": 16000, "temperature": 1.0, "top_p": 0.95, "n": 1, "timeout": 1800, "stream": true, "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "'$EFFORT'"}}}' \
  --eval-batch-size 64 --ignore-errors --work-dir $D/$TAG/gsm8k > gsm8k.log 2>&1
echo "=== done $(date)"
grep -a -E "gsm8k.*\|" gsm8k.log | tail -n 5
