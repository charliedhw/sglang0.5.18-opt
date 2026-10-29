#!/bin/bash
# GLM-5.3 accuracy check: gsm8k (full test set, 1319) + aime26 (30 problems x repeats)
# against the sglang server on :30000 (served name glm5.3-fp8).  Run in the
# glm5.2_evalscope container (evalscope 1.10.0, host network):
#   docker exec -d glm5.2_evalscope bash /data/home/daihw/scripts/glm5.3/eval/run_eval.sh <tag> [gsm8k_limit] [aime_repeats]
# Results: eval/<tag>/{gsm8k,aime26}/<timestamp>/reports, logs eval/<tag>/*.log
# Sampling follows the model's generation_config (temperature 1.0, top_p 0.95), thinking on.
# EFFORT (env, default max) is the chat template's reasoning_effort: the GLM-5.3 template
# only knows low / high and maps anything else (or nothing) to max, its default.  "high"
# gives one-line reasoning on aime26, so use max like glm5.2_shca/run_aime26.sh.
TAG=${1:-opt}
GSM_LIMIT=${2:-0}       # 0 = whole test set
AIME_REPEATS=${3:-4}
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

# max_tokens 65536 (glm5.2 used 128000): one request lives on one DP rank, whose KV pool is
# ~103K tokens, so a 128K-token answer could never finish here.
echo "=== aime26 $(date) effort=$EFFORT"
evalscope eval --model glm5.3-fp8 --api-url $API --api-key EMPTY --eval-type openai_api \
  --datasets aime26 \
  --dataset-args '{"aime26": {"local_path": "/data/home/daihw/scripts/glm5.2_shca/aime26", "subset_list": ["default"]}}' \
  --repeats $AIME_REPEATS \
  --generation-config '{"max_tokens": 65536, "temperature": 1.0, "top_p": 0.95, "n": 1, "timeout": 7200, "stream": true, "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "'$EFFORT'"}}}' \
  --eval-batch-size 32 --ignore-errors --work-dir $D/$TAG/aime26 > aime26.log 2>&1

echo "=== done $(date)"
