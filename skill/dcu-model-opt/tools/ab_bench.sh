#!/bin/bash
# vllm bench scenarios for A/B (run INSIDE the container against the server on :30000).
# usage: [NP_MULT=2] ab_bench.sh <tag> [scenario ...]
#   scenario = name:input_len:prefix_len:concurrency   (prefix_len 0 = no shared prefix;
#   58982/65536 = 90% prefix-cache hit), prompts = NP_MULT x concurrency, output 1024, ignore-eos.
# Prefix cache with dp-attention: every DP rank has its own radix cache and requests go
# round-robin, so --num-warmups must be >= dp (8 here) or some ranks never see the prefix.
# cache_salt isolates the runs from older cached prefixes.  If vllm bench fails on an openai
# import (sglang pins an old openai), put a newer openai into a separate dir and set BENCHLIB.
[ -n "$BENCHLIB" ] && export PYTHONPATH=$BENCHLIB
TAG=$1; shift
O=${O:-$(dirname "$0")/ab}; mkdir -p $O
M=${MODEL_PATH:-/data/model/your-model}     # tokenizer path
NAME=${SERVED_NAME:-default}
SALT=${SALT:-ab-001}
SC=${@:-"64k_c1:65536:58982:1 64k_c8:65536:58982:8"}
for s in $SC; do
  IFS=: read -r name in pre c <<< "$s"
  extra=(); [ "$pre" -gt 0 ] && extra=(--random-prefix-len "$pre")
  vllm bench serve --host 127.0.0.1 --port 30000 --model $M --served-model-name $NAME \
    --dataset-name random --random-input-len $((in - pre)) --random-output-len 1024 \
    --random-range-ratio 0 --num-warmups ${WARMUPS:-8} --num-prompts $((c * ${NP_MULT:-2})) --max-concurrency $c \
    --percentile-metrics ttft,tpot,itl,e2el --temperature 0.01 --ignore-eos --request-rate inf \
    --extra-body "{\"cache_salt\":\"$SALT\"}" "${extra[@]}" > $O/${TAG}_${name}.txt 2>&1
  echo "[$TAG $name] $(grep -E 'Output token throughput|Mean TTFT|Mean TPOT|Median TPOT|Mean ITL' $O/${TAG}_${name}.txt | sed 's/  */ /g' | tr '\n' ';')"
done
