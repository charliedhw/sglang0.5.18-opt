#!/bin/bash
# A/B decode benchmark (same vllm bench flags as bench/vllm_bench-pc.sh).
# usage: [NP_MULT=2] ab_bench.sh <tag> [scenario ...]   scenario = name:input:prefix:conc, prompts = NP_MULT x conc
export PYTHONPATH=/data/home/daihw/scripts/glm5.3/opt/benchlib
TAG=$1; shift
O=/data/home/daihw/scripts/glm5.3/opt2/ab; mkdir -p $O
M=/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015/
SC=${@:-"64k_c8:65536:58982:8 128k_c8:131072:117965:8 64k_c32:65536:58982:32"}
for s in $SC; do
  IFS=: read -r name in pre c <<< "$s"
  extra=(); [ "$pre" -gt 0 ] && extra=(--random-prefix-len "$pre")
  timeout -k 30 ${BENCH_TO:-1500} vllm bench serve --host 127.0.0.1 --port 30000 --model $M --served-model-name glm5.3-fp8 \
    --dataset-name random --random-input-len $((in - pre)) --random-output-len 1024 \
    --random-range-ratio 0 --num-warmups 8 --num-prompts $((c * ${NP_MULT:-2})) --max-concurrency $c \
    --percentile-metrics ttft,tpot,itl,e2el --temperature 0.01 --ignore-eos --request-rate inf \
    --extra-body '{"cache_salt":"ab-001"}' "${extra[@]}" > $O/${TAG}_${name}.txt 2>&1
  rc=$?
  echo "[$TAG $name] $(grep -E 'Output token throughput|Mean TTFT|Mean TPOT|Median TPOT|Mean ITL' $O/${TAG}_${name}.txt | sed 's/  */ /g' | tr '\n' ';')"
  [ $rc -ge 124 ] && { echo "[$TAG $name] TIMEOUT / killed (rc=$rc), skipping the rest"; break; }
done
