#!/bin/bash
# Prefill TTFT probe through the PD router: random prompts, 4 output tokens, fresh seed/cache_salt
# per run so nothing hits the P radix cache.
# usage: bash ttft_probe.sh TAG [IN=32768] [CONC=1] [NUM=4]
TAG=$1; IN=${2:-32768}; CONC=${3:-1}; NUM=${4:-4}
W=/data/home/daihw/scripts/glm5.3/pd/work
export PYTHONPATH=/data/home/daihw/scripts/glm5.3/opt/benchlib
SEED=$RANDOM
vllm bench serve --host 127.0.0.1 --port 30000 \
    --model /data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015/ --served-model-name glm5.3-fp8 \
    --backend vllm --endpoint /v1/completions \
    --dataset-name random --random-input-len $IN --random-output-len 4 --random-range-ratio 0 \
    --num-warmups 1 --num-prompts $NUM --max-concurrency $CONC --seed $SEED \
    --percentile-metrics ttft,e2el --temperature 0.01 --ignore-eos \
    --extra-body "{\"cache_salt\":\"probe-$TAG-$SEED\"}" > $W/probe_${TAG}_${IN}_c${CONC}.log 2>&1
grep -E "Mean TTFT|Median TTFT|Request throughput|Total input tokens|Benchmark duration" $W/probe_${TAG}_${IN}_c${CONC}.log
