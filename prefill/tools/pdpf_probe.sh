#!/bin/bash
# GLM-5.3 PD prefill probe, driven through the router on dcu4.
# Output length 1 so TTFT is dominated by the P node's prefill; the D node only has to emit one
# token, which keeps the decode side out of the measurement.
#   bash pdpf_probe.sh <tag> [points]
set -u
G=/data/home/daihw/scripts/glm5.3
P=$G/pd
M=/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015
N=glm5.3-fp8
TAG=$1; SEL=${2:-all}
OUT=$P/pf/results.csv
mkdir -p $P/pf/raw
[ -f $OUT ] || echo "tag,point,in_tok,conc,ttft_ms,p99ttft_ms,tot_tps,ok,secs" > $OUT

POINTS='
i40k_c16|40960|16
i40k_c32|40960|32
i40k_c64|40960|64
i64k_c16|65536|16
i64k_c32|65536|32
i64k_c64|65536|64
i128k_c16|131072|16
i128k_c32|131072|32
'

echo "$POINTS" | grep -v '^$' | while IFS='|' read -r id tok c; do
    [ "$SEL" = all ] || echo ",$SEL," | grep -q ",$id," || continue
    f=$P/pf/raw/${TAG}_${id}.txt
    echo "[$TAG] $id in=$tok c=$c $(date +%H:%M:%S)"
    t0=$(date +%s)
    docker exec glm5.3 bash -c "cd $G && timeout 3000 python3 -m sglang.bench_serving \
        --backend sglang --base-url http://127.0.0.1:30000 --model $M --served-model-name $N \
        --dataset-name generated-shared-prefix --gsp-num-groups 1 --gsp-prompts-per-group $((c*2)) \
        --gsp-system-prompt-len 0 --gsp-question-len $tok --gsp-output-len 1 \
        --num-prompts $((c*2)) --max-concurrency $c --request-rate inf --seed 897 \
        --warmup-requests 0" < /dev/null > $f 2>&1
    t1=$(date +%s)
    g() { grep -iP "$1" $f 2>/dev/null | grep -oP '[\d.]+' | $2 -1; }
    ttft=$(g "Mean TTFT" head); p99=$(g "P99 TTFT" tail); tot=$(g "Total token throughput" head)
    ok=$(grep -iP "Successful requests" $f 2>/dev/null | grep -oP '\d+' | head -1)
    echo "$TAG,$id,$tok,$c,${ttft:-NA},${p99:-NA},${tot:-NA},${ok:-0},$((t1-t0))" >> $OUT
    tail -1 $OUT
done
echo "[$TAG] done $(date +%H:%M:%S)"