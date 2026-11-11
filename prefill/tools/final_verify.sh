#!/bin/bash
# Final verification of the optimised prefill config at the shapes that were asked for:
# 40K / 64K input, 16 / 32 / 64 concurrency, output 1 so TTFT is the prefill cost.
# p_opt.sh now defaults both switches on, so this measures the shipped configuration.
set -u
G=/data/home/daihw/scripts/glm5.3
P=$G/pd
M=/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015
N=glm5.3-fp8
OUT=$P/pf/final.csv
mkdir -p $P/pf/final
[ -f $OUT ] || echo "tag,point,in_tok,conc,ttft_ms,p99ttft_ms,tot_tps,ok,secs" > $OUT

echo "=== restore P ($(date +%H:%M:%S))"
docker exec glm5.3 bash -c 'pkill -f "[s]glang serve"; sleep 5; pkill -9 -f "[s]glang::"' < /dev/null
docker exec glm5.3 bash -c 'pkill -f "[l]aunch_router"' < /dev/null 2>/dev/null
sleep 8
docker exec -d glm5.3 bash -c "cd $P && nohup bash p_opt.sh > $P/logs/nohup_final.log 2>&1" < /dev/null
for i in $(seq 1 80); do sleep 15; curl -s -m 5 http://127.0.0.1:30001/health >/dev/null 2>&1 && { echo "P ready"; break; }; done
echo "=== restore router"
docker exec -d glm5.3 bash -c "cd $P && nohup bash route.sh > $P/logs/route_nohup.log 2>&1" < /dev/null
sleep 25
curl -s -m 300 http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm5.3-fp8","prompt":"hi","max_tokens":4,"temperature":0}' > /dev/null 2>&1 && echo "smoke ok"
echo "=== confirm both switches are live"
cat /proc/$(pgrep -f "[s]glang serve" | head -1)/environ 2>/dev/null | tr '\0' '\n' | grep -E "FAST_DEQUANT|BETA_FP8"

for spec in "40960 16" "40960 32" "40960 64" "65536 16" "65536 32" "65536 64"; do
    set -- $spec; tok=$1; c=$2
    f=$P/pf/final/opt_${tok}_c${c}
    echo "[opt] in=$tok c=$c $(date +%H:%M:%S)"
    t0=$(date +%s)
    docker exec glm5.3 bash -c "cd $G && timeout 3000 python3 -m sglang.bench_serving \
        --backend sglang --base-url http://127.0.0.1:30000 --model $M --served-model-name $N \
        --dataset-name generated-shared-prefix --gsp-num-groups 1 --gsp-prompts-per-group $((c*2)) \
        --gsp-system-prompt-len 0 --gsp-question-len $tok --gsp-output-len 1 \
        --num-prompts $((c*2)) --max-concurrency $c --request-rate inf --seed 897 \
        --warmup-requests 0" < /dev/null > $f.txt 2>&1
    t1=$(date +%s)
    g() { grep -iP "$1" $f.txt 2>/dev/null | grep -oP '[\d.]+' | $2 -1; }
    echo "opt,i$((tok/1024))k_c$c,$tok,$c,$(g 'Mean TTFT' head),$(g 'P99 TTFT' tail),$(g 'Total token throughput' head),$(grep -iP 'Successful requests' $f.txt | grep -oP '\d+' | head -1),$((t1-t0))" >> $OUT
    tail -1 $OUT
done
echo "=== FINAL VERIFY DONE $(date)"