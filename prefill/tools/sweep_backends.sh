#!/bin/bash
# Sweep the selectable attention/indexer backends on the GLM-5.3 prefill node.
#
# Metric is NEW-token throughput = (total_input - cached) / duration, i.e. the rate at which the P
# node does real work.  Total-input throughput is misleading under cache: at 90% hit it reads 72k
# tok/s while actually computing only 7.3k.
#
# Each arm is measured twice: once with the radix cache flushed (no reuse) and once at 90% hit,
# because the two scenarios stress different kernels -- _mqa_logits and _dequantize_k_cache_paged
# grow from 8.3%/3.7% to 18.3%/12.2% when a 59K prefix is cached.
set -u
G=/data/home/daihw/scripts/glm5.3
P=$G/pd
B=$P/benchmark
M=/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015
IN=${IN:-65536}
CONC=${CONC:-16}
OUT=$P/pf/sweep.csv
mkdir -p $P/pf/sweep
[ -f $OUT ] || echo "arm,env,pct,new_tok_s,all_tok_s,ttft_ms,hit_pct,ok,secs" > $OUT

restart() {
    local envs="$1"
    docker exec glm5.3 bash -c 'pkill -f "[s]glang serve"; sleep 5; pkill -9 -f "[s]glang::"' < /dev/null
    docker exec glm5.3 bash -c 'pkill -f "[l]aunch_router"' < /dev/null 2>/dev/null
    sleep 8
    docker exec -d glm5.3 bash -c "cd $P && $envs nohup bash p_opt.sh > $P/logs/nohup_sweep.log 2>&1" < /dev/null
    for i in $(seq 1 120); do
        sleep 15
        curl -s -m 5 http://127.0.0.1:30001/health > /dev/null 2>&1 && break
        # a scheduler that raised will never answer /health; stop waiting out the full 30 min
        if grep -aqE "ValueError|Scheduler hit an exception|is not supported" $P/logs/nohup_sweep.log 2>/dev/null; then
            echo "  start error:"; grep -aE "ValueError|is not supported" $P/logs/nohup_sweep.log | tail -2
            return 1
        fi
    done
    curl -s -m 5 http://127.0.0.1:30001/health > /dev/null 2>&1 || { echo "  START FAILED"; tail -12 $P/logs/nohup_sweep.log; return 1; }
    docker exec -d glm5.3 bash -c "cd $P && nohup bash route.sh > $P/logs/route_nohup.log 2>&1" < /dev/null
    sleep 25
    curl -s -m 300 http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"glm5.3-fp8","prompt":"hi","max_tokens":4,"temperature":0}' > /dev/null 2>&1
    return 0
}

measure() {   # arm env pct
    local arm=$1 envs=$2 pct=$3
    local script=bench_prefix90.py
    [ "$pct" = 0 ] && script=bench_prefix0.py
    local f=$P/pf/sweep/${arm}_p${pct}
    # a flushed radix tree is what makes the 0% arm actually 0%
    curl -s -m 120 -X POST http://127.0.0.1:30001/flush_cache > /dev/null 2>&1; sleep 3
    local t0=$(date +%s)
    docker exec glm5.3 bash -c "cd $B && PYTHONPATH=$G/benchmark/sglang-das/python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PREFIX_SEED=4242 PREFIX_CACHE_PCT=$pct timeout 2400 python3 $script \
      --backend sglang --base-url http://localhost:30000 --model $M --tokenizer $M \
      --served-model-name GLM-5.3-Channel-FP8-w8a8 --dataset-name random-ids --tokenize-prompt \
      --random-input-len $IN --random-output-len 1 --random-range-ratio 1.0 --request-rate inf \
      --num-prompts $((CONC*2)) --max-concurrency $CONC --warmup-requests 0 --cache-report \
      --output-details --disable-tqdm --output-file $f.jsonl" < /dev/null > $f.log 2>&1
    local t1=$(date +%s)
    python3 - "$f.jsonl" "$arm" "$envs" "$pct" "$((t1-t0))" "$OUT" <<'PYEOF'
import json,sys,os
fn,arm,envs,pct,secs,out=sys.argv[1:7]
try:
    o=json.load(open(fn))
except Exception:
    open(out,'a').write(f"{arm},\"{envs}\",{pct},FAIL,FAIL,FAIL,FAIL,0,{secs}\n"); sys.exit()
ti=o.get('total_input_tokens',0); du=o.get('duration',1)
ca=o.get('cached_tokens',0)
ca=sum(ca) if isinstance(ca,list) else ca
new=(ti-ca)/du if du else 0
allt=ti/du if du else 0
hit=100*ca/ti if ti else 0
row=f"{arm},\"{envs}\",{pct},{new:.0f},{allt:.0f},{o.get('mean_ttft_ms',0):.0f},{hit:.1f},{o.get('completed',0)},{secs}"
open(out,'a').write(row+"\n"); print("  "+row)
PYEOF
}

run_arm() {   # arm "env string"
    local arm=$1 envs=$2
    echo "=== [$arm] $envs  $(date +%H:%M:%S)"
    restart "$envs" || { echo "$arm,\"$envs\",-,STARTFAIL,,,,0,0" >> $OUT; return; }
    measure "$arm" "$envs" 0
    measure "$arm" "$envs" 90
}

# The indexer backend sweep is dead on arrival: paged_mqa_logits_backend.py rejects everything
# except lightop on HCU, and "auto" already resolves to it.  Only the prefill attention backend
# has real alternatives, and those are exactly the ones that decide whether the fp8 KV cache has
# to be dequantized to bf16 first (12.2% of the 90%-cache profile).
run_arm pf_q8   "PREFILL_BE=flashmla_sparse_q8"
run_arm pf_kv   "PREFILL_BE=flashmla_kv"
run_arm pf_auto "PREFILL_BE=flashmla_auto"
echo "=== SWEEP DONE $(date)"