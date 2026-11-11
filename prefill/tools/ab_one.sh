#!/bin/bash
# One arm: restart P with the given env, then measure new-token throughput at 0% and 90% hit.
set -u
G=/data/home/daihw/scripts/glm5.3
P=$G/pd
B=$P/benchmark
M=/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015
IN=${IN:-65536}
CONC=${CONC:-16}
ARM=$1; ENVS=${2:-}
OUT=$P/pf/fastdq.csv
mkdir -p $P/pf/fastdq
[ -f $OUT ] || echo "arm,env,pct,new_tok_s,all_tok_s,ttft_ms,hit_pct,ok" > $OUT

docker exec glm5.3 bash -c 'pkill -f "[s]glang serve"; sleep 5; pkill -9 -f "[s]glang::"' < /dev/null
docker exec glm5.3 bash -c 'pkill -f "[l]aunch_router"' < /dev/null 2>/dev/null
sleep 8
docker exec -d glm5.3 bash -c "cd $P && $ENVS nohup bash p_opt.sh > $P/logs/nohup_fastdq.log 2>&1" < /dev/null
for i in $(seq 1 80); do
    sleep 15
    curl -s -m 5 http://127.0.0.1:30001/health > /dev/null 2>&1 && break
    grep -aqE "ValueError|Scheduler hit an exception" $P/logs/nohup_fastdq.log 2>/dev/null && { echo "[$ARM] START FAILED"; exit 1; }
done
docker exec -d glm5.3 bash -c "cd $P && nohup bash route.sh > $P/logs/route_nohup.log 2>&1" < /dev/null
sleep 25
curl -s -m 300 http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm5.3-fp8","prompt":"hi","max_tokens":4,"temperature":0}' > /dev/null 2>&1
echo "[$ARM] $ENVS  $(date +%H:%M:%S)"

for pct in 0 90; do
    script=bench_prefix90.py; [ "$pct" = 0 ] && script=bench_prefix0.py
    f=$P/pf/fastdq/${ARM}_p${pct}
    curl -s -m 120 -X POST http://127.0.0.1:30001/flush_cache > /dev/null 2>&1; sleep 3
    docker exec glm5.3 bash -c "cd $B && PYTHONPATH=$G/benchmark/sglang-das/python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PREFIX_SEED=4242 PREFIX_CACHE_PCT=$pct timeout 2400 python3 $script \
      --backend sglang --base-url http://localhost:30000 --model $M --tokenizer $M \
      --served-model-name GLM-5.3-Channel-FP8-w8a8 --dataset-name random-ids --tokenize-prompt \
      --random-input-len $IN --random-output-len 1 --random-range-ratio 1.0 --request-rate inf \
      --num-prompts $((CONC*2)) --max-concurrency $CONC --warmup-requests 0 --cache-report \
      --output-details --disable-tqdm --output-file $f.jsonl" < /dev/null > $f.log 2>&1
    python3 - "$f.jsonl" "$ARM" "$ENVS" "$pct" "$OUT" <<'PYEOF'
import json,sys
fn,arm,envs,pct,out=sys.argv[1:6]
try: o=json.load(open(fn))
except Exception:
    open(out,'a').write(f"{arm},\"{envs}\",{pct},FAIL,FAIL,FAIL,FAIL,0\n"); sys.exit()
ti=o.get('total_input_tokens',0); du=o.get('duration',1) or 1
ca=o.get('cached_tokens',0); ca=sum(ca) if isinstance(ca,list) else ca
row=(f"{arm},\"{envs}\",{pct},{(ti-ca)/du:.0f},{ti/du:.0f},"
     f"{o.get('mean_ttft_ms',0):.0f},{100*ca/ti if ti else 0:.1f},{o.get('completed',0)}")
open(out,'a').write(row+"\n"); print("  "+row)
PYEOF
done