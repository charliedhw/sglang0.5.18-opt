#!/bin/bash
# Profile the GLM-5.3 P node under 0% and 90% prefix-cache hit, same input length and concurrency,
# so the two kernel rankings can be diffed directly.
#
# Why: with a 59K cached prefix the node processes NEW tokens at 7336 tok/s versus 10622 tok/s with
# no cache -- 31% slower per token of real work.  DSA attention is top-k bounded so it should not
# care about context length, but the indexer that picks the top-k scores against the whole context
# and does.  Back-of-envelope says ~49% of per-token cost scales with context, which is far more
# than the indexer's 8% share in the 0% profile, so something else grows too.  Measure, do not guess.
set -u
G=/data/home/daihw/scripts/glm5.3
P=$G/pd
B=$P/benchmark
M=/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015
IN=${IN:-65536}
CONC=${CONC:-16}
STAMP=$(date +%m%d-%H%M)

SELF=$$
for pid in $(pgrep -f "pdpf_probe\.sh" 2>/dev/null); do [ "$pid" = "$SELF" ] || kill "$pid" 2>/dev/null; done
sleep 2; pkill -9 -f "[b]ench_prefix" 2>/dev/null; pkill -9 -f "[b]ench_serving" 2>/dev/null; sleep 2

restart_stack() {
    local pdir=$1
    docker exec glm5.3 bash -c 'pkill -f "[s]glang serve"; sleep 5; pkill -9 -f "[s]glang::"' < /dev/null
    docker exec glm5.3 bash -c 'pkill -f "[l]aunch_router"' < /dev/null 2>/dev/null
    sleep 8
    docker exec -d glm5.3 bash -c "cd $P && ${PENV:-} SGLANG_TORCH_PROFILER_DIR=$pdir nohup bash p_opt.sh > $P/logs/nohup_p_prof.log 2>&1" < /dev/null
    for i in $(seq 1 120); do
        sleep 15
        curl -s -m 5 http://127.0.0.1:30001/health > /dev/null 2>&1 && { echo "  P ready ${i}0s"; break; }
    done
    docker exec -d glm5.3 bash -c "cd $P && nohup bash route.sh > $P/logs/route_nohup.log 2>&1" < /dev/null
    sleep 25
    curl -s -m 300 http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"glm5.3-fp8","prompt":"hi","max_tokens":4,"temperature":0}' > /dev/null && echo "  smoke ok"
}

run_case() {
    local pct=$1
    # bench_prefix90.py raises on pct=0 ("gives cached=0"); the no-prefix path is in bench_prefix0.py
    local script=bench_prefix90.py
    [ "$pct" = 0 ] && script=bench_prefix0.py
    local PDIR=$P/pf/prof-${TAG:-x}-pct${pct}-${IN}-c${CONC}-$STAMP
    mkdir -p $PDIR
    echo "=== case ${pct}% cache  -> $PDIR"
    restart_stack $PDIR

    # warm: for the 90% case this is what plants the shared prefix in the radix tree
    docker exec glm5.3 bash -c "cd $B && PYTHONPATH=$G/benchmark/sglang-das/python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PREFIX_SEED=4242 PREFIX_CACHE_PCT=$pct timeout 1800 python3 $script \
      --backend sglang --base-url http://localhost:30000 --model $M --tokenizer $M \
      --served-model-name GLM-5.3-Channel-FP8-w8a8 --dataset-name random-ids --tokenize-prompt \
      --random-input-len $IN --random-output-len 1 --random-range-ratio 1.0 --request-rate inf \
      --num-prompts 4 --max-concurrency 4 --warmup-requests 0 --cache-report --output-details \
      --disable-tqdm --output-file $PDIR/warm.jsonl" < /dev/null > $PDIR/warm.log 2>&1

    curl -s -m 60 -X POST http://127.0.0.1:30001/start_profile -H 'Content-Type: application/json' \
      -d '{"output_dir":"'"$PDIR"'","num_steps":12,"activities":["GPU"]}'; echo
    docker exec glm5.3 bash -c "cd $B && PYTHONPATH=$G/benchmark/sglang-das/python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PREFIX_SEED=4242 PREFIX_CACHE_PCT=$pct timeout 2400 python3 $script \
      --backend sglang --base-url http://localhost:30000 --model $M --tokenizer $M \
      --served-model-name GLM-5.3-Channel-FP8-w8a8 --dataset-name random-ids --tokenize-prompt \
      --random-input-len $IN --random-output-len 1 --random-range-ratio 1.0 --request-rate inf \
      --num-prompts $((CONC*2)) --max-concurrency $CONC --warmup-requests 0 --cache-report \
      --output-details --disable-tqdm --output-file $PDIR/load.jsonl" < /dev/null > $PDIR/load.log 2>&1

    for i in $(seq 1 40); do
        sleep 15
        n=$(ls $PDIR 2>/dev/null | grep -c "trace.json" || true)
        [ "$n" -gt 0 ] && { sleep 30; break; }
    done
    curl -s -m 60 -X POST http://127.0.0.1:30001/stop_profile > /dev/null 2>&1 || true
    sleep 10
    grep -E "Cache hit rate|Input token throughput|Mean TTFT" $PDIR/load.log | head -3
    echo "  traces: $(ls $PDIR | grep -c trace.json)"
}

run_case ${ONLY:-0}
[ -z "${ONLY:-}" ] && run_case 90
echo "=== BOTH CASES DONE $(date)"