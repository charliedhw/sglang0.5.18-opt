#!/bin/bash
# Decode-phase GPU trace of a running SGLang server (run INSIDE the container):
#   prof_decode.sh <out_name> <input_len> <output_len> <concurrency>
# Starts a vllm bench load, waits until every DP rank decodes its share of requests and no new
# prefill showed up for 8 s (pure decode), then POST /start_profile for 10 steps (GPU only; with_stack
# off - it slows the run and bloats the trace).  Traces land in $W/prof/<out_name>/ (one per rank).
# Analyse with kagg.py / kstat.py / kseq.py / tracediff.py.  The server may misbehave after a
# profile: restart it before benchmarking.  Server must be started with SGLANG_TORCH_PROFILER_DIR set.
W=${W:-$(pwd)}
OUTD=$1; IN=$2; OUT=$3; C=$4
DP=${DP:-8}
LOG=${SRV_LOG:-$(ls -t $W/srv_*.out | head -1)}     # the server's stdout log
M=${MODEL_PATH:-/data/model/your-model}; NAME=${SERVED_NAME:-default}
[ -n "$BENCHLIB" ] && export PYTHONPATH=$BENCHLIB
[ -d $W/prof/$OUTD ] && mv $W/prof/$OUTD /tmp/prof_${OUTD}_old_$(date +%s)
mkdir -p $W/prof/$OUTD
L0=$(wc -l < $LOG)
vllm bench serve --host 127.0.0.1 --port 30000 --model $M --served-model-name $NAME \
  --dataset-name random --random-input-len $IN --random-output-len $OUT --ignore-eos --max-concurrency $C --num-prompts $C \
  --seed $RANDOM --temperature 0.01 --percentile-metrics ttft,tpot,itl,e2el > $W/prof/bench_$OUTD.out 2>&1 &
BP=$!
PER=$(( (C + DP - 1) / DP ))
for t in $(seq 1 400); do
  sleep 3
  LINE=$(tail -n +$L0 $LOG | grep -a 'Decode batch' | grep 'DP0 ' | tail -1)
  NPF=$(tail -n +$L0 $LOG | grep -a 'Prefill batch' | wc -l)
  R=$(echo "$LINE" | grep -o '#running-req: [0-9]*' | grep -o '[0-9]*$')
  if [ -n "$R" ] && [ "$R" -ge "$PER" ]; then
     sleep 8
     NPF2=$(tail -n +$L0 $LOG | grep -a 'Prefill batch' | wc -l)
     [ "$NPF2" = "$NPF" ] && break
  fi
done
echo "start profile at t=$t: $LINE" | cut -c1-300
curl -s -m 600 http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"$W/prof/$OUTD\",\"num_steps\":10,\"activities\":[\"GPU\"]}"
echo
wait $BP
grep -E "Mean TPOT|Mean TTFT|Output token throughput|Successful" $W/prof/bench_$OUTD.out
ls -la $W/prof/$OUTD | head -n 4
