#!/bin/bash
# Host-side gsm8k driver for the final start script: start it in glm5.3-pd, wait until ready,
# pre-warm the sampling path on every DP rank, run eval/run_gsm8k.sh <tag> in glm5.2_evalscope,
# stop the server.  A server that fails to start / pre-warm is restarted (up to 3 starts).
# usage (on the host):
#   setsid nohup bash drive_gsm8k.sh <tag> <start script under glm5.3/> [limit] < /dev/null > drive_gsm8k.log 2>&1 &
G=/data/home/daihw/scripts/glm5.3
E=$G/eval
TAG=$1; START=$2; LIMIT=${3:-0}

stop_server() {
  # [s] / [g]: the patterns must not match this bash -c command line itself
  docker exec glm5.3-pd bash -c 'pkill -f "sglang [s]erve"; sleep 20; pkill -9 -f "sglan[g]::"; pkill -9 -f "sglang [s]erve"'
  sleep 15
}

# same as drive_eval.sh: 16 requests per setting reach each of the 8 DP ranks (round_robin) twice
prewarm() {
  docker exec glm5.3-pd bash -c '
    for tp in "1.0 0.95" "0.01 1.0"; do
      set -- $tp
      for k in $(seq 1 16); do
        curl -s -m 300 http://127.0.0.1:30000/v1/chat/completions -H "Content-Type: application/json" \
          -d "{\"model\":\"glm5.3-fp8\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup $k: 1+$k=?\"}],\"max_tokens\":64,\"temperature\":$1,\"top_p\":$2}" > /dev/null &
      done
      wait
    done'
}

echo "=== $TAG start=$START $(date '+%F %T')"
ok=0
for try in 1 2 3; do
  SRV=$E/srv_$TAG.$try.out
  docker exec -d glm5.3-pd bash -c "cd $G && bash $START > $SRV 2>&1"
  up=0
  for i in $(seq 1 120); do
    sleep 10
    grep -aq "fired up and ready" $SRV 2>/dev/null && { up=1; break; }
  done
  echo "start $try: ready=$up after $((i * 10))s; errors: $(grep -aciE 'Traceback|Error' $SRV)"
  [ $up = 1 ] && prewarm
  if [ $up = 1 ] && ! grep -aq "Scheduler hit an exception" $SRV; then
    echo "pre-warm OK $(date '+%T')"
    ok=1; break
  fi
  echo "start $try failed: $(grep -a -m1 -E 'Error:|error:' $SRV | cut -c1-200)"
  stop_server
done
[ $ok = 1 ] && docker exec glm5.2_evalscope bash $E/run_gsm8k.sh $TAG $LIMIT
echo "eval end $(date '+%F %T'); server crash during eval: $(grep -ac 'Scheduler hit an exception' $SRV)"
stop_server
echo "stopped $(date '+%F %T')"
echo DONE
