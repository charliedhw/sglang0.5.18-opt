#!/bin/bash
# Host-side accuracy driver: for each "tag:ENV=V,ENV=V" arm start the DP8 server
# (opt2/start_k.sh) in glm5.3-pd with the extra env, wait until ready, pre-warm the sampling
# path on every DP rank, run eval/run_eval.sh <tag> in glm5.2_evalscope, then stop the server.
# A server that crashes during start/pre-warm is restarted (up to 3 starts per arm).
# usage (on the host):
#   nohup bash drive_eval.sh "opt:SGLANG_HCU_SKINNY_GEMM=1,..." "r1:SGLANG_HCU_FAST_MASK_TOPK=0" > drive.log 2>&1 &
# GSM_LIMIT / AIME_REPEATS (env) are passed through to run_eval.sh.
G=/data/home/daihw/scripts/glm5.3
O=$G/opt2
E=$G/eval

stop_server() {
  # [s] / [g]: the patterns must not match this bash -c command line itself
  docker exec glm5.3-pd bash -c 'pkill -f "sglang [s]erve"; sleep 20; pkill -9 -f "sglan[g]::"; pkill -9 -f "sglang [s]erve"'
  sleep 15
}

# Server/bench warmups are greedy, so without this the first T>0 request of each rank is
# where the sampling kernels are first launched.  round_robin dispatch -> 16 requests per
# setting reach every one of the 8 DP ranks twice.
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

for arm in "$@"; do
  tag=${arm%%:*}; envs=${arm#*:}
  echo "=== $tag env=[$envs] $(date '+%F %T')"
  ok=0
  for try in 1 2 3; do
    SRV=$E/srv_$tag.$try.out
    docker exec -d glm5.3-pd bash -c "cd $O && env ${envs//,/ } setsid bash start_k.sh > $SRV 2>&1"
    up=0
    for i in $(seq 1 120); do
      sleep 10
      grep -q "fired up and ready" $SRV 2>/dev/null && { up=1; break; }
    done
    echo "start $try: ready=$up after $((i * 10))s; errors: $(grep -ciE 'Traceback|Error' $SRV)"
    [ $up = 1 ] && prewarm
    if [ $up = 1 ] && ! grep -q "Scheduler hit an exception" $SRV; then
      echo "pre-warm OK $(date '+%T'); sampler warmup retries: $(grep -c 'HCU sampler warmup attempt' $SRV)"
      ok=1; break
    fi
    echo "start $try failed: $(grep -m1 -E 'Error:|error:' $SRV | cut -c1-200)"
    stop_server
  done
  [ $ok = 1 ] && docker exec glm5.2_evalscope bash $E/run_eval.sh $tag "${GSM_LIMIT:-0}" "${AIME_REPEATS:-4}"
  echo "eval end $(date '+%F %T'); server crash during eval: $(grep -c 'Scheduler hit an exception' $SRV)"
  stop_server
  echo "stopped $(date '+%F %T')"
done
echo DONE
