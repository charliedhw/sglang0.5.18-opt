#!/bin/bash
# ab_env.sh with the start script as a parameter (ab_env.sh itself is fixed to
# sglang_start_opt2_final.sh, which hard-sets the batch-1 switches):
#   START=sglang_start_glm53_opt.sh [REF0=ab/<tag>_greedy.json] ab_env2.sh "tag:VAR=v[,VAR=v...] ..." [scenario ...]
G=/data/home/daihw/scripts/glm5.3; W=$G/opt2
START=${START:-sglang_start_glm53_opt.sh}
ARMS=$1; shift
SC=${@:-"64k_c1:65536:58982:1 64k_c8:65536:58982:8 64k_c16:65536:58982:16 64k_c32:65536:58982:32"}
nsg() { docker exec glm5.3-pd bash -c "ps aux | grep -c '[s]glang'"; }
REF=${REF0:-}
for arm in $ARMS; do
  tag=${arm%%:*}; envs=${arm#*:}
  LOG=$W/srv_$tag.out
  docker exec -d glm5.3-pd bash -c "cd $G && export ${envs//,/ } && bash $START > $LOG 2>&1"
  ok=0
  for i in $(seq 1 120); do
    sleep 15
    grep -aq 'The server is fired up' $LOG && { ok=1; break; }
    [ $i -gt 4 ] && [ "$(nsg)" = 0 ] && break
  done
  echo "[$tag] start=$START ${envs} ready=$ok after $((i * 15)) s"
  if [ $ok = 1 ]; then
    docker exec glm5.3-pd python3 $W/greedy_chk.py $W/ab/${tag}_greedy.json $REF
    [ -z "$REF" ] && REF=$W/ab/${tag}_greedy.json
    L0=$(wc -l < $LOG)
    docker exec glm5.3-pd bash $W/ab_bench.sh $tag $SC
    echo "[$tag] mean accept len (DP0): $(tail -n +$L0 $LOG | grep -a 'Decode batch' | grep 'DP0 ' \
      | grep -o 'accept len: [0-9.]*' | awk '{s += $3; n++} END {if (n) printf "%.3f over %d lines", s / n, n}')"
    grep -a 'SGLANG_HCU' $LOG | sed 's/.*\] //' | cut -c1-160 | sort | uniq -c | sed "s/^/[$tag] /"
  fi
  docker exec glm5.3-pd pkill -f "[s]glang serve"
  for i in $(seq 1 60); do sleep 5; [ "$(nsg)" = 0 ] && break; done
  [ "$(nsg)" = 0 ] || docker exec glm5.3-pd pkill -9 -f '[s]glang'
  echo "[$tag] stopped ($(nsg) sglang procs left)"
  sleep 20
done
