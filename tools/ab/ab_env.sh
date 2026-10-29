#!/bin/bash
# DP8 end-to-end A/B of env settings of sglang_start_opt2_final.sh (run on the host, ~25 min per arm):
#   [REF0=ab/<tag>_greedy.json] ab_env.sh "tag:VAR=v[,VAR=v...] ..." [scenario ...]   (scenario as in ab_bench.sh)
# per arm: start the server with the envs, greedy check (vs REF0 or the first arm), ab_bench.sh,
# mean accept len, stop.  Same flow as ab_lmhead.sh, which takes a single VAR.
G=/data/home/daihw/scripts/glm5.3; W=$G/opt2
ARMS=$1; shift
SC=${@:-"64k_c1:65536:58982:1 64k_c8:65536:58982:8 64k_c16:65536:58982:16 64k_c32:65536:58982:32"}
nsg() { docker exec glm5.3-pd bash -c "ps aux | grep -c '[s]glang'"; }
REF=${REF0:-}
for arm in $ARMS; do
  tag=${arm%%:*}; envs=${arm#*:}
  LOG=$W/srv_$tag.out
  docker exec -d glm5.3-pd bash -c "cd $G && export ${envs//,/ } && bash sglang_start_opt2_final.sh > $LOG 2>&1"
  ok=0
  for i in $(seq 1 120); do
    sleep 15
    grep -aq 'The server is fired up' $LOG && { ok=1; break; }
    [ $i -gt 4 ] && [ "$(nsg)" = 0 ] && break
  done
  echo "[$tag] ${envs} ready=$ok after $((i * 15)) s"
  if [ $ok = 1 ]; then
    docker exec glm5.3-pd timeout -k 30 900 python3 $W/greedy_chk.py $W/ab/${tag}_greedy.json $REF
    [ -z "$REF" ] && REF=$W/ab/${tag}_greedy.json
    L0=$(wc -l < $LOG)
    docker exec glm5.3-pd bash $W/ab_bench.sh $tag $SC
    echo "[$tag] mean accept len (DP0): $(tail -n +$L0 $LOG | grep -a 'Decode batch' | grep 'DP0 ' \
      | grep -o 'accept len: [0-9.]*' | awk '{s += $3; n++} END {if (n) printf "%.3f over %d lines", s / n, n}')"
    grep -aoE 'SGLANG_HCU_[A-Z_]+: .*' $LOG | sort | uniq -c | sed "s/^/[$tag] /"
  fi
  docker exec glm5.3-pd pkill -f "[s]glang serve"
  for i in $(seq 1 60); do sleep 5; [ "$(nsg)" = 0 ] && break; done
  [ "$(nsg)" = 0 ] || docker exec glm5.3-pd pkill -9 -f '[s]glang'
  echo "[$tag] stopped ($(nsg) sglang procs left)"
  sleep 20
done
