#!/bin/bash
# End-to-end A/B of env switches of one start script (run on the HOST; ~25 min per arm at 4 scenarios):
#   [REF0=<greedy.json>] ab_env.sh "tag:VAR=v[,VAR=v...] tag2:VAR=v ..." [scenario ...]
# per arm: start the server in the container with the envs, wait until ready, greedy check
# (vs REF0, or vs the first arm), ab_bench.sh, mean accept len of the bench window, stop.
# Arms run one after another; the start script must take the switches as ${VAR:-default}.
# Edit the variables below.  Run it detached:  setsid nohup bash ab_env.sh "..." < /dev/null > ab.log 2>&1 &
G=${G:-/data/home/user/scripts/model}         # work dir (mounted in the container)
W=${W:-$G/opt}                                # tools dir: greedy_chk.py, ab_bench.sh; results in $W/ab
CT=${CT:-model-container}                     # container name
START=${START:-sglang_start_opt.sh}           # start script under $G (never edit it while an arm runs)
ARMS=$1; shift
SC=${@:-"64k_c1:65536:58982:1 64k_c8:65536:58982:8 64k_c16:65536:58982:16 64k_c32:65536:58982:32"}
mkdir -p $W/ab
nsg() { docker exec $CT bash -c "ps aux | grep -c '[s]glang'"; }
REF=${REF0:-}
for arm in $ARMS; do
  tag=${arm%%:*}; envs=${arm#*:}
  LOG=$W/ab/srv_$tag.out
  docker exec -d $CT bash -c "cd $G && export ${envs//,/ } && bash $START > $LOG 2>&1"
  ok=0
  for i in $(seq 1 120); do
    sleep 15
    grep -aq 'The server is fired up' $LOG && { ok=1; break; }
    [ $i -gt 4 ] && [ "$(nsg)" = 0 ] && break
  done
  echo "[$tag] ${envs} ready=$ok after $((i * 15)) s"
  if [ $ok = 1 ]; then
    docker exec $CT python3 $W/greedy_chk.py $W/ab/${tag}_greedy.json $REF
    [ -z "$REF" ] && REF=$W/ab/${tag}_greedy.json
    L0=$(wc -l < $LOG)
    docker exec $CT bash $W/ab_bench.sh $tag $SC
    echo "[$tag] mean accept len (DP0): $(tail -n +$L0 $LOG | grep -a 'Decode batch' | grep 'DP0 ' \
      | grep -o 'accept len: [0-9.]*' | awk '{s += $3; n++} END {if (n) printf "%.3f over %d lines", s / n, n}')"
    echo "[$tag] hooks: $(grep -a -o 'SGLANG_HCU_[A-Z_]*:[^,]*' $LOG | sort -u | tr '\n' ';')"
  else
    echo "[$tag] start failed: $(grep -a -m1 -E 'Error:|error:' $LOG | cut -c1-200)"
  fi
  docker exec $CT pkill -f "[s]glang serve"
  for i in $(seq 1 60); do sleep 5; [ "$(nsg)" = 0 ] && break; done
  [ "$(nsg)" = 0 ] || docker exec $CT pkill -9 -f '[s]glang'
  echo "[$tag] stopped ($(nsg) sglang procs left)"
  sleep 20
done
