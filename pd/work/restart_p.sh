#!/bin/bash
# host-side helper (dcu4): stop P + router, start P (env passed through: DEEPEP_CFG CHUNK MEM P_EXTRA),
# wait for :30001 health, then start the router and smoke-test it.  usage: bash restart_p.sh TAG
# ([s]glang patterns so pkill does not match the bash -c running it)
TAG=${1:-base}
PD=/data/home/daihw/scripts/glm5.3/pd
docker exec glm5.3 bash -c 'pkill -f "[s]glang serve"; pkill -f "[p]_opt.sh"; pkill -f "[r]oute.sh"; pkill -f "[s]glang_router"; sleep 5; pkill -9 -f "[s]glang::"' < /dev/null
sleep 10
docker exec glm5.3 bash -c 'ps -eo pid,args | grep -E "[s]glang"' < /dev/null && echo "WARN: sglang processes still alive"
[ -f $PD/logs/p_nohup.log ] && mv $PD/logs/p_nohup.log /tmp/p_nohup_$(date +%H%M%S).log
[ -f $PD/logs/route_nohup.log ] && mv $PD/logs/route_nohup.log /tmp/route_nohup_$(date +%H%M%S).log
ENVS="DEEPEP_CFG=${DEEPEP_CFG:-} CHUNK=${CHUNK:-} MEM=${MEM:-}"
setsid nohup docker exec glm5.3 bash -c "cd $PD && export $ENVS P_EXTRA='${P_EXTRA:-}'; [ -z \"\$DEEPEP_CFG\" ] && unset DEEPEP_CFG; [ -z \"\$CHUNK\" ] && unset CHUNK; [ -z \"\$MEM\" ] && unset MEM; bash p_opt.sh" < /dev/null > $PD/logs/p_nohup.log 2>&1 &
for i in $(seq 1 90); do
    sleep 10
    if curl -s -m 5 http://127.0.0.1:30001/health > /dev/null 2>&1 && grep -q "fired up" $PD/logs/p_nohup.log; then
        echo "P up after $((i*10))s ($TAG)"; break
    fi
    if grep -qE "Traceback|OutOfMemory" $PD/logs/p_nohup.log; then echo "P ERROR"; grep -E "Error" $PD/logs/p_nohup.log | tail -5; exit 1; fi
done
setsid nohup docker exec glm5.3 bash -c "cd $PD && bash route.sh" < /dev/null > $PD/logs/route_nohup.log 2>&1 &
sleep 20
for i in $(seq 1 12); do
    r=$(curl -s -m 60 http://127.0.0.1:30000/v1/completions -H 'Content-Type: application/json' \
        -d '{"model":"glm5.3-fp8","prompt":"Hello, my name is","max_tokens":4}')
    echo "$r" | grep -q '"choices"' && { echo "router OK"; exit 0; }
    sleep 10
done
echo "router smoke FAILED: $r"
