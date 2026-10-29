#!/bin/bash
# host-side (dcu4): restart P with the current p_opt.sh defaults, then run the long-context bench scenarios;
# 128K last (dcu5 went down during 128K PC90 c16 on 10/26, so collect 64K / 16K first)
W=/data/home/daihw/scripts/glm5.3/pd/work
cd $W
bash restart_p.sh final2 > restart_final2.log 2>&1
grep -q "router OK" restart_final2.log || { echo "restart failed"; exit 1; }
for s in "长文64K" "长文分析" "超长上下文128K"; do
    echo "=== $s $(date)"
    docker exec glm5.3 bash -c "cd /data/home/daihw/scripts/glm5.3/pd/bench && PYTHONPATH=/data/home/daihw/scripts/glm5.3/opt/benchlib bash vllm_bench-pc_1015.sh --scenario $s" < /dev/null
done
echo "=== done $(date)"
