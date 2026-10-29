#!/bin/bash
# host-side (dcu4): wait for run_long2.sh, then compare its TTFT / throughput with bench_20261025_211619
# -> work/compare_long2.txt   (columns: scenario, PC, conc, old TTFT, new TTFT, change, old tput, new tput)
PD=/data/home/daihw/scripts/glm5.3/pd
OUT=$PD/work/compare_long2.txt
for i in $(seq 1 480); do
    grep -aq "=== done" $PD/logs/bench_long_opt2.log && break
    sleep 60
done
{
    echo "generated $(date); run_long2 finished: $(grep -ac '=== done' $PD/logs/bench_long_opt2.log)"
    grep -a "=== " $PD/logs/bench_long_opt2.log
    NEW=$(ls -d $PD/bench/bench_20261026_0[7-9]* $PD/bench/bench_20261026_1* 2>/dev/null)
    echo "new dirs: $NEW"
    printf "%-16s %-4s %4s %10s %10s %8s %8s %8s\n" scenario PC conc oldTTFT newTTFT chg oldTput newTput
    for d in $NEW; do tail -n +2 $d/summary.csv; done | tr -d '\r' | awk -F, -v OLD=$PD/bench/bench_20261025_211619/summary.csv '
        BEGIN { while ((getline l < OLD) > 0) { sub(/\r/, "", l); split(l, f, ","); k = f[1] "," f[2] "," f[3]; ot[k] = f[15]; op[k] = f[12] } }
        { k = $1 "," $2 "," $3; c = (ot[k] > 0 && $15 > 0) ? sprintf("%+.1f%%", ($15 / ot[k] - 1) * 100) : "NA";
          printf "%-16s %-4s %4s %10.0f %10.0f %8s %8.1f %8.1f\n", $1, $2, $3, ot[k], $15, c, op[k], $12 }'
} > $OUT 2>&1
