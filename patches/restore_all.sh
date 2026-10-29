#!/bin/bash
# Undo the round-2 patches: every patched sglang file goes back to its round-2 base (its earliest
# backup *.bak_20261022 / *.bak_20261023 / *.bak_qcat / *.bak_fuse2), i.e. the round-1 state.
# With ROUND1=1 logits_processor.py / unquant.py also go back to the pristine image versions
# (*.bak_20261021).  The current files are kept as *.pre_restore_<time>; all backups stay, so
# apply_all.sh can re-apply.  hcu_ops stays installed (nothing imports it after the restore).
# usage (inside the container, no sglang server up): [ROUND1=1] bash restore_all.sh
set -e
D=$(cd "$(dirname "$0")" && pwd)
SP=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
S=$SP/sglang/srt
TS=$(date +%Y%m%d%H%M%S)
if pgrep -f "sglang serve|sglang.launch_server" > /dev/null; then echo "sglang server running, abort"; exit 1; fi
for rel in $(cat "$D/FILES"); do
    base=""
    order="bak_20261022 bak_20261023 bak_qcat bak_fuse2"
    [ "${ROUND1:-0}" = 1 ] && order="bak_20261021 $order"
    for s in $order; do
        [ -e "$S/$rel.$s" ] && { base=$S/$rel.$s; break; }
    done
    if [ -z "$base" ]; then echo "keep     $rel (no backup)"; continue; fi
    if cmp -s "$base" "$S/$rel"; then echo "same     $rel"; continue; fi
    cp -p "$S/$rel" "$S/$rel.pre_restore_$TS"
    cp "$base" "$S/$rel"
    echo "restored $rel <- ${base##*.}"
done
