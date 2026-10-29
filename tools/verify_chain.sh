#!/bin/bash
# Template: reproducibility check of a patch chain (writes only to a fresh /tmp mirror).
# For every patched file: copy its BASE (earliest backup of this round) into the mirror tree,
# replay the patch scripts in apply order with SGL_SRT_DIR pointing into the mirror (or with the
# site-packages path sed-rewritten for scripts that hardcode it), then cmp each result with the
# live file.  "verify fail=0" == the live files are exactly what the shipped scripts produce.
# Fill: FILES (one path under sglang/srt per line), BASES (backup suffixes, earliest first),
#       CHAIN (patch scripts in apply order), extra artifact checks at the end.
# usage (inside the container): bash verify_chain.sh
D=$(cd "$(dirname "$0")" && pwd)
SP=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
S=$SP/sglang/srt
T=/tmp/verify_chain_$(date +%Y%m%d%H%M%S)
BASES="bak_round2a bak_round2b"                  # earliest first
CHAIN="patch_a.py patch_b.py patch_c.py"         # apply order
mkdir -p $T
fail=0
for rel in $(cat "$D/FILES"); do
    base=""
    for s in $BASES; do [ -e "$S/$rel.$s" ] && { base=$S/$rel.$s; break; }; done
    if [ -z "$base" ]; then echo "NOBAK $rel"; fail=1; continue; fi
    mkdir -p "$(dirname "$T$S/$rel")"
    cp -p "$base" "$T$S/$rel"
done
export SGL_SRT_DIR=$T$S/
for p in $CHAIN; do
    sed "s#$SP#$T$SP#g" "$D/$p" > "$T/$p"      # for scripts that hardcode site-packages
    python3 "$T/$p" > /dev/null || { echo "FAIL running $p"; fail=1; }
done
unset SGL_SRT_DIR
for rel in $(cat "$D/FILES"); do
    if cmp -s "$T$S/$rel" "$S/$rel"; then echo "same   $rel"; else echo "DIFF   $rel"; fail=1; fi
done
# artifacts, e.g. the prebuilt extension:
# cmp -s "$D/../hcu_ops/hcu_ops/_C.cpython-310-x86_64-linux-gnu.so" "$SP/hcu_ops/_C.cpython-310-x86_64-linux-gnu.so" || fail=1
echo "verify fail=$fail  (mirror: $T)"
