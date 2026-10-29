#!/bin/bash
# Incremental in-place build of opt2/hcu_ops (keeps build/, only changed sources recompile),
# niced so a running A/B bench is not disturbed.  The previous in-place .so is kept as
# _C*.so.bak_<tag> (tag = $1, default nq).  CPU only, no GPU use.
set -e
G=/data/home/daihw/scripts/glm5.3
cd $G/opt2/hcu_ops
TAG=${1:-nq}
export PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=${MAX_JOBS:-8}
for f in hcu_ops/_C*.so; do [ -e "$f.bak_$TAG" ] || cp -p "$f" "$f.bak_$TAG"; done
date
nice -n 19 python3 setup.py build_ext --inplace > build.log 2>&1 || { tail -60 build.log; exit 1; }
date
ls -la hcu_ops/_C*.so
grep -E "warning|error" build.log | grep -v -i deprecat | head -20 || true
