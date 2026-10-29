#!/bin/bash
# Reproducibility check of the whole patch chain; writes only to a fresh /tmp/glm53_pt_<time>.
#   - every patched sglang file: its round-2 base (earliest backup: *.bak_20261022 / *.bak_20261023 /
#     *.bak_qcat / *.bak_fuse2) is copied into the mirror tree, the 13 patch scripts run there in
#     apply_all.sh's order (paths rewritten), and each result is cmp'ed with the live file
#   - round 1: round1/*.py vs the live unquant.py and logits_processor.py's round-1 backup
#   - hcu_ops: the package's _C.so / __init__.py / skinny_cfg.json vs the installed ones
# usage (inside the container): bash verify_all.sh     -> "verify fail=0" when all match
D=$(cd "$(dirname "$0")" && pwd)
R=$(dirname "$D")
SP=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
S=$SP/sglang/srt
T=/tmp/glm53_pt_$(date +%Y%m%d%H%M%S)
mkdir -p $T
fail=0
R2FILES=$(grep -v -E "unquant.py" "$D/FILES")
for rel in $R2FILES; do
    base=""
    for s in bak_20261022 bak_20261023 bak_qcat bak_fuse2; do
        [ -e "$S/$rel.$s" ] && { base=$S/$rel.$s; break; }
    done
    if [ -z "$base" ]; then echo "NOBAK $rel"; fail=1; continue; fi
    mkdir -p "$(dirname "$T$S/$rel")"
    cp -p "$base" "$T$S/$rel"
done
run() {  # run <patch script> [args]: the script with site-packages paths rewritten into $T
    local p=$1
    shift
    sed "s#$SP#$T$SP#g" "$D/$p" > "$T/$p"
    python3 "$T/$p" "$@" > /dev/null || { echo "FAIL running $p"; fail=1; }
}
unset SGL_SRT_DIR
for p in patch_topk.py patch_skinny.py patch_rope.py patch_moe.py patch_bmm.py; do run $p; done
F=$T$S/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py
cp -p "$F" "$F.bak_20261023"
run patch_mla_rocm_bmm.py "$F"
for p in patch_sampler_warmup.py patch_qcat.py patch_fuse2.py patch_fuse3.py patch_kvfuse.py \
         patch_mqa.py patch_lm_head_nn.py; do
    run $p
done
for rel in $R2FILES; do
    if cmp -s "$T$S/$rel" "$S/$rel"; then echo "same   $rel"; else echo "DIFF   $rel"; fail=1; fi
done
chk() {  # chk <label> <a> <b>
    if cmp -s "$2" "$3"; then echo "same   $1"; else echo "DIFF   $1"; fail=1; fi
}
chk "round1 unquant.py" "$D/round1/unquant.py" "$S/layers/quantization/unquant.py"
chk "round1 logits_processor.py" "$D/round1/logits_processor.py" "$S/layers/logits_processor.py.bak_20261023"
for f in _C.cpython-310-x86_64-linux-gnu.so __init__.py skinny_cfg.json; do
    chk "hcu_ops/$f" "$R/hcu_ops/hcu_ops/$f" "$SP/hcu_ops/$f"
done
echo "verify fail=$fail  (mirror: $T)"
