#!/bin/bash
# GLM-5.3 FP8 on DCU (BW1100 / gfx938): install every optimization patch (round 1 + round 2) into
# the sglang of this container.  Run inside the container with no sglang server up:
#     bash apply_all.sh            # install the prebuilt hcu_ops (built first if absent) + patch sglang
#     bash apply_all.sh --build    # rebuild hcu_ops from ../hcu_ops/csrc for gfx938 first
#
#   1. round 1: layers/logits_processor.py and layers/quantization/unquant.py are replaced by the
#      round-1 versions (round1/) only if the live file is the known pristine one (md5 check);
#      pristine copy kept as *.bak_20261021.  Already round-1 files are left alone.
#   2. hcu_ops: _C.so / __init__.py / skinny_cfg.json copied into site-packages/hcu_ops (a
#      differing installed file is kept as *.bak_apply_<time>).
#   3. sglang patch chain, fixed order.  Each script backs its files up once (*.bak_20261022,
#      *.bak_20261023, *.bak_qcat, *.bak_fuse2, *.bak_fuse3, *.bak_kvfuse, *.bak_mqa) and always
#      re-patches from that backup, so re-running this script is idempotent.
#   4. py_compile of every touched file + import check with every switch on.
# Everything new is behind SGLANG_HCU_* switches (default off unless noted in the report); the
# start script decides what is on.  Undo: restore_all.sh.  Reproducibility check: verify_all.sh.
set -eo pipefail
D=$(cd "$(dirname "$0")" && pwd)
R=$(dirname "$D")
SP=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
S=$SP/sglang/srt
TS=$(date +%Y%m%d%H%M%S)
unset SGL_SRT_DIR   # the patch scripts honour it; apply_all always patches the live sglang
if pgrep -f "sglang serve|sglang.launch_server" > /dev/null; then echo "sglang server running, abort"; exit 1; fi

# 1. round 1 -------------------------------------------------------------------------------------
r1() {  # r1 <path under sglang/srt> <marker of the round-1 change>
    local f=$S/$1 n
    n=$(basename "$1")
    if grep -q "$2" "$f"; then echo "round1 present  $1"; return 0; fi
    if [ "$(md5sum < "$f" | cut -d' ' -f1)" != "$(grep " $n\$" "$D/round1/MD5SUMS.pristine" | cut -d' ' -f1)" ]; then
        echo "round1: $f is neither the pristine nor the round-1 version, abort"; exit 1
    fi
    [ -e "$f.bak_20261021" ] || cp -p "$f" "$f.bak_20261021"
    cp "$D/round1/$n" "$f"
    echo "round1 applied  $1"
}
r1 layers/logits_processor.py SGLANG_HCU_LM_HEAD_NN
r1 layers/quantization/unquant.py SGLANG_HCU_LINEAR_NN

# 2. hcu_ops -------------------------------------------------------------------------------------
H=$R/hcu_ops
# no prebuilt _C.so in this tree (the public repo ships sources only) -> build it
if [ "$1" = "--build" ] || [ ! -e "$H/hcu_ops/_C.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "building hcu_ops (log: $H/build.log) ..."
    (cd "$H" && PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=${MAX_JOBS:-16} python3 setup.py build_ext --inplace > build.log 2>&1) \
        || { tail -n 60 "$H/build.log"; exit 1; }
fi
mkdir -p "$SP/hcu_ops"
for f in _C.cpython-310-x86_64-linux-gnu.so __init__.py skinny_cfg.json; do
    if ! cmp -s "$H/hcu_ops/$f" "$SP/hcu_ops/$f"; then
        [ -e "$SP/hcu_ops/$f" ] && cp -p "$SP/hcu_ops/$f" "$SP/hcu_ops/$f.bak_apply_$TS"
        cp -p "$H/hcu_ops/$f" "$SP/hcu_ops/$f"
        echo "installed hcu_ops/$f"
    fi
done

# 3. sglang patch chain (order matters: qcat -> fuse2 -> kvfuse on forward_mla_rocm.py,
#    fuse2 -> fuse3 -> mqa on dsa_indexer.py / layernorm.py / deepseek_v2.py / CT w8a8_fp8) ----------
cd "$D"
for p in patch_topk.py patch_skinny.py patch_rope.py patch_moe.py patch_bmm.py; do python3 $p; done
F=$S/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py
[ -e "$F.bak_20261023" ] || cp -p "$F" "$F.bak_20261023"
python3 patch_mla_rocm_bmm.py "$F"   # in place, prints "already patched" on re-runs
for p in patch_sampler_warmup.py patch_qcat.py patch_fuse2.py patch_fuse3.py patch_kvfuse.py \
         patch_mqa.py patch_lm_head_nn.py; do
    python3 $p
done

# 4. checks --------------------------------------------------------------------------------------
for f in $(cat "$D/FILES"); do python3 -m py_compile "$S/$f"; done
cd /tmp
SGLANG_HCU_NORM_QUANT=1 SGLANG_HCU_SILU_QUANT=1 SGLANG_HCU_MLA_KV_FUSED=1 SGLANG_HCU_MQA_LOGITS=1 \
SGLANG_HCU_MQA_HIST=1 SGLANG_HCU_SKINNY_LM_HEAD=1 SGLANG_HCU_QUANT_REUSE=1 SGLANG_HCU_STRIDED_RMS=1 \
SGLANG_HCU_MLA_QCAT=1 SGLANG_HCU_SKINNY_BMM=1 SGLANG_HCU_SKINNY_GEMM=1 python3 -c "
import hcu_ops, sglang.srt.models.deepseek_v2 as d, sglang.srt.layers.layernorm as ln
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm as m
from sglang.srt.layers.attention.dsa import dsa_indexer as i
from sglang.srt.layers.attention import dsa_backend as b
from sglang.srt.layers import logits_processor as lp
from sglang.srt.speculative import draft_lm_head_vp as vp
print('import ok', hcu_ops.__file__)
print('qcat', m._hcu_qcat is not None and b._hcu_qcat is not None, 'quant_reuse', m._hcu_quant_reuse,
      'norm_quant', ln._hcu_norm_quant, 'silu_quant', d._hcu_silu_quant, 'kv_fused', m._hcu_kv is not None,
      'mqa', i._hcu_mqa is not None, 'mqa_hist', i._hcu_mqa_hist, 'lm_head', lp._hcu_lm_head is not None,
      'draft_vp', vp._hcu_uniform_rows)
" 2>&1 | grep -v -i -E "warn|setattr|_float_to_str|plugin"
echo "apply_all done"
