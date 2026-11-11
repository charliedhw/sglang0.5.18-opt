#!/bin/bash
# Install every GLM-5.3 optimisation in this repository into the sglang of the current container.
#
# Two independent rounds, both idempotent and both reversible with uninstall.sh:
#
#   decode  (2026-10)  hcu_ops operator package + 15 sglang patches.
#                      64K PC90% decode step time -23.2% at c1, -11.1% at c32.
#                      Everything sits behind SGLANG_HCU_* switches; the start script decides
#                      which are on.  See patches/apply_all.sh and docs/GLM5.3_DCU_优化报告_20261023.md.
#
#   prefill (2026-11)  Vectorised paged K-cache dequant kernel (12.3x, bit-identical).
#                      90% prefix-cache hit: +36.5% new-token throughput, TTFT -27.2%.
#                      Behind SGLANG_DSA_FAST_DEQUANT_PAGED.  See prefill/README.md.
#                      The second prefill change (LIGHTOP_MQA_LOGITS_BETA_FP8) is a pure
#                      environment variable and needs no installation -- it is already set by
#                      prefill/p_opt.sh.
#
# Usage, inside the container, with no sglang server running:
#     bash install.sh                 # both rounds
#     bash install.sh --decode-only   # 2026-10 only
#     bash install.sh --prefill-only  # 2026-11 only
#     bash install.sh --build         # force rebuilding hcu_ops from source first
#
# Re-running is safe: each patch backs its target up once and always re-patches from that backup.
set -eo pipefail

D=$(cd "$(dirname "$0")" && pwd)
DO_DECODE=1
DO_PREFILL=1
BUILD=""

for arg in "$@"; do
    case "$arg" in
        --decode-only)  DO_PREFILL=0 ;;
        --prefill-only) DO_DECODE=0 ;;
        --build)        BUILD="--build" ;;
        -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
        *)              echo "unknown option: $arg"; exit 1 ;;
    esac
done

if pgrep -f "sglang serve|sglang.launch_server" > /dev/null; then
    echo "a sglang server is running -- stop it first, the patches rewrite files it has imported"
    exit 1
fi

python3 - <<'PYEOF'
import sysconfig, os, sys
p = sysconfig.get_paths()["purelib"]
if not os.path.isdir(os.path.join(p, "sglang")):
    sys.exit(f"sglang not found under {p}; run this inside the serving container")
print(f"target sglang: {os.path.join(p, 'sglang')}")
PYEOF

if [ "$DO_DECODE" = 1 ]; then
    echo
    echo "=== decode round (2026-10): hcu_ops + sglang patch chain ==="
    bash "$D/patches/apply_all.sh" $BUILD
fi

if [ "$DO_PREFILL" = 1 ]; then
    echo
    echo "=== prefill round (2026-11): paged K-cache dequant kernel ==="
    python3 "$D/prefill/patches/patch_fast_dequant.py"
fi

echo
echo "=== verifying ==="
if [ "$DO_DECODE" = 1 ]; then
    bash "$D/patches/verify_all.sh" || { echo "decode verify FAILED"; exit 1; }
fi
if [ "$DO_PREFILL" = 1 ]; then
    python3 - <<'PYEOF'
import torch
from sglang.kernels.ops.attention.dsa import dequant_k_cache as m

assert hasattr(m, "_dequantize_k_cache_paged_v2_kernel"), "prefill patch not present"
pool, n = 1 << 16, 8192
cache = (torch.randn(pool, 656, device="cuda") * 0.1).to(torch.float8_e4m3fn).view(pool, 1, 656)
page = torch.randint(0, pool, (n,), device="cuda", dtype=torch.int32)
ref = m._dequantize_k_cache_paged_orig(cache, page, 128)
import os
os.environ["SGLANG_DSA_FAST_DEQUANT_PAGED"] = "1"
m._FD_ON = True
got = m.dequantize_k_cache_paged(cache, page)
assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), "fast dequant is NOT bit-exact"
print("prefill verify: fast dequant installed and bit-identical to the original")
PYEOF
fi

cat <<'EOF'

=== done ===

Switches, all revertible per run without reinstalling:

  decode   SGLANG_HCU_* -- see the table in docs/GLM5.3_DCU_优化报告_20261023.md section 4.
           start/sglang_start_glm53_opt.sh turns on the validated set.
           Never enable SGLANG_HCU_SKINNY_DRAFT_VP with --enable-dp-attention (NCCL hang).

  prefill  SGLANG_DSA_FAST_DEQUANT_PAGED=1  (prefill/p_opt.sh sets it; FAST_DEQUANT=0 disables)
           LIGHTOP_MQA_LOGITS_BETA_FP8=1    (env var only; BETA_FP8=0 disables)

Start scripts:
  single node DP8   start/sglang_start_glm53_opt.sh
  PD prefill node   prefill/p_opt.sh      (export NODE_IP=<this node's ib7 IP> first)
  PD decode node    pd/d_opt.sh
  PD router         pd/route.sh           (must run inside the container; the host has no python)

Undo: bash uninstall.sh [--decode-only|--prefill-only] [ROUND1=1 for the pristine image state]
EOF
