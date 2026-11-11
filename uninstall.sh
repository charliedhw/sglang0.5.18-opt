#!/bin/bash
# Undo what install.sh installed, in reverse order.
#
#   prefill (2026-11)  Restores dsa/dequant_k_cache.py from its *.bak_fastdequant backup.
#   decode  (2026-10)  Returns every patched file to its round-2 base, i.e. the round-1 state.
#                      With ROUND1=1 logits_processor.py and unquant.py also go back to the
#                      pristine image versions, leaving sglang exactly as the image shipped it.
#
# Nothing is deleted: the current files are kept as *.pre_restore_<time> and every backup stays,
# so install.sh can re-apply afterwards.  hcu_ops stays in site-packages -- after the restore no
# code imports it, and leaving it avoids a rebuild if the patches go back on.
#
# Usage, inside the container, with no sglang server running:
#     bash uninstall.sh                      # both rounds, back to the round-1 state
#     ROUND1=1 bash uninstall.sh             # both rounds, back to the pristine image
#     bash uninstall.sh --prefill-only       # 2026-11 only
#     bash uninstall.sh --decode-only        # 2026-10 only
set -eo pipefail

D=$(cd "$(dirname "$0")" && pwd)
DO_DECODE=1
DO_PREFILL=1

for arg in "$@"; do
    case "$arg" in
        --decode-only)  DO_PREFILL=0 ;;
        --prefill-only) DO_DECODE=0 ;;
        -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
        *)              echo "unknown option: $arg"; exit 1 ;;
    esac
done

if pgrep -f "sglang serve|sglang.launch_server" > /dev/null; then
    echo "a sglang server is running -- stop it first"
    exit 1
fi

if [ "$DO_PREFILL" = 1 ]; then
    echo "=== prefill round: restoring dequant_k_cache.py ==="
    python3 "$D/prefill/patches/patch_fast_dequant.py" --revert
fi

if [ "$DO_DECODE" = 1 ]; then
    echo
    echo "=== decode round: restoring the sglang patch chain ==="
    bash "$D/patches/restore_all.sh"
fi

echo
python3 - <<'PYEOF'
import sysconfig, os
p = os.path.join(sysconfig.get_paths()["purelib"], "sglang")
hits = []
for root, _, files in os.walk(p):
    for f in files:
        if f.endswith(".py"):
            try:
                s = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if "_dequantize_k_cache_paged_v2_kernel" in s:
                hits.append(os.path.relpath(os.path.join(root, f), p))
print("prefill patch still present in:" if hits else "prefill patch removed", *hits)
PYEOF

cat <<'EOF'

=== done ===

Remember to drop the switches from whatever start script you use, otherwise the server will set
environment variables for code that is no longer patched (harmless, but misleading):
  SGLANG_DSA_FAST_DEQUANT_PAGED   (prefill/p_opt.sh: FAST_DEQUANT=0 also does it per run)
  SGLANG_HCU_*                    (start/sglang_start_glm53_opt.sh)

LIGHTOP_MQA_LOGITS_BETA_FP8 needs no uninstall -- it only selects an implementation that already
ships inside lightop.
EOF
