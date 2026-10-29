#!/bin/bash
# GLM-5.3 FP8 (-1015 weights) PD disaggregation - DECODE node (dcu5), optimized image glm5.3 (glm5.3_v1023).
# Based on d_debug.sh.  D_MODE selects the attention layout:
#   dp (default)  the d_debug.sh layout, DP8 dp-attention + EP8: ~135K KV per DP rank at mem 0.88 (one 128K
#                 request per rank, 8 per node), much faster decode (TP8 is +61% ITL at c64).
#   tp            TP8 + EP8 DeepEP-LL: one KV pool for the whole node (MLA latent replicated per rank), so
#                 128K+ requests fit; the 8 TP ranks share every request. Use for the 128K scenarios.
# hcu_ops switches as in sglang_start_glm53_opt.sh; any can be overridden per run, e.g.
#   SGLANG_HCU_MQA_LOGITS=0 D_MODE=tp bash d_opt.sh
D=$(cd "$(dirname "$0")" && pwd)
D_MODE=${D_MODE:-dp}

# -------------------- network / RDMA (from d_debug.sh) --------------------
export NCCL_SOCKET_IFNAME=ib7
export GLOO_SOCKET_IFNAME=ib7
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1,shca_4:1,shca_5:1,shca_6:1,shca_7:1
export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1,shca_4:1,shca_5:1,shca_6:1,shca_7:1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3,shca_4,shca_5,shca_6,shca_7
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=0
export ROCSHMEM_IPC_MNVL=1
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="shca_0,shca_1,shca_2,shca_3,shca_4,shca_5,shca_6,shca_7"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

# -------------------- runtime --------------------
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export GLIBC_TUNABLES="glibc.rtld.optional_static_tls=0x40000"
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export USE_DCU_CUSTOM_ALLREDUCE=${USE_DCU_CUSTOM_ALLREDUCE:-1}
export HIP_BUFFER_EXTRA_SIZE=0
export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=1
export GPU_MAX_HW_QUEUES=3

# -------------------- sglang / LightOp / FlashMLA --------------------
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1
export LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA=3
export FLASH_MLA_SPARSE_DECODE_DISABLE_SPLITKV=0
export FLASH_MLA_SPARSE_PAGE64_SPECIALIZE=1
export FLASH_MLA_SPARSE_DECODE_NUM_SM_PARTS=${FLASH_MLA_SPARSE_DECODE_NUM_SM_PARTS:-32}
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1

# -------------------- hcu_ops optimization switches (0 = original kernel) --------------------
export SGLANG_HCU_LM_HEAD_NN=${SGLANG_HCU_LM_HEAD_NN:-1}
export SGLANG_HCU_LINEAR_NN=${SGLANG_HCU_LINEAR_NN:-1}
export SGLANG_HCU_FAST_MASK_TOPK=${SGLANG_HCU_FAST_MASK_TOPK:-1}
export SGLANG_HCU_SKINNY_GEMM=${SGLANG_HCU_SKINNY_GEMM:-1}
export SGLANG_HCU_FUSED_ROPE=${SGLANG_HCU_FUSED_ROPE:-1}
export SGLANG_HCU_FUSED_GATE=${SGLANG_HCU_FUSED_GATE:-1}
export SGLANG_HCU_FUSED_ACT=${SGLANG_HCU_FUSED_ACT:-1}
export SGLANG_HCU_SKINNY_BMM=${SGLANG_HCU_SKINNY_BMM:-1}
export SGLANG_HCU_SAMPLER_WARMUP=${SGLANG_HCU_SAMPLER_WARMUP:-1}
export SGLANG_HCU_MLA_QCAT=${SGLANG_HCU_MLA_QCAT:-1}
export SGLANG_HCU_STRIDED_RMS=${SGLANG_HCU_STRIDED_RMS:-1}
export SGLANG_HCU_QUANT_REUSE=${SGLANG_HCU_QUANT_REUSE:-1}
export SGLANG_HCU_NORM_QUANT=${SGLANG_HCU_NORM_QUANT:-1}
export SGLANG_HCU_SILU_QUANT=${SGLANG_HCU_SILU_QUANT:-1}
export SGLANG_HCU_MLA_KV_FUSED=${SGLANG_HCU_MLA_KV_FUSED:-1}
export SGLANG_HCU_MQA_LOGITS=${SGLANG_HCU_MQA_LOGITS:-1}
export SGLANG_HCU_MQA_HIST=${SGLANG_HCU_MQA_HIST:-1}
export SGLANG_HCU_SKINNY_LM_HEAD=${SGLANG_HCU_SKINNY_LM_HEAD:-1}
# never with --enable-dp-attention (NCCL hang); TP8 has no draft lm_head VP
export SGLANG_HCU_SKINNY_DRAFT_VP=0

MODEL_PATH=${MODEL_PATH:-/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015}
time=$(date "+%m%d-%H%M")

if [ "$D_MODE" = dp ]; then
    # leave room for eager (non-graph) fallback steps: the masked MoE buffer is 768MB
    MEM=${MEM:-0.88}
    MAX_RUN=${MAX_RUN:-128}
    PAR="--tp-size 8 --dp-size 8 --ep-size 8 --moe-dense-tp-size 1 --enable-dp-attention --enable-dp-lm-head"
    PAR+=" --speculative-draft-lm-head-vp-size 8 --cuda-graph-max-bs ${CG_BS:-16}"
else
    # TP8 (for >100K contexts only; slower than DP8 at c>=8): 91GB weights/card.
    # 0.9 OOMs in bs64 graph capture; 0.85 OOMs on eager steps once bs > graph max,
    # so keep max-running = graph bs and some headroom
    MEM=${MEM:-0.82}
    MAX_RUN=${MAX_RUN:-${CG_BS:-64}}
    PAR="--tp-size 8 --ep-size 8 --cuda-graph-max-bs ${CG_BS:-64}"
fi

sglang serve \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3,shca_4,shca_5,shca_6,shca_7 \
    --dist-init-addr ${NODE_IP:?export NODE_IP=<this node ib7 IP>}:5123 --nnodes 1 --node-rank 0 \
    --numa-node 3 1 1 0 7 5 5 4 \
    --model-path $MODEL_PATH \
    --served-model-name glm5.3-fp8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30002 \
    --tokenizer-worker-num 16 \
    $PAR \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --dsa-prefill-backend flashmla_sparse \
    --dsa-decode-backend flashmla_kv \
    --context-length 1048576 \
    --dtype bfloat16 \
    --dist-timeout 10000 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --watchdog-timeout 3600 \
    --page-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static $MEM \
    --chunked-prefill-size -1 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --max-running-requests $MAX_RUN \
    $D_EXTRA \
    2>&1 | tee $D/logs/glm53-decode-opt-${D_MODE}_$time.log
