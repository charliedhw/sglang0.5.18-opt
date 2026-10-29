#!/bin/bash
# GLM-5.3 FP8 (-1015 weights) PD disaggregation - PREFILL node (dcu4), optimized image glm5.3 (glm5.3_v1023).
# Based on p_debug.sh: CP8 prefill (attn-cp-size 8, DeepEP normal, no cuda graph) - one KV pool sharded by
# sequence, so 128K+ prompts fit.  Added: -1015 weights, hcu_ops switches of sglang_start_glm53_opt.sh
# (only rope / NORM_QUANT / SILU_QUANT / FUSED_GATE / NN layout act on prefill; the decode-only ones are inert
# here), runtime env, 64K chunks + static EPLB (128K TTFT -12%).  Any switch can be overridden per run,
# e.g. SGLANG_HCU_NORM_QUANT=0 bash p_opt.sh;  CHUNK=32768 EPLB_INIT= bash p_opt.sh = previous layout
D=$(cd "$(dirname "$0")" && pwd)

# -------------------- network / RDMA (from p_debug.sh) --------------------
export NCCL_SOCKET_IFNAME=ib7
export GLOO_SOCKET_IFNAME=ib7
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1,shca_4:1,shca_5:1,shca_6:1,shca_7:1
export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1,shca_4:1,shca_5:1,shca_6:1,shca_7:1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3,shca_4,shca_5,shca_6,shca_7
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ROCSHMEM_IPC_MNVL=1
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="shca_0,shca_1,shca_2,shca_3,shca_4,shca_5,shca_6,shca_7"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0

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

# -------------------- sglang / LightOp --------------------
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
export SGLANG_HCU_SKINNY_DRAFT_VP=0

MODEL_PATH=${MODEL_PATH:-/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015}
DEEPEP_CFG=${DEEPEP_CFG:-/data/home/daihw/configs/deepep.json}
# 64K chunks: 128K prompt in 2 chunks instead of 4 (131072 OOMs at mem 0.86)
CHUNK=${CHUNK:-65536}
MEM=${MEM:-0.86}
# Static EPLB: 8 redundant experts placed from a recorded expert-load distribution (cuts the DeepEP
# combine wait on the busiest rank).  The default file was recorded on the random-token bench load;
# for real traffic record a new one (--expert-distribution-recorder-mode stat, then
# /start_expert_distribution_record, /stop_..., /dump_... -> /tmp/*.pt).  EPLB_INIT= disables it.
EPLB_INIT=${EPLB_INIT-$D/eplb/rand_c64k.pt}
EPLB_ARGS=""
[ -n "$EPLB_INIT" ] && EPLB_ARGS="--init-expert-location $EPLB_INIT --ep-num-redundant-experts ${EPLB_REDUNDANT:-8}"
time=$(date "+%m%d-%H%M")

sglang serve \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3,shca_4,shca_5,shca_6,shca_7 \
    --dist-init-addr ${NODE_IP:?export NODE_IP=<this node ib7 IP>}:5123 --nnodes 1 --node-rank 0 \
    --attn-cp-size 8 --cp-strategy interleave --enable-dsa-cache-layer-split --disable-cuda-graph \
    --nsa-prefill-cp-mode round-robin-split --enable-nsa-prefill-context-parallel \
    --numa-node 3 1 1 0 7 5 5 4 \
    --model-path $MODEL_PATH \
    --served-model-name glm5.3-fp8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30001 \
    --tokenizer-worker-num 16 \
    --tp-size 8 \
    --ep-size 8 \
    --moe-dense-tp-size 1 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --deepep-config $DEEPEP_CFG \
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
    --chunked-prefill-size $CHUNK \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --max-running-requests 128 \
    $EPLB_ARGS \
    $P_EXTRA \
    2>&1 | tee $D/logs/glm53-prefill-opt_$time.log
