#!/bin/bash
# GLM-5.3 FP8 (Channel-FP8 w8a8, -1015 weights) on 8x BW1100 - optimized DP8 + EP8 + EAGLE start script.
# Needs the optimization patches in the container (bash /opt/glm5.3_opt/patches/apply_all.sh once;
# glm5.3-pd as of 2026-10-23 already has them, check with verify_all.sh).  Every hcu_ops switch below is
# "${VAR:-default}", so one can be turned off for a run without editing this file, e.g.
#   SGLANG_HCU_MQA_LOGITS=0 bash sglang_start_glm53_opt.sh
# Report: GLM5.3_DCU_优化报告_20261023.md (switch table: section 4).
D=$(cd "$(dirname "$0")" && pwd)

# Python runtime
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000

# LightOp MQA and paired sparse-mask top-k
export LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA=3
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=1

export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1

# GEMM backends and Decode tuning
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1

# Network and collectives
export NCCL_IB_DISABLE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

# HCU runtime and graph dispatch
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

# DeepEP and ROCSHMEM
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

# MTP, KV allocation and scheduler
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

# round 1: bf16 lm_head / linear in NN layout (default on in the patched files)
export SGLANG_HCU_LM_HEAD_NN=${SGLANG_HCU_LM_HEAD_NN:-1}
export SGLANG_HCU_LINEAR_NN=${SGLANG_HCU_LINEAR_NN:-1}

# round 2, batch 1: hcu_ops kernels; 0 = original kernel
export SGLANG_HCU_FAST_MASK_TOPK=${SGLANG_HCU_FAST_MASK_TOPK:-1}   # DSA indexer topk-2048, split histogram
export SGLANG_HCU_SKINNY_GEMM=${SGLANG_HCU_SKINNY_GEMM:-1}         # decode fp8 dense GEMM (q_b / o_proj / idx wq_b / dense gate_up)
export SGLANG_HCU_FUSED_ROPE=${SGLANG_HCU_FUSED_ROPE:-1}           # rope: 8 launches -> 1
export SGLANG_HCU_FUSED_GATE=${SGLANG_HCU_FUSED_GATE:-1}           # MoE gate + DeepEP id postprocess
export SGLANG_HCU_FUSED_ACT=${SGLANG_HCU_FUSED_ACT:-1}             # masked SiLU*mul + fp8 quant over masked_m rows
export SGLANG_HCU_SKINNY_BMM=${SGLANG_HCU_SKINNY_BMM:-1}           # MLA absorb bmm (w_kc / w_vc)
export SGLANG_HCU_SAMPLER_WARMUP=${SGLANG_HCU_SAMPLER_WARMUP:-1}   # T>0 sampling kernels warmed at startup ("no kernel image" guard)

# round 2, batch 2 (A): MLA q path, all bit-exact
export SGLANG_HCU_MLA_QCAT=${SGLANG_HCU_MLA_QCAT:-1}               # absorb bmm + rope write q_nope/q_pe in place (no concat_mla_absorb_q)
export SGLANG_HCU_STRIDED_RMS=${SGLANG_HCU_STRIDED_RMS:-1}         # q_a / kv_a RMSNorm on strided views (no .contiguous copies)
export SGLANG_HCU_QUANT_REUSE=${SGLANG_HCU_QUANT_REUSE:-1}         # one fp8 quant of q_lora feeds q_b and indexer wq_b

# round 2, batch 3 (B): fusions, all bit-exact
export SGLANG_HCU_NORM_QUANT=${SGLANG_HCU_NORM_QUANT:-1}           # fused_add_rms_norm + per-token fp8 quant (M <= NORM_QUANT_MAX_M)
export SGLANG_HCU_SILU_QUANT=${SGLANG_HCU_SILU_QUANT:-1}           # dense / shared-expert silu*mul + fp8 quant
export SGLANG_HCU_MLA_KV_FUSED=${SGLANG_HCU_MLA_KV_FUSED:-1}       # kv_a norm + k_pe rope + fp8 KV-cache store in one kernel
export SGLANG_HCU_MQA_LOGITS=${SGLANG_HCU_MQA_LOGITS:-1}           # DSA indexer logits: hcu_ops.mqa_logits (rpc 3, 64 CTAs/request)
export SGLANG_HCU_MQA_HIST=${SGLANG_HCU_MQA_HIST:-1}               # topk histogram pass 1 fused into the logits kernel (single rows, <= 8 rows)
export SGLANG_HCU_SKINNY_LM_HEAD=${SGLANG_HCU_SKINNY_LM_HEAD:-1}   # target verify / draft extend lm_head (M <= 64): skinny NN GEMM
# never with --enable-dp-attention: NCCL hang (DP ranks gather different sizes)
export SGLANG_HCU_SKINNY_DRAFT_VP=0

host=$(hostname -I | awk '{print $1}')
time=$(date "+%m%d-%H%M")
MODEL_PATH=${MODEL_PATH:-/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015}

sglang serve \
    --model-path $MODEL_PATH \
    --random-seed 536624121 \
    --served-model-name glm5.3-fp8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --tokenizer-worker-num 16 \
    --dist-init-addr $host:5001 \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 8 \
    --dp-size 8 \
    --ep-size 8 \
    --moe-dense-tp-size 1 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --deepep-config $D/deepep_cfg40.json \
    --enable-dp-lm-head \
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
    --mem-fraction-static 0.89 \
    --chunked-prefill-size 16384 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --cuda-graph-max-bs 32 \
    --max-running-requests 256 \
    --speculative-draft-lm-head-vp-size 8 \
    --enable-prefill-delayer \
    --prefill-delayer-token-usage-low-watermark 0.05 \
    2>&1 | tee glm53-opt_$time.log
