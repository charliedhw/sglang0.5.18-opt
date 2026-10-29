# GLM-5.3 FP8 推理优化（SGLang 0.5.18 dtk / Hygon DCU BW1100）

本仓库汇总 GLM-5.3 FP8（`GLM-5.3-Channel-FP8-w8a8-1015`）在 Hygon DCU（BW1100，gfx938）上的全部优化改动，包括：

- 对 SGLang 0.5.18（dtk 镜像）的补丁；
- 新增算子包 `hcu_ops`；
- 单机和 PD 分离两种部署的启动脚本；
- 测试与分析工具。

工作时间 2026-10-21 ~ 10-26，在 dcu4 / dcu5 两台 8 卡节点上完成。优化镜像 `glm5.3_v1023` 已包含本仓库的全部补丁。

| 阶段 | 日期 | 内容 | 详细文档 |
|---|---|---|---|
| 第一轮 | 10-21 | 启动参数调整，bf16 GEMM 改走 NN 布局 | `docs/GLM5.3_opt_README_20261021.md` |
| 第二轮 | 10-22 ~ 10-23 | 新建 `hcu_ops` 算子包，13 个 sglang 补丁，分批次 1、A、B 三批 | `docs/GLM5.3_opt2_README_20261023.md` |
| 合并报告 | 10-23 | 第一轮 + 第二轮完整报告：开关表、A/B 数据、精度 | `docs/GLM5.3_DCU_优化报告_20261023.md` |
| PD 分离 | 10-25 ~ 10-26 | P（CP8）和 D（DP8）启动脚本；P 侧改 64K chunk 并加静态 EPLB | `docs/GLM5.3_PD分离测试与优化报告_20261026.md` |

## 1. 目录结构

| 目录 | 内容 |
|---|---|
| `patches/` | 安装与回滚脚本：`apply_all.sh`、`verify_all.sh`、`restore_all.sh`<br>`FILES`：被修改的 17 个 sglang 文件<br>13 个 `patch_*.py`<br>`round1/`：第一轮的 2 个完整文件，以及镜像原始文件的 md5 |
| `diffs/` | 每个被修改文件相对镜像原始文件的统一 diff，分 `round1/`、`round2/`，路径相对 `sglang/srt/` |
| `hcu_ops/` | 新增算子包<br>源码：`csrc/*.cu`、`bind.cpp`、`setup.py`、`build_inc.sh`<br>Python 包：`hcu_ops/__init__.py`、`skinny_cfg.json`。不含预编译 `_C.so`，需在 gfx938 + DTK 环境下编译<br>`tests/`：逐 bit 单测与计时；`tune/`：skinny GEMM / bmm 调优脚本 |
| `start/` | 单机 DP8 启动脚本 `sglang_start_glm53_opt.sh`，DeepEP normal 配置 `deepep_cfg40.json` |
| `pd/` | PD 分离：`p_opt.sh`（prefill，CP8）、`d_opt.sh`（decode，DP8/TP8）<br>`eplb/rand_c64k.pt`：静态 EPLB 专家分布<br>`work/`：重启、TTFT 探针、长上下文 bench、对比、微基准脚本 |
| `tools/` | profiling 分析（`kagg.py`、`tracediff.py`、`kseq.py`、`kstat.py`）、A/B（`ab_bench.sh`、`ab_env.sh`）、排除 prefill 的步时统计（`ana_clean.py`）、greedy 一致性检查（`greedy_chk.py`）等 |
| `eval/` | gsm8k 评测：`drive_gsm8k.sh` 在宿主机运行，`run_gsm8k.sh` 在 evalscope 容器内运行 |
| `skill/dcu-model-opt/` | DCU 模型优化 skill：流程、工具、踩坑清单、GLM5.3 案例 |
| `docs/` | 上表中的各份报告 |

## 2. 代码改动汇总

### 2.1 sglang 补丁（路径相对 `sglang/srt/`）

每项都有独立的环境变量开关，设为 0 即走原实现。

| 开关 | 补丁 | 修改的文件 | 作用 | 数值 |
|---|---|---|---|---|
| `SGLANG_HCU_LM_HEAD_NN` | round1 | `layers/logits_processor.py` | lm_head 权重重排为 NN 布局（rocBLAS NN GEMM） | 布局等价 |
| `SGLANG_HCU_LINEAR_NN` | round1 | `layers/quantization/unquant.py` | bf16 线性层（N、K ≥ 1024）的权重重排为 NN 布局 | 布局等价 |
| `SGLANG_HCU_FAST_MASK_TOPK` | `patch_topk.py` | `layers/attention/dsa/dsa_indexer_metadata.py` | DSA topk-2048 + page table 变换，用 split 直方图实现 | 选出集合一致 |
| `SGLANG_HCU_SKINNY_GEMM` | `patch_skinny.py` | `layers/quantization/fp8_utils.py`、`layers/linear.py` | decode 小 M 的 fp8 dense GEMM 走 skinny GEMM，按调优表 `skinny_cfg.json` 选择 | 与 hipBLASLt 同量级 |
| `SGLANG_HCU_FUSED_ROPE` | `patch_rope.py` | `layers/rotary_embedding/base.py` | rope 每层 8 次 launch 合成 1 次 | 逐 bit |
| `SGLANG_HCU_FUSED_GATE` | `patch_moe.py` | `layers/moe/topk.py` | MoE gate + DeepEP id 后处理 | 逐 bit |
| `SGLANG_HCU_FUSED_ACT` | `patch_moe.py` | `layers/moe/ep_moe/layer.py` | MoE masked SiLU×mul + fp8 量化 | 逐 bit |
| `SGLANG_HCU_SKINNY_BMM` | `patch_bmm.py`、`patch_mla_rocm_bmm.py` | `forward_mla.py`、`forward_mla_rocm.py` | MLA absorb bmm（w_kc / w_vc） | ≤ 1.25 倍 torch 误差 |
| `SGLANG_HCU_SAMPLER_WARMUP` | `patch_sampler_warmup.py` | `layers/sampler.py` | 启动时预热 T>0 的采样 kernel，规避 "no kernel image" | 不影响计算 |
| `SGLANG_HCU_MLA_QCAT` | `patch_qcat.py` | `forward_mla_rocm.py`、`layers/attention/dsa_backend.py` | absorb bmm 和 q rope 直接写进同一个 q 缓冲，去掉 `concat_mla_absorb_q` | 逐 bit |
| `SGLANG_HCU_STRIDED_RMS` | `patch_fuse2.py` | `layers/layernorm.py` 等 | q_a / kv_a 的 RMSNorm 直接作用于 strided 视图，去掉 2 次 `.contiguous()` | 逐 bit |
| `SGLANG_HCU_QUANT_REUSE` | `patch_fuse2.py` | `compressed_tensors_w8a8_fp8.py`、`models/deepseek_v2.py`、`dsa_indexer.py` | 同一份 fp8 量化供多个下游 GEMM 共用 | 逐 bit |
| `SGLANG_HCU_SKINNY_LM_HEAD` | `patch_lm_head_nn.py` | `layers/logits_processor.py`、`speculative/draft_lm_head_vp.py` | verify / draft extend 的 lm_head（M ≤ 64）走 skinny NN GEMM | bf16 舍入级 |
| `SGLANG_HCU_NORM_QUANT` | `patch_fuse3.py` | `layernorm.py`、`compressed_tensors_w8a8_fp8.py`、`deepseek_v2.py` | fused_add_rms_norm + per-token fp8 量化 | 逐 bit |
| `SGLANG_HCU_SILU_QUANT` | `patch_fuse3.py` | 同上 | dense / shared expert 的 silu×mul + fp8 量化 | 逐 bit |
| `SGLANG_HCU_MLA_KV_FUSED` | `patch_kvfuse.py` | `forward_mla_rocm.py` | kv_a RMSNorm + q/k rope + fp8 KV cache 写入，合成 1 个 kernel | 逐 bit |
| `SGLANG_HCU_MQA_LOGITS` | `patch_mqa.py` | `layers/attention/dsa/dsa_indexer.py` | DSA indexer logits 改用 `hcu_ops.mqa_logits`（每个 CTA 处理 3 行，每个请求 64 个 CTA） | 逐 bit |
| `SGLANG_HCU_MQA_HIST` | `patch_mqa.py` | 同上 | 把 topk 的第 1 遍直方图融进 logits kernel（仅 G=1 且 ≤ 8 行） | 逐 bit |
| `SGLANG_HCU_SKINNY_DRAFT_VP` | `patch_lm_head_nn.py` | `speculative/draft_lm_head_vp.py` | draft lm_head VP 按捕获行数裁剪。**dp-attention 下会导致 NCCL hang，固定为 0** | — |

两个 round1 开关在代码里默认为 1，其余开关代码默认为 0；启动脚本把这些开关（除 `SKINNY_DRAFT_VP` 外）全部显式打开。

补丁按固定顺序叠加，同一文件被多个补丁修改：topk → skinny → rope → moe → bmm → mla_rocm_bmm → sampler_warmup → qcat → fuse2 → fuse3 → kvfuse → mqa → lm_head_nn。每个文件的最终改动见 `diffs/`。

### 2.2 hcu_ops 算子（`hcu_ops/csrc/`）

| 文件 | 算子 |
|---|---|
| `sparse_mask_topk.cu` | DSA topk-2048：12/12/8 位 3 遍直方图加 1 遍压缩，每行切到最多 8 个 CTA，兼容 CUDA graph |
| `skinny_gemm.cu` | 小 M fp8 dense GEMM（mmac 16x16x32，按 `skinny_cfg.json` 选择 tile），以及 MLA absorb bmm（`bmm_skinny`） |
| `skinny_nn.cu` | lm_head 小 M NN GEMM |
| `rope.cu` | 多 head rope 一次 launch；`rope_qout` 把 q rope 直接写进 absorb q 缓冲（qcat） |
| `moe_ops.cu` | MoE gate topk + DeepEP id 后处理，masked SiLU×mul + fp8 量化 |
| `norm_quant.cu` | fused_add_rms_norm + fp8 量化、silu×mul + fp8 量化 |
| `mla_kv.cu` | kv_a RMSNorm + q/k rope + fp8 KV cache 写入融合（`rope_qout_kv_store`） |
| `mqa_hist.cu` | indexer MQA logits，以及融合的 topk 第 1 遍直方图 |

strided RMSNorm 和量化复用只改了补丁层的调用方式，没有新增 kernel。

除 GEMM 类以外，所有 kernel 都与原实现逐 bit 一致（`hcu_ops/tests/` 中有对应单测）。

### 2.3 启动参数

- **单机 DP8（`start/sglang_start_glm53_opt.sh`）**
  - `--max-running-requests` 32 → 256，`--mem-fraction-static` 0.89；
  - DeepEP LL 每个 rank 256 token，DeepEP normal 40 SM（`deepep_cfg40.json`）；
  - `--chunked-prefill-size 16384`，开启 `--enable-prefill-delayer`；
  - 2.1 节的全部 hcu 开关。
- **PD 分离 P（`pd/p_opt.sh`，dcu4）**
  - CP8 prefill，打开 hcu 开关。prefill 路径上实际生效的只有 rope、LINEAR_NN、FUSED_GATE、SILU_QUANT、NORM_QUANT 和 sampler warmup。
  - 新增：`--chunked-prefill-size 65536`（原 32768），以及静态 EPLB `--init-expert-location pd/eplb/rand_c64k.pt --ep-num-redundant-experts 8`。
  - 回到旧配置：`CHUNK=32768 EPLB_INIT= bash p_opt.sh`。
- **PD 分离 D（`pd/d_opt.sh`，dcu5）**
  - 默认 DP8 dp-attention + EP8 DeepEP-LL，mem 0.88，每个 DP rank 约 135K KV，能放下 128K 请求。
  - 打开全部 decode 开关。
  - `D_MODE=tp` 为 TP8 备选。

PD 分离阶段只修改了启动参数，没有新增代码补丁。

## 3. 效果

### 3.1 单机 DP8

| 项目 | 结果 |
|---|---|
| 第一轮（参数），旧权重 | 1K/1K c128 输出吞吐 667 → 1808 tok/s（+171%），TTFT 102.7 s → 4.1 s<br>3.5K/1.5K c128 541 → 1574 tok/s（+191%） |
| 第一轮（NN 布局） | lm_head M=96：40.9 ms → 1.6 ms，ITL 约 −5% |
| 第二轮（算子），-1015 权重，64K PC90% decode 步时 | c1 81.6 → 62.7 ms（**−23.2%**），c8 −19.5%，c16 −13.6%，c32 −11.1%<br>c1 输出吞吐 59.7 → 74.9 tok/s（+25.5%） |
| 精度 | -1015 + 最终脚本：gsm8k 97.27%（1319 题），与优化前的差异在统计误差内 |

### 3.2 PD 分离（1P dcu4 + 1D dcu5）

| 对比 | 结果 |
|---|---|
| 优化镜像 vs 10/18 旧镜像 | 对话 c128 ITL −34%，吞吐 +25%<br>对话4k c64 吞吐 +101%<br>64K PC90 c32 TTFT −57%，吞吐 +80%<br>128K 从 12 项全部失败变为全部通过 |
| P 侧 64K chunk + 静态 EPLB vs 32K chunk | 单请求 128K TTFT −12%<br>64K PC50 TTFT −14% ~ −36%，64K PC90 −3% ~ −25%<br>128K c1~c4 −7% ~ −11%<br>128K c≥8 持平：瓶颈在 D 侧 KV 容量 |

需要注意的点：16K c64 TTFT +79%，尚未 A/B 确认原因。详见 PD 报告 3.4 节和第 8 节。

## 4. 使用

以下命令均在容器内执行，执行时不能有 sglang 服务在运行。镜像为 SGLang 0.5.18 dtk 版。

```bash
bash patches/apply_all.sh            # 编译 hcu_ops（无 _C.so 时自动编译）+ 安装 + 第一轮 + 第二轮补丁链（可重复执行）
bash patches/apply_all.sh --build    # 强制从源码重新编译 hcu_ops
bash patches/verify_all.sh           # 核对补丁链，应输出 "verify fail=0"
bash patches/restore_all.sh          # 回到第一轮代码；ROUND1=1 时回到镜像原始代码

bash start/sglang_start_glm53_opt.sh                 # 单机 DP8
SGLANG_HCU_MQA_LOGITS=0 bash start/sglang_start_glm53_opt.sh   # 临时关闭某一项
```

PD 分离按 P → D → router 的顺序启动，见 PD 报告第 2 节。`p_opt.sh` / `d_opt.sh` 启动前需要 `export NODE_IP=<本节点 ib7 IP>`（用于 `--dist-init-addr`）。

## 5. 未做 / 后续方向

- 按约定未做：W4A8 MoE、TBO（two-batch overlap）、shared expert fp8。
- 后续方向：
  - 扩 D 节点；
  - 2P + cache_aware 路由；
  - 用真实流量重录 EPLB 分布；
  - sparse prefill attention kernel；
  - 修复 sglang `load_aware` 的 bug；
  - prefill 使用 128K chunk。

  详见 PD 报告第 8 节。
