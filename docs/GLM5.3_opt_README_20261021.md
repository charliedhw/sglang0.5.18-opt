# GLM-5.3 FP8 SGLang 推理性能优化说明（dcu4，2026-10-21）

## 1. 概述

| 项目 | 内容 |
|---|---|
| 服务器 | dcu4，8 × HCU BW1100（gfx938） |
| 容器 | `glm5.3-pd`（SGLang 0.5.18 dtk，torch 2.11） |
| 模型 | `/data/model/GLM/GLM-5.3-Channel-FP8-w8a8`（served name `glm5.3-fp8`） |
| 部署形态 | TP8 / DP8 / EP8，dp-attention，DeepEP auto，EAGLE MTP（5 步 / 6 draft token，spec v2），DSA 稀疏 MLA + fp8 KV |
| 原始启动脚本 | `sglang_start.sh_1017`（未修改） |
| 测试脚本 | `bench/vllm_bench-pc.sh`（未修改）：random 数据集，1024/1024 与 3584/1536 两个场景，并发 1–128，num_prompts = 2×并发，temperature 0.01，seed 10801 |
| 最终启动脚本 | `sglang_start_opt_final.sh` + `deepep_cfg40.json` |

**结论：**
- **高并发：**
  - 1K/1K 场景，128 并发时输出吞吐从 667 提升到 1808 tok/s（2.7 倍），64 并发从 713 提升到 1225 tok/s。
  - 3584/1536 场景，128 并发时从 541 提升到 1574 tok/s（2.9 倍），64 并发从 553 提升到 1155 tok/s。
  - 首 token 延迟（TTFT）大幅下降：128 并发时 1K 场景从 102.7s 降到 4.1s，4K 场景从 204.7s 降到 13.8s。
- **低并发（≤16）：** 与原始配置持平。表中个别点的高低是 MTP 接受率随机波动造成的，见 7.2 节。

优化分两部分：
1. **启动参数和调度。** 这是收益的主体，主要解除了原配置最多 32 个运行请求的限制。
2. **基于 profiling 的算子层优化。**
   - 两个框架代码补丁，让 bf16 GEMM 走 NN 布局，避开 rocBLAS 的慢 kernel；
   - DeepEP 通信 kernel 的 SM 数配置。

本轮**没有修改算子实现，也没有做 kernel 融合**，这部分留作后续工作（见第 9 节）。

---

## 2. 交付文件

都在 `/data/home/daihw/scripts/glm5.3/` 下：

| 文件 | 说明 |
|---|---|
| `sglang_start_opt_final.sh` | **最终启动脚本**。在原脚本基础上改了 6 处（见 4.1 节），日志前缀为 `glm53-opt-final_` |
| `deepep_cfg40.json` | DeepEP normal 模式配置（40 SM），启动脚本通过 `--deepep-config` 引用这个绝对路径 |
| `GLM5.3_opt_README_20261021.md` | 本文档 |
| `opt_final_20261021/patches/` | 框架代码补丁：`*.diff`（统一 diff）、`*.patched`（改后的完整文件）、`apply_patches.sh`（在新容器中一键打补丁） |
| `opt_final_20261021/results/` | 原始配置和最终配置的完整压测日志、低并发 A/B 日志、实测时使用的启动脚本（`.as_tested`，与最终脚本仅 deepep 配置路径和日志名不同） |
| `opt_final_20261021/tools/` | profiling 解析和单算子微基准脚本及其输出（见第 6 节），以及低并发 A/B 压测脚本 `vllm_bench-c8.sh` |
| `opt/` | 全部试验过程文件：各轮启动脚本 `start_opt*.sh`、服务日志、压测日志、profiling trace（`opt/prof/dec`、`opt/prof/dec128`、`opt/prof/pre64`） |

---

## 3. 使用方法

```bash
# 宿主机上进入容器
docker exec -it glm5.3-pd bash
cd /data/home/daihw/scripts/glm5.3
nohup bash sglang_start_opt_final.sh > /dev/null 2>&1 &      # 日志写在 glm53-opt-final_<时间>.log

# 等待服务就绪（日志中出现 "The server is fired up"），之后可以查询
curl -s http://127.0.0.1:30000/v1/models

# 压测（原测试脚本，未修改）
cd bench && bash vllm_bench-pc.sh
```

**框架补丁开关：** 两个补丁默认开启，可在启动脚本里用环境变量关闭，不需要改代码：
```bash
export SGLANG_HCU_LM_HEAD_NN=0     # 关闭 lm_head NN 布局
export SGLANG_HCU_LINEAR_NN=0      # 关闭 bf16 线性层 NN 布局
```

---

## 4. 优化项明细

### 4.1 启动参数（`sglang_start_opt_final.sh` 与 `sglang_start.sh_1017` 的差异）

| 参数 | 原值 | 新值 | 作用 / 依据 |
|---|---|---|---|
| `--max-running-requests` | 32 | **256** | 原配置全系统最多 32 个请求同时运行（DP8 下每个 rank 4 个），64、128 并发时其余请求排队，TTFT 达到 33–205s。改为 256（每个 rank 32 个）后排队消失。**这是吞吐提升的主要来源。** |
| `--mem-fraction-static` | 0.86 | **0.89** | 增加 KV cache 容量，支撑 256 个运行请求 |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 128 | **256** | DeepEP 低延迟模式每个 rank 单次最多分发的 token 数。每个 rank 32 个请求 × 6 个 draft token = 192，必须调大 |
| `--chunked-prefill-size` | 32768 | **16384** | 每个 rank 的单次 prefill 从 4096 token 降到 2048 token，减少 prefill 对 decode 的阻塞；与 DeepEP 配置一起调优（见 4.2 节） |
| `--enable-prefill-delayer` | — | **开启** | 新请求的 prefill 延迟到可以合批时再执行，减少 prefill 打断 decode 的次数 |
| `--prefill-delayer-token-usage-low-watermark` | — | **0.05** | KV 占用低于 5% 时不延迟，避免低负载时白等 |
| `--deepep-config` | 默认 | **`deepep_cfg40.json`** | 见 4.2 节 |

`--cuda-graph-max-bs 32` 保持不变。它是每个 rank 的 batch 上限，和每个 rank 最多 32 个请求一致。

### 4.2 DeepEP 配置（`deepep_cfg40.json`）

```json
{"normal_dispatch": {"num_sms": 40, "num_max_nvl_chunked_send_tokens": 6, "num_max_nvl_chunked_recv_tokens": 256,
                     "num_max_rdma_chunked_send_tokens": 6, "num_max_rdma_chunked_recv_tokens": 128},
 "normal_combine":  {"num_sms": 40, ...同上}}
```

- **依据：** prefill profiling 显示 DeepEP normal 模式的 notify/dispatch/combine 合计约占 GPU 时间的 34%（见 6.3 节）。默认只给通信分配 24 个 SM。
- **单算子测试**（每个 rank 2048 token，即调整后的单次 prefill 规模）：

| SM 数 | dispatch | combine |
|---|---|---|
| 24（默认） | 1117µs | 1334µs |
| 32 | 900µs | 1212µs |
| **40** | **795µs** | **1175µs** |
| 48 | 765µs | 1304µs（combine 变差） |

  40 SM 时两项都比较好，所以选 40。chunk 参数（6/256，6/128）在 12/512 下没有更好的表现，因此保持默认。
- 这个配置只影响 prefill 用的 normal 模式。decode 用的低延迟模式不受影响。

### 4.3 框架代码补丁（算子层面：bf16 GEMM 改为 NN 布局）

修改了容器内 `/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/` 下的 2 个文件。**只改了权重的显存布局，没有修改任何 kernel，也没有改 .so 库**。原文件备份为 `*.bak_20261021`。

**问题：**
- 这个模型的 lm_head（154880×6144）、前 3 层 dense、indexer、eh_proj、shared experts、router gate 都是 bf16 未量化。
- 它们走 `F.linear(x, W)` 或 `torch.matmul(x, W.T)`，权重按 `[N, K]` 行连续存放，所以实际调用 rocBLAS 的 **NT 布局 GEMM**。
- profiling 发现，在 gfx938 上 rocBLAS 给大 N 的 bf16 NT GEMM 选了很差的 kernel（`Cijk_Alik_Bljk_BBH_MT32x32x4…` 等小 tile kernel），见 6.1 节。

**解决：** 在第一次调用时执行一次 `w.data = w.data.t().contiguous().t()`。
- 张量的逻辑形状和数值都不变，其它代码不受影响。
- 底层存储变成 `[K, N]` 连续，`weight.T` 就是连续张量，rocBLAS 走 NN GEMM。

| 补丁 | 文件 | 作用范围 | 开关（默认开启） |
|---|---|---|---|
| ① lm_head | `logits_processor.py`：新增 `_maybe_relayout_lm_head_weight()`，在 `_compute_lm_head` 的 `torch.matmul` 之前调用 | 目标模型 lm_head | `SGLANG_HCU_LM_HEAD_NN` |
| ② bf16 线性层 | `quantization/unquant.py`：新增 `_maybe_relayout_linear_weight_nn()`，在 `UnquantizedLinearMethod.apply()` 的 `F.linear` 之前调用 | 输出维度 N ≥ 1024 且输入维度 K ≥ 1024 的 bf16/fp16 二维权重 | `SGLANG_HCU_LINEAR_NN`，阈值 `SGLANG_HCU_LINEAR_NN_MIN_N`（默认 1024） |

**安全措施：**
- 每个权重只处理一次（lm_head 用全局开关，线性层用 `layer._hcu_nn_checked` 标记），之后零开销。
- 如果第一次调用发生在 CUDA graph 捕获中，就不移动存储（图里可能已经记录了旧地址）。实际上重排都发生在捕获前的 eager warmup 阶段，启动日志里每个 rank 会打印一行 `LM-head weight relaid out for NN GEMM: shape=(154880, 6144)`。
- N < 1024 的小权重（router gate、indexer 的 wk）在 NT 下更快，不做重排。
- FP8 量化层走另一条 GEMM 路径，不受影响。MTP 草稿模型的 lm_head 用 draft LM-head VP 路径，自己缓存权重分片，也不受影响。
- 启动时重排 lm_head 会临时多占约 1.9GB 显存，随后通过 `empty_cache()` 释放。

**效果：**
- 单算子：lm_head 在 M=96（128 并发时每个 rank 的 verify 规模）从 40.9ms 降到 1.6ms；dense o_proj、down、eh_proj 大约快 2 倍。详见 6.1 节。
- 端到端：
  - 只上补丁①时，1K 场景快速压测 c16 从 345 提升到 427 tok/s，c64 从 1008 提升到 1130 tok/s（+12%）；
  - 两个补丁同时开关做 A/B，decode 单步时间（ITL）从 107.6–108.9ms 降到 101–104ms（约 5%）。
- 精度：单算子对比中 NN 与 NT 结果的最大绝对误差 ≤0.031，属于 bf16 舍入范围。只改了内存布局，数学上等价。

**在新容器中打补丁 / 回滚：**
```bash
# 打补丁（同版本镜像；已打过会跳过；自动备份 .bak_20261021）
docker exec <容器> bash /data/home/daihw/scripts/glm5.3/opt_final_20261021/patches/apply_patches.sh

# 回滚
docker exec <容器> bash -c 'cd /usr/local/lib/python3.10/dist-packages/sglang/srt/layers && \
  cp logits_processor.py.bak_20261021 logits_processor.py && \
  cp quantization/unquant.py.bak_20261021 quantization/unquant.py'
```

**注意：** 补丁对容器内所有 SGLang 服务都生效，不只是 GLM-5.3。

---

## 5. 性能结果

完整跑 `bench/vllm_bench-pc.sh`。原始配置的日志是 `results/bench_base.log`，最终配置的是 `results/bench_final12.log`。

### 5.1 场景 1：输入 1024 / 输出 1024

| 并发 | 原始 输出 tok/s | 最终 输出 tok/s | 提升 | 原始 TTFT ms | 最终 TTFT ms | 原始 TPOT ms | 最终 TPOT ms |
|---|---|---|---|---|---|---|---|
| 1 | 56.62 | 54.68 | 持平* | 699 | 572 | 16.99 | 17.75 |
| 2 | 89.59 | 94.43 | +5% | 709 | 635 | 20.94 | 18.27 |
| 4 | 125.78 | 152.83 | +22% | 678 | 661 | 25.53 | 22.09 |
| 8 | 306.10 | 223.72 | 持平* | 832 | 731 | 23.51 | 25.62 |
| 16 | 410.95 | 387.09 | 持平* | 1363 | 1128 | 29.18 | 29.82 |
| 32 | 602.82 | **722.55** | **+20%** | 3319 | 2104 | 37.15 | 31.64 |
| 64 | 713.07 | **1225.38** | **+72%** | 33081 | 3191 | 39.18 | 35.86 |
| 128 | 666.84 | **1807.73** | **+171%** | 102683 | 4134 | 42.91 | 48.69 |

### 5.2 场景 2：输入 3584 / 输出 1536

| 并发 | 原始 输出 tok/s | 最终 输出 tok/s | 提升 | 原始 TTFT ms | 最终 TTFT ms | 原始 TPOT ms | 最终 TPOT ms |
|---|---|---|---|---|---|---|---|
| 1 | 66.30 | 51.35 | 持平* | 1738 | 1809 | 13.96 | 18.31 |
| 2 | 120.93 | 124.18 | +3% | 1816 | 1945 | 14.94 | 14.50 |
| 4 | 210.03 | 220.56 | +5% | 1812 | 1844 | 17.18 | 16.43 |
| 8 | 280.14 | 284.68 | +2% | 2012 | 2707 | 22.15 | 21.81 |
| 16 | 396.70 | 484.88 | +22% | 3900 | 3484 | 29.93 | 25.17 |
| 32 | 540.96 | **811.30** | **+50%** | 12187 | 5815 | 38.98 | 29.59 |
| 64 | 552.77 | **1154.81** | **+109%** | 72610 | 8480 | 49.34 | 40.87 |
| 128 | 540.66 | **1573.94** | **+191%** | 204679 | 13831 | 53.49 | 59.46 |

\* 标“持平”的点见 7.2 节：同一天对原始配置做了 A/B 复测，结果与最终配置在同一波动区间内。

另外，128 并发时最终配置的 TPOT 略高于原始配置（1K：42.9→48.7ms，4K：53.5→59.5ms）。原因是原始配置同一时刻只有 32 个请求在 decode，其余请求在排队，所以单个请求的 TPOT 低，但 TTFT 极长。最终配置所有请求同时 decode，单步 batch 更大。从用户感知的端到端延迟（E2EL）看，最终配置大幅更优：1K 场景 146.6s → 53.9s，4K 场景 286.8s → 105.1s。

### 5.3 各轮试验汇总（输出 tok/s，“q” 表示只跑了部分并发点的快速压测）

| 轮次 | 相对原始配置的改动 | 1K c16 | 1K c64 | 1K c128 | 4K c32 | 4K c128 | 结论 |
|---|---|---|---|---|---|---|---|
| base | 原始 | 411 | 713 | 667 | 541 | 541 | — |
| opt1 | max-running 256、mem 0.89、DeepEP 256 + `--enforce-shared-experts-fusion` | — | — | — | — | — | 崩溃（Invalid address access / Segfault），放弃 shared-experts 融合 |
| opt2 | max-running 256、mem 0.89、DeepEP 256 | 345 | 1008 | 1244 | 577 | 1087 | 主要收益来源 |
| opt3 | opt2 + local-control-broadcast + `--speculative-skip-dp-mlp-sync` | — | — | — | — | — | 崩溃（MLP sync padding 时 `input_ids=None`），放弃 |
| opt4 | opt2 + `--enable-dp-attention-local-control-broadcast` | 344 | 958 | 1267 | — | — | 无收益，不采用 |
| opt5 (q) | opt2 + 补丁① lm_head NN | 427 | 1130 | 1623 | 715 | 1331 | 采用 |
| opt6 (q) | opt5 + 补丁② 线性层 NN | 420 | 1094 | 1587 | 711 | 1349 | 采用（ITL −5%） |
| opt7 (q) | opt6 + prefill-delayer | 430 | 1195 | — | — | — | 采用 |
| opt8 | opt2 + prefill-delayer，mem 0.86 | — | — | — | — | — | 未完成压测 |
| opt9 (q) | opt7 + chunked-prefill 16384 | 378 | 1252 | 1858 | 756 | 1511 | 采用 |
| opt10 (q) | opt9 + delayer 低水位 0.05 | 402 | 1266 | 1796 | 694 | 1551 | 采用 |
| opt11 (q) | opt10 + MTP 7 步 / 8 draft token | 369 | 1099 | 1553 | — | — | 变差，不采用 |
| **opt12 (q)** | **opt10 + `deepep_cfg40.json`（即最终配置）** | 425 | 1222 | 1636 | 738 | 1600 | **最终版** |
| opt13 (q) | opt12 + `--load-balance-method total_requests` | — | — | — | — | — | c16 时崩溃，见 7.1 节 |
| opt14/15 | opt13 关闭补丁①② / 只关闭补丁① | 386 / 350 | 1016 / 980 | — | — | — | 不崩，但比 round-robin 慢 |
| opt16 | opt12 关闭补丁①②（A/B 用） | — | — | — | — | — | ITL 回升到 107–109ms |

快速压测的 c128 单点波动较大（比如 opt12 是 1636，而完整压测是 1808）。最终结论以 5.1、5.2 节的完整压测为准。

---

## 6. Profiling 分析过程

**方法：**
- 用 SGLang 自带的 torch profiler：启动时设置 `SGLANG_TORCH_PROFILER_DIR`，然后调用 `curl -X POST :30000/start_profile -d '{"output_dir":..., "num_steps":12, "activities":["GPU","CPU"]}'`。8 个 rank 各导出一份 `*.trace.json.gz`。
- 用 `tools/anatrace.py` 按 kernel 名统计总耗时、占比、调用次数和平均耗时，并按单个 rank 看一层内 kernel 的先后顺序。
- 对可疑的算子，用单算子微基准复现并验证改法（`tools/mb_*.py`，用 CUDA graph 计时）。

注意：在这个环境里，profiling 结束后服务可能卡死或崩溃，抓完 trace 后建议重启服务。不要开 `with_stack=true`，trace 会无限膨胀。

### 6.1 Trace ①：decode，1 并发，原始参数（`opt/prof/dec`）

| kernel | 占比 | 平均耗时 |
|---|---|---|
| FP8 MoE grouped GEMM（`m_grouped_fp8_gemm_nt_masked`） | 14.8% | 74µs |
| **bf16 `Cijk_Alik_Bljk_BBH_MT32x32x4`** | **11.1%** | **104µs** |
| FP8 dense GEMM（`F8BS_MT48x16x256`） | 10.5% | 54µs |
| DeepEP 低延迟 combine | 8.8% | 44µs |
| 稀疏 MLA decode attention | 6.2% | 60µs |
| 其它 bf16 GEMM（MT32x16x32/256/128、MT256x256x16、MT32x32x32） | 约 14% | — |
| rotary_embedding（每层 8 次调用） | 4.4% | 5.3µs |

bf16 GEMM 合计约占 25%，与它在模型里的计算量明显不成比例。逐层看 kernel 顺序后，定位到 dense 层的 o_proj、gate_up 和 lm_head 落在 `MT32x32x4` / `MT32x32x32` kernel 上，单次 218–308µs。

**单算子复现**（`tools/mb_lmh2.py`、`tools/mb_lin2.py`、`tools/mb_nn.py`，单位 µs）：

| 权重 | 布局 | M=6 | M=36 | M=96 | M=192 | M=1024 |
|---|---|---|---|---|---|---|
| lm_head 154880×6144 | rocBLAS NT（原始） | 3424 | 15327 | 40879 | 2141 | — |
| | **rocBLAS NN（补丁后）** | **1595** | **1590** | **1599** | **1613** | — |
| | hipBLASLt NT | 1147 | 2504 | 2140 | 2361 | — |
| dense o_proj | NT → NN | 230 → 113 | 480 → 138 | 458 → 203 | 593 → 393 | 1330 → 1538 |
| dense gate_up | NT → NN | 329 → 182 | 321 → 376 | 312 → 242 | 565 → 322 | 1179 → 1088 |
| dense down / eh_proj | NT → NN | 177 → 86 | 133 → 104 | 238 → 136 | 368 → 243 | 762 → 847 |
| shared gate_up | NT → NN | 37 → 34 | 48 → 43 | 66 → 56 | 85 → 73 | 193 → 168 |
| router gate（N=256） | NT → NN | 8 → 9 | 11 → 16 | 12 → 16 | 16 → 24 | 45 → 39 |

结论：
- decode 常见的小 M 下，NN 布局普遍更快，lm_head 尤其明显。M=1024（prefill）时个别形状 NN 稍慢，但 prefill 中占比很小。
- hipBLASLt 只在 M=6 时更好，整体不如 rocBLAS NN 稳定，所以没有切换 BLAS 后端。
- N 小的权重（router、indexer wk）NT 更快，所以补丁②设了 N≥1024 的门槛。

### 6.2 Trace ②：decode，128 并发，补丁①②之后（`opt/prof/dec128`）

| kernel | 占比 | 平均耗时 |
|---|---|---|
| FP8 MoE grouped GEMM（masked） | 27.0% | 350µs |
| 稀疏 MLA decode attention（`flash_fwd_fixed_splitkv_mla_fp8_sparse`） | 24.0% | 600µs |
| DeepEP 低延迟 combine | 11.3% | 147µs |
| FP8 dense GEMM（多种 tile） | 约 7% | — |
| DeepEP 低延迟 dispatch | 4.0% | 52µs |

bf16 GEMM 合计降到约 11%，换成了 MT128x64x32、MT256x256x16 等大 tile kernel，`MT32x32x4` 已经不出现。瓶颈转到 MoE GEMM、attention 和通信。

**MoE GEMM 单算子测试**（`tools/mb_moe.py`）：N=4096、K=6144，32 个专家，decode 规模下 12 种 kernel 模式（1100–1111）都在约 470µs，等效带宽约 1.71TB/s。作为对比，实测 D2D 拷贝带宽约 1.35TB/s（读+写）。所以 decode 下 MoE GEMM 已经卡在显存带宽上，换 kernel 配置没有收益。要进一步提速只能减少权重读取量（见第 9 节）。

### 6.3 Trace ③：prefill，64 个约 1K token 请求（`opt/prof/pre64`）

| kernel | 占比 | 平均耗时 |
|---|---|---|
| FP8 MoE grouped GEMM（contiguous） | 20.8% | 1412µs |
| 稀疏 attention fwd | 11.1% | 1732µs |
| DeepEP cached_notify_combine | 10.5% | 1430µs |
| DeepEP combine | 8.6% | 1172µs |
| DeepEP notify_dispatch | 7.7% | 1046µs |
| DeepEP dispatch | 7.5% | 1011µs |
| bf16 GEMM MT256x256x16 | 4.5% | 301µs |

DeepEP normal 模式合计约 34%，于是做了 4.2 节的 SM 数调优。

---

## 7. 已知问题与注意事项

### 7.1 不要使用 `--load-balance-method total_requests`

- **现象：** 和补丁①（lm_head NN）同时使用时，在 16 并发的第一个 batch，某个 DP rank 在 `sampler.py:743 torch.multinomial` 报 `torch.AcceleratorError: no kernel image is available`（hipErrorNoBinaryForGpu），可稳定复现（opt13、opt13b 两次）。
- **排查：**
  - 关闭补丁①（opt15）或两个补丁都关闭（opt14）就不崩；
  - 用 `mb_nn.py` 扫描了所有 bf16 线性层和 lm_head 在 NN 布局下 M=1..4200 的形状，全部正常；
  - 根因尚未定位。
- **结论：** 最终配置用默认的 round-robin，已跑过完整压测和多轮快速压测，都没有出现问题。total_requests 本身也更慢（1K c64 约 1000 tok/s，round-robin 为 1222）。如果以后必须用 total_requests，请先设置 `SGLANG_HCU_LM_HEAD_NN=0`。

### 7.2 低并发结果波动大

MTP 接受长度强烈依赖生成内容：同一个 prompt 在 T=0.01 下，每个请求的接受长度可以在 2.5 到 6.0 之间。低并发时样本少（num_prompts=2×并发），单次结果的波动可以达到 ±20%。

同一天、同一个 A/B 脚本（`tools/vllm_bench-c8.sh`）的复测：

| 配置 | 1K c8 | 1K c16 | 4K c1 |
|---|---|---|---|
| 原始参数，关闭补丁（`results/benchq_ab_base_np.log`） | 256 / 264 | 404 | 66.6 / 46.3 |
| 最终配置（`results/benchq_ab_opt12b.log` 及其它重复） | 243–283 | 379–415 | 46.8 / 70.5 |
| 最终配置，关闭补丁（`results/benchq_ab_opt16.log`） | 259 / 283 | 397 | 51.7 / 60.0 |

两者在同一区间内，低并发没有退化。原始完整压测中的 1K c8 = 306 属于偏高的单次样本。

### 7.3 其它

- **固化镜像：** 补丁和 `.bak_20261021` 备份都在容器的 site-packages 里，`docker commit` 会一并带上。启动脚本里的 `deepep_cfg40.json` 是 `/data` 挂载路径，新容器需要保持相同的挂载。
- **NUMA balancing：** 宿主机开着 NUMA balancing，aiter 启动时会告警（日志里的 "error" 字样都来自这条告警）。可以考虑 `echo 0 > /proc/sys/kernel/numa_balancing`。这是宿主机级设置，本轮没有修改。
- **chunked-prefill-size：** 试过 32768（每个 rank 4096）配合 delayer 和 mem 0.89，在 128 并发时 OOM，因此最终用 16384。
- **评估后没有采用的：**
  - MTP 7 步 / 8 draft token（opt11，变差）；
  - MoE GEMM kernel 配置调优（已到带宽上限）；
  - EPLB 专家重均衡；
  - topk_length 调整；
  - shared-experts 融合（opt1 崩溃）；
  - skip-dp-mlp-sync（opt3 崩溃）；
  - local-control-broadcast（opt4 无收益）；
  - hipBLASLt 后端。

---

## 8. 改动清单汇总

| 类别 | 改动 | 位置 |
|---|---|---|
| 启动参数 | max-running-requests 256、mem 0.89、chunked-prefill 16384、prefill-delayer + 低水位 0.05 | `sglang_start_opt_final.sh` |
| 通信配置 | DeepEP 低延迟 dispatch token 数 256；normal 模式 40 SM | `sglang_start_opt_final.sh`、`deepep_cfg40.json` |
| 框架代码 | lm_head 权重 NN 布局 | 容器内 `sglang/srt/layers/logits_processor.py` |
| 框架代码 | bf16 线性层（N、K ≥ 1024）权重 NN 布局 | 容器内 `sglang/srt/layers/quantization/unquant.py` |
| 未改动 | 原始启动脚本、测试脚本、模型文件、所有 kernel 和 .so 库（aiter、DeepEP、deepgemm、lightop、rocBLAS） | — |

---

## 9. 后续深挖方向（待确认后再执行）

以下内容**本轮都没有执行**，需要先确认方案。耗时占比来自 6.2 节 128 并发 decode 的 trace。

1. **FP8 MoE grouped GEMM（27%，已到带宽上限）**
   - 唯一出路是减少读取的权重字节数：比如 W4A8 / INT4 专家权重，或者在专家负载不均时跳过空专家的权重读取（masked kernel 要检查是否仍会读取 M=0 专家的权重）。
   - 也可以考虑和前后的 `act_and_mul_fp8_quant`、`per_token_quant_fp8` 融合，减少中间激活的读写。
2. **稀疏 MLA decode attention（24%）**
   - `flash_fwd_fixed_splitkv_mla_fp8_sparse` 在 128 并发时每次调用约 600µs，需要分析 split-kv 的切分数、occupancy 和 fp8 解量化开销；
   - 可以考虑与 `flash_fwd_mla_combine`、`concat_mla_absorb_q` 融合。
3. **DeepEP 低延迟 combine（11%）**
   - 尝试低延迟模式的 SM 和 QP 参数（`ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX` 等）；
   - 尝试 combine 与 shared expert 计算重叠（two-batch overlap / SBO）。
4. **小算子融合候选（decode 1 并发时较明显）**
   - 每层 8 次 `rotary_embedding_kernel`（各约 5µs，合计占 4.4%），可以合并成一次；
   - `fused_add_rms` + `per_token_quant_fp8` 可以融合；
   - 零散的 `elementwise` / `vectorized_elementwise` kernel（合计约 4–5%）。
5. **lm_head**
   - 补丁后每步约 1.6ms，相当于完整读一遍 1.9GB 权重。因为开了 `--enable-dp-lm-head`，每个 rank 都持有完整词表。
   - 可以评估改成词表并行切分（每个 rank 读 1/8 + all-gather），或者把 lm_head 量化为 FP8。
6. **根因排查：** total_requests + lm_head NN 下 `torch.multinomial` 报 "no kernel image" 的问题（7.1 节）。
7. **低并发 4K：** chunked-prefill 改为每个 rank 2048 后，单请求 3584 token 的 prefill 要分两段，TTFT 略高。可以评估动态 chunk 大小。
