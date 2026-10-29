# GLM-5.3 FP8 第二轮优化说明：算子改写与 kernel 融合（dcu4，2026-10-23）

## 1. 概述

| 项目 | 内容 |
|---|---|
| 服务器 / 容器 | dcu4，8 × BW1100（gfx938，64 CU）；容器 `glm5.3-pd`（第一轮固化镜像 `glm5.3_v1022` 之上继续修改） |
| 模型 | `/data/model/GLM/GLM-5.3-Channel-FP8-w8a8`（served name `glm5.3-fp8`） |
| 部署形态 | DP8 dp-attention + EP8，DeepEP LL + DeepGEMM masked MoE，EAGLE 5/1/6，DSA 稀疏 MLA + fp8 KV（同第一轮 `sglang_start_opt_final.sh`） |
| 目标负载 | `bench/vllm_bench-pc.sh`（未修改）：64K/128K 输入、1024 输出、PC 90%、并发 1–32 |
| 精度验证 | `glm5.2_evalscope` 容器（evalscope 1.10.0）：gsm8k 全量 1319 题、aime26 30 题 × 4 次 |
| 最终启动脚本 | `sglang_start_opt2_final.sh`（= 第一轮最终脚本 + 7 个 `SGLANG_HCU_*` 开关：6 个算子开关和 1 个 sampler 预热开关） |

本轮没有再调启动参数，全部工作都在算子层：
- 根据 64K decode 的 profiling，新写了一个算子包 **`hcu_ops`**（HIP C++ 扩展，gfx938），共 6 类 kernel，替换 lightop / hipBLASLt / torch 里在 decode 小 batch 下效率低的实现。
  - 其中 3 个是 kernel 融合：rope 8 次 launch 合成 1 次；MoE gate 与 id 后处理合并；DSA topk 的 4 轮 radix 与 2 次扫描合并成 3 遍直方图加 1 遍压缩。
- 通过环境变量开关接入 SGLang，每个开关都能单独关闭回退到原实现，代码改动都有备份。
- 数值上：
  - rope、gate、act 三个 kernel 与原实现**逐 bit 一致**；
  - topk 选出的 2048 个 token 集合与原实现完全相同；
  - GEMM/bmm 的误差与 hipBLASLt / torch.bmm 处于同一量级。

**结论：**（待 A/B 与精度结果出来后填写，见第 6、7 节）

---

## 2. 交付文件

都在 `/data/home/daihw/scripts/glm5.3/` 下：

| 文件 | 说明 |
|---|---|
| `sglang_start_opt2_final.sh` | 最终启动脚本：第一轮 `sglang_start_opt_final.sh` 加上第 3.1 节的 7 个 `SGLANG_HCU_*=1`，日志前缀 `glm53-opt2-final_` |
| `GLM5.3_opt2_README_20261023.md` | 本文档 |
| `opt2_final_20261023/hcu_ops/` | 算子包完整源码：`csrc/*.cu`、`hcu_ops/__init__.py`、`skinny_cfg.json`、`setup.py`、`build.sh`；正确性/性能测试 `test_*.py`；调优脚本与数据 `tune_skinny.py`、`skinny_tune.json`、`make_cfg.py`、`bmm_tune.txt`；探测程序 `probe_*.hip` |
| `opt2_final_20261023/patches/` | SGLang 补丁：`*.diff`、各 `patch_*.py`（从备份重新打补丁）、`apply_patches.sh`（编译安装 hcu_ops + 打全部补丁）、`verify_patches.sh`（核对补丁） |
| `opt2_final_20261023/results/` | 端到端 A/B 结果（`ab/`）、各组服务日志（`server_logs/`）、profiling 日志（`prof/`）、精度验证结果与脚本（`eval/`，含步时统计 `ana_clean.py`、`ana_steps.py`） |
| `opt2_final_20261023/tools/` | profiling 与崩溃排查工具：`prof_arm.sh`、`prof_dec2.sh`、`tracediff.py`、`repro.sh`、`scan_bundles.py`、`lazy_oom.py`、`lazy_race.py/.sh` 及其日志（见第 6.5 节、第 9 节） |
| `opt2/` | 全部试验过程文件：profiling trace（`opt2/prof/d64k_c8*`）、A/B 脚本 `run_ab.sh` / `ab_bench.sh` / `start_k.sh`、各轮服务日志 |
| `eval/` | 精度验证脚本 `run_eval.sh`（evalscope 容器内执行）、`drive_eval.sh`（宿主机上起停服务并调度）及输出 |

---

## 3. 使用方法

```bash
docker exec -it glm5.3-pd bash
cd /data/home/daihw/scripts/glm5.3
nohup bash sglang_start_opt2_final.sh > /dev/null 2>&1 &     # 日志 glm53-opt2-final_<时间>.log
```

### 3.1 开关

| 环境变量 | 默认 | 最终脚本 | 作用 | 修改的文件（容器内 `sglang/srt/` 下） |
|---|---|---|---|---|
| `SGLANG_HCU_FAST_MASK_TOPK` | **1** | 1 | DSA indexer 的 topk-2048 + page table 变换，替换 lightop `fast_topk_transform_sparse_mask_fused` | `layers/attention/dsa/dsa_indexer_metadata.py`（`.bak_20261022`） |
| `SGLANG_HCU_SKINNY_GEMM` | 0 | 1 | FP8 dense 线性层（q_b、o_proj、indexer wq_b、dense gate_up）在小 M 下改走 skinny GEMM | `layers/quantization/fp8_utils.py`、`layers/linear.py`（`.bak_20261023`） |
| `SGLANG_HCU_FUSED_ROPE` | 0 | 1 | rope 每层 8 次 launch 合成 1 次 | `layers/rotary_embedding/base.py`（`.bak_20261023`） |
| `SGLANG_HCU_FUSED_GATE` | 0 | 1 | MoE 路由 gate 与 DeepEP id 后处理融合 | `layers/moe/topk.py`（`.bak_20261023`） |
| `SGLANG_HCU_FUSED_ACT` | 0 | 1 | MoE 两次 grouped GEMM 之间的 masked SiLU×mul + fp8 量化 | `layers/moe/ep_moe/layer.py`（`.bak_20261023`） |
| `SGLANG_HCU_SKINNY_BMM` | 0 | 1 | MLA absorb 的 `q_nope @ w_kc`、`attn @ w_vc` 两个 bf16 bmm | `models/deepseek_common/attention_forward_methods/forward_mla.py`（`.bak_20261023`） |
| `SGLANG_HCU_SAMPLER_WARMUP` | 0 | 1 | 创建 sampler 时把 T>0 的采样路径（multinomial、top-p）各跑一次，失败重试；规避首个采样请求的 "no kernel image" 崩溃（第 9 节） | `layers/sampler.py`（`.bak_20261023`） |

辅助变量（一般不用改）：
- `HCU_TOPK_IMPL=split|single`（默认 split）
- `HCU_TOPK_SPLITS`（默认 8）
- `HCU_SKINNY_CFG=<json>`：替换 skinny GEMM 的调优表

第一轮的两个开关 `SGLANG_HCU_LM_HEAD_NN`、`SGLANG_HCU_LINEAR_NN` 仍然默认开启，不受影响。

### 3.2 设计原则

- **按形状接入，不支持就回退。** 每个 hook 先检查 dtype、形状、stride、对齐和 M 是否在调优表里，任何一项不满足就走原来的实现。所以开关打开后，prefill、大 batch、TP8 等没有调过的情况仍然走原算子。
- **兼容 CUDA graph。** 所有 kernel 都没有 host 同步，launch 配置只取决于静态形状，可以被 graph 捕获。act 和 topk 的切分由 device 端根据实际行数决定。
- **结果确定。** 所有归约都按固定顺序进行，同样的输入每次 replay 输出相同。

---

## 4. Profiling 定位

用 64K、PC 90%、c8 的 decode 抓 trace（`opt2/prof/d64k_c8`，TP0，10 步）。GPU 忙碌 946ms 中：

| kernel | 合计 | 占比 | 次数 | 平均 |
|---|---|---|---|---|
| MoE grouped fp8 masked GEMM | 193 ms | 20.4% | 1476 | 131 µs |
| DeepEP LL combine | 95 ms | 10.1% | — | 64.6 µs |
| **lightop topk（`fast_topk_transform_sparse_mask`）** | **93.9 ms** | **9.9%** | 214 | **439 µs** |
| hipBLASLt F8BS MT48x16x256（FP8 dense GEMM） | 77.6 ms | 8.2% | — | 54 µs |
| flash sparse MLA decode | 50.7 ms | 5.4% | — | 65.9 µs |
| BBH MT256x256x16（lm_head） | 45.9 ms | 4.9% | 58 | 792 µs |
| BBH MT64x16x128（shared expert gate_up 等） | 40 ms | 4.2% | — | — |
| DeepEP LL dispatch | 32.6 ms | 3.4% | — | — |
| **rotary_embedding** | **32.4 ms** | 3.4% | **6152** | 5.3 µs |
| paged_mqa_logits_grouped（indexer） | 19.6 ms | 2.1% | — | 91.5 µs |

按单个 rank 的 kernel 顺序拆出一层（每个 rank 6 个 token，即 1 请求 × 6 draft）：

| 算子 | 耗时 | 问题 |
|---|---|---|
| fused_qkv_a（FP8，6144→2624，16MB） | 16.6 µs | 约 1 TB/s |
| q_b（FP8，2048→16384，33.5MB） | 33.8 µs | 约 1 TB/s |
| w_kc bmm（bf16，BBH MT32x16x32） | 18.2 µs | |
| rotary_embedding × 8 | 约 42 µs | 64 个 head 被拆成 8 次 launch |
| sparse MLA fixed-split + combine | 68.5 + 7.5 µs | |
| w_vc bmm（bf16） | 24.5 µs | |
| o_proj（FP8，16384→6144，100MB） | 91.8 µs | 约 1.1 TB/s |
| moe_fused_gate + id 后处理 | 17.3 µs | 两个 kernel；前者有 scratch |
| act_and_mul_fp8_quant | 17.4 µs | 4096 个 CTA，绝大多数是空的 |
| MoE gate_up / down（masked fp8） | 171–278 / 92–147 µs | 约 3–3.7 TB/s，已接近带宽 |
| DeepEP dispatch / combine | 36 / 62–131 µs | 主要在等其他 rank |

**结论：**
- 最大的单项是 topk：每步约 21 次调用 × 439µs，占 9.9%。
- 其次是 dense 小 M GEMM。hipBLASLt 在 M=6 时只跑到约 1TB/s，每层约 270µs，相当于每步约 21ms。
- 再往下是一批小 kernel：rope 每层 8 次 launch、gate、act、两个 absorb bmm。在 HIP graph 下每个 kernel 至少约 5µs，所以合并 launch 本身就有收益。
- MoE GEMM 已接近带宽上限（第一轮 6.2 节结论），combine 主要在等其他 rank，这两项本轮没有动。

---

## 5. 算子优化明细

所有 kernel 都在 `hcu_ops`（`opt2_final_20261023/hcu_ops/csrc/`）里。下面的单算子时间都是在空闲 GPU7 上用 CUDA graph 计时的，单位 µs。

### 5.1 DSA indexer topk-2048（`sparse_mask_topk.cu`，`SGLANG_HCU_FAST_MASK_TOPK`）

**原实现的问题**（lightop `fast_topk_transform_sparse_mask_fused`）：
- 每行只有 1 个 CTA。
- 做 4 轮 8-bit radix，再加 2 次扫描。
- 正的 logits 最高字节几乎相同，所以 shared atomic 冲突严重。
- 每 16 个 token 走一次串行 bit 循环。
- 在 CUDA graph 下，score 宽度是 page table 容量（1M token），行越长越慢：64K 时约 380µs，128K 时约 740µs。

**新实现：**
- **single 版**（每行 1 个 CTA，1024 线程）：
  - pass 1：key 高 12 位的直方图，同时统计正数和零的个数。如果阈值正好是 +0，后两遍直接跳过。
  - pass 2：阈值 bin 内 key 中间 12 位的直方图。
  - pass 3：低 8 位的 per-wave 直方图，同时统计“严格大于阈值”的 token 数。
  - pass 4：用 ballot 前缀和把选中的 token 下标压缩到 LDS，再做一次合并访存的 page table gather。
  - 所有 per-token 操作都不分支：不在当前直方图范围内的 token 加到每个 lane 的 trash 槽。按 token 分支要慢约 4 倍。
- **split 版（默认）**：decode 只有几行，single 版只能用到几个 CU。split 版把每行切到最多 8 个 CTA 上，分 4 个 kernel：
  - K1：每个 slice 的 12-bit 直方图，合并成每行的全局直方图。
  - K2：每个 CTA 推出阈值 bin，按 token 顺序压缩两张表：一定入选的 A 表，以及阈值 bin 里的候选 C 表（带 key）。
  - K3：确定 24-bit 前缀，统计每个 slice 中高于前缀的数量和低 8 位直方图。
  - K4：确定精确阈值，按 A、C 中高于阈值的、C 中等于阈值的（下标小的优先）顺序写出，并经 page table 转换。
  - slice 数在 device 端按行长度算出，所以兼容 CUDA graph 的 padding 宽度。CTA 按 slice-major 编号，让同一行的空闲尾部 slice 分散到不同 shader engine 上。
- 行数 ≥ CU 数的一半时，每行自动退回 1 个 CTA。

**正确性**（`test_topk.py`）：
- 覆盖 10 种分布：pos、sparse30、sparse2、relu_sum、mixed、ties、zeros、neg、narrow、const。
- 覆盖 B ∈ {6, 24}，L 从 1000 到 131072，以及 ragged/padding 宽度，最后加 1M 宽度的生产布局。
- 全部与 lightop 选出**相同的 token 集合**（`ALL_VALID True`）。
- 输出顺序不同：先按 token 升序写严格大于阈值的，再写等于阈值的，每次 replay 结果固定。下游稀疏 MLA 把这些下标当集合用，所以数学上等价；只是 KV 的累加顺序可能变化，带来舍入级的差异。

**性能**（生产布局：宽度 1M、relu_sum 分布，lightop → split）：

| 行数 | 16K | 64K | 128K |
|---|---|---|---|
| 6 | 115 → 48 | 379 → 49 | 739 → 61 |
| 12 | 115 → 49 | 383 → 61 | 737 → 83 |
| 24 | 117 → 49 | 388 → 88 | 763 → 134 |
| 48 | 146 → 43 | 532 → 116 | 1042 → 211 |
| 192 | 460 → 127 | 1977 → 341 | 4028 → 626 |

- 64K、c8 时每次调用从 439µs 降到约 50µs。按每步约 21 次估算，每个 decode step 省约 8ms。
- 已知小差距：
  - 16K 且行数很少时，split 比 single 慢约 5µs。
  - 几乎全零或极稀疏的行有约 25µs 的固定下限，lightop 在这种情况下是 17–25µs。生产中的行是 relu_sum 分布，不会碰到这种情况。

### 5.2 Skinny FP8 GEMM（`skinny_gemm.cu`，`SGLANG_HCU_SKINNY_GEMM`）

**问题：** decode 时 dense 线性层的 M 是 6 × 每个 rank 的请求数。hipBLASLt 按启发式选的 tile（MT48x16x256 等）只能跑到约 1TB/s。例如 o_proj 读 100MB 权重要 91.8µs，而 HBM 读带宽峰值约 1.9TB/s。

**设计：**
- **mmac 布局：** 用 `probe_mmac.hip` 实测 `__builtin_hcu_mmac_f32_16x16x32_fp8_fp8` 的 lane 布局：lane l 提供 A 的第 l%16 行和 B 的第 l%16 列，k 范围是 (l/16)×8 起的 8 个元素。
  - A 放 W 的 16 行，B 放 X 的 16 个 token。
  - fp8 版输出 = Σ(W·X) × w_scale[n] × x_scale[m]，即 per-channel × per-token，和 deepgemm 的 w8a8 语义相同。
- **划分：** 每个 CTA 负责 NT 个 16 行的 W tile，跑完整个 K。NW 个 wave 切分 K，部分和按固定顺序经 LDS 归约，结果是确定的。
- **v2：W 先放进 LDS。** `probe_bw2.hip` 实测：按 mmac 布局直接加载（每行 64B）时，HBM 读最多约 1.1TB/s；每行连续 256B 时能到 1.6–1.8TB/s。
  - 所以每个 wave 每条指令加载 4 行 × 256B 到私有 LDS slab，再按 mmac 布局读回。
  - 提前预取 1–2 个 chunk，避免 vmcnt 等待。
- **v4：X 也放进 LDS。** `diag_skinny.py` 定位到，M 超过约 8 以后，按 mmac 布局反复从 L2 读 X 成了瓶颈。
  - v4 每个 chunk 只用 ceil(M/4) 条指令按行连续加载 X，所有 tile 共用。
  - LDS 大小在 launch 时按实际 M 分配。
- **调优：**
  - `tune_skinny.py` 对 GLM-5.3 在 DP8 下的每个形状和 M 扫描 ver、nw、nt，与 hipBLASLt 比较，结果存到 `skinny_tune.json`。
  - `make_cfg.py` 只保留快 ≥5% 且 ≥1µs 的点，生成 `skinny_cfg.json`。
  - 其余 M 和形状一律回退 hipBLASLt。

**结果（hipBLASLt → skinny）：**

| 形状（N×K） | 接入的 M | 结果 |
|---|---|---|
| o_proj 6144×16384 | 3、5–24 | M=6–16：约 96.5 → 65.7–71.4；M=18：102.7 → 83.1；M=24：103.4 → 92.3（M=1、2、4 时 hipBLASLt 已经选到好 tile，不接入） |
| q_b 16384×2048 | 1–12 | 29.9 → 23.8–28.1 |
| indexer wq_b 4096×2048 | 1–4 | 10.4 → 8.6–8.8 |
| dense gate_up 24576×6144（前 3 层） | 1–8 | 100 → 91.7–95.3 |
| qkv_a 2624×6144、dense down 6144×12288 | — | 从来没赢过，不接入 |

- **精度：** 与 fp32 参考的最大绝对误差 1.6–1.7e-3，和 hipBLASLt 本身的误差相同。接入后的 `_hcu_fp8_gemm` 与 hipBLASLt 输出的相对差 ≤2.1e-5。
- **收益：** c8 时 M=6，每层 o_proj 省约 31µs，q_b 省约 4µs，78 层合计每次 verify forward 省约 2.7ms。M>24（即每个 rank 超过 4 个请求）后 hipBLASLt 切到大 tile，skinny 不再占优，自动回退。
- **bf16：** 也调过 bf16 版（shared expert down、eh_proj，见 `skinny_tune.json`），收益只有约 2µs。而且第一轮的 NN 布局补丁改变了 bf16 权重的 stride，所以 `try_skinny_bf16` 没有接入 SGLang。

### 5.3 rope 融合（`rope.cu`，`SGLANG_HCU_FUSED_ROPE`）

- **问题：** sgl_kernel 的 `rotary_embedding` 每个 token 一个 CTA，线程数为 min(nq × rot_dim / 2, 512)。但 HCU 版编译时 launch bound 是 256 线程，所以 SGLang 的 HCU 回退路径把 MLA 的 64 个 query head 拆成 8 次 launch，每次约 5.3µs，每层约 42µs。整个 trace 里有 6152 次调用。
- **新实现：** head 铺在 gridDim.y 上，q 和 k 在一次 launch 里完成。每个元素的计算表达式和标量类型与 sgl_kernel 完全相同，所以结果**逐 bit 一致**（`test_rope.py`，neox 与非 neox 两种）。
- **结果：** T=6 时 8 次 launch 共 20.3µs → 1 次 2.4µs。
- **接入条件：** bf16/fp16，cos_sin_cache 与 query 同类型，2D 或 3D 张量且最后一维连续。否则走原路径。

### 5.4 MoE gate 融合（`moe_ops.cu`：`moe_gate_topk`，`SGLANG_HCU_FUSED_GATE`）

- **问题：** lightop `moe_fused_gate` 约 10.3µs。
  - 它用动态下标访问 per-lane 私有数组，编译器把数组放进 scratch。
  - 每轮选 expert 要做 11 次串行 shuffle。
  - 之后还有一个单独编译的后处理 kernel，约 5.4µs：把 padding 行的 id 置 -1 并转 int64。
- **新实现：**
  - 每个 token 用 16 个 lane（一个 DPP row），每个 lane 负责 E/16 个 expert。
  - 每轮先取 lane 内最大值（严格大于，相同分数取下标小的），再用 DPP 对 (score, −index) 做 max 归约，不经过 LDS。
  - 胜出项用展开的比较清除，不用动态下标，所以没有 scratch。
  - sigmoid、correction bias 选择、renormalize、routed_scaling_factor 与 lightop 一致；exp 用 lightop 在 gfx938 上的 exp2 写法。
  - 后处理合并进同一个 kernel。
- **结果：** 10.3 + 5.4µs → 6.8–7.3µs。weights 和 ids **逐 bit 一致**，tie-breaking 也一致（`test_moe_ops.py`）。
- **接入条件：** 只有 1 个 expert group、topk=8、有 correction bias、renormalize、没有 fused shared experts 和 EPLB remap，并且走 HCU DeepEP + DeepGEMM 路径。这些正是 GLM-5.3 的配置；其他情况走原路径。

### 5.5 MoE masked SiLU×mul + fp8 量化（`moe_ops.cu`：`silu_mul_fp8_quant_masked`，`SGLANG_HCU_FUSED_ACT`）

- **问题：** lightop `fuse_silu_mul_fp8_quant_ep` 固定 launch E × 128 个 CTA（256 线程），每个 CTA 处理一行。
  - decode 时每个 rank 有 32 个 expert × 256 的容量，但有效行通常只有几十到两百，4096 个 CTA 里绝大多数读完 masked_m 就退出了。
  - 算子本身受 VALU 延迟限制：每个元素约 30 条 VALU，exp 和 rcp 都是 1/4 速率。所以即使只有 32 行也要 17µs。
- **新实现：**
  - 用 masked_m 的前缀和给有效行全局编号，再分给固定的 512 个 CTA。
  - 一行可以拆给 1、2 或 4 个 wave，跨 wave 的 abs-max 经 LDS 归约。WPR=0 时由 kernel 按实际行数选择让每个 SIMD 工作量最少的拆法：行少时多拆，铺满所有 SIMD；行多时每行 1 个 wave，没有同步开销。所以一套 launch 配置适用于 CUDA graph 下的所有 batch。
  - **数值与 lightop 完全相同：**
    - silu 用 bf16(x / (1 + 2^(−x·log2e)))，乘积取 bf16；
    - 行 abs-max；scale = max(amax/448, 1/(448·512))；q = cvt_pk_fp8(y × (1/scale))。
  - **硬件 bf16 转换：** 用 gfx938 的 `v_cvt_bf16_f32`（1 条指令）代替软件 RNE 的 3 条。`bf16_round_check` 遍历了全部 2^32 个 float，非 NaN 结果与 RNE 完全相同；NaN 最终统一成 0x7FC0，与 lightop 相同。
  - **放弃的方案：** 用 rcp（带或不带 Newton 迭代）代替 IEEE 除法。遍历全部 65536 个 bf16 输入后发现，它在 −0、−87.5..−88.5（rcp flush）、5.9375（单纯 rcp）这几个点舍入不同；而且只有几千行以上才有收益。所以保留 IEEE 除法。
- **结果**（**逐 bit 一致**，包括 NaN/inf，见 `test_moe_ops.py`）：

| 有效行分布 | lightop | hcu |
|---|---|---|
| 每个 expert 1 行（32 行） | 16.7 | 4.6 |
| 约 100–128 行 | 17.2–18.4 | 5.4 |
| 150–192 行 | 17.8–18.5 | 6.5–6.6 |
| 193–256 行 | 19.1–19.3 | 6.7–6.8 |
| 384 行 | 20.7 | 8.8 |
| 1536 行 | 39.7 | 23.3 |
| 满容量 8192 行 | 2639 | 104 |

- **试过但没有用：**
  - 让 lightop 走小 grid 的 `limit=inf` 技巧：小 batch 没有变快，大 batch 反而更慢。
  - 用 `amdgpu_waves_per_eu(5)` 把 VGPR 压到 48：会 spill，反而更慢。
- **接入条件：** bf16、连续存储、d 是 1024 的倍数且在 1024–8192 之间（GLM-5.3 的 d=2048）、E ≤ 256、没有 swiglu limit。

### 5.6 MLA absorb bmm（`skinny_gemm.cu`：`bmm_skinny`，`SGLANG_HCU_SKINNY_BMM`）

- **问题：** `q_nope @ w_kc`（每个 head N=512、K=192）和 `attn_out @ w_vc`（N=256、K=512）是 64 个 head 的 bf16 bmm。torch.bmm 在 decode 时选 BBH MT32x16x32 kernel，耗时 18.2 / 24.5µs。
- **新实现：**
  - batched skinny GEMM：K 很短，所以不切 K。每个 wave 负责一个 16 行 W tile 的整个 K，按 CW 字节分 chunk 流水，预取 1 个 chunk。
  - W 按行连续的 16B 片加载进 wave 私有 LDS（w_kc 整个 tile 是一块连续内存），再按 mmac 布局读回。
  - X 和输出支持任意 stride，所以可以直接读写 q_nope、attn_output 的 head 切片，不需要额外 transpose 或拷贝。
- **结果**（hcu vs torch.bmm；`test_bmm.py`、`bmm_tune.txt`）：

| 形状 | 接入的 M | hcu | torch.bmm |
|---|---|---|---|
| kc，64 head（DP8） | 1–12 | 10.6–13.9 | 14.0–15.6 |
| vc，64 head（DP8） | 1–12 | 13.7–18.0 | 17.0–20.0 |
| kc，8 head（TP8） | 1–16 | 3.3–4.3 | 4.2–5.2 |
| vc，8 head（TP8） | 1–32 | 4.1–7.4 | 4.7–9.0 |

- M 再大时，X 按 mmac 布局反复读取成为瓶颈，torch.bmm 更快，所以回退。调优表是 `hcu_ops._BMM_CFG`。
- **精度：** 与 fp32 参考的误差 ≤ torch.bmm 自身误差的 1.25 倍；所有配置都通过。

### 5.7 本轮新增 kernel 汇总

以下是 c8 时每个 rank M=6 的单算子微基准数据。括号里是 trace 中的实测值；trace 里每个 kernel 至少约 5µs，所以 launch 多的算子实际省得更多。

| 算子 | 原实现 | 新实现 | 数值 | 每层节省 |
|---|---|---|---|---|
| topk-2048（每 4 层 1 次） | lightop 约 380µs（trace 439µs） | 49–61µs | 选出的集合相同 | 按层平均约 90–100µs |
| o_proj + q_b | hipBLASLt 126.8µs（trace 125.6µs） | 91.8µs | 与 hipBLASLt 同量级 | 约 35µs |
| rope | 8 次 launch 20.3µs（trace 约 42µs） | 1 次 2.4µs | 逐 bit 一致 | 18–40µs |
| absorb bmm kc + vc | torch 32.6µs（trace 42.7µs） | 26.1µs | ≤1.25 倍 torch 误差 | 约 6.5µs |
| gate + 后处理 | 10.3 + 5.4µs | 约 7µs | 逐 bit 一致 | 约 9µs |
| act + 量化 | 16.7–18.4µs | 4.6–5.4µs | 逐 bit 一致 | 约 12µs |

合计每层约 180–200µs。78 层加上 MTP draft，每个 decode cycle 理论上能省 14–16ms，实际收益以第 6 节端到端结果为准。

---

## 6. 端到端性能（DP8，64K 输入 / 1024 输出，PC 90%）

### 6.1 测试方法

- **服务：** `opt2/run_ab.sh` 每组用 `opt2/start_k.sh` 起 DP8 服务，开关通过环境变量传入。`start_k.sh` 与最终脚本的启动参数完全相同，只是去掉了 `SGLANG_HCU_*` 的 export，并设置了 profiler 目录。每组跑完就停掉服务，下一组重新启动。
- **压测：** `opt2/ab_bench.sh` 的 vllm bench 参数与 `bench/vllm_bench-pc.sh` 相同：random 数据集，输入 65536，其中 prefix 58982（即 PC 90%），输出 1024，`--ignore-eos`，`temperature 0.01`，`--request-rate inf`。
  - 不同之处：`--num-warmups 8`，保证 8 个 DP rank 都预热到、都建好前缀缓存（原因见第一轮 PC 未命中的分析）；`--num-prompts` 为并发数的 2 倍（复测时 6 倍）。
- **分组：**

| 组 | 开关 | 含义 |
|---|---|---|
| r1 | `SGLANG_HCU_FAST_MASK_TOPK=0`，其余关 | 第一轮最终状态 |
| base | 只开 topk（新默认值） | 本轮 topk 单独的收益 |
| skinny | topk + `SKINNY_GEMM` | |
| all | 全部开关（= 最终脚本） | |

### 6.2 vllm bench 结果

Mean TPOT（ms）/ Mean ITL（ms）/ 输出吞吐（tok/s），num-prompts = 2 × 并发：

| 组 | c1 | c8 | c16 | c32 |
|---|---|---|---|---|
| r1 | 14.65 / 84.20 / 57.95 | 22.31 / 127.02 / 264.88 | 28.87 / 157.39 / 369.41 | 40.91 / 212.12 / 499.29 |
| base | 13.12 / 75.42 / 63.54 | 18.26 / 105.60 / 314.40 | 26.92 / 142.66 / 340.11 | 34.70 / 182.64 / 541.55 |
| base（第 1 次） | 13.43 / 75.70 / 62.47 | 21.64 / 123.36 / 269.78 | 22.93 / 131.69 / 471.90 | 42.58 / 216.75 / 450.00 |
| skinny | 12.69 / 72.73 / 65.42 | 21.11 / 114.24 / 239.44 | 23.27 / 129.08 / 389.28 | 34.66 / 184.46 / 539.66 |
| all | **12.20 / 70.10 / 67.78** | 25.08 / 127.45 / 196.15 | 23.73 / 131.28 / 382.13 | 36.50 / 192.45 / 522.17 |

- **c1 结果稳定，可以直接比较：** r1 → all 的 TPOT 从 14.65 降到 12.20ms（−16.7%），吞吐从 57.95 升到 67.78 tok/s（+17%）。逐项看，topk 贡献 1.5ms，skinny GEMM 0.4ms，rope、gate、act、bmm 合计 0.5ms。
- **c8–c32 的 TPOT 噪声比各组之间的差距还大。** base 用相同配置跑了两次：c8 是 21.64 / 18.26，c16 是 22.93 / 26.92，c32 是 42.58 / 34.70，相差 15–20%；c1 两次是 13.43 / 13.12，很稳定。
  - 原因是 prefill 干扰。dp-attention + EP 下所有 rank 每一步都要同步做 MoE，任何一个 rank 在做 64K prefill（PC 命中后也还有约 6.5K 新 token，即一个 chunk），其他 rank 的 decode 都要等。
  - num-prompts 只有 2 倍并发时，第二批请求是在第一批 decode 过程中陆续进来的，prefill 插在 decode 中间的次数和位置取决于请求完成的先后，每次运行都不一样。TPOT 是每个请求的 (E2E − TTFT) / 输出数，会把这些等待计进去。
  - 所以 c8 时 all 的 25.08ms 并不说明 all 比 r1 慢，要看下面排除了 prefill 的步时。

### 6.3 排除 prefill 后的 decode 步时

`eval/ana_clean.py` 从服务日志里统计 decode 步时：
- 按 10s 切窗口，丢掉任何 rank 有 prefill 的窗口；
- 剩下的窗口按每个 DP rank 的 `#running-req` 分类：`1rank_r1` 是只有 1 个 rank 在 decode 1 个请求（c1 场景），`all8_rK` 是 8 个 rank 都在 decode K 个请求（均衡的 c8 / c16 / c32）；
- 每条 decode 日志的步时 = `#running-req × accept_len / gen_throughput`，即一次 verify + draft 循环的时间；只统计上下文 ≥50K token 的行。

中位数（ms）：

| 场景 | r1 | base | skinny | all | r1 → all |
|---|---|---|---|---|---|
| 1rank_r1（c1） | 84.3 | 75.5 | 73.6 | **71.0** | −13.3ms（−15.8%） |
| all8_r1（c8） | 99.8 | 90.6 | 89.8 | **90.7**（p25 83.6） | −9.1ms（−9.1%） |
| all8_r2（c16） | 116.2 | — | 103.4 | **101.9** | −14.3ms（−12.3%） |
| all8_r4（c32） | 137.7 | 127.4 | 126.3 | **127.0** | −10.7ms（−7.8%） |

（base 那次 c16 没有出现全 8 个 rank 都是 2 个请求的干净窗口。）bench 中的 accept len 都是 6.00（random 数据下模型输出很快进入重复，5 个 draft token 几乎全部被接受），所以各组的步时可以直接比较，不受接受率影响。

- **topk 是主要收益：** r1 → base 每步省 9–10ms，与第 5.1 节估算的约 8ms 一致，且与并发无关，因为每个 rank 每步都要做 topk。
- **其余 5 个算子：** 单 rank 时每步再省 4.5ms（75.5 → 71.0），与第 5.7 节的单算子估算（每层约 80µs × 78 层 ≈ 6ms）同量级。
- **8 个 rank 都在 decode 时，其余 5 个算子的收益基本看不出来**（all8_r1 是 90.6 → 90.7，all8_r4 是 127.4 → 127.0）：
  - 每层 MoE 的 dispatch / combine 要等所有 rank 到齐，单个 rank 省下的 dense 时间会变成等最慢 rank 的时间。
  - MoE masked GEMM 的耗时随路由内容变化：两次 trace 中同一个 kernel 平均每次 131µs 和 149µs，相差 14%（见第 6.5 节）。8 个 rank 每层都取最慢者，这部分波动比省下的几 ms 还大。
  - all8_r1 下 all 组的步时分布是双峰的：p25 是 83.6ms，比 base 的中位数低 7ms，说明在 MoE 负载均衡的步里收益是在的。
- **结论：** 端到端收益以 topk 为主，在所有并发下都有；skinny GEMM、rope、gate、act、bmm 在低并发（c1、少量 rank 有请求）时有 4–5ms/步的收益，高并发下被 EP 同步掩盖。

### 6.4 复测（num-prompts = 6 × 并发）

（待填：`post_eval.sh` 在精度测试后用 r1 与 all 两组复测 64K c8、c16，减小 prefill 干扰占比）

### 6.5 profiling 对比

（待填：all 组 64K c8 的 decode trace 与 base 对比，`tracediff.py`）

base（只开 topk）与第 4 节原始 trace 的对比（TP0，10 步，`tools/tracediff.py`）：
- topk：93.9ms → 15.5ms（新实现是 4 个 kernel，合计）。
- MoE masked GEMM：同一个 kernel 每次 131µs vs 149µs，两次 trace 的差别来自路由内容，不是代码改动。

---

## 7. 精度验证（glm5.2_evalscope）

### 7.1 方法

- **流程：** `eval/drive_eval.sh` 在宿主机上运行，每组依次做以下几步：
  1. 在 `glm5.3-pd` 里用 `opt2/start_k.sh` 加该组开关起 DP8 服务；
  2. 预热每个 DP rank 的采样路径（第 9.1 节）；
  3. 在 `glm5.2_evalscope` 容器（evalscope 1.10.0）里运行 `eval/run_eval.sh <组名>`；
  4. 停服务。
- **分组：**
  - opt：最终脚本的全部开关；
  - r1：第一轮状态，即 `SGLANG_HCU_FAST_MASK_TOPK=0`，其余算子开关关闭。两组都开 `SAMPLER_WARMUP`，它只影响启动，不影响计算结果。
- **采样参数：** 按模型 `generation_config`，temperature 1.0、top_p 0.95，开启 thinking，`reasoning_effort=max`。

| 数据集 | 题数 | max_tokens | 并发（eval batch） |
|---|---|---|---|
| gsm8k | 测试集全部 1319 题 × 1 次 | 16000 | 64 |
| aime26 | 30 题 × 4 次（`--repeats 4`），数据为 `glm5.2_shca/aime26` | 65536 | 32 |

- **与 glm5.2 的 `run_aime26.sh` 的差别：**
  - max_tokens 是 65536，不是 128000。DP8 下一个请求只在一个 rank 上，单 rank 的 KV 池约 103K token，128K 的回答无法完成。
  - 重复 4 次取平均。AIME 只有 30 题，单次结果 ±1 题就是 ±3.3 个百分点。glm5.2 两次单跑分别是 83.33% 和 86.67%。
- **关于 `reasoning_effort`：** GLM-5.3 的 chat template 只认 `low` 和 `high`，其他值或不传都按 `max` 处理，`max` 也是默认值。
  - 第一次评测误用了 `high`：模型在 aime26 上基本不思考，reasoning 只有一句话就直接作答，所以中途停掉了。
  - 那次已经跑完的 gsm8k（opt 组 96.66%，平均输出 162 token）保留在 `eval/opt_high/`，不作为正式结果。

### 7.2 结果

（r1 组运行中，结果出来后补全对比）

| 组 | gsm8k（1319） | aime26（30 × 4） | aime26 每次 | 未截断的答对率 | 截断条数 | 服务崩溃 |
|---|---|---|---|---|---|---|
| opt（最终脚本） | **97.42%** | **84.17%** | 76.67 / 90.00 / 86.67 / 83.33 | 101/102（99.0%） | 18/120（全错） | 0 |
| r1（第一轮） | | | | | | |
| 参考：glm5.2 FP8（128K，单次） | — | 83.33% / 86.67%（两次） | | | 4/30 顶到 128K | |

- aime26 输出 token 中位数 13.4K、平均 23.2K（opt 组），评测统计的平均 TPOT 38.9ms。
- **aime26 的失分全部来自 max_tokens 截断：** opt 组没被截断的 102 条答对 101 条，顶到 65536 的 18 条全部判错：它们都还在 thinking 阶段，正文为空（stop_reason=max_tokens），抽不出答案。
- 同一组 4 次之间相差 13 个百分点（76.67%–90.00%），单次 30 题的成绩不能用来比较算子改动。

（结论待 r1 组结果）

---

## 8. 改动清单与回滚

### 8.1 容器内改动

| 类别 | 改动 | 备份 |
|---|---|---|
| 新增 Python 包 | `/usr/local/lib/python3.10/dist-packages/hcu_ops/`（`_C.so`、`__init__.py`、`skinny_cfg.json`）及 `hcu_ops-0.1.0.dist-info` | `hcu_ops.bak_20261023` 是 10-22 晚上只含 topk 的早期版本，不是镜像原有组件 |
| SGLang 补丁 | 第 3.1 节表中的 8 个文件 | 同目录下的 `*.bak_20261022`（topk）和 `*.bak_20261023`（其余） |
| 第一轮补丁 | `logits_processor.py`、`quantization/unquant.py` | `*.bak_20261021`（未变动） |
| 未改动 | 模型文件、`bench/vllm_bench-pc.sh`、lightop、deepgemm、DeepEP、aiter、rocBLAS/hipBLASLt 等 .so | — |

### 8.2 回滚

- **只关闭某一项优化：** 在启动脚本中把对应开关设为 0，或删掉那一行 export。topk 要显式设 `SGLANG_HCU_FAST_MASK_TOPK=0`，因为它默认开启。
- **完全恢复到第一轮代码：**

```bash
docker exec glm5.3-pd bash -c '
S=/usr/local/lib/python3.10/dist-packages/sglang/srt
cp $S/layers/attention/dsa/dsa_indexer_metadata.py.bak_20261022 $S/layers/attention/dsa/dsa_indexer_metadata.py
for f in layers/quantization/fp8_utils.py layers/linear.py layers/rotary_embedding/base.py \
         layers/moe/topk.py layers/moe/ep_moe/layer.py \
         models/deepseek_common/attention_forward_methods/forward_mla.py layers/sampler.py; do
  cp $S/$f.bak_20261023 $S/$f
done'
```

  恢复这 8 个文件后，SGLang 不再 import hcu_ops，包留在 site-packages 里也不影响运行，不需要卸载。

- **在新容器里重新部署**（同版本镜像）：

```bash
docker exec <容器> bash /data/home/daihw/scripts/glm5.3/opt2_final_20261023/patches/apply_patches.sh
```

  脚本会先编译安装 hcu_ops（需要 dtk 编译环境，约 3 分钟），再从备份打全部补丁，可以重复执行。它按 `patches/../hcu_ops` 找源码，所以要保持导出目录的结构。

- **检查补丁是否完整：** `bash opt2_final_20261023/patches/verify_patches.sh`。脚本在 `/tmp/pt` 下用备份重新打一遍补丁，再与容器内的现行文件比较，8 个文件都应输出 `same`，最后一行是 `verify fail=0`。它不改动 `/tmp/pt` 以外的任何文件。

---

## 9. 注意事项与已知问题

- **TP8 没做端到端测试。** 按约定，本轮没有重启 TP8 做 A/B。各算子在 TP8 下的情况：
  - topk、rope、gate、act 与并行方式无关，可以直接用。
  - bmm 已为 8 head 调过表（5.6 节）。
  - skinny GEMM 的调优表只有 DP8 的形状。TP8 下 o_proj、q_b 等按 TP 切分后形状不同，会自动全部回退 hipBLASLt，不会出错，但也没有收益。
  - 128K 场景只能用 TP8 跑（DP8 每个 rank 的 KV 只有约 104K token）。要拿到 skinny GEMM 的收益，需要先对 TP8 形状跑 `tune_skinny.py` 和 `make_cfg.py`。
- **hcu_ops 只为 gfx938 编译**（`setup.py` 里的 `PYTORCH_ROCM_ARCH`），换卡型要重新编译和调优。
- **补丁对容器内所有 SGLang 服务都生效。** 但除 topk 以外的开关都默认关闭，hook 遇到不支持的形状也会自动回退。
- **低并发吞吐波动大**：MTP 接受长度随内容变化，DP8 下请求少时各 rank 负载不均。c8 及以上的 TPOT 还受 prefill 干扰（第 6.2 节），A/B 以 c1 的 TPOT 和排除 prefill 后的步时（第 6.3 节）为准。
- **日志里的 "errors: 27"** 来自 NUMA balancing 告警等启动信息，与本轮改动无关，每组都相同。

### 9.1 首个采样请求的 "no kernel image" 崩溃（已规避，根因未定位）

- **现象：** 一共出现 3 次。服务启动、CUDA graph 捕获、greedy warmup 都正常；第一个 temperature>0 的请求到达后，某个 DP rank 在 `sampler.py` 的 `sampling_from_probs_torch`（`torch.multinomial`）报错，scheduler 退出，整个服务挂掉：
  ```
  torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
  Search for `hipErrorNoBinaryForGpu' in ROCm documentation for more information.
  ```
  - 第一轮调参时的 opt13、opt13b（DP3）。那时还没有 hcu_ops，补丁只有第一轮的两个，所以**与本轮新写的算子无关**。
  - 本轮 profiling 的 all 组（`opt2/srv_prof_all_crash1023-0442.out`）：DP3 和 DP4 在同一秒报错。
- **触发条件：** 服务的 warmup 请求和 bench 的 warmup 都是 greedy，所以 multinomial 用到的 kernel（max/min 检查、sum、`exponential_`、div、argmax）在第一个采样请求时才第一次 launch，HIP runtime 此时才懒加载对应的 code object。`bench/vllm_bench-pc.sh` 用的是 `TEMPERATURE=0.01`，同样走这条路径；评测用的 temperature 1.0 也一样。
- **排查（工具都在 `opt2_final_20261023/tools/`）：**
  - `scan_bundles.py`：检查 torch 等 .so 里的每个 clang offload bundle，相关 kernel 都有 gfx938 的 code object，不是真的缺架构。
  - `repro.sh`：用崩溃那次（all 组）的开关起服务，不做 warmup，直接发 3 轮 8 × 64K 的 T=0.01 请求，未复现（`repro_all.log`）。
  - `lazy_oom.py`：把显存占到只剩 0 MB 再第一次调用 multinomial，正常。排除了“显存不足导致懒加载失败”。
  - `lazy_race.py` / `lazy_race.sh`：8 张卡上的 8 个进程在同一时刻第一次调用 multinomial，无后台线程 10 轮、有后台 GPU 线程 2 轮，共 96 次首次调用，0 失败（`lazy_race.log`）。
  - 结论：只在完整服务里偶发（两轮几十次服务启动中出现 3 次），推测与多个 rank 并发懒加载 code object 时的 runtime 状态有关，但单独构造不出来，根因没有定位。
- **规避（两层）：**
  - 服务端：`SGLANG_HCU_SAMPLER_WARMUP=1`（`patch_sampler_warmup.py`）。创建 sampler 时用 float32 logits 把 multinomial 和 top-k/top-p 两条采样路径在 bs=1、8 下各跑一次，让懒加载发生在启动阶段（此时没有其他任务），失败自动重试最多 3 次，不会等到请求时才失败。
    - 代价：这些 code object 占的约 70MB 显存在 KV 池分配之前就占用了，每个 rank 的 KV 从 104512 降到 103104 token（−1.3%）。
  - 评测驱动 `eval/drive_eval.sh`：服务就绪后先在每个 DP rank 上各发 T=1.0 和 T=0.01 的请求预热；如果此时服务崩溃，自动重启，最多 3 次。
  - 加上这两层以后，精度评测的各次启动都没有再出现这个错误（第 7 节）。
- **建议：** 生产启动脚本保留 `SGLANG_HCU_SAMPLER_WARMUP=1`。如果用别的启动脚本，至少在服务就绪后先发一轮 T>0 的请求，覆盖到每个 DP rank（round_robin 下发 2×dp 个即可）。

---

## 10. 评估过但没做，以及后续方向

- **MoE masked grouped GEMM（20.4%）：** 实测已接近 3–3.7TB/s 的等效带宽，换 kernel 配置没有收益。下一步只能减少权重字节数，比如 W4A8 专家权重。
- **DeepEP LL combine（10.1%）：** 主要时间在等待其他 rank 的 MoE 计算，需要 two-batch overlap 这类调度层的改动。
- **qkv_a、dense down：** skinny GEMM 没有赢过 hipBLASLt。
- **shared expert：** hipBLASLt 冷启动下已是最优，并且在 alt stream 上与 MoE 重叠，不是瓶颈。
- **lm_head（每次约 0.8–1.6ms）：** 可以评估 FP8 量化或词表并行。
- **小 kernel 融合候选：** `fused_add_rms` + `per_token_quant_fp8`、`concat_mla_absorb_q`、`fused_quantize_and_store_mla_kv` 等，每个 5–13µs。
- **topk：** 可以把 pass-1 直方图融合进 indexer 的 `paged_mqa_logits` kernel，省掉一次 logits 读取。
- **HYGON-AI 开源仓库的参考情况：**
  - lightop 是本轮 rope、gate、act、topk 逐 bit 对齐的基准实现；
  - FlashMLA-das 和 hytlass 下载到 `opt2/build` 作为 MLA decode 和 gfx938 mmac 指令的参考，本轮没有直接复用它们的代码。
