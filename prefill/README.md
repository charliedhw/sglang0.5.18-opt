# GLM-5.3 PD 分离 prefill 优化(2026-11)

这一轮针对 **PD 分离模式下 prefill 节点(CP8)** 的性能,与仓库根目录那套 10 月的 decode 优化相互独立,可以叠加使用。

部署:dcu4 = P(`p_opt.sh`,CP8,chunk 32768),dcu5 = D(`d_opt.sh`,DP8),router 在 P 侧。
镜像 `sglang:0.5.18-...-glm5.3_v1023`,模型 `GLM-5.3-Channel-FP8-w8a8-1015`。

---

## 1. 结果

| 场景 | 指标 | 优化前 | 优化后 | 变化 |
|---|---|---|---|---|
| **90% 前缀缓存命中** | 新 token 吞吐 | 7206 | **9833** tok/s | **+36.5%** |
| | TTFT | 14241 | **10361** ms | **−27.2%** |
| **无缓存(6 点平均)** | TTFT | — | — | **−8%** |
| | 吞吐 | — | — | **+8.4%** |

无缓存场景逐点(output=1,TTFT 即 prefill 耗时):

| 输入 | 并发 | TTFT 优化前 → 后 | 变化 |
|---|---|---|---|
| 40K | 16 / 32 / 64 | 52491→49643 / 92822→85514 / 193029→178107 ms | −5.4% / −7.9% / −7.7% |
| 64K | 16 / 32 / 64 | 84207→76665 / 162690→145872 / 313926→290083 ms | −9.0% / −10.3% / −7.6% |

原始数据见 `results/`。

---

## 2. 两项改动

### 2.1 `SGLANG_DSA_FAST_DEQUANT_PAGED` —— paged K-cache dequant 重写(12.3 倍)

**问题。** `_dequantize_k_cache_paged_kernel` 的 launch 形状是 `grid = (num_tokens, 5)`:每个
program 只搬 128 字节,392832 个 token 要起 **196 万个 workgroup**,而 `block[256]` 里只有 128 条
lane 有效。实测 **97 GB/s**,而本机 copy 上限实测 **1361 GB/s**——只跑到 7%。

**修法。** 同一个源文件里已经有正确的范式,其注释写着:

> `_GATHER_TOKENS_PER_PROG = 4` — 4 tokens × 512 fp8 = 每线程 16 字节,一次向量化访存

把这个形状套到 paged dequant:`grid = (cdiv(num_tokens, T),)`,每个 program 处理 T 个完整 token
(512 nope + 4 个 fp32 scale + 64 rope)。**只改任务划分,数值路径一行没动**,所以输出逐 bit 一致。

**微基准**(392832 token,搬 710 MB,空闲卡):

| | 耗时 | 带宽 | 加速 |
|---|---|---|---|
| 原 kernel | 7309 µs | 97 GB/s | — |
| T=8, warps=4 | **592 µs** | **1200 GB/s** | **12.3×** |

达到 copy 上限的 88%,`torch.equal` 验证一致。复现:`microbench/mb_dequant.py`。

**适用范围。** 这个 kernel 只在**有缓存前缀**时被调用:

```python
if _has_prefix:
    kv_cache = dequantize_k_cache_paged(...)
else:
    kv_cache = _cat([k, k_rope], dim=-1)
```

所以它是缓存路径的修复:90% 命中时值 +14.9%,无缓存时只有 +1.5%。

### 2.2 `LIGHTOP_MQA_LOGITS_BETA_FP8` —— 纯环境变量

lightop 中较新的 fp8 mqa_logits 实现,其自带的错误信息写明 **"requires gfx938/wave64"**——正是
BW1100。零代码改动。在 2.1 之上再 **+18.8%**(90% 命中)。

---

## 3. 用法

```bash
# 安装 kernel 补丁(容器内,不能有服务在跑)
python3 prefill/patches/patch_fast_dequant.py
python3 prefill/patches/patch_fast_dequant.py --revert   # 还原

# 启动 P 节点;两个开关在 p_opt.sh 里默认开启
export NODE_IP=<本节点 ib7 IP>
bash prefill/p_opt.sh

# 单独回退其中一项
FAST_DEQUANT=0 bash prefill/p_opt.sh
BETA_FP8=0     bash prefill/p_opt.sh
```

补丁是幂等的:首次运行备份一次,之后每次都从备份重打。条件不满足(group_size≠128、
dim≠656)时自动回退到原实现。

`route.sh` **必须在容器内执行**,宿主机没有 `python` 命令。启动顺序是 P → D → router。

---

## 4. 实测排除的方案

这些都试过并测过,不要重复:

| 方案 | 结果 |
|---|---|
| `--dsa-prefill-backend flashmla_kv` | 无缓存场景 **−28%**。免了 dequant 但 attention 更慢,净亏。sglang 里 `total_kv < total_q*512 → sparse` 的启发式在这个形状下判断正确 |
| `--dsa-prefill-backend flashmla_sparse_q8` | 起不来:`Cannot find CUTLASS headers required for JIT compilation`,镜像缺依赖 |
| `--dsa-prefill-backend flashmla_auto` | 解析结果就是 `flashmla_sparse`,数字逐项吻合 |
| `--dsa-paged-mqa-logits-backend deepgemm` / `aiter` | `not supported on HCU; only 'lightop' is implemented` |
| `SGL_USE_LIGHTOP_TOPK_BACKAND=1` | 更差(8025 vs 8280);与 BETA_FP8 组合也拖后腿(9591 vs 9833) |
| `SGL_USE_LIGHTOP_TOPK_BACKAND=2` | 服务起不来 |
| `SGLANG_LIGHTOP_DEQUANTIZE_K_CACHE_PAGED` | 对 GLM-5.3 无效:`dsa_backend.py` 不读这个开关,只有 DSv4 的后端读 |
| 静态 EPLB | 按要求未做 |

---

## 5. 已到硬件上限的项

| kernel | 无缓存占比 | 判定依据 |
|---|---|---|
| MoE grouped fp8 GEMM | 13.7% | 走 deepgemm 的 `wasp_pair32_bm256_bn256` 变体,符号 `pair32_wasp_is_gfx938` 表明是厂商针对本架构的特化实现。精确测量被 7 维打包权重格式阻塞(`packed_weight must be 7D`) |
| `sparse_attn_fwd` | 24.2% | 已在厂商 H64 快路径上(`run_h64_fast_path`) |
| DeepEP 合计 | 20.9% | 通信开销,不是算子问题 |

---

## 6. 唯一剩下的空间

GLM-5.3 的 `index_topk` 是 **2048**,所以稀疏 prefill 走的是:

```cpp
KernelTemplate_B_H_64<576, HAVE_TOPK_LENGTH, IS_TOPK_2048=true, SINK, CACHE_INDICES_IN_LDS=false>
```

而 `HYGON-AI/FlashMLA-das`(开源)的 `prefill_kernel_opt_readme` 记录过一次平均 **59.45%** 的延迟
下降,但它明确写的是 "Avoided index LDS prefetch overhead **when IS_TOPK_2048 is false**"——
**`true` 这条分支没有被优化过。**

代价:改 `csrc/gfx93/prefill/sparse/phase1.cuh`(75 KB HIP 模板)、重编译 flash_mla、通过 958 个
回归用例。属独立的 kernel 工程任务,尚未投入。

---

## 7. 方法学要点

- **prefill 吞吐与并发、输入长度无关**,稳定在 10.4~11.4K tok/s(每 rank 约 1325),说明算力已
  饱和、**TTFT 就是排队时间**。因此"TTFT 降 20%"等价于"prefill 吞吐提 20%"。
- **缓存场景要看"新 token 吞吐"而不是总输入吞吐。** 90% 命中时总输入吞吐是 72931 tok/s,看起来
  很高,但真正计算的只有 7336 tok/s——退化就藏在这里。
- **GPU 微基准不能和服务并跑**,否则会 OOM(即使显存看起来够)。
- 逐 kernel 对比(同输入同并发、分别抓 0% 和 90% 两份 trace)是定位缓存退化的唯一可靠手段;
  两个增长项 `_mqa_logits`(8.3%→18.3%)和 `_dequantize_k_cache_paged`(3.7%→12.2%)合计
  +18.5 个百分点,正好解释了 31% 的每 token 效率下降。

---

## 8. 目录

| 路径 | 内容 |
|---|---|
| `patches/patch_fast_dequant.py` | dequant kernel 补丁,幂等、可 `--revert` |
| `patches/dequant_k_cache.py.diff` | 相对镜像原始文件的 diff |
| `microbench/mb_dequant.py` | dequant 正确性 + 性能微基准(含逐 bit 校验) |
| `microbench/mb_bw.py` | 本机 HBM copy 带宽上限 |
| `microbench/mb_moe.py` / `mb_moe2.py` | MoE GEMM 带宽 / 算力核查 |
| `tools/pdpf_probe.sh` | prefill 抽点探针(走 router,output=1) |
| `tools/ab_one.sh` | 单组 A/B:重启 P → flush 缓存 → 测 0% 和 90% |
| `tools/sweep_backends.sh` / `sweep_lightop.sh` | 后端与 lightop 开关扫描 |
| `tools/prof_cache_ab.sh` | 0% vs 90% 的 GPU trace 对比抓取 |
| `tools/final_verify.sh` | 最终配置在 40K/64K × c16/32/64 的验证 |
| `results/*.csv` | 全部实测数据 |
| `p_opt.sh` / `route.sh` | P 节点启动脚本与路由脚本(内网地址已替换为环境变量) |

详细报告见 `../docs/GLM5.3_prefill优化报告_20261111.md`。
