# GLM-5.3 PD 分离 prefill 性能优化报告

节点:dcu4(P,CP8 prefill)+ dcu5(D,DP8 decode),router 在 dcu4
镜像:`sglang:0.5.18-...-20261016-0200-glm5.3_v1023`,模型 `GLM-5.3-Channel-FP8-w8a8-1015`
日期:2026-11-10 ~ 11-11

---

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| **最终配置** | `SGLANG_DSA_FAST_DEQUANT_PAGED=1` + `LIGHTOP_MQA_LOGITS_BETA_FP8=1`,已写入 `p_opt.sh` |
| **缓存命中场景(90%)** | 新 token 吞吐 7206 → **9833 tok/s(+36.5%)**,TTFT 14241 → **10361 ms(−27.2%)** |
| **无缓存场景(0%)** | 六点全测:TTFT **−8%**,吞吐 **+8.4%**(见 1.1) |
| **"缓存命中下吞吐下降"问题** | 退化从 **−34% 收窄到 −13%**,基本解决 |
| **整体 prefill 20% 目标** | **未达成**。在不改 FlashMLA 闭源级 HIP 模板、不用 EPLB 的约束下,实测可达上限约 +8% |

### 1.1 无缓存场景完整验证(40K / 64K × c16 / c32 / c64,output=1)

| 输入 | 并发 | TTFT 基线 → 优化后 | 变化 | 吞吐 基线 → 优化后 |
|---|---|---|---|---|
| 40K | 16 | 52491 → 49643 ms | −5.4% | 10650 → 11266 |
| 40K | 32 | 92822 → 85514 ms | −7.9% | 11358 → 12255 |
| 40K | 64 | 193029 → 178107 ms | −7.7% | 10815 → 11660 |
| 64K | 16 | 84207 → 76665 ms | −9.0% | 10374 → 11345 |
| 64K | 32 | 162690 → 145872 ms | −10.3% | 10407 → 11501 |
| 64K | 64 | 313926 → 290083 ms | −7.6% | 10563 → 11404 |

早期用 64K/c16 单点探针得到的 +3.3% 低估了收益;上表是六点完整测量,应以此为准。

---

## 2. 问题定位:为什么缓存命中反而更慢

### 2.1 现象

| | 0% 命中 | 90% 命中 |
|---|---|---|
| 总输入 tok/s | 10622 | 72931 |
| **新 token tok/s** | **10622** | **7336** |

90% 命中时,208.7 万输入 token 中 188.6 万来自缓存,真正要算的只有 21.1 万,除以 28.76 秒 = 7336 tok/s。
**端到端 TTFT 其实大幅改善(84.2 s → 13.9 s),缓存整体是赚的**,但没达到应有收益——若每 token 效率不变,TTFT 本该降到约 8.5 s。

### 2.2 根因(逐 kernel 对比确认)

同输入同并发下抓两份 prefill trace:

| kernel | 12.5% 命中 | 90% 命中 | 变化 |
|---|---|---|---|
| `sparse_attn_fwd` | 23.2% | 17.1% | 下降 |
| MoE grouped fp8 GEMM | 13.1% | 10.0% | 下降 |
| DeepEP cached_notify_combine | 11.1% | 8.4% | 下降 |
| **`_mqa_logits`(indexer)** | **8.3%** | **18.3%** | **+10.0 点** |
| **`_dequantize_k_cache_paged`** | **3.7%** | **12.2%** | **+8.5 点** |

两个增长项合计 **+18.5 个百分点**,正是退化来源。其余 kernel 占比全部下降(随新 token 数线性缩减),而这两项随**上下文长度**走。

代码层面是明确的:

```python
if _has_prefix:
    kv_cache = dequantize_k_cache_paged(...)   # 只在有缓存时发生
else:
    kv_cache = _cat([k, k_rope], dim=-1)
```

KV 以 fp8 存储(`--kv-cache-dtype fp8_e4m3`),而 `flashmla_sparse` 后端要 bf16,每次读缓存都要把 59K token 反量化一遍。**缓存省下计算,换来按缓存长度计的带宽开销。** indexer 则因为要对全上下文打分,本身就是 O(上下文长度)。

---

## 3. 优化项一:paged K-cache dequant 重写(12.3 倍)

### 3.1 问题

`_dequantize_k_cache_paged_kernel` 是 Triton 实现,launch 配置为 `grid = (num_tokens, 5)`:每个 program 只处理 128 个元素,392832 token 要起 **196 万个 workgroup**,且 `block[256]` 中只有 128 条 lane 有效。

实测:**7309 µs,97 GB/s**,而本机 copy 上限实测 **1361 GB/s** —— 只跑到 7%。

### 3.2 修法

同一个文件里已有现成范式,其注释写着:

> `_GATHER_TOKENS_PER_PROG = 4` — 4 tokens × 512 fp8 = 每线程 16 字节,一次向量化访存

把这个形状套到 paged dequant:`grid = (cdiv(num_tokens, T),)`,每 program 处理 T 个完整 token(512 nope + 4 scales + 64 rope)。

### 3.3 结果(微基准,392832 token,搬 710 MB)

| | 耗时 | 带宽 | 加速 |
|---|---|---|---|
| 现有 kernel | 7309 µs | 97 GB/s | — |
| **v2(T=8, warps=4)** | **592 µs** | **1200 GB/s** | **12.3×** |

达到 copy 上限的 88%,`torch.equal` 验证**逐 bit 一致**(只改任务划分,数值路径未动)。

### 3.4 端到端(各 2 次重复)

| 场景 | base | fastdq | 变化 |
|---|---|---|---|
| 90% 命中 吞吐 | 7156 / 7256 | **8203 / 8208** | **+13.9%** |
| 90% 命中 TTFT | 14342 / 14140 ms | **12512 / 12504 ms** | **−12.2%** |
| 0% 命中 吞吐 | 10929 / 10921 | 11092 / 11095 | +1.5% |

补丁:`pd/pf/patch_fast_dequant.py`,环境变量 `SGLANG_DSA_FAST_DEQUANT_PAGED` 控制,默认关闭,条件不满足自动回退原实现,`--revert` 可还原。

---

## 4. 优化项二:lightop BETA_FP8 indexer(纯环境变量)

`LIGHTOP_MQA_LOGITS_BETA_FP8` 启用 lightop 中较新的 fp8 mqa_logits 实现,其错误信息写明 **"requires gfx938/wave64"** —— 正是本机 GPU。

| 90% 命中 | fastdq 单独 | + BETA_FP8 | 变化 |
|---|---|---|---|
| 新 token 吞吐 | 8280 | **9833** | **+18.8%** |
| TTFT | 12282 ms | **10361 ms** | **−15.6%** |

零代码改动、零风险。

---

## 5. 实测排除的方案

| 方案 | 结果 |
|---|---|
| `--dsa-prefill-backend flashmla_kv` | 0% 场景 **−28%**。免了 dequant 但 attention 更慢,净亏。代码里的启发式(`total_kv < total_q*512` 选 sparse)在我们的形状下判断正确 |
| `--dsa-prefill-backend flashmla_sparse_q8` | 启动失败:`Cannot find CUTLASS headers required for JIT compilation`,镜像缺依赖 |
| `--dsa-prefill-backend flashmla_auto` | 解析结果就是 `flashmla_sparse`,数字逐项吻合,无变化 |
| `--dsa-paged-mqa-logits-backend deepgemm/aiter` | `not supported on HCU; only 'lightop' is implemented` |
| `SGL_USE_LIGHTOP_TOPK_BACKAND=1` | 更差(8025 vs 8280);与 BETA 组合也拖后腿(9591 vs 9833) |
| `SGL_USE_LIGHTOP_TOPK_BACKAND=2` | 服务起不来 |
| `SGLANG_LIGHTOP_DEQUANTIZE_K_CACHE_PAGED` | 对 GLM5.3 无效,`dsa_backend.py` 不读这个开关,只有 DSv4 后端读 |
| 静态 EPLB | 按要求未做 |

---

## 6. 剩余项与可达上限

0% 场景优化后的 kernel 占比:

| kernel | 占比 | 判定 |
|---|---|---|
| **`sparse_attn_fwd`** | **24.2%** | 已在厂商 H64 快路径,但走 **`IS_TOPK_2048=true` 分支**——`FlashMLA-das` 的优化 readme 明说"Avoided index LDS prefetch overhead **when IS_TOPK_2048 is false**",即这条分支未被优化。源码开源,但需改 75KB HIP 模板 + 过 958 回归用例 |
| MoE grouped fp8 GEMM | 13.7% | 厂商 gfx938 特化 `wasp_pair32` kernel,无明显空间 |
| DeepEP 合计 | 20.9% | 通信问题,非算子;EPLB 已排除 |
| `_mqa_logits` | 7.9% | 已用 BETA_FP8 优化 |
| dequant | — | 已优化掉 |

**结论:0% 场景的 +20% 目标在当前约束下达不到。** 唯一能显著推动的路径是改 FlashMLA 的 `IS_TOPK_2048` 分支,属独立的 kernel 工程任务。

---

## 7. 关键事实与踩坑

- **prefill 吞吐与并发、输入长度无关**,稳定在 10.4~11.4K tok/s(每 rank 约 1325),说明算力饱和,**TTFT 就是排队时间**。因此"TTFT 降 20%"等价于"prefill 吞吐提 20%"。
- **GLM5.3 的 `index_topk` 是 2048**(不是 512),决定了走 `IS_TOPK_2048=true` 这条未优化分支。
- **dcu5 重启后只重启 D 不够**:P 持有重启前的 KV 传输连接,router 会报 "No available decode workers" 而网络本身正常。**必须 P 和 D 都重启,再起 router。**
- **`route.sh` 必须在容器内跑**,宿主机没有 `python` 命令。
- **`p_up.sh` 的清理步骤会连带杀掉 router**,每次重启 P 后都要重新拉 router。
- **GPU 微基准不能与服务并跑**(十月在 GLM5.3 上即使有 34 GB 空闲也会 OOM)。
- `prefill_prefix0.sh` 的输出目录名误写成 `prefill_prefix90_*`;`bench_prefix90.py` 必须传 `--output-file`,否则在写 warmup.json 时崩溃。

---

## 8. 交付物

| 位置 | 内容 |
|---|---|
| dcu4 `pd/p_opt.sh` | 最终启动脚本,两个开关默认开启,可用 `FAST_DEQUANT=0` / `BETA_FP8=0` 单独回退;备份 `p_opt.sh.bak_prefillopt_*` |
| dcu4 `pd/pf/patch_fast_dequant.py` | dequant kernel 补丁,可 `--revert` |
| dcu4 `pd/pf/mb_dequant.py` | dequant 微基准(正确性 + 性能) |
| dcu4 `pd/pf/mb_bw.py` / `mb_moe2.py` | 带宽上限 / MoE GEMM 微基准 |
| dcu4 `pd/work/` | 探针与实验脚本(`pdpf_probe.sh`、`ab_one.sh`、`sweep_*.sh`、`final_verify.sh`) |
| dcu4 `pd/pf/` | 全部实测数据(`sweep.csv`、`fastdq.csv`、`final.csv`)与 GPU trace |

---

## 9. 待决策

**是否投入改 FlashMLA-das 的 `IS_TOPK_2048=true` 分支?**

- 收益:`sparse_attn_fwd` 占 0% 场景 24.2%,是唯一有实质空间的大项
- 依据:厂商自己的 readme 记录过同类优化在 D512 分支上拿到 **平均 59.45% 延迟下降**
- 成本:修改 75KB HIP 模板、重编译 flash_mla、通过 958 个回归用例,保守估计数天
- 风险:收益不确定,且会偏离厂商发布版本
