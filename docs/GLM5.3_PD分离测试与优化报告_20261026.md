# GLM-5.3 FP8 PD 分离测试与优化报告（优化镜像 glm5.3_v1023）

日期：2026-10-25 ~ 10-26
节点：dcu4（P，prefill + router）、dcu5（D，decode），BW1100 ×8 / 节点
模型：`/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015`
测试目录：`/data/home/daihw/scripts/glm5.3/pd`，bench 工具 `bench/vllm_bench-pc_1015.sh`（`vllm_bench-pc.sh` 的副本，仅 MODEL_PATH 改为 -1015 权重；原脚本未改动）

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 最终脚本 | P：`pd/p_opt.sh`（dcu4），D：`pd/d_opt.sh`（dcu5），router：原 `route.sh` 不变 |
| 新镜像 vs 10/18 基线（D=DP8） | 对话 c128 ITL 267→178 ms（-34%），输出吞吐 +25%；对话4k c64 吞吐 869→1752 tok/s（+101%）；64K PC90 c32 TTFT 55.0→23.8 s（-57%）、吞吐 +80% |
| 128K 场景 | 10/18 基线全部失败 → 现在 12/12 全部通过（D DP8 mem 0.88，每 DP rank ≈135K KV） |
| P 侧优化（64K chunk + 静态 EPLB） | 单请求 128K TTFT -12%（13.45→11.83 s）；bench：128K PC50 c1/c2/c4 TTFT -11% / -7% / -9%，64K PC50 全并发 -14% ~ -36%，64K PC90 -3% ~ -25% |
| 高并发 128K TTFT | 受 D 容量限制（每 D 节点同时只能装 8 个 128K 请求），P 侧优化对 c≥8 基本无收益（±5% 波动内），需要增加 D 节点 |
| 待确认 | 16K 长文分析 c64 TTFT +79%（31.0→55.3 s）、c2 +36%，其余并发持平或下降，需 A/B（`CHUNK=32768 EPLB_INIT=`）确认是否为 64K chunk 的副作用 |
| 异常 | 10/26 01:46 128K PC90 c16 测试中 dcu5 整机硬挂死（见第 6 节），06:56 重启后补测全部完成，未复现 |

## 2. 部署方式

### 2.1 P（dcu4，`p_opt.sh`，端口 30001）

- 基于 `p_debug.sh`：CP8 prefill（`--attn-cp-size 8 --enable-nsa-prefill-context-parallel`，round-robin-split），DeepEP normal，无 cuda graph，mem 0.86
- 加入优化镜像的 hcu_ops 开关（与 `sglang_start_glm53_opt.sh` 一致；对 prefill 生效的是 rope / NORM_QUANT / SILU_QUANT / FUSED_GATE / NN layout，decode 专用开关在 P 上不起作用）
- **本次新增优化**：`--chunked-prefill-size 65536`（原 32768）+ 静态 EPLB `--init-expert-location pd/eplb/rand_c64k.pt --ep-num-redundant-experts 8`
- 所有开关均可在启动时用环境变量覆盖，回到旧配置：`CHUNK=32768 EPLB_INIT= bash p_opt.sh`

### 2.2 D（dcu5，`d_opt.sh`，端口 30002）

- `D_MODE=dp`（默认）：DP8 dp-attention + EP8 DeepEP-LL，mem 0.88，每 DP rank ≈135K token KV，可容纳单个 128K 请求，cuda graph bs 16
- `D_MODE=tp`：TP8 + EP8，mem 0.82，max-running = graph bs 64；KV 是单一大池，但 decode ITL 明显更差（见第 4 节），仅作备选
- 不能开 `SGLANG_HCU_SKINNY_DRAFT_VP`（dp-attention 下 NCCL 挂死）

### 2.3 启停注意事项

- 启动顺序：P → D → router（`route.sh`，端口 30000，round_robin），起来后先用 :30000 发一条 completion 冒烟
- 进程名是 `sglang serve` / `sglang::scheduler_*` / `sglang::router`；在 `docker exec bash -c` 中 kill 要用 `pkill -f "[s]glang serve"` 这类写法，否则会匹配到自身 bash，导致旧 router 残留、新 router 端口冲突（Prometheus exporter Address already in use），表现为 503 “No available prefill workers”
- P 重启后 router 熔断器会短暂把 P 标记为不可用，需要等冒烟请求成功再开测
- 辅助脚本：`pd/work/restart_p.sh`（重启 P + router 并冒烟），`pd/work/ttft_probe.sh`（单场景 TTFT 探针）

## 3. 性能测试结果

基线 = 10/18 旧镜像（`bench_20261018_103341` / `_132048`，D=DP8）；新镜像 = `bench_20261025_211619`（P chunk 32K，无 EPLB，全量 48 项）；新 P = `bench_20261026_072246` / `_074910` / `_081119`（P chunk 64K + EPLB，长上下文三场景，见 3.4）。3.1~3.3 为新镜像相对旧镜像的对比。

### 3.1 对话 / 对话4k（PC off）

| 场景 | 并发 | 基线 ITL / 吞吐 / TTFT | 新镜像 ITL / 吞吐 / TTFT |
|---|---|---|---|
| 对话 | 1 | 75.3 ms / 34.7 / 566 ms | 56.7 ms / 74.3 / 572 ms |
| 对话 | 16 | 121.3 / 470 / 1349 | 96.8 / 507 / 992 |
| 对话 | 64 | 191.0 / 1117 / 2353 | 140.4 / 1588 / 2001 |
| 对话 | 128 | 267.4 / 1600 / 3847 | 177.6 / 2002 / 3120 |
| 对话4k | 1 | 76.4 / 70.3 / 569 | 58.2 / 65.2 / 592 |
| 对话4k | 16 | 119.5 / 508 / 2585 | 99.3 / 587 / 1377 |
| 对话4k | 64 | 181.0 / 869 / 6056 | 141.1 / 1752 / 5203 |
| 对话4k | 128 | 266.0 / 1410 / 12557 | 183.5 / 2336 / 9005 |

（吞吐为输出 tok/s；ITL 为 Mean ITL，MTP 开启时每步多 token，TPOT 更低）

### 3.2 长文分析 16K（PC off，基线 `bench_20261018_132048`）

| 并发 | 基线 ITL / 吞吐 / TTFT | 新镜像 ITL / 吞吐 / TTFT |
|---|---|---|
| 1 | 78.5 / 68.4 / 1127 | 58.6 / 59.4 / 1032 |
| 8 | 89.7 / 257 / 5130 | 73.6 / 366 / 4210 |
| 32 | 128.3 / 506 / 22605 | 98.3 / 762 / 13577 |
| 64 | 134.0 / 579 / 61551 | 126.8 / 812 / 30969 |
| 128 | 143.6 / 615 / 134388 | 102.1 / 652 / 130244 |

### 3.3 长文64K

| PC | 并发 | 基线 ITL / 吞吐 / TTFT | 新镜像 ITL / 吞吐 / TTFT |
|---|---|---|---|
| 50% | 1 | 87.0 / 56.4 / 2492 | 59.4 / 77.6 / 2436 |
| 50% | 8 | 87.3 / 217 / 17933 | 55.4 / 251 / 19358 |
| 50% | 32 | 91.0 / 232 / 89572 | 61.5 / 274 / 80940 |
| 90% | 2 | 92.4 / 107 / 1970 | 63.4 / 154 / 2076 |
| 90% | 8 | 99.4 / 328 / 6881 | 72.0 / 422 / 5748 |
| 90% | 16 | 100.3 / 293 / 21397 | 80.7 / 471 / 8605 |
| 90% | 32 | 102.9 / 330 / 55026 | 86.4 / 594 / 23773 |

### 3.4 新 P（64K chunk + EPLB）vs 新镜像（32K chunk）：长上下文三场景

新 P 补测：`bench_20261026_072246`（64K）、`_074910`（16K）、`_081119`（128K），对比 `bench_20261025_211619`；对比表 `pd/work/compare_long2.txt`。TTFT 单位 ms，吞吐为输出 tok/s。

| 场景 | PC | 并发 | 旧 TTFT | 新 TTFT | 变化 | 旧吞吐 | 新吞吐 |
|---|---|---|---|---|---|---|---|
| 128K | 50% | 1 | 4785 | 4246 | -11.3% | 66.0 | 56.5 |
| 128K | 50% | 2 | 11710 | 10940 | -6.6% | 79.7 | 91.8 |
| 128K | 50% | 4 | 20334 | 18428 | -9.4% | 116.1 | 120.9 |
| 128K | 50% | 8 | 44891 | 47440 | +5.7% | 129.6 | 135.4 |
| 128K | 50% | 16 | 90532 | 91143 | +0.7% | 128.4 | 139.1 |
| 128K | 50% | 32 | 180895 | 177142 | -2.1% | 134.1 | 141.5 |
| 128K | 90% | 1 | 1945 | 1895 | -2.6% | 82.0 | 81.0 |
| 128K | 90% | 2 | 3563 | 3344 | -6.2% | 137.4 | 137.8 |
| 128K | 90% | 4 | 5813 | 6692 | +15.1% | 208.8 | 214.2 |
| 128K | 90% | 8 | 12082 | 11567 | -4.3% | 297.6 | 305.5 |
| 128K | 90% | 16 | 25901 | 25704 | -0.8% | 331.2 | 344.7 |
| 128K | 90% | 32 | 52598 | 53933 | +2.5% | 372.4 | 378.6 |
| 64K | 50% | 1 | 2436 | 2085 | -14.4% | 77.6 | 71.1 |
| 64K | 50% | 2 | 5611 | 3567 | -36.4% | 120.6 | 109.9 |
| 64K | 50% | 4 | 10998 | 9188 | -16.5% | 162.6 | 191.2 |
| 64K | 50% | 8 | 19358 | 14726 | -23.9% | 251.3 | 288.1 |
| 64K | 50% | 16 | 40919 | 31744 | -22.4% | 262.3 | 292.1 |
| 64K | 50% | 32 | 80940 | 67361 | -16.8% | 273.9 | 323.1 |
| 64K | 90% | 1 | 1141 | 1044 | -8.5% | 89.1 | 87.0 |
| 64K | 90% | 2 | 2076 | 1912 | -7.9% | 153.6 | 147.0 |
| 64K | 90% | 4 | 3364 | 2518 | -25.2% | 255.2 | 253.7 |
| 64K | 90% | 8 | 5748 | 5573 | -3.0% | 422.0 | 359.8 |
| 64K | 90% | 16 | 8604 | 7802 | -9.3% | 471.2 | 506.7 |
| 64K | 90% | 32 | 23773 | 22740 | -4.3% | 593.7 | 575.5 |
| 16K | off | 1 | 1032 | 1090 | +5.7% | 59.4 | 87.3 |
| 16K | off | 2 | 1296 | 1757 | +35.6% | 120.9 | 158.3 |
| 16K | off | 4 | 3068 | 2973 | -3.1% | 272.9 | 224.6 |
| 16K | off | 8 | 4210 | 3371 | -19.9% | 366.1 | 385.4 |
| 16K | off | 16 | 7255 | 7719 | +6.4% | 489.4 | 501.2 |
| 16K | off | 32 | 13577 | 12886 | -5.1% | 762.3 | 767.5 |
| 16K | off | 64 | 30969 | 55312 | +78.6% | 811.6 | 677.3 |
| 16K | off | 128 | 130244 | 121660 | -6.6% | 652.5 | 712.0 |

说明：
- 64K 场景收益最明显（PC50 全并发 -14% ~ -36%，c8~c32 输出吞吐 +15% ~ +18%）：64K 请求的新 token 正好整块进一个 chunk。
- 128K 低并发（c1~c4，PC50）-7% ~ -11%；c≥8 TTFT 不随 P 优化变化：D 每 DP rank 只能装 1 个 128K 请求（≈135K KV），每 D 节点 8 个并发，多余请求在 D 侧排队等 KV，TTFT 主要是排队时间（c16 ≈ 2×c8，c32 ≈ 4×c8）。
- 128K PC90 c4 两次测试均 +14~15%，但 bench 总时长下降（39.2→37.7 s）：64K chunk 把 4 个请求各约 13K 的新 token 合并到同一 chunk，吞吐更高但首 token 同时返回。PC90 c8 两次分别 -21% / -4%，单次测试波动在 ±10% 量级。
- 16K c64 TTFT +79% 且吞吐下降（中位 TTFT 20.6→62.4 s，总时长 161→194 s），而 c8 -20%、c128 -7%（总时长 402→368 s），方向不一致。推测主要原因在 D 侧：c64 时每个 DP rank 约 8 个 16K+1K 请求 ≈136K token，正好压在每 rank ≈135K 的 KV 容量边界上，请求是否需要在 D 排队对时序非常敏感，单次测试波动大；c2 只有 4 个样本（P99 2.85 s 拉高均值）。是否与 64K chunk 有关尚未 A/B 确认（见第 8 节）。
- 10/26 01:46 的第一次测试（`bench_20261026_011149`，dcu5 宕机中断）128K 已完成部分与上表一致（c1 -9%、c4 -9%、c16 +0.4%、c32 -2%）。

## 4. D 模式选择：DP8 vs TP8

| 项目 | DP8（默认） | TP8 |
|---|---|---|
| KV 容量 | 每 DP rank ≈135K（mem 0.88），128K 请求可放下，每节点最多 8 个 | 单一 KV 池（MLA latent 各 rank 复制），容量 ≈ 单 rank，但一个请求不受 rank 切分限制 |
| decode ITL | 好 | c64 时 ITL +61% |
| 稳定性 | 稳定 | bs 超过 cuda graph 最大 bs 走 eager 的 draft 步骤会 OOM，且 /health 仍返回 200，bench 直接挂住；只能 max-running = graph bs |

结论：生产/测试默认 DP8；128K 以上高并发瓶颈在 D 的 KV 容量，建议按 128K 并发需求横向扩 D 节点（1P:2D 等），而不是改 TP8。

## 5. P 侧 128K TTFT 分析

### 5.1 现象

128K PC50 c1 TTFT ≈4.8 s，其中实际要 prefill 的新 token ≈64K。P 的计算代价约 88 µs/token/节点，与 chunk 大小基本无关，即 P 是纯计算瓶颈。

### 5.2 profile（64K chunk，rank0，span 6400 ms，GPU busy 5740 ms）

| kernel | 时间 (ms) | 占比 | 说明 |
|---|---|---|---|
| sparse_attn_fwd（flash_mla sparse prefill） | 1519 | 26% | asm .so，约 122 TFLOPS，无源码 |
| MoE fp8 grouped GEMM | 821 | 14% | |
| cached_notify_combine | 687 | 12% | DeepEP combine 等待，本质是 expert 负载不均衡 |
| combine / dispatch | 366 / 208 | 10% | |
| dense fp8 GEMM | ~507 | 9% | |
| mqa_logits / adaptive_topk | 294 / 155 | 8% | indexer |
| bf16 absorb BMM（w_kc / w_vc） | 151 + 133 | 5% | 大 M 时 hcu_ops skinny bmm 不接管，走 rocBLAS |
| ep_gather / ep_scatter / act | ~260 | 5% | |
| fused_add_rmsnorm | 93 | 2% | ~680 GB/s |

结论：最大的可调项是 combine 等待（专家负载不均）和 chunk 粒度；attention 内核为闭源 asm，暂无替换手段。

### 5.3 优化尝试（探针：32K×24 请求 c8 总时长；128K×4 c2 总时长；128K c1 中位 TTFT）

| 配置 | 32K c8 总时长 | 128K c2 总时长 | 128K c1 TTFT | 结论 |
|---|---|---|---|---|
| 基线（chunk 32K） | 72.0 s | 39.84 s | 13.45 s | |
| DeepEP num_sms 64 | 70.95 s | 39.90 s | | 无收益 |
| 动态 EPLB（--enable-eplb，每 40 iter 重排） | 71.2 s | 40.35 s | | 每次重排要 reload 权重 3.3 s，抵消收益 |
| chunk 64K | 66.5 s | 38.55 s | 12.79 s | 有效 |
| chunk 128K | OOM | | | mem 0.86 下申请 1.8 GiB 失败 |
| **chunk 64K + 静态 EPLB（8 冗余专家）** | **60.15 s（-16%）** | **36.38 s（-9%）** | **11.83 s（-12%）** | **采用** |
| chunk 64K + 静态 EPLB（16 冗余专家） | 59.8 s | 35.67 s | 11.82 s | KV 降到 354K token，收益不足，不采用 |
| `--ep-dispatch-algorithm load_aware` | 启动失败 | | | sglang bug：`LoadAwareReplicaSolver.__init__() got an unexpected keyword argument 'static_probabilities'` |

静态 EPLB 文件说明：`pd/eplb/rand_c64k.pt` 是在 random token 的 bench 负载上录制的专家分布。真实业务流量的专家分布会不同，上线前建议用真实流量重录：

```bash
# P 启动时加 P_EXTRA="--expert-distribution-recorder-mode stat"（同时 EPLB_INIT= 关闭旧分布）
curl -X POST http://127.0.0.1:30001/start_expert_distribution_record
# ... 跑一段真实流量 ...
curl -X POST http://127.0.0.1:30001/stop_expert_distribution_record
curl -X POST http://127.0.0.1:30001/dump_expert_distribution_record   # 容器内 /tmp/expert_distribution_recorder_*.pt
# 拷贝到 pd/eplb/ 后：EPLB_INIT=pd/eplb/xxx.pt bash p_opt.sh
```

或使用 `--enable-eplb` 并把重排间隔设得很大（避免频繁 reload 权重）。

### 5.4 后续方向

见第 8 节。

## 6. 问题记录

1. **dcu5 整机失联（10/26 01:46）**：128K PC90 c16 测试中，P 日志出现 `KVTransferError: Failed to send kv chunk ... to <D 节点 IP>`，此后 dcu5 ping 不通、ssh “No route to host”，router 持续 health check 失败。新镜像同一用例（c16）此前曾正常跑完。dcu5 于 06:56 重启恢复，上一次启动的内核日志在 01:46:09 戛然而止，没有 panic / OOM / MCE 记录（整机硬挂死）；唯一异常是 01:40:02 RDMA 网卡 `shca_core 0000:ad:00.0`（推测为 ib7，NCCL / KV 传输所用）上报 `SHCA_EVENT_TYPE_MONITOR_HW health monitor errcode 0x7e00004`。其余大量 `ib*: failed to attach to multicast group` 为常驻噪声。建议硬件侧检查该网卡及 BMC 日志。恢复后按 64K → 16K → 128K 顺序补测（`pd/work/run_long2.sh`，07:22~08:50），全部完成，128K PC90 c16/c32 未再复现宕机。
2. router 残留导致 503：见 2.3。
3. vllm bench 的 warmup 与第 0 个请求是同一个 prompt，第 0 个请求会命中 P 的 radix cache，探针测试应看中位数 / 总时长，不看均值。
4. `load_aware` 专家分发算法在该镜像的 sglang 中有 bug（见 5.3），不可用。

## 7. 交付文件

| 文件 | 位置 |
|---|---|
| P 启动脚本 | dcu4 `pd/p_opt.sh`（备份 `/tmp/p_opt.sh.bak_0110`） |
| EPLB 分布 | dcu4 `pd/eplb/rand_c64k.pt` |
| D 启动脚本 | dcu5 `pd/d_opt.sh`（10/26 仅更新头部容量注释，备份 `/tmp/d_opt.sh.bak_0712`） |
| router | dcu4 `pd/route.sh`（未改动） |
| 辅助脚本 | dcu4 `pd/work/`：restart_p.sh、ttft_probe.sh、run_long.sh / run_long2.sh（长上下文三场景）、compare_long2.sh（自动对比）、rankspan.py、gaps.py、mb_absorb_bmm.py |
| bench 结果 | dcu4 `pd/bench/`：`bench_20261025_211619`（新镜像全量 48 项）、`bench_20261026_072246` / `_074910` / `_081119`（新 P：64K / 16K / 128K）、`bench_20261026_011149`（新 P 第一次，dcu5 宕机中断）；对比表 `pd/work/compare_long2.txt` |

## 8. 后续可能的优化方向

按预期收益从高到低排列；标注“约定不做”的为本次范围外。

### 8.1 系统 / 部署层面（收益最大）

1. **扩 D 节点（1P:2D 或更多）**：128K c≥8、16K c64 这类场景 TTFT 主要是在 D 侧排队等 KV（每 D 节点 ≈1.08M token KV，最多 8 个 128K 请求）。P 侧再优化对这些点基本无效，加 D 节点可近似线性降低排队时间。
2. **扩 P 节点（2P）+ 缓存感知路由**：64K PC50 c8~c32 TTFT 15~67 s 主要是 P 的计算排队（≈88 µs/token/节点）；多 P 时 router 从 round_robin 改为 cache_aware，同一前缀落到同一 P，还能提高前缀命中、减少重复 prefill。
3. **按业务形态选择 P 的 chunk**：长上下文（≥64K）为主用 64K chunk；若以 16K 级短请求高并发为主，建议先用 `CHUNK=32768 EPLB_INIT=` 做 A/B（尤其 16K c64）再定，必要时 P 按场景分池。
4. **用真实流量重录 EPLB 分布**：当前 `rand_c64k.pt` 来自随机 token 负载，真实文本专家分布不同，重录后 combine 等待（12%）可能进一步下降；也可尝试 `--enable-eplb` + 很大的重排间隔。

### 8.2 P 侧算子（单项收益 2~10%）

5. **sparse prefill attention（26%，最大单项）**：当前为闭源 asm（≈122 TFLOPS）。可向厂商申请针对 prefill 形状（64 头、topk 2048、大 M）调优的版本；或对镜像内 boltops triton / aiter tilelang 实现按 prefill 形状重新 tune（目前只调过 decode 形状）。
6. **MoE 负载均衡 / 通信**：修复 sglang `load_aware` 的 `static_probabilities` 参数 bug（小改动，可开关控制）后试动态副本选择；DeepEP normal 的 combine/dispatch 参数（chunk size、num_sms 组合）细调。
7. **MoE fp8 grouped GEMM（14%）与 dense fp8 GEMM（9%）**：针对 prefill 大 M、CP8 下每 rank 8192 token 的形状做 tile 调优。
8. **增大 chunk 到 128K**：当前 mem 0.86 下 OOM（差 1.8 GiB）；降低激活峰值（如 indexer / MoE 中间 buffer 分块）或略降 mem-fraction 后，128K 请求可一个 chunk 算完，预计 128K TTFT 再降约 5%。
9. **absorb BMM（5%）**：大 M 时走 rocBLAS（微测 ≈94 TFLOPS），可做 fp8 路径或专用大 M kernel，最多省 2~3%。
10. **fused_add_rmsnorm（2%）**：大 M 时带宽仅 ≈680 GB/s，可调优。

### 8.3 约定不做的项（记录备查）

- W4A8 MoE、TBO（two-batch overlap，可重叠 DeepEP 通信与计算，对 combine/dispatch 22% 有潜在收益）、shared expert fp8。

### 8.4 稳定性

- dcu5 RDMA 网卡 `0000:ad:00.0` 硬件告警与整机挂死的关联需硬件侧确认，否则长时间高负载 PD 测试存在再次中断风险。
