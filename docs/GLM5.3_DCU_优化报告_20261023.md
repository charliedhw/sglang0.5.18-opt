# GLM-5.3 FP8 推理优化报告（Hygon DCU BW1100，dcu4，2026-10-23）

本报告合并了两轮工作：
- 第一轮（10-21）：启动参数与 bf16 GEMM 布局；
- 第二轮（10-22 至 10-23）：算子改写与 kernel 融合，分三批，即批次 1、A、B。

两份阶段说明 `GLM5.3_opt_README_20261021.md`、`GLM5.3_opt2_README_20261023.md` 放在交付包的 `docs/` 目录，是本报告的详细版。

---

## 1. 概述与结论

| 项目 | 内容 |
|---|---|
| 服务器 | dcu4，8 × BW1100（gfx938，64 CU，wave64） |
| 容器 | `glm5.3-pd`（SGLang 0.5.18 dtk，torch 2.11，lightop / DeepEP / DeepGEMM / FlashMLA） |
| 模型 | **`/data/model/GLM/GLM-5.3-Channel-FP8-w8a8-1015`**（最终权重）。第一轮和第二轮批次 1 用的是旧权重 `GLM-5.3-Channel-FP8-w8a8`，它有精度问题，已被替换 |
| 部署形态 | DP8 dp-attention + EP8，DeepEP（auto：prefill 走 normal，decode 走 LL）+ DeepGEMM masked MoE，EAGLE MTP 5 步 / topk 1 / 6 draft token，DSA 稀疏 MLA + fp8 KV，page 64 |
| 测试负载 | `bench/vllm_bench-pc.sh`（未修改）。A/B 用其中的 64K 输入 / 1024 输出、PC 90%、并发 1–32 |
| 最终启动脚本 | `sglang_start_glm53_opt.sh`（第 3 节） |
| 精度验证 | `glm5.2_evalscope` 容器跑 gsm8k；aime26 按用户要求由用户自测 |

### 1.1 三轮工作一览

| 阶段 | 内容 | 主要收益 |
|---|---|---|
| 第一轮：参数 | max-running 32→256、mem 0.89、DeepEP LL 256 token/rank、chunked-prefill 16384、prefill-delayer、DeepEP normal 40 SM | 1K/4K 高并发吞吐提升到 2.7–2.9 倍，c128 的 TTFT 从 103s / 205s 降到 4s / 14s |
| 第一轮：布局 | bf16 lm_head / 线性层改走 NN 布局 GEMM（只改权重存储，不改 kernel） | lm_head M=96：40.9ms → 1.6ms |
| 第二轮批次 1 | 新建 `hcu_ops` 算子包，包括 topk-2048、skinny fp8 GEMM、rope、MoE gate、MoE act+quant、absorb bmm，以及 sampler 预热 | 旧权重 64K c1：TPOT 14.65 → 12.20ms（−16.7%），c1 步时 84.3 → 71.0ms |
| 第二轮 A | MLA q 路径：去掉 concat（qcat）、strided RMSNorm、量化复用、-1015 新形状 skinny 调优行、lm_head skinny NN | -1015 64K 步时 c8 83.2 → 79.2ms（−4.8%），c1–c32 降低 0.9–4.8% |
| 第二轮 B | 融合：fused_add_rms+quant、silu_mul+quant、kv_a norm+rope+KV 写入、indexer MQA logits 重写、topk 直方图融进 logits | -1015 64K 步时（含 lm_head skinny）c1 65.4 → 62.7ms（−4.1%），c16 −3.1% |
| **第二轮合计** | 批次 1 + A + B（-1015 权重，w15r1 → 最终脚本） | **64K decode 步时 c1 −23.2%、c8 −19.5%、c16 −13.6%、c32 −11.1%；c1 输出吞吐 59.7 → 74.9 tok/s（+25.5%）** |
| 未做（C） | W4A8 MoE、TBO、shared-expert fp8 | 按用户要求不做（第 15 节） |

### 1.2 结论

- **吞吐（第一轮）：** 主要来自调度参数。原配置全系统最多 32 个运行请求，导致高并发时排队。
  - 1K/1K：c128 输出吞吐 667 → 1808 tok/s，c64 713 → 1225 tok/s。
  - 3.5K/1.5K：c128 541 → 1574 tok/s，c64 553 → 1155 tok/s。
- **64K PC90% 长上下文 decode（第二轮）：** 收益来自算子。
  - 最大的一项是 DSA topk：64K 每次调用 439 → 约 50µs，每步省约 9ms。
  - 其余算子在低并发时每步再省 4–8ms。
  - 旧权重上 c1 的 decode 步时从 84.3ms 降到 71.0ms（−15.8%）。
  - -1015 权重上，A、B 两批在批次 1 的基础上再降低步时 2.3–6.0%（第 11.2 节）。
  - 第二轮总收益（第 11.3 节，-1015 权重，w15r1 → 最终脚本）：decode 步时 c1 81.6 → 62.7ms（**−23.2%**）、c8 −19.5%、c16 −13.6%、c32 −11.1%；中位 ITL c1 81.1 → 62.4ms；c1 输出吞吐 59.7 → 74.9 tok/s（**+25.5%**）。折算 decode 吞吐提升 c1 +30% 至 c32 +12.5%。
- **数值：**
  - 第二轮新写的 kernel 中，除 GEMM 类以外都与原实现**逐 bit 一致**。逐 bit 一致的有 rope、gate、act、qcat、strided RMS、quant 复用、norm+quant、silu+quant、kv 融合、MQA logits、直方图融合。topk 选出的集合也与原实现一致。
  - GEMM 类误差与 hipBLASLt / rocBLAS 同量级，包括 skinny fp8 GEMM、absorb bmm、lm_head skinny。
  - A/B 的 greedy 检查：所有逐 bit 一致的开关打开后，输出与参考全长一致。
- **精度：**
  - 旧权重：gsm8k 97.42%（第一轮 97.57%），aime26 84.17%（第一轮 80.83%），差异在统计波动内。
  - -1015 权重 + 最终脚本：gsm8k **97.27%**（1319 题，用时 11 分 51 秒，评测期间服务无崩溃），aime26 由用户自测。

---

## 2. 交付物

| 位置 | 内容 |
|---|---|
| `/data/home/daihw/scripts/glm5.3/sglang_start_glm53_opt.sh` | **最终启动脚本**（-1015 权重，全部优化开关打开） |
| `/data/home/daihw/scripts/glm5.3/deepep_cfg40.json` | DeepEP normal 模式配置（40 SM），启动脚本按脚本所在目录引用 |
| `/data/home/daihw/scripts/glm5.3/GLM5.3_DCU_优化报告_20261023.md` | 本报告 |
| `/data/home/daihw/scripts/glm5.3/glm53_opt_final_20261023/` | **完整交付包**，目录结构见下表 |
| 容器 `glm5.3-pd` 内 `/opt/glm5.3_opt/` | 交付包的副本，固化镜像时会一起带上 |
| 容器 `glm5.3-pd` 的 site-packages | 已经安装好的补丁和 `hcu_ops`，可用 `verify_all.sh` 核对 |
| 本地 `~/.claude/skills/dcu-model-opt/`（包内 `skill/dcu-model-opt/` 有副本） | DCU 模型优化 skill：流程、工具、踩坑清单、本案例 |

交付包目录：

| 目录 | 内容 |
|---|---|
| `README.md` | 本报告 |
| `start/` | `sglang_start_glm53_opt.sh`、`deepep_cfg40.json` |
| `patches/` | `apply_all.sh`、`verify_all.sh`、`restore_all.sh`、`FILES`（17 个被改文件）、13 个 `patch_*.py`、`round1/`（第一轮的 2 个文件和原始 md5） |
| `diffs/` | `round1/`、`round2/`：每个被改文件相对镜像原始文件的统一 diff |
| `hcu_ops/` | 算子包：`csrc/*.cu`、`bind.cpp`、`setup.py`、`build_inc.sh`、预编译的 `_C.so`、`__init__.py`、`skinny_cfg.json`；`tests/`（逐 bit 单测和计时，`cd tests && python3 test_xxx.py`）；`tune/`（skinny / bmm 调优脚本） |
| `tools/` | profiling 分析、A/B、步时统计、greedy 检查等脚本 |
| `results/` | `ab/` A/B 日志；`steps/` 排除 prefill 后的步时统计；`hooks/` 服务日志中的开关摘录；`prof/` 优化前后 kernel 对比（第 11.4 节）；`eval/` 精度结果 |
| `eval/` | gsm8k 评测脚本：`drive_gsm8k.sh` 在宿主机上运行，`run_gsm8k.sh` 在 evalscope 容器内运行 |
| `docs/` | 两轮的阶段说明 |
| `skill/dcu-model-opt/` | skill 副本 |

---

## 3. 使用方法

### 3.1 启动

```bash
docker exec -it glm5.3-pd bash
cd /data/home/daihw/scripts/glm5.3          # 或 /opt/glm5.3_opt/start
nohup bash sglang_start_glm53_opt.sh > /dev/null 2>&1 &   # 日志：glm53-opt_<时间>.log
# 就绪标志："The server is fired up"
```

- 换权重：`MODEL_PATH=/path/to/model bash sglang_start_glm53_opt.sh`。
- 临时关闭某一项（不用改脚本）：`SGLANG_HCU_MQA_LOGITS=0 bash sglang_start_glm53_opt.sh`。
- 回到“第一轮状态 + -1015 权重”：把第 4 节表中的 15 个 hcu 开关都设为 0（`results/` 里的 w15r1 组就是这样跑的）。
- 启动参数与第二轮的 `sglang_start_opt2_final.sh` 完全相同，只是增加了开关，并把默认权重换成 -1015。

### 3.2 补丁的安装、核对和回滚

均在容器内执行，执行时不能有 sglang 服务在运行：

```bash
bash /opt/glm5.3_opt/patches/apply_all.sh          # 安装预编译 hcu_ops + 第一轮补丁 + 第二轮补丁链（可重复执行）
bash /opt/glm5.3_opt/patches/apply_all.sh --build  # 先在 gfx938 上从源码编译 hcu_ops（需要 dtk 编译环境）
bash /opt/glm5.3_opt/patches/verify_all.sh         # 在 /tmp 镜像目录中用备份重放补丁链，并与现行文件比较，应输出 "verify fail=0"
bash /opt/glm5.3_opt/patches/restore_all.sh        # 回到第一轮代码（每个文件取最早的第二轮备份）
ROUND1=1 bash /opt/glm5.3_opt/patches/restore_all.sh   # 连第一轮也回滚，恢复镜像原始文件
```

- **`apply_all.sh`：**
  - 第一轮的两个文件只有在 md5 等于镜像原始版本时才会被替换；已经是第一轮版本的，直接跳过。
  - 第二轮的 13 个补丁按固定顺序执行：topk → skinny → rope → moe → bmm → mla_rocm_bmm → sampler_warmup → qcat → fuse2 → fuse3 → kvfuse → mqa → lm_head_nn。
  - 每个补丁第一次运行时备份所改文件，以后每次都从备份重打。备份名为 `*.bak_20261022`、`*.bak_20261023`、`*.bak_qcat`、`*.bak_fuse2`、`*.bak_fuse3`、`*.bak_kvfuse`、`*.bak_mqa`。
  - 最后对所有改动文件做 `py_compile`，并在所有开关打开的情况下做一次 import 检查。
- **`restore_all.sh`：**
  - 当前文件另存为 `*.pre_restore_<时间>`，所有备份都保留，之后可以再次执行 `apply_all.sh`。
  - `hcu_ops` 包留在 site-packages 中。回滚后没有代码再 import 它。
- **适用范围：** 补丁对容器内所有 SGLang 服务生效。除了第一轮的两个开关和 topk 以外，其余开关在代码里**默认关闭**，由启动脚本决定是否打开。另外，每个 hook 遇到不支持的形状、dtype 或 stride，都会自动回退到原实现。
- **全新容器验证（10-23）：**
  - 用原始镜像 `sglang:0.5.18-…-scalex40-20261016-0200` 起一个临时容器（`docker run --rm`，挂 `/opt/hyhal` 和 GPU 设备），里面没有任何改动，执行交付包的 `apply_all.sh`。
  - 结果：第一轮两个文件 md5 校验通过并替换，hcu_ops 安装完成，13 个补丁全部打上，`py_compile` 通过，import 检查所有开关为 True（draft_vp 为 False，符合预期），`verify_all.sh` 输出 fail=0，再执行一次 `apply_all.sh` 结果不变。
  - 同一容器内在 GPU 上跑 hcu_ops 单测：norm_quant、kv_fused、mqa_hist（`ALL_OK True`，logits 逐 bit 一致）、qcat（逐 bit 一致）都通过。
  - 没有 GPU 的容器也能做补丁安装和 `verify_all`，但 import 检查会失败：deepgemm 在 import 时要读 GPU 的 CU 数。

---

## 4. 开关表

“代码默认”指不设环境变量时补丁代码里的取值，“启动脚本”指 `sglang_start_glm53_opt.sh` 里的取值。所有开关都可以单独设为 0，回到原实现。

| 环境变量 | 代码默认 | 启动脚本 | 批次 | 作用 | 数值 |
|---|---|---|---|---|---|
| `SGLANG_HCU_LM_HEAD_NN` | 1 | 1 | 第一轮 | lm_head 权重重排为 NN 布局（rocBLAS NN GEMM） | 布局等价 |
| `SGLANG_HCU_LINEAR_NN` | 1 | 1 | 第一轮 | bf16 线性层（N、K ≥ 1024）的权重重排为 NN 布局 | 布局等价 |
| `SGLANG_HCU_FAST_MASK_TOPK` | 1 | 1 | 批次 1 | DSA indexer 的 topk-2048 + page table 变换（split 直方图） | 选出集合一致 |
| `SGLANG_HCU_SKINNY_GEMM` | 0 | 1 | 批次 1 / A | decode 小 M 的 fp8 dense GEMM 走 skinny GEMM（按调优表 `skinny_cfg.json` 选择） | 与 hipBLASLt 同量级 |
| `SGLANG_HCU_FUSED_ROPE` | 0 | 1 | 批次 1 | rope 每层 8 次 launch 合成 1 次 | 逐 bit |
| `SGLANG_HCU_FUSED_GATE` | 0 | 1 | 批次 1 | MoE gate + DeepEP id 后处理 | 逐 bit |
| `SGLANG_HCU_FUSED_ACT` | 0 | 1 | 批次 1 | MoE masked SiLU×mul + fp8 量化 | 逐 bit |
| `SGLANG_HCU_SKINNY_BMM` | 0 | 1 | 批次 1 | MLA absorb bmm（w_kc / w_vc） | ≤ 1.25 倍 torch 误差 |
| `SGLANG_HCU_SAMPLER_WARMUP` | 0 | 1 | 批次 1 | 启动时预热 T>0 的采样 kernel，规避 "no kernel image" | 不影响计算 |
| `SGLANG_HCU_MLA_QCAT` | 0 | 1 | A | absorb bmm 和 q rope 直接写进同一个 q 缓冲，去掉 `concat_mla_absorb_q` | 逐 bit |
| `SGLANG_HCU_STRIDED_RMS` | 0 | 1 | A | q_a / kv_a 的 RMSNorm 直接作用于 strided 视图，去掉 2 次 `.contiguous()` 拷贝 | 逐 bit |
| `SGLANG_HCU_QUANT_REUSE` | 0 | 1 | A | q_lora、hidden_states 的 fp8 量化同时供 q_b / qkv_a 和 indexer wq_b / wk 使用 | 逐 bit |
| `SGLANG_HCU_SKINNY_LM_HEAD` | 0 | 1 | A | 目标模型 verify / draft extend 的 lm_head（M ≤ 64）走 skinny NN GEMM | bf16 舍入级 |
| `SGLANG_HCU_SKINNY_DRAFT_VP` | 0 | **0（固定）** | A | draft lm_head VP 按捕获行数裁剪。**DP 下会 NCCL hang，禁止打开** | — |
| `SGLANG_HCU_NORM_QUANT` | 0 | 1 | B | fused_add_rms_norm + per-token fp8 量化，量化结果给下游 CHANNEL-fp8 线性层复用 | 逐 bit |
| `SGLANG_HCU_SILU_QUANT` | 0 | 1 | B | dense / shared expert 的 silu×mul + fp8 量化 | 逐 bit |
| `SGLANG_HCU_MLA_KV_FUSED` | 0 | 1 | B | kv_a RMSNorm + q/k rope + fp8 KV cache 写入，合成 1 个 kernel | 逐 bit |
| `SGLANG_HCU_MQA_LOGITS` | 0 | 1 | B | DSA indexer logits 改用 `hcu_ops.mqa_logits`（每个 CTA 处理 3 行，每个请求 64 个 CTA） | 逐 bit |
| `SGLANG_HCU_MQA_HIST` | 0 | 1 | B | 把 topk 的第 1 遍直方图融进 logits kernel（仅 G=1 且 ≤ 8 行） | 逐 bit |

辅助变量（一般不需要修改）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `HCU_TOPK_IMPL` / `HCU_TOPK_SPLITS` | split / 8 | topk 实现，以及每行最多切成几个 CTA |
| `HCU_SKINNY_CFG` | 包内 `skinny_cfg.json` | 替换 skinny GEMM 调优表。回到批次 1 的表：指向 `skinny_cfg.json.bak_cfg2` |
| `HCU_MQA_PCTAS` / `HCU_MQA_PCTAS_G1` | −1 | 每个请求的 persistent CTA 数。−1 表示行数 ≤ 12（G6）或 ≤ 4（G1）时取 64，否则用 lightop 默认 |
| `HCU_MQA_RPC` | 3 | G=6 时每个 CTA 处理的行数 |
| `HCU_MQA_HIST_MAX_ROWS` / `_G6` | 8 / 0 | 直方图融合的行数上限。G6 不划算，默认关闭 |
| `HCU_NORM_RSQRT_MODE` | 0 | rsqrt 写法，0 与 lightop 逐 bit 一致 |
| `SGLANG_HCU_NORM_QUANT_MAX_M` | 4096 | norm+quant 融合的最大行数 |
| `SGLANG_HCU_MQA_PCTAS` | 0 | 走 lightop 路径时 MQA logits 的 persistent CTA 数（0 为 lightop 默认） |

---

## 5. 方法

整体流程：摸底基线 → 调启动参数 → profiling 定位 → 写逐 bit 对齐的 kernel → 补丁加开关接入 → 单测 → greedy 一致性 → E2E A/B → 排除 prefill 后的步时 → evalscope 精度。下面是各环节的具体做法。

- **单算子：** 在空闲卡上，用 CUDA graph 连续调用 100 次取平均。正确性用 `torch.equal`（fp8 先 view 成 uint8），覆盖 NaN/inf/±0、全零行、tie、离群值和 ragged 宽度。
- **greedy 一致性：** 每组服务起来以后，3 个 prompt 在 T=0 下各跑 2 次，与参考组比较公共前缀。
  - 逐 bit 一致的改动必须全长相同。
  - GEMM 类改动（skinny 调优行、bmm）会让 greedy 在几十到几百个字符后分叉，这是累加顺序不同造成的正常现象。
- **E2E A/B：** 每组单独起一次 DP8 服务，跑 64K PC90% 的 c1 / c8 / c16 / c32，然后停服务。
  - vllm bench 参数与 `vllm_bench-pc.sh` 相同，只是 `--num-warmups 8`，保证 8 个 DP rank 都建好前缀缓存。
  - **看 ITL，不看 TPOT。** ITL 是一次 verify + draft 循环的时间，也就是步时。TPOT 等于 ITL / 接受长度，而接受长度随生成内容变化：例如 w15base 的 c1 TPOT 是 16.89ms，ITL 却只有 66.21ms，因为那次接受长度偏低。c1 的吞吐同样受接受长度影响。
- **排除 prefill 的步时（`tools/ana_clean.py`）：**
  - 背景：dp-attention + EP 下，任何一个 rank 在做 64K prefill，其他 rank 的 decode 都要等，所以 c8–c32 的 E2E 数字有 ±15–20% 的噪声。
  - 做法：从服务日志中按 10s 切窗口，丢掉有 prefill 的窗口，按每个 rank 的运行请求数分类：`1rank_r1` 对应 c1，`all8_rK` 对应均衡的 c8/c16/c32。然后统计 decode 步时的中位数。
- **精度：** `glm5.2_evalscope` 容器（evalscope 1.10.0）跑 gsm8k 全部 1319 题。
  - 采样参数按模型 `generation_config`：T=1.0、top_p 0.95、thinking 开启、`reasoning_effort=max`，max_tokens 16000。
  - 评测驱动在服务就绪后，先给每个 DP rank 发 T>0 的请求预热；如果服务崩溃，自动重启。

---

## 6. 第一轮：启动参数与 NN 布局（10-21）

### 6.1 启动参数

| 参数 | 原值 | 新值 | 依据 |
|---|---|---|---|
| `--max-running-requests` | 32 | **256** | DP8 下原值相当于每个 rank 只能跑 4 个请求，c64/c128 时 TTFT 达到 33–205s。**这是吞吐提升的主要来源** |
| `--mem-fraction-static` | 0.86 | 0.89 | 为 256 个请求提供 KV 容量 |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 128 | 256 | 每个 rank 32 个请求 × 6 个 draft token = 192 |
| `--chunked-prefill-size` | 32768 | 16384 | 每个 rank 单次 prefill 降到 2048 token，减少对 decode 的阻塞。设 32768 时 c128 会 OOM |
| `--enable-prefill-delayer`，低水位 0.05 | — | 开启 | 让 prefill 合批，减少对 decode 的打断 |
| `--deepep-config deepep_cfg40.json` | 24 SM | 40 SM | prefill 时 DeepEP normal 模式占约 34%。单算子扫描 SM 数：24/32/40/48 时 dispatch 为 1117/900/795/765µs，combine 为 1334/1212/1175/1304µs |

试过但没有采用的：
- `--enforce-shared-experts-fusion`：崩溃。
- `--speculative-skip-dp-mlp-sync`：崩溃。
- local-control-broadcast：无收益。
- MTP 7/8：变差。
- `--load-balance-method total_requests`：c16 崩溃，而且更慢。
- hipBLASLt 后端、EPLB：没有收益。

完整的 opt1–opt16 试验表见 `docs/GLM5.3_opt_README_20261021.md` 5.3 节。

### 6.2 bf16 GEMM 改 NN 布局

- **问题：** lm_head（154880×6144）、前 3 层 dense、indexer、eh_proj、shared expert 等 bf16 权重走 `F.linear`，实际调用 rocBLAS 的 NT GEMM。在 gfx938 上，rocBLAS 给这些形状选了 `MT32x32x4` 这类小 tile kernel：lm_head 在 M=96 时要 40.9ms。
- **做法：** 第一次调用时执行一次 `w.data = w.data.t().contiguous().t()`。逻辑形状和数值不变，只是存储变成 [K, N] 连续，rocBLAS 改走 NN GEMM。
- **效果：**
  - lm_head：M=6 时 3424 → 1595µs，M=96 时 40879 → 1599µs。
  - dense o_proj / down 大约快 2 倍。
  - 两个补丁一起，ITL 降低约 5%。

### 6.3 第一轮结果（旧权重，完整 `vllm_bench-pc.sh`）

| 场景 | c32 | c64 | c128 | c128 TTFT |
|---|---|---|---|---|
| 1K/1K 输出 tok/s | 603 → **723**（+20%） | 713 → **1225**（+72%） | 667 → **1808**（+171%） | 102.7s → 4.1s |
| 3.5K/1.5K 输出 tok/s | 541 → **811**（+50%） | 553 → **1155**（+109%） | 541 → **1574**（+191%） | 204.7s → 13.8s |

c16 及以下与原配置持平。差异在 MTP 接受长度的波动范围内，详见第一轮说明 7.2 节。

---

## 7. 第二轮 profiling 定位（64K PC90%，c8 decode）

一个 rank 10 步，GPU 忙碌 946ms：

| kernel | 占比 | 平均 | 处理 |
|---|---|---|---|
| MoE grouped fp8 masked GEMM | 20.4% | 131µs | 已到 3–3.7 TB/s 等效带宽，改 kernel 没有空间（要减少字节数只能走 C 项 W4A8） |
| DeepEP LL combine | 10.1% | 64.6µs | 主要在等其他 rank（C 项 TBO） |
| **lightop topk（sparse mask）** | **9.9%** | **439µs** | 批次 1：split 直方图 topk |
| hipBLASLt F8BS MT48x16x256（dense fp8 GEMM） | 8.2% | 54µs | 批次 1 / A：skinny GEMM |
| flash sparse MLA decode | 5.4% | 65.9µs | B：评估固定 splits，没有收益 |
| lm_head（BBH MT256x256x16） | 4.9% | 792µs | A：skinny NN |
| rotary_embedding（6152 次） | 3.4% | 5.3µs | 批次 1：8 次 launch 合成 1 次 |
| paged_mqa_logits（indexer） | 2.1% | 91.5µs | B：重写 + 直方图融合 |

单层 kernel 序列中还有一批 5–17µs 的小 kernel：concat_mla_absorb_q、2 次 `.contiguous()`、4 次 per-token quant、kv_a norm、KV 写入、silu×mul 等。在 HIP graph 下每个 kernel 至少约 5µs，所以**合并 launch 本身就有收益**。这就是 A、B 两批的目标。

优化后的 profile 对比见第 11.4 节：每步 kernel 时间 83.1 → 76.2ms，kernel 数减少 20%。

---

## 8. 第二轮批次 1：hcu_ops 基础算子（详见 `docs/GLM5.3_opt2_README_20261023.md` 第 5 节）

单算子时间单位 µs，在空闲卡上用 CUDA graph 计时，形状为 c8 时每个 rank M=6。

| 算子 | 原实现 | 新实现 | 数值 | 要点 |
|---|---|---|---|---|
| DSA topk-2048（64K） | lightop 379（trace 439） | **49** | 选出集合一致 | 原实现每行 1 个 CTA、4 轮 radix。新实现是 12/12/8 位的 3 遍直方图加 1 遍压缩；每行切到最多 8 个 CTA，slice 数在 device 端计算，兼容 CUDA graph |
| o_proj fp8（6144×16384） | hipBLASLt 96.5 | 65.7–71.4 | 同量级 | mmac 16x16x32 fp8；W 按每行连续 256B 读进 LDS，X 也放进 LDS；按 (N, K, M) 生成调优表 |
| q_b fp8（16384×2048） | 29.9 | 23.8–28.1 | 同量级 | 同上 |
| rope（64 head） | 8 次 launch 20.3 | 1 次 2.4 | 逐 bit | head 铺在 gridDim.y 上，q 和 k 在同一次 launch 里完成 |
| MoE gate + 后处理 | 10.3 + 5.4 | 6.8–7.3 | 逐 bit | 每个 token 16 个 lane，用 DPP 做 max 归约，没有 scratch |
| MoE act + 量化 | 16.7–18.4 | 4.6–5.4 | 逐 bit（含 NaN/inf） | 用 masked_m 前缀和给有效行编号，固定 512 个 CTA，按行数自动决定每行拆给几个 wave |
| absorb bmm kc + vc | 32.6 | 26.1 | ≤ 1.25 倍 torch 误差 | 批量 skinny GEMM，支持任意 stride |
| sampler 预热 | — | — | 不影响计算 | 规避第一个 T>0 请求偶发的 `hipErrorNoBinaryForGpu` |

旧权重 64K A/B，c1 结果：

| 组 | 内容 | TPOT（ms） | 吞吐（tok/s） | 1rank_r1 步时 |
|---|---|---|---|---|
| r1 | 第一轮状态 | 14.65 | 57.95 | 84.3ms |
| base | + topk | 13.12 | 63.54 | 75.5ms |
| skinny | + skinny GEMM | 12.69 | 65.42 | 73.6ms |
| all | + 批次 1 全部 | 12.20 | 67.78 | 71.0ms |

---

## 9. 第二轮 A：MLA q 路径（-1015 权重）

### 9.1 qcat：去掉 `concat_mla_absorb_q`（`SGLANG_HCU_MLA_QCAT`，`patch_qcat.py`）

- **原流程：**
  1. absorb bmm 输出 q_nope [T, 64, 512]；
  2. rope 原地旋转 q_pe [T, 64, 64]；
  3. `concat_mla_absorb_q` 把两者拼成 [T, 64, 576] 送给 FlashMLA，约 12.5µs/层。
- **新流程：**
  - 先分配一个 [T, 64, 576] 缓冲：bmm 直接写 `[..., :512]`，`hcu_ops.rope_qout` 把旋转后的 q_pe 写到 `[..., 512:]`。rope_qout 的计算表达式与 FUSED_ROPE 相同，k_pe 仍然原地旋转。
  - `dsa_backend.py` 的 flashmla_kv 路径用 `hcu_ops.adjacent_q_all` 检查两个视图在内存中是否相邻。相邻就直接当成 q_all 使用（零拷贝），否则仍走原 concat。
  - 适用于 decode、target verify、draft extend。
- **结果：** bmm + rope + concat 从 23.0µs 降到 13.6µs，每层省约 9.4µs，逐 bit 一致。

### 9.2 strided RMSNorm（`SGLANG_HCU_STRIDED_RMS`，`patch_fuse2.py`）

- **原流程：** q_a（宽 2048）和 kv_a（宽 512）是打包的 qkv latent 的行切片，`RMSNorm.forward_hip` 先 `.contiguous()` 再调用 lightop `rms_norm_opt`，每层多出 2 个拷贝 kernel。
- **新流程：** lightop `rms_norm_opt` 本身支持行 stride，所以直接传入 strided 视图。结果逐 bit 一致，每层少 2 个 kernel。

### 9.3 量化复用（`SGLANG_HCU_QUANT_REUSE`，`patch_fuse2.py`）

- **背景：** -1015 权重是 compressed-tensors 格式，CHANNEL fp8 w8a8 FP8_DYNAMIC，indexer 的 wq_b / wk 也是 fp8。
- **原流程：** indexer wq_b 的输入是 q_lora，wk 的输入是 hidden_states，这两个 bf16 张量已经被 q_b_proj 和 fused_qkv_a_proj 各量化过一次（per-token fp8），indexer 又各量化一次。
- **新流程：** 每个输入只量化一次，(q, scale) 同时交给两个 GEMM，每层少 2 个量化 kernel。
  - 使用前检查源张量是否就是 GEMM 的输入，不是就重新量化。
  - 改动的文件：`compressed_tensors_w8a8_fp8.py`、`deepseek_v2.py`、`forward_mla_rocm.py`、`dsa_indexer.py`。
- 结果逐 bit 一致。

### 9.4 -1015 新增 fp8 形状的 skinny 调优行（`skinny_cfg.json`，随 `SGLANG_HCU_SKINNY_GEMM` 生效）

- **背景：** -1015 权重把 indexer wq_b、wk 和 shared expert down 也量化成了 fp8，这些形状在批次 1 的调优表里没有。
- **做法：** 重新扫描后，加入快 ≥ 5% 的点：

| 形状（N×K） | 用途 | 接入的 M | 每次节省 |
|---|---|---|---|
| 4096×2048 | indexer wq_b | 1–8 | 约 1–1.8µs |
| 6144×2048 | shared expert down | 1–8 | 约 1–1.8µs |
| 128×6144 | indexer wk | 3–5 | 约 1µs |

- **数值：** 与 fp32 参考的误差为 1.66e-3，和 hipBLASLt 相同，但不是逐 bit 一致。所以 greedy 输出在几十个字符后分叉（A/B 中的 f2cfg 组），属于预期现象。
- 要回到批次 1 的表：`HCU_SKINNY_CFG=<site-packages>/hcu_ops/skinny_cfg.json.bak_cfg2`。

### 9.5 lm_head skinny NN GEMM（`SGLANG_HCU_SKINNY_LM_HEAD`，`patch_lm_head_nn.py`）

- **背景：** 经过第一轮的 NN 重排后，lm_head 每次 verify 约 1.58–1.6ms，相当于完整读一遍 1.9GB 权重，约 1.2 TB/s。`--enable-dp-lm-head` 下每个 rank 都持有完整词表。
- **新实现：** `hcu_ops.skinny_nn` 直接读 n 连续的 NN 权重。每个 CTA 负责 128 列，每个 lane 每个 chunk 处理 h 个 16 行的 k 切片，预取 pf 个 chunk。

| 形状 | M | skinny NN | rocBLAS matmul |
|---|---|---|---|
| 154880×6144（目标模型 lm_head） | ≤ 48 | 1110–1216µs | 1583–1595µs |
| | 49–64 | 1274–1326µs | 1583–1595µs |
| 19360×6144（draft VP 分片） | ≤ 48 | 156–201µs | 208–392µs |

- **数值：** 与 matmul 的最大差 ≤ 0.031，属于 bf16 舍入级。A/B 中 greedy 输出与参考全长一致（w15lm、f3allb）。
- **E2E：** c1 ITL 降低 2.4–3ms（w15base 66.21 → w15lm 63.78；f2all 64.62 → f2lm 61.66）。
- **问题与处理：**
  - 最初这个开关还包含 draft lm_head VP 的“按捕获行数裁剪”。结果 f2lm 组在 c8 时 NCCL hang，压测超时被杀。
  - 原因：DP 的各个 rank 可能同时处于不同模式，一个在重放 decode graph，另一个在跑 eager 的 prefill / draft extend（用固定大小的缓冲），all_gather 的尺寸就不一致了。
  - 处理：这部分拆成独立开关 `SGLANG_HCU_SKINNY_DRAFT_VP`，默认关闭，启动脚本里固定为 0。
  - `SGLANG_HCU_SKINNY_LM_HEAD` 现在只替换 GEMM 本身，不改变任何通信尺寸。f3allb 组（最终脚本）复核了 c1/c8/c16/c32，全部完成，没有 hang，服务日志里没有 Traceback，greedy 输出全长一致。

---

## 10. 第二轮 B：kernel 融合与 indexer（-1015 权重）

### 10.1 fused_add_rms_norm + fp8 量化（`SGLANG_HCU_NORM_QUANT`，`patch_fuse3.py`）

- **原流程：** decoder 层的 input / post-attention RMSNorm 用 lightop `fused_add_rms_norm_opt`，输出 bf16 x；随后 qkv_a + indexer wk、shared expert / dense 的 gate_up 各自对 x 做 per-token fp8 量化。
- **新流程：**
  - `hcu_ops.fused_add_rms_quant` 在一个 kernel 里完成 `residual += x`、RMSNorm 和 per-token fp8 量化，然后把 (x, x._version, q, s) 暂存起来。
  - 下游的 CHANNEL-fp8 线性层如果发现输入正是这个 x，并且之后没有被修改过（`_version` 相同），就直接用 (q, s)。
  - 每层少 2 个量化 kernel。M 上限由 `SGLANG_HCU_NORM_QUANT_MAX_M` 控制，默认 4096。
- **数值：** x、residual、q、s 都与“lightop + scaled_fp8_quant”逐 bit 一致。rsqrt 写法由 `HCU_NORM_RSQRT_MODE=0` 控制，与 lightop 编译出的 c10 rsqrt 相同。
- **单算子：** M=8 时 10.29 → 8.45µs，M=192 时 20.62 → 19.70µs（不计省掉的下游量化 kernel）。

### 10.2 silu×mul + fp8 量化（`SGLANG_HCU_SILU_QUANT`，`patch_fuse3.py`）

- **范围：** dense 层和 shared expert 的 `DeepseekV2MLP`。
- **做法：** SiluAndMul 和 down_proj 的输入量化合成一个 kernel，bf16 乘积不写回显存，down_proj 直接用 (q, s)。每层少 1 个 kernel。
- **数值：** 与 lightop `fuse_silu_and_mul` + `per_token_quant_fp8` 逐 bit 一致。
- **单算子：** d=2048、M=8 时 5.20 → 3.10µs；d=12288、M=192 时 25.72 → 18.04µs。

### 10.3 kv_a norm + q/k rope + fp8 KV 写入（`SGLANG_HCU_MLA_KV_FUSED`，`patch_kvfuse.py`）

- **原流程：** 3 个 kernel：
  1. kv_a_layernorm（lightop `rms_norm_opt`）；
  2. q / k rope；
  3. `fused_quantize_and_store_mla_kv_cache`（DSA fp8 656 字节的 slot）。
- **新流程：** `hcu_ops.rope_qout_kv_store` 一次 launch 完成全部三步：
  - latent 在打包缓冲里原地归一化，k_pe 原地旋转；
  - q_pe 旋转后写进 qcat 缓冲的 rope 段；
  - 写入本层的 KV slot；
  - 之后 attn_mqa 拿到 `k=None, save_kv_cache=False`，不再重复写 KV。
- **适用范围：** decode、target verify、draft extend，并且必须是 flashmla_kv + lightop fp8 KV 存储。不满足条件就走原路径。
- **数值：** q_out、kv_c、k_pe 和 KV slot 都逐 bit 一致。
- **单算子：** T=4 时 12.69 → 3.45µs，T=192 时 23.03 → 9.70µs。
- **接入时的坑：** dtk 0.5.18 的 `ForwardBatch` 没有 `attn_backend` 属性，要用 `forward_context.get_attn_backend()`。

### 10.4 indexer MQA logits 重写（`SGLANG_HCU_MQA_LOGITS`，`patch_mqa.py`）

- **原实现：** lightop `page_mqa_logits_sparse_mask[_grouped]`。G=6 是 MTP verify，G=1 是单行 decode / draft。
  - 默认 persistent CTA 数为 min(chunks, 32)，请求少时用不满 64 个 CU。
  - G=6 时每行单独读一遍 K tile。
- **新实现（`hcu_ops.mqa_logits`，`mqa_hist.cu`）：**
  - 计算表达式与 lightop 相同，logits 和 mask 逐 bit 一致。
  - G=6 时每个 CTA 处理 3 行（`HCU_MQA_RPC=3`），一次 K tile 加载供 3 行使用。
  - 请求少时（G6 ≤ 12 行，G1 ≤ 4 行）每个请求用 64 个 persistent CTA，铺满所有 CU；请求多时退回 lightop 的默认切分。
- **单算子（µs，graph 内计时，`ut/mqa_fair.out`）：**
  - 公平对比：两边都设 `LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA=3`（原启动脚本 `sglang_start.sh_1017` 本来就设了），lightop 再扫 persistent CTA 数（`LT_PCTAS` 0/64/128，取最好的）。
  - hcu 列按上线逻辑取值：G6 ≤ 12 行、G1 ≤ 4 行时每请求 64 个 CTA，否则用默认切分。
  - 所有场景 logits / mask 逐 bit 一致，5 组 `ALL_OK True`。

| 场景 | 生产 lightop（默认 CTA 数） | lightop rpc3 + 调过 CTA 数 | hcu | hcu 相对调过的 lightop |
|---|---|---|---|---|
| G6 16K，1 请求 | — | 33.4 | **21.6** | −35% |
| G6 16K，4 / 8 请求 | — | 47.4 / 74.0 | 46.0 / 73.8 | 持平 |
| G6 64K，1 请求 | 93.2 | 47.3–56.6 | **43.3** | −8% |
| G6 64K，2 请求 | — | 80.8 | **71.5** | −12% |
| G6 64K，4 请求 | — | 142.9 | **121.1** | −15% |
| G6 64K，8 请求 | — | 257.2–272.4 | **210.9** | −18% |
| G6 128K，1 / 2 请求 | — | 80.7 / 143.8 | **72.5 / 121.0** | −10% / −16% |
| G1 64K，1 / 2 请求 | — | 31.0 / 31.2 | **20.3 / 21.8** | −35% / −30% |
| G1 64K，4 / 8 请求 | — | 37.1 / 70.8 | 39.0 / 73.0 | +5% / +3%（但加上直方图融合后，logits+topk 合计仍省 4–5µs，见 10.5） |

  - 结论：收益主要来自 rpc3 之外的两点：请求少时铺满 CU，以及 G6 下 3 行共用一次 K tile。G1 在 4–8 个请求时单看 logits 略慢，靠直方图融合补回。

### 10.5 topk 直方图融进 logits kernel（`SGLANG_HCU_MQA_HIST`，`patch_mqa.py`）

- **做法：**
  - 在 logits kernel 写出 logits 的同时，构建 split topk 的第 1 遍（12 位直方图），并代替 topk 的两次 workspace memset。
  - topk 只再跑第 2–4 遍，从而省掉一次对 logits 的完整读取。
  - 两次调用之间通过 (logits 张量, lengths 指针) 配对；配对失败时 topk 自己重做第 1 遍。
- **结果：**
  - G=1 时，直方图让 logits kernel 多花 3–5µs，但 topk 省下约 10µs，净省 4–9µs。只在行数 ≤ 8 时打开（`HCU_MQA_HIST_MAX_ROWS=8`）。
  - **G=6 不划算：** R=2/3 的 kernel 只有 occupancy 1，LDS 原子操作要多花 15–100µs，超过节省的时间，所以默认关闭（`HCU_MQA_HIST_MAX_ROWS_G6=0`）。
  - 启动日志里的 `group_size=6 hist=True` 只是打印开关的值，G6 实际上不会走融合路径。
- **数值：** 直方图和 topk 结果与非融合版逐 bit 一致（`test_mqa_hist.py`）。

### 10.6 评估后放弃的项

| 项 | 结论 | 原因 |
|---|---|---|
| FlashMLA 固定 num_splits / persistent CTA | 放弃 | 64K decode 下没有收益，而且改变 split 会改变累加顺序，不是逐 bit 一致 |
| draft lm_head VP 按捕获行数裁剪 | 放弃（固定关闭） | DP 下 c ≥ 8 时 NCCL hang（第 9.5 节） |
| G=6 直方图融合 | 默认关闭 | occupancy 1 加上 LDS 原子操作，得不偿失 |
| qkv_a、dense down 的 skinny GEMM | 不接入 | 从来没有赢过 hipBLASLt |

---

## 11. 端到端结果

### 11.1 旧权重（参考）

见第 6.3 节（第一轮）和第 8 节（批次 1）。批次 1 在旧权重上排除 prefill 后的步时（中位数，ms）：

| 场景 | r1 | all | 变化 |
|---|---|---|---|
| 1rank_r1（c1） | 84.3 | 71.0 | −15.8% |
| all8_r1（c8） | 99.8 | 90.7 | −9.1% |
| all8_r2（c16） | 116.2 | 101.9 | −12.3% |
| all8_r4（c32） | 137.7 | 127.0 | −7.8% |

### 11.2 -1015 权重：A / B 分组 A/B（64K 输入 / 1024 输出，PC90%，num-prompts = 2 × 并发）

| 组 | 开关（在批次 1 全开的基础上） |
|---|---|
| w15r1 | 15 个 hcu 开关全关，即第一轮状态（换成 -1015 权重） |
| w15base | 批次 1 全开（第二轮批次 1 的最终状态），skinny 表还没有 -1015 新行 |
| w15lm | w15base + SKINNY_LM_HEAD（只跑了 c1） |
| f2cfg | + -1015 skinny 调优行 |
| f2qcat | f2cfg + MLA_QCAT |
| f2all / f2allb | f2cfg + A 的 3 个开关（QCAT、STRIDED_RMS、QUANT_REUSE），同一配置跑了两次 |
| f2lm | f2all + SKINNY_LM_HEAD（当时还包含 draft VP 裁剪），c8 hang |
| **f3allb** | f2all + B 全部 + SKINNY_LM_HEAD（draft VP 关闭），**即最终脚本** |

**Median ITL（ms，越低越好；对 prefill 卡顿不敏感，是最可靠的 E2E 指标）**

| 组 | c1 | c8 | c16 | c32 | 接受长度（DP0） | greedy 与参考 |
|---|---|---|---|---|---|---|
| w15r1 | 81.11 | 98.21 | 113.29 | 135.03 | 5.384 | 与 w15base 分叉（批次 1 的 GEMM 类不是逐 bit 一致，属预期） |
| w15base | 66.26 | 83.14 | 98.67 | 123.24 | 5.654 | 参考 |
| w15lm | 63.63 | — | — | — | — | 全长一致 |
| f2cfg | 65.54 | 83.01 | 97.14 | 124.53 | 5.129 | 分叉（调优行不是逐 bit 一致，属预期） |
| f2qcat | 64.71 | 81.10 | 97.50 | 123.75 | 5.465 | 与 f2cfg 一致 |
| f2all | 64.59 | 79.20 | 94.17 | 122.32 | 5.344 | 与 f2cfg 一致 |
| f2allb | 64.26 | 79.26 | 95.09 | 120.51 | 5.325 | 与 f2all 全长一致 |
| f2lm | 61.69 | hang | — | — | — | — |
| **f3allb** | **62.38** | **77.59** | **95.13** | **119.54** | 5.175 | 与 f2allb 全长一致 |

**Mean ITL（ms）**：会被 prefill 卡顿拉高（P99 ITL 达到 0.3–3.2s），只作参考。

| 组 | c1 | c8 | c16 | c32 |
|---|---|---|---|---|
| w15r1 | 81.06 | 123.92 | 145.41 | 213.34 |
| w15base | 66.21 | 101.24 | 129.21 | 213.89 |
| f2cfg | 65.49 | 102.59 | 131.26 | 185.79 |
| f2qcat | 64.73 | 99.17 | 130.09 | 195.39 |
| f2all | 64.62 | 94.43 | 122.73 | 187.33 |
| f2allb | 64.36 | 97.72 | 118.09 | 179.99 |
| **f3allb** | **62.00** | 119.01* | 127.03* | 193.04* |

\* f3allb 的 c8/c16/c32 分别有一次 1.3s、0.7s、3.2s 的 prefill 卡顿（P99 ITL），把平均值拉高了；中位 ITL 和步时都不受影响。

**输出吞吐（tok/s）**

| 组 | c1 | c8 | c16 | c32 |
|---|---|---|---|---|
| w15r1 | 59.66 | 207.74 | 363.70 | 478.62 |
| w15base | 51.40** | 304.58 | 345.03 | 495.97 |
| f2cfg | 72.13 | 297.75 | 366.44 | 550.57 |
| f2qcat | 71.72 | 316.88 | 347.19 | 531.43 |
| f2all | 72.50 | 343.07 | 496.16 | 530.14 |
| f2allb | 72.26 | 339.25 | 409.05 | 581.58 |
| **f3allb** | **74.89** | 287.52 | 366.52 | 570.03 |

\*\* w15base 的 c1 那次接受长度偏低（TPOT 16.89ms，但 ITL 只有 66.21ms），吞吐不能直接比较。c1 应该看 ITL。

**排除 prefill 后的 decode 步时（中位数，ms）**

| 场景 | w15r1 | w15base | f2allb | f3allb | w15base → f3allb |
|---|---|---|---|---|---|
| 1rank_r1（c1） | 81.6 | 66.6 | 65.4 | **62.7** | −5.9% |
| all8_r1（c8） | 97.2 | 83.2 | 79.2 | **78.2** | −6.0% |
| all8_r2（c16） | 109.3 | 98.3 | 97.4（p25 93.0） | **94.4** | −4.0% |
| all8_r4（c32） | 134.5 | 122.4 | 120.2 | **119.6** | −2.3% |

解读：
- **批次 1（w15r1 → w15base）：** 步时降低 9.0–18.4%（c1 81.6 → 66.6ms），与旧权重上的 r1 → all（−7.8% 至 −15.8%）一致。
- **A 批（w15base → f2allb）：** 步时降低 0.9–4.8%（c16 的 f2allb 分布偏宽，p25 为 93.0ms）。
- **B 批 + lm_head skinny（f2allb → f3allb）：**
  - 步时：c1 −4.1%、c8 −1.3%、c16 −3.1%、c32 −0.5%。
  - 中位 ITL：c1 64.26 → 62.38ms，c8 79.26 → 77.59ms。
  - c1 的约 2.7ms 中，lm_head skinny 约占 2.2ms（参照 w15base → w15lm 的 66.6 → 64.4ms），其余来自 B 批的融合。
- **c8–c32 的 E2E 吞吐波动很大**（f3allb 的 c8 287 vs f2allb 的 339 tok/s）：
  - 原因是 prefill 卡顿。8 个 rank 中任意一个在做 64K prefill（每个 chunk 2048 token），其余 rank 的 decode 都会停一下。卡顿落在哪一组、哪次运行是随机的。
  - 同一配置的 f2all 和 f2allb，c16 吞吐分别是 496 和 409 tok/s。
  - 所以可信的量是中位 ITL 和步时。
- **EP 同步掩盖：** dp-attention + EP 下每层的 dispatch / combine 都要等最慢的 rank。单个 rank 省下的 dense 时间，在 8 个 rank 都有请求时只能部分体现。所以 1rank_r1 的收益最干净；并发越高，MoE 占比越大，相对收益越小。
- **为什么 B 批单算子收益大，E2E 收益小：**
  - 单算子收益是按每层或每次调用计算的。例如 MQA logits 每次省约 50µs，但 GLM-5.3 每步只有约 22 次 indexer 调用（profile 结果，见第 7 节：2.1% / 91.5µs），合计约 1ms。
  - 融合省掉的小 kernel 每个 3–10µs，乘上层数后，每步合计约 1–2ms。
  - 一步 62–120ms 中，MoE + DeepEP 已经占 30% 以上，这部分属于 C 项范围，本轮没有动。

### 11.3 第二轮总收益（-1015 权重，w15r1 → 最终脚本）

w15r1 与最终脚本（f3allb）用同一启动脚本、同一权重，只差 15 个 hcu 开关。

| 指标 | 场景 | w15r1 | 最终（f3allb） | 变化 |
|---|---|---|---|---|
| decode 步时（排除 prefill，ms） | c1 | 81.6 | **62.7** | **−23.2%** |
| | c8 | 97.2 | **78.2** | **−19.5%** |
| | c16 | 109.3 | **94.4** | **−13.6%** |
| | c32 | 134.5 | **119.6** | **−11.1%** |
| Median ITL（ms） | c1 | 81.11 | **62.38** | **−23.1%** |
| | c8 | 98.21 | **77.59** | **−21.0%** |
| | c16 | 113.29 | **95.13** | **−16.0%** |
| | c32 | 135.03 | **119.54** | **−11.5%** |
| Median TPOT（ms） | c1 | 14.18 | **10.79** | −23.9% |
| | c8 | 22.29 | **21.30** | −4.4% |
| | c16 | 26.41 | **22.99** | −13.0% |
| | c32 | 38.77 | **34.84** | −10.1% |
| 输出吞吐（tok/s） | c1 | 59.66 | **74.89** | +25.5% |
| | c8 | 207.74 | **287.52** | +38.4% |
| | c16 | 363.70 | **366.52** | +0.8% |
| | c32 | 478.62 | **570.03** | +19.1% |

- **按步时换算的 decode 吞吐提升**（接受长度相同时，吞吐与步时成反比）：c1 +30%、c8 +24%、c16 +16%、c32 +12.5%。这是最能代表算子优化效果的数字。
- **E2E 吞吐 / TPOT 的波动：** 它们同时受 prefill 卡顿和接受长度影响（w15r1 的接受长度 5.384，f3allb 5.175；GEMM 类不逐 bit 一致，生成内容不同，接受长度随之变化）。所以 c8 的 +38% 和 c16 的 +0.8% 都不能单独看，以步时和中位 ITL 为准。
- **各批贡献（步时，c1 / c8 / c16 / c32）：**

| 批次 | c1 | c8 | c16 | c32 |
|---|---|---|---|---|
| 批次 1（w15r1 → w15base） | −15.0ms | −14.0ms | −11.0ms | −12.1ms |
| A（w15base → f2allb） | −1.2ms | −4.0ms | −0.9ms | −2.2ms |
| B + lm_head（f2allb → f3allb） | −2.7ms | −1.0ms | −3.0ms | −0.6ms |
| 合计 | −18.9ms | −19.0ms | −14.9ms | −14.9ms |

- 收益随并发下降：c1 时一步里 dense 算子（attention、indexer、GEMM、norm）占比最高；c32 时 MoE + DeepEP 的占比上升，而这部分属于 C 项，本轮没有动。

### 11.4 优化后 kernel 对比（profile，64K PC90%，c8 稳态 decode，DP0）

两份 profile 都用 `prof_decode.sh` 采集（`start_profile` num_steps=10），条件相同：-1015 权重，64K，c8（每个 rank 1 个请求），纯 decode。

- 基线 `prof/d64k_c8_w1015`（10-23 10:22）：批次 1 开关全开。当时 ROCm MLA 路径上的 absorb bmm 还没接上 hcu_ops（`patch_mla_rocm_bmm.py` 是看了这份 profile 之后 10:40 加的），所以基线里 bmm 仍走 hipBLASLt。
- 最终 `prof/d64k_c8_final`：最终脚本。
- 两份 trace 的 verify 步数不同（基线 9 步，最终 10 步，以 `VerifyTreeGreedy` 次数计），下表都换算成每步。
- 原始输出：`results/prof/`（`kagg_*.txt` 为 kernel 排名，`tracediff_*.txt` 为逐 kernel 差异）。

| 类别 | 基线 kernel | 最终 kernel | 基线 ms/步 | 最终 ms/步 | 变化 |
|---|---|---|---|---|---|
| norm / quant / rope / KV 写入 / silu / concat / copy 等小 kernel | per_token_quant ×4/层、fused_add_rms、concat_mla_absorb_q、`.contiguous()` copy、rms_small_hidden、fused_quantize_and_store_mla_kv、rope、silu_and_mul | add_rms_quant、silu_mul_quant、rope_qout_kv_store，剩余 per_token_quant 约 2/层 | 8.75 | 4.53 | **−4.22** |
| lm_head | BBH MT256x256x16（792µs） | skinny_nn（1008µs，每步 2 次）+ BBH（355µs） | 5.09 | 3.44 | **−1.65** |
| indexer MQA logits | lightop paged_mqa_logits（91.6µs） | mqa_logits（42.9µs）+ mqa_init | 2.16 | 1.11 | **−1.05** |
| MLA absorb bmm | hipBLASLt BBH MT32x16x32（21.2µs） | bmm_skinny（15.5µs） | 3.57 | 2.57 | **−1.00** |
| dense fp8 GEMM | skinny + hipBLASLt F8BS（MT96/48/64） | skinny（加 -1015 调优行后多数形状转到 skinny） | 13.06 | 12.66 | −0.40 |
| DSA topk（split 直方图） | split_hist / select / compact / refine | 同左 | 1.58 | 1.53 | −0.05 |
| sparse MLA attention | flash_fwd sparse + combine | 同左（未改） | 6.08 | 6.03 | ≈0 |
| MoE masked GEMM | grouped fp8 masked GEMM | 同左（未改，C 项） | 19.65 | 21.19 | +1.54 |
| DeepEP LL dispatch + combine | | 同左（未改，C 项） | 13.38 | 13.89 | +0.51 |
| **全部 kernel** | | | **83.1** | **76.2** | **−6.9（−8.3%）** |
| kernel 数 / 步 | | | 3805 | 3035 | **−20.2%** |

- 本轮改动的类别合计每步省约 8.4ms，与第 11.2 节 c8 步时（w15base 83.2 → f3allb 78.2ms）的方向和量级一致。
- MoE GEMM 和 DeepEP 两份 trace 之间多了约 2ms，这两部分没有改动。差异来自专家路由（每次生成内容不同，各专家收到的 token 数不同）以及 combine 里等最慢 rank 的时间。
- 预期中被替换掉的 kernel 在最终 trace 中都已消失或只剩少量调用：`concat_mla_absorb_q`、`fused_quantize_and_store_mla_kv`、`rope_kernel`、`rms_kernel_small_hidden_multi_token`、`silu_and_mul_kernel_opt1`、lightop `paged_mqa_logits`、hipBLASLt bmm。
- 最终 trace 中 MoE GEMM（27.8%）+ DeepEP（18.2%）已占 GPU 时间的 46%，是剩余的主要部分（第 15 节）。

---

## 12. 精度

| 权重 | 组 | gsm8k（1319 题） | aime26（30 题 × 4 次） | 说明 |
|---|---|---|---|---|
| 旧 | r1（第一轮） | 97.57% | 80.83% | |
| 旧 | opt（批次 1 全开） | 97.42% | 84.17%（每次分别为 76.67 / 90.00 / 86.67 / 83.33） | aime 失分全部来自 65536 截断：18/120 条被截断且全错，未截断的 102 条中答对 101 条 |
| 旧 | opt_high（误用 `reasoning_effort=high`，仅作参考） | 96.66% | — | high 几乎不思考 |
| **-1015** | **最终脚本** | **97.27%** | 用户自测 | 10-23 17:08 – 17:20，effort=max，评测期间服务无崩溃 |

- -1015 最终脚本的 gsm8k 97.27% 与旧权重的 97.42% / 97.57% 相差 0.15–0.30 个百分点（2–4 题）。1319 题下准确率的标准误约 0.45 个百分点，差异在统计误差内。
- 旧权重上 opt 与 r1 的 gsm8k 相差 0.15 个百分点（约 2 题），aime26 单次之间就有 13 个百分点的波动，都在统计误差范围内。
- 第二轮除 GEMM 类以外的改动都是逐 bit 一致的。GEMM 类的误差与 hipBLASLt / rocBLAS 同量级，不会系统性影响精度。
- -1015 的 gsm8k 由 `eval/drive_gsm8k.sh final1015 sglang_start_glm53_opt.sh` 运行，结果在 `/data/home/daihw/scripts/glm5.3/eval/final1015/`。
- **aime26 自测注意事项：**
  - DP8 下单个请求只在一个 rank 上，单 rank 的 KV 池约 103K token，所以 max_tokens 最多约 65536，128K 跑不完。
  - 建议 `--repeats 4` 取平均。
  - GLM-5.x 的 `reasoning_effort` 只认 low / high，其余值都按 max 处理；不要传 high。

---

## 13. 改动清单与回滚

### 13.1 容器内改动

| 类别 | 内容 | 备份 |
|---|---|---|
| 新增 Python 包 | site-packages 下的 `hcu_ops/`（`_C.so`、`__init__.py`、`skinny_cfg.json`） | `skinny_cfg.json.bak_cfg2` 是批次 1 的表；`hcu_ops.bak_20261023` 是 10-22 的早期版本 |
| 第一轮 | `layers/logits_processor.py`、`layers/quantization/unquant.py` | `*.bak_20261021`（镜像原始文件） |
| 第二轮 | `patches/FILES` 中除 `unquant.py` 以外的 16 个文件（见下表；`logits_processor.py` 是在第一轮版本上再修改） | 各文件最早的 `*.bak_20261022` / `*.bak_20261023` / `*.bak_qcat` / `*.bak_fuse2` |
| 未改动 | 模型文件、`bench/vllm_bench-pc.sh`、原始启动脚本 `sglang_start.sh_1017`，以及 lightop、DeepEP、DeepGEMM、FlashMLA、aiter、rocBLAS / hipBLASLt 等 .so | — |

第二轮各文件与补丁的对应关系（路径相对 `sglang/srt/`）：

| 文件 | 补丁 |
|---|---|
| `layers/attention/dsa/dsa_indexer_metadata.py` | topk |
| `layers/quantization/fp8_utils.py`、`layers/linear.py` | skinny |
| `layers/rotary_embedding/base.py` | rope |
| `layers/moe/topk.py`、`layers/moe/ep_moe/layer.py` | moe（gate / act） |
| `models/deepseek_common/attention_forward_methods/forward_mla.py` | bmm |
| `models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` | mla_rocm_bmm → qcat → fuse2 → kvfuse |
| `layers/sampler.py` | sampler_warmup |
| `layers/attention/dsa_backend.py` | qcat |
| `layers/layernorm.py` | fuse2 → fuse3 |
| `layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` | fuse2 → fuse3 |
| `models/deepseek_v2.py` | fuse2 → fuse3 |
| `layers/attention/dsa/dsa_indexer.py` | fuse2 → fuse3 → mqa |
| `layers/logits_processor.py`（在第一轮版本上）、`speculative/draft_lm_head_vp.py` | lm_head_nn |

### 13.2 回滚

- **只关闭某一项：** 启动前把对应开关设为 0（第 4 节）。
- **回到第一轮代码：** `bash /opt/glm5.3_opt/patches/restore_all.sh`。
- **回到镜像原始代码：** `ROUND1=1 bash /opt/glm5.3_opt/patches/restore_all.sh`。
- **在新容器中部署（同版本镜像）：** 把交付包拷进去，执行 `bash patches/apply_all.sh`（加 `--build` 会从源码编译），然后执行 `bash patches/verify_all.sh`。

---

## 14. 已知问题与注意事项

- **draft VP 裁剪（`SGLANG_HCU_SKINNY_DRAFT_VP`）在 DP 下会 NCCL hang**，禁止打开。TP（非 DP）下理论上可以用，但没有测试。
- **TP8 部署没有做端到端测试**（按约定没有重启 TP8）：
  - 与并行方式无关、可以直接用的：topk、rope、gate、act、qcat、strided RMS、norm/silu+quant、kv 融合、MQA logits。
  - bmm 已经为 8 head 调过表。
  - skinny GEMM 和 lm_head skinny 的调优表只有 DP8 的形状。TP8 下形状不同，会自动回退，没有收益，也不会出错。
- **hcu_ops 只为 gfx938 编译和调优。** 换卡型要用 `apply_all.sh --build` 重新编译，并重新跑调优。
- **首个 T>0 采样请求偶发 "no kernel image"（`hipErrorNoBinaryForGpu`）：**
  - 根因没有定位：code object 齐全，单独构造并发懒加载也复现不出来。
  - 规避：`SGLANG_HCU_SAMPLER_WARMUP=1`（代价是每个 rank 的 KV 少 1.3%）。用别的启动脚本时，至少在服务就绪后给每个 DP rank 发一个 T>0 的请求。
  - 第一轮发现的 `--load-balance-method total_requests` 会稳定触发这个问题，不要使用。
- **低并发 E2E 波动：** MTP 接受长度随内容变化，c8–c32 还受 prefill 干扰，单次 E2E 数字有 ±15–20% 的噪声。A/B 结论以 ITL 和排除 prefill 后的步时为准。
- **PC 命中测试：** dp-attention 下 warmup 次数要 ≥ dp（`--num-warmups 8` 或 16），否则有的 rank 没建好前缀缓存，看起来就像“没有命中”。
- **固化镜像：** 补丁、备份和 `hcu_ops` 都在容器的 site-packages 中，`/opt/glm5.3_opt/` 也在容器内，`docker commit` 会一并带上。`deepep_cfg40.json` 按启动脚本所在目录引用。
- 宿主机开着 NUMA balancing，启动日志里的 "error" 字样来自这条告警，与本次改动无关。
- **确认开关是否生效：** 启动日志里只有 KV_FUSED、MQA_LOGITS、NORM_QUANT、QUANT_REUSE 会打印 `SGLANG_HCU_*` 行（最终脚本的摘录在 `results/hooks/`）；SILU_QUANT、SKINNY_LM_HEAD 和批次 1 的开关不打印。要确认它们，可以抓一次 profile，看 `silu_mul_quant_kernel`、`skinny_nn_kernel`、`bmm_skinny_kernel` 等是否出现（第 11.4 节）。

---

## 15. 未做的项（C）与后续方向

按用户要求，C 项**没有实施**。下面只记录评估结论，供后续决策：

| 项 | 预期收益 | 风险 / 代价 |
|---|---|---|
| W4A8 MoE 专家权重 | MoE masked GEMM 在最终配置中占 decode GPU 时间的 28%（第 11.4 节），已到带宽上限，字节数减半可节省其中 30–45% | 需要量化校准，影响精度，要重新做精度验证 |
| TBO（two-batch overlap）/ SBO | 掩盖 DeepEP LL dispatch + combine（最终配置中约 18%）和跨 rank 等待 | 改动调度层，与 MTP、dp-attention 的兼容性需要验证 |
| shared expert 改 fp8 | 每层 shared expert GEMM 字节数减半 | 影响精度；-1015 权重中 shared expert down 已是 fp8 |

其他方向：
- TP8 形状的 skinny / lm_head 调优（128K 只能用 TP8 部署）。
- lm_head 词表并行或改 fp8：每步约 1.1–1.6ms。
- prefill 侧：DSA 稀疏 attention fwd 和 DeepEP normal 模式。
