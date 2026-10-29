---
name: dcu-model-opt
description: |
  Use when optimizing LLM inference performance on Hygon DCU (BW1100 / K100 / gfx93x, DTK,
  HIP) with SGLang or vLLM: tuning launch parameters, profiling decode/prefill, rewriting or
  fusing kernels (HIP C++ extensions), hooking them into the framework behind env switches,
  A/B benchmarking with vllm bench, and accuracy checks with evalscope. Also covers the
  remote-server workflow (ssh + docker exec from Windows PowerShell) and the delivery package
  (patches / apply / verify / restore / start script / report).
---

# DCU 模型推理优化（SGLang / vLLM on Hygon DCU）

这套方法来自 GLM-5.3 FP8 在 dcu4（8 × BW1100 gfx938）上的两轮优化。完整案例见 `reference/glm53_case.md`。

核心原则：**先调参数，再 profiling 定位，然后做逐 bit 可验证的算子替换，用开关接入，最后 A/B 和精度双重验证。**

## 0. 工作守则（每次都要遵守）

- **改文件前先备份。**
  - 容器内框架文件第一次修改时 `cp -p f f.bak_<tag>`，备份只做一次。
  - 补丁脚本永远**从备份重新打**，保证幂等、可重放（模板见 `tools/patch_template.py`）。
- **新功能默认关闭。** 每项优化一个 `SGLANG_HCU_*` 环境变量。不支持的形状、dtype、stride 自动回退原实现。启动脚本决定打开哪些。
- **远程命令不写 `rm`。** 要删除就 `mv` 到 `/tmp`。
- **不动用户的原始脚本和测试脚本。** 修改都另存新文件，例如 `*_opt.sh`。
- **重启服务要先确认影响面。**
  - 只能用用户允许的部署形态做 A/B。
  - 服务或 A/B 压测在跑时，不在同一张卡上跑 GPU 单测。
  - 不要修改正在执行的 bash 脚本：bash 是边读边执行的。
- **结果如实报告。**
  - 端到端噪声通常有 ±10%，结论要附上测量方法。
  - 失败和放弃的项也要写进报告。

## 1. 总流程

| 步 | 做什么 | 产出 |
|---|---|---|
| 1 | 摸底：部署形态（TP/DP/EP、dp-attention、MTP）、负载（输入/输出长度、PC 命中率、并发）、基线 bench、服务日志 | 基线表 |
| 2 | 启动参数和调度（收益通常最大） | 新启动脚本 |
| 3 | profiling：decode 和 prefill 各抓一份 trace，做 kernel 排行，再拆出单层的 kernel 序列 | kernel 占比表、单层耗时表 |
| 4 | 选目标：每步占比 × 可节省比例。先判断是带宽瓶颈还是 launch/occupancy 瓶颈 | 候选清单 |
| 5 | 写 kernel：与参考实现逐 bit 对齐，兼容 CUDA graph，按形状门控 | `hcu_ops` 扩展 + 单测 |
| 6 | 接入：补丁脚本、开关、补丁链、mirror 重放校验 | `patch_*.py`、`apply/verify/restore` |
| 7 | 验证：单测（bit-exact + graph 计时）→ import 检查 → greedy 一致性 → E2E A/B → 排除 prefill 后的步时 → evalscope 精度 | A/B 表、精度表 |
| 8 | 交付：启动脚本、补丁包、报告（中文）、回滚方法 | 交付目录 |

### 1.1 基线与压测

- **PC 命中：** dp-attention 下每个 DP rank 有自己的 radix cache，请求按 round_robin 分配。
  - warmup 次数要 ≥ dp（建议 2×dp），否则有的 rank 没建好前缀，"PC 90%" 实际上不命中。
  - vllm bench 用 `--num-warmups 8`（DP8）。
- **低并发噪声：** MTP 接受长度随内容变化，num_prompts = 2×并发时 c≤16 的波动可达 ±20%。
  - **开了 MTP 时看 ITL，不看 TPOT。** ITL 是一个 verify + draft 循环的时间，也就是步时；TPOT = ITL / 接受长度，会随生成内容变化。例如 GLM-5.3 有一组 c1 的 TPOT 是 16.9ms，ITL 却正常（66ms），只是那次接受长度偏低。c1 的吞吐同理。
  - c8 及以上要看排除 prefill 后的 decode 步时（`tools/ana_clean.py`）。
- **vllm bench 报 openai 兼容错误：** 容器里 sglang 锁定的 openai 版本太旧时，把新版 openai 装到单独目录，跑 bench 时用 `PYTHONPATH=<该目录>` 指过去。

### 1.2 启动参数（SGLang DP + EP + MTP 的常见收益点）

- **`--max-running-requests`：** 原值偏小会导致高并发排队，TTFT 暴涨。DP8 下每个 rank 要能容纳 并发 / dp 个请求。
- **配套项：**
  - `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` ≥ 每个 rank 的请求数 × draft token 数；
  - `--cuda-graph-max-bs` 与每个 rank 的请求上限一致；
  - `--mem-fraction-static` 随之上调。
- **prefill 干扰 decode：** `--chunked-prefill-size` 设为每个 rank 2K，加上 `--enable-prefill-delayer` 和低水位 0.05。
- **DeepEP normal 模式（prefill）：** 先用单算子测试扫 SM 数。BW1100 上 40 SM 优于默认的 24。
- **试过但不要用：**
  - `--load-balance-method total_requests`：可能崩溃，而且更慢；
  - `--enforce-shared-experts-fusion`、`--speculative-skip-dp-mlp-sync`：会崩溃；
  - MTP 7/8 比 5/6 更差。

## 2. Profiling

- **抓 trace：**
  - 启动时设置 `SGLANG_TORCH_PROFILER_DIR`，然后：
    ```bash
    curl -X POST :30000/start_profile -d '{"output_dir":"...","num_steps":10,"activities":["GPU"]}'
    ```
  - 不要开 `with_stack`。
  - 抓完建议重启服务，因为 profiler 之后服务可能卡住。
- **分析工具：**
  - `tools/kagg.py trace.json.gz`：按 kernel 名统计总耗时、占比、次数、平均。
  - `tools/kstat.py trace.json.gz <regex>`：同名 kernel 的 grid/block 分布，用于区分不同调用点。
  - `tools/kseq.py trace.json.gz <anchor_regex> <occ> <n>`：两个锚点 kernel（例如 attention）之间的 kernel 序列，用来拆出一层。
  - `tools/tracediff.py a b`：两份 trace 逐 kernel 对比，看改动前后。
- **判断瓶颈的方法：**
  - **算等效带宽** = 权重字节 / 耗时，与实测 HBM 读带宽比较。gfx938 上每行连续 256B 读约 1.6–1.8 TB/s，按 mmac 布局每行 64B 读约 1.1 TB/s。已接近上限的，例如 MoE grouped GEMM，不要再动 kernel，只能减少字节数（W4A8 等）。
  - **launch 下限：** HIP graph 下每个 kernel 至少约 5µs。每层几十个小 kernel 时，**合并 launch 本身就有收益**。比如 rope 8 次 launch 合成 1 次：20µs → 2.4µs。
  - **空转的 grid：** 固定 E×128 个 CTA，其中大多数一读 masked_m 就退出（act quant）；或者每行只有 1 个 CTA，只能用到几个 CU（topk）。这类都是改 launch 结构的机会。
  - **EP 同步掩盖：** dp-attention + EP 时每层都要等最慢的 rank，单 rank 的 dense 节省在高并发下会被 MoE 负载波动掩盖。所以要分别看 1rank_r1 和 all8_rK 的步时。

## 3. 写 kernel（HIP C++ 扩展，gfx938）

- **工程：** `torch.utils.cpp_extension` + `setup.py`，`PYTORCH_ROCM_ARCH=gfx938`。
  - 增量编译用 `build_inc.sh`（只重新编译改动的 `.cu`）。
  - 不要用带 `rm -rf` 的全量构建脚本。
- **逐 bit 对齐参考实现**（lightop / sgl_kernel / torch），这是能放心默认打开的前提：
  - 使用相同的表达式和标量类型，比如 silu 用 `bf16(x / (1 + 2^(−x·log2e)))`，fp8 scale 用 `max(amax/448, 1/(448·512))`，量化用 `q = cvt(y × (1/scale))`；
  - 归约按固定顺序进行；tie-breaking 一致（相同分数取下标小的）；
  - `v_cvt_bf16_f32` 是硬件 RNE（遍历 2^32 个 float 验证过）；rcp 不是 IEEE，不能代替除法；
  - rsqrt 的不同写法会改变末位：做成 mode 开关，单测确认 mode 0/1 逐 bit 一致后再用。
- **兼容 CUDA graph：**
  - 不做 host 同步；
  - launch 配置只取决于静态形状；
  - 依赖实际行数的切分在 device 端计算，例如前缀和编号、按行长算 slice 数。
- **小 M（decode）GEMM：**
  - mmac 16x16x32 fp8：W 的 16 行放 A，X 的 16 个 token 放 B。
  - W 先按行连续 256B 读进 LDS，再按 mmac 布局读回；X 也放进 LDS，所有 tile 共用。
  - 按 (N, K, M) 扫描 ver / nw / nt 生成调优表，只保留快 ≥5% 且 ≥1µs 的点，其余回退 hipBLASLt。
- **occupancy / VGPR：**
  - 用 `amdgpu_waves_per_eu` 压 VGPR 时检查是否 spill（看 ISA 或 `.s`）。
  - 动态下标访问私有数组会被放进 scratch，改成展开的比较。
- **单测：** 每个 kernel 一个 `test_*.py`。
  - 正确性：`torch.equal`（fp8 先 view 成 uint8），覆盖 NaN/inf/±0、全零行、tie、离群值、ragged/padding。
  - 性能：`tools/ut_bench.py` 的 `graph_us()`（CUDA graph 内连续调用 100 次取平均）。
  - 要在空闲的卡上跑。

## 4. 接入框架（补丁约定）

- 一个补丁脚本负责一组文件：
  - `assert src.count(old) == 1` 保证锚点唯一；
  - 第一次运行时备份为 `.bak_<tag>`，之后每次都从备份打补丁；
  - 前置依赖用 `need` 字符串检查。
- **补丁链：**
  - 同一个文件被多个补丁修改时，后一个补丁的备份就是前一个补丁的结果，所以**顺序固定**，比如 qcat → fuse2 → fuse3 → kvfuse → mqa。
  - 用 `apply_all.sh` 固化顺序。
  - `verify_all.sh` 把每个文件最早的备份复制到 `/tmp` 镜像目录，按同样顺序重放（用 sed 改写路径），再与现行文件逐个 `cmp`，应得到 `fail=0`。
- **hook 写法：**
  - import 阶段读取开关；
  - 运行时先做 `*_supported(...)` 检查，不满足就走原路径；
  - 第一次命中时打一行 `logger.info`，用来确认 hook 真的生效（A/B 时 grep 日志）。
- **SGLang（dtk 0.5.18）的坑：**
  - `ForwardBatch` 没有 `attn_backend`，要用 `sglang.srt.model_executor.forward_context.get_attn_backend()`。
  - **DP 集合通信的尺寸必须一致。**
    - 不同 DP rank 可能同时处于不同模式：一个在 replay decode graph，另一个在跑 eager prefill 或 draft extend。
    - 任何“按捕获行数裁剪”的 all_gather 都会导致尺寸不一致，NCCL 卡死。draft lm_head VP 裁剪就是这样，c≥8 时 hang。
  - **采样 kernel 懒加载：** 第一个 T>0 的请求偶发 `no kernel image (hipErrorNoBinaryForGpu)`。
    - 服务端在启动时预热采样路径（`SGLANG_HCU_SAMPLER_WARMUP`）。
    - 评测驱动在服务就绪后，给每个 DP rank 发 T>0 的请求预热，并自动重启。
  - 第一次调用时重排权重（NN 布局）要避开 CUDA graph 捕获阶段。

## 5. 验证与 A/B

1. **单测：** bit-exact，graph 计时。
2. **import 检查：** 所有开关打开后能 import，并打印各 hook 的状态。
3. **greedy 一致性：** `tools/greedy_chk.py OUT.json REF.json`，每个 prompt 在 T=0 下跑 2 次，与 REF 的公共前缀应为全长。bit-exact 的改动必须全长一致。
4. **E2E A/B：** `tools/ab_env.sh "tag:VAR=v,..."`。每组起一次服务，做 greedy 检查，跑 `ab_bench.sh`（64K PC90%，c1/c8/c16/c32），统计 accept len，然后停服务。
   - 结论以 c1 的 ITL 和 `ana_clean.py` 的步时为准；TPOT 和吞吐只作参考，因为它们受接受长度影响。
   - 同一配置连续跑两次，c16/c32 的吞吐也能差 20%（GLM-5.3 的 f2all 是 496，f2allb 是 409 tok/s）。不要用单次 E2E 的高并发数字下结论。
   - **改动通信尺寸、调度或 graph 的开关，必须在 c8/c16/c32 下跑通。** c1 通过不代表没问题：draft VP 裁剪在 c1 正常，c8 才 hang。
   - **对照组要包括“全部新开关关闭”的一组**（GLM-5.3 的 w15r1），用来给出整轮总收益。
5. **profile 复核：** 最终配置再抓一次 profile，用 `tracediff.py` 与优化前对比（按 `VerifyTreeGreedy` 次数换算成每步），确认新 kernel 出现、旧 kernel 消失。不打日志的 hook 只能这样确认；GLM-5.3 的 ROCm 路径 bmm 没接上，就是这样发现的。
6. **精度：** evalscope（gsm8k 全量；aime 30 题 × 4 次，单次波动 ±13%）。
   - 采样参数按模型 `generation_config`。
   - GLM-5.x 的 `reasoning_effort` 只认 low/high，其他值都按 max 处理，`high` 几乎不思考。
   - DP8 下单个 rank 的 KV 容量限制了 max_tokens。

## 6. 远程操作（Windows PowerShell → ssh → docker）

- **`tools/remote.py`**（paramiko，凭据读环境变量 `DCU_HOST`/`DCU_USER`/`DCU_PW`）：
  - `python remote.py -f q.sh`：用 `bash -s` 执行本地脚本。
  - `python remote.py put|get <local> <remote>`。
- **避免 PowerShell 引号问题：** 远程命令一律写成脚本文件（`Set-Content -Encoding ascii`），再用 `-f` 执行。
- **后台任务：** `setsid nohup bash x.sh < /dev/null > log 2>&1 &` 单独占一行。
  - 前面**不能**有 `cd ... &&`，否则 ssh 通道被后台子 shell 占住，命令一直不返回。
- **docker exec：** 末尾加 `< /dev/null`，防止吃掉 `bash -s` 的 stdin。
  - 容器内复杂命令写成 `.sh` 放到挂载目录再执行。
- 输出里 `ERROR: Directory '/public/software/modules' not found` 是 dcu4 登录脚本的噪声，可以忽略。

## 7. 交付物清单

- **启动脚本：** 基于用户原脚本另存。开关写成 `export VAR=${VAR:-1}`，方便单独覆盖。配置文件按脚本所在目录引用。
- **`patches/`：**
  - `patch_*.py`；
  - `apply_all.sh`（检查原始文件的 md5 → 安装预编译的扩展，或 `--build` → 补丁链 → py_compile + import 检查）；
  - `verify_all.sh`；
  - `restore_all.sh`（回到每个文件最早的备份，当前文件另存为 `.pre_restore_<ts>`）；
  - `FILES`；
  - `diffs/`。
- **`hcu_ops/`：** 源码、预编译 `.so`、调优表、单测、调优脚本。
- **`results/`：** A/B、单测、服务日志、trace 摘要、精度。
- **报告（中文）：**
  - 概述和结论；
  - 每项优化：问题 → 设计 → 数值验证 → 单算子收益 → E2E 收益；
  - 放弃的项和原因；开关表；使用、回滚、复现方法；已知问题。

## 8. GLM-5.3 案例速查（DP8 + EP8，EAGLE 5/1/6，64K PC90%）

| 项 | 单算子 | 数值 | 结论 |
|---|---|---|---|
| 参数：max-running 32→256 等 | — | — | 1K/4K 高并发吞吐 2.7–2.9 倍 |
| bf16 lm_head / 线性层 NN 布局 | lm_head M=96：40.9ms→1.6ms | 布局等价 | 采用 |
| DSA topk-2048（split 直方图） | 64K：380→49µs | 选出的集合相同 | 主要收益：每步 −9ms |
| skinny fp8 GEMM（o_proj/q_b） | 96→66µs | ≈hipBLASLt | c1 有收益 |
| rope 8→1 launch | 20.3→2.4µs | bit-exact | 采用 |
| MoE gate+后处理 | 15.7→7µs | bit-exact | 采用 |
| masked SiLU×mul+fp8 量化 | 17→5µs | bit-exact | 采用 |
| MLA absorb bmm | 32.6→26µs | ≤1.25 倍误差 | 采用 |
| qcat（去掉 concat_mla_absorb_q） | 23→13.6µs/层 | bit-exact | 采用 |
| strided RMSNorm / quant 复用 | 每层少 2 次 copy / 2 次 quant | bit-exact | 采用 |
| fused_add_rms+quant / silu+quant | 10.3→8.5µs / 5.2→3.1µs | bit-exact（rsqrt mode 0） | 采用 |
| kv_a norm+rope+fp8 KV store | 12.7→3.5µs | bit-exact | 采用 |
| indexer MQA logits（rpc3、64 CTA） | 93→43µs | bit-exact | 采用 |
| topk 直方图融进 logits（G1） | 每次省 4–9µs | bit-exact | 只开 G1、≤8 行；G6 不划算 |
| lm_head skinny NN（M≤64） | 1583→1110–1326µs | ≈bf16 舍入 | 采用（只用于目标模型；c1 ITL −2.4–3ms） |
| FlashMLA 固定 splits | 无收益 | 不是 bit-exact | 放弃 |
| draft lm_head VP 裁剪 | — | — | DP 下 NCCL hang，放弃 |
| MoE masked GEMM | 已到 3–3.7TB/s | — | 不动，要用 W4A8 |
| **第二轮合计（-1015，64K PC90%）** | 每步 kernel 83.1→76.2ms，kernel 数 −20% | gsm8k 97.27% | 步时 c1 −23.2%、c8 −19.5%、c16 −13.6%、c32 −11.1%；c1 吞吐 +25.5% |

收益规律：低并发时 dense 算子（attention、indexer、GEMM、norm）占比高，算子优化收益最大；并发升高后 MoE + DeepEP 占比上升，收益递减。要继续提高高并发，需要动 MoE（W4A8）和通信重叠（TBO）。

## 9. 工具（`tools/`）

| 文件 | 用途 |
|---|---|
| `remote.py` | ssh/sftp 助手（凭据读环境变量） |
| `prof_decode.sh` | 加压到纯 decode 稳态后 `start_profile` 抓 10 步 GPU trace |
| `kagg.py` / `kstat.py` / `kseq.py` / `tracediff.py` | torch profiler trace 分析 |
| `ana_clean.py` | 从 SGLang dp-attention 日志统计排除 prefill 后的 decode 步时 |
| `greedy_chk.py` | greedy 数值一致性检查 |
| `ab_env.sh` / `ab_bench.sh` | 按环境变量分组的 E2E A/B 和 vllm bench 场景（改顶部的路径变量） |
| `patch_template.py` | 补丁脚本模板（备份一次，从备份重打，锚点唯一） |
| `verify_chain.sh` | 补丁链 mirror 重放校验模板 |
| `ut_bench.py` | 单测辅助：`graph_us()` 计时，`same()`/`ndiff()` 逐 bit 比较 |

参考资料：`reference/pitfalls.md`（踩坑清单，动手前先过一遍），`reference/glm53_case.md`（GLM-5.3 完整优化报告）。
