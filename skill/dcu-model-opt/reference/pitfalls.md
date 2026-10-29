# DCU 推理优化踩坑清单

每条格式：现象 → 原因 → 做法。

## 部署与压测

- **PC 90% 场景日志里没有缓存命中**
  - 原因：dp-attention 下每个 DP rank 各有一份 radix cache，请求按 round_robin 分配；warmup 只有 2 条时，只有 2 个 rank 建好了前缀。
  - 做法：`--num-warmups` ≥ dp（8 或 16）。每次测试换 `cache_salt` 或 seed，测前 `/flush_cache`。

- **vllm bench 报 `NamespaceTool` / openai 导入错误**
  - 原因：sglang 锁定的 openai 版本（如 2.6.1）比 vllm bench 需要的旧。
  - 做法：`pip install --target benchlib openai==<新版>`，跑 bench 时设 `PYTHONPATH=benchlib`，不要动 sglang 环境。

- **低并发 E2E 数字来回跳 ±10–20%**
  - 原因：MTP 接受长度随随机内容变化；c≤16、num_prompts=2c 时，prefill 和 decode 重叠的比例不稳定。
  - 做法：c1 看 ITL（步时），不看 TPOT：TPOT = ITL / 接受长度，接受长度一变 TPOT 就跳。c8 及以上用 `ana_clean.py` 统计排除 prefill 后的步时；多跑几次或者加大 NP。

- **换权重后 skinny GEMM 的 greedy 输出分叉**
  - 原因：新权重把更多层量化成 fp8（例如 indexer wq_b/wk、shared expert down），调优表新加的行与 hipBLASLt 同精度但不是 bit-exact。
  - 做法：属于预期现象。用 fp32 参考比较误差量级，确认与 hipBLASLt 同级；之后的 bit-exact 开关改用“加了新行的组”作为 greedy 参考。

- **高并发时 TTFT 爆炸**
  - 原因：`--max-running-requests` 太小，导致排队。
  - 做法：每个 rank 的上限 ≥ 并发 / dp，同时调大 `--cuda-graph-max-bs` 和 DeepEP 每个 rank 的 dispatch token 数。

- **几种参数会崩溃或变慢，不要用**
  - `--load-balance-method total_requests`
  - `--enforce-shared-experts-fusion`
  - `--speculative-skip-dp-mlp-sync`
  - 在 DP 下打开 draft lm_head VP 裁剪

## 框架接入（SGLang dtk 0.5.18）

- **`AttributeError: 'ForwardBatch' object has no attribute 'attn_backend'`**
  - 做法：用 `from sglang.srt.model_executor.forward_context import get_attn_backend`。
  - 教训：hook 用到的框架对象属性，要在 import 检查之外再做一次真实前向，或者至少在启动日志里确认。

- **DP 下 NCCL hang（c≥8 才出现）**
  - 原因：DP 的各个 rank 可能同时处于不同模式（decode graph replay / eager extend / idle）。只要某个 all_gather / all_reduce 的尺寸按本 rank 的实际行数或捕获行数裁剪，rank 之间尺寸就会不一致。
  - 做法：DP 集合通信只用全局统一的尺寸（例如 `global_num_tokens` 的最大值）；改动通信尺寸的优化必须在 c8/c16/c32 下压测。

- **第一个 T>0 的请求偶发 `no kernel image is available` / `hipErrorNoBinaryForGpu`**
  - 原因：采样相关 kernel 的 code object 懒加载，第一次加载时偶发失败。
  - 做法：服务启动时预热采样路径；评测驱动在就绪后给每个 rank 发 T>0 的请求，失败就自动重启服务。

- **CUDA graph 捕获期间报错，或者重放结果不对**
  - 原因：hook 里有 host 同步（`.item()`、`.tolist()`、`nonzero`）、按实际值分配内存，或在第一次调用时重排权重。
  - 做法：launch 配置只依赖静态形状；动态切分放到 device 端；权重重排放在 `process_weights_after_loading`，或者在捕获之前完成。

- **补丁重复执行导致代码叠加，或者锚点找不到**
  - 做法：从 `.bak_<tag>` 重新打；`assert count == 1`；后续补丁用 `need` 检查前置补丁；`verify_all` 做 mirror 重放。

- **开关打开了，但 kernel 没有生效**
  - 例子：GLM-5.3 的 absorb bmm 开关在 `forward_mla.py` 里接上了，但 DCU 实际走的是 `forward_mla_rocm.py`，所以开关打开后 profile 里仍然是 hipBLASLt bmm。
  - 原因：同一功能在框架里常有 CUDA / ROCm / HCU 多条路径；而且很多 hook 不打日志。
  - 做法：每批改动之后抓一次 profile，确认新 kernel 出现、旧 kernel 消失（`tracediff.py`）。hook 第一次生效时打印一行 `SGLANG_HCU_*` 日志。

- **在没有 GPU 的容器里检查补丁，import 失败**
  - 现象：`librocm_smi64.so.2` 找不到；挂上 `/opt/hyhal` 之后，又在 deepgemm import 时报 `DEEPGEMM_GPU_CUS` 为 None。
  - 做法：补丁应用、`py_compile` 和 `verify_all` 都可以在无 GPU 容器里做（`-v /opt/hyhal:/opt/hyhal:ro`）；import 检查要挂上 `/dev/kfd`、`/dev/dri`，并且等 GPU 空闲时再做。

## Kernel（gfx938）

- **“融合后反而更慢”**
  - 原因：
    - VGPR 用量上升，occupancy 下降；
    - 动态下标访问私有数组被放进 scratch；
    - 每行只有 1 个 CTA，用不满 64 个 CU。
  - 做法：
    - 看 ISA 里的 `vgpr_count`、`scratch`；
    - 展开比较代替动态下标；
    - 按行数选择 rows_per_cta 和 persistent CTAs；
    - 融合只在小 M 下做，大 M 回退。

- **不是 bit-exact**
  - 常见原因：
    - rsqrt / rcp 写法不同：硬件 rcp 不是 IEEE；
    - 归约顺序不同；
    - bf16 中间值有没有舍入；
    - fp8 scale 下限；
    - `x * (1/s)` 和 `x / s` 的差别。
  - 做法：逐个算子对照参考实现的源码或 ISA；把 rsqrt 写法做成 mode 开关；单测覆盖 NaN/inf/0/离群值/tie。

- **topk 不一致**
  - 原因：分数相同时的 tie-breaking，以及 -inf 和 padding 的处理。
  - 做法：比较选出的集合（排序后的下标），不要求顺序一致；明确规定分数相同时取下标小的。

- **小 M GEMM 打不过 hipBLASLt**
  - 原因：W 不是按行连续读；或者 tile 数少于 CU 数。
  - 做法：W 每行连续读 256B 进 LDS；按 (N, K, M) 生成调优表，只在快 ≥5% 的点启用。

- **MoE grouped GEMM 已经到 3–3.7 TB/s**
  - 改 kernel 没有空间，只能减少字节数：W4A8，或者 shared expert 改 fp8。
  - 这类改动影响精度，需要用户确认。

## 远程操作

- **ssh 命令一直不返回**
  - 原因：后台进程继承了通道的 stdout/stdin。
  - 做法：后台命令单独占一行，写成 `setsid nohup cmd < /dev/null > log 2>&1 &`，前面不要接 `cd ... &&`。

- **`docker exec` 吃掉了 `bash -s` 后面的脚本**
  - 做法：`docker exec ... < /dev/null`。

- **PowerShell 引号和 `$` 被展开**
  - 做法：远程脚本用 here-string `@' ... '@` 写到本地 `.sh`（`Set-Content -Encoding ascii`），再用 `remote.py -f` 执行。

- **修改正在运行的 bash 脚本导致执行错乱**
  - 原因：bash 是边读边执行的。
  - 做法：改成新文件名，或者等脚本跑完再改。

- **服务或压测在跑时做 GPU 单测**
  - 后果：单测结果和压测结果都不可信，还可能 OOM。
  - 做法：等 A/B 跑完；或者明确空出一张卡，并且不是 DP 服务正在用的卡。
