# Trace 算子性能优化记录

本文记录 Trace 算子在 Ascend 910B1 上的性能分析过程与优化结果。
数字分两类，**必须区分**：

- **CAModel 周期级仿真**（`tests/bench_910b.sh`）：用来比较指令条数、
  标量簿记、向量流水停顿。同一 build 内完全可复现。
- **真机耗时**（`tests/bench_real.sh`，`msprof op`）：唯一能反映 GM 访存、
  DMA 效率、任务调度开销的口径。**CAModel 不能替代它**，见 §8。

- 目标平台：Ascend 910B1（`dav_2201`），CANN 9.2.0
- 测量工具：`tests/bench_910b.sh`（CAModel 周期）、`tests/bench_real.sh`（真机 + 管线归因）
- 参考文档：`docs/CANN_9.0.0_AscendC_算子开发指南.pdf` 第 2.5.2、3.7、3.8、6.2.3.7 章

---

## 1. 结论速览

| shape / dtype | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 1 × 1 float | 5026 | **3968** | −21.1% |
| 128 × 129 float | 6779 | **4538** | −33.1% |
| **256 × 385 float** | 6936 | **5040** | **−27.3%** |
| 1024 × 1025 float | 7097 | **5090** | −28.3% |
| 4096 × 4097 float | 8339 | **6247** | −25.1% |
| 256 × 385 half | 7275 | **4799** | −34.0% |
| 4096 × 4097 half | — | **6171** | — |

单位：kernel cycle 数（`end_cycle - start_cycle`，跨所有 AIV 核）。
`ops/trace/tests/ut` 全绿（含 fp16 精度比对）。

---

## 2. 基线分解

优化前实现包含 5 个分阶段探针（`TRACE_STAGE=0..4` 提前返回），
在同一 build 内测得各阶段累计周期：

| 阶段 | 内容 | 累计周期 | 增量 |
|---|---|---:|---:|
| Stage 0 | kernel 入口 + TilingData 加载 | 3104 | 3104 |
| Stage 1 | + 6 次标量/中转 `InitBuffer` | 3839 | +735 |
| Stage 2 | + 4 次稠密 `InitBuffer` + `BuildMask` | 4497 | +658 |
| Stage 3 | + 本地 CopyIn/Compute 循环 | 5541 | +1044 |
| Stage 4 | + 跨核汇总（MTE3 + `SyncAll`） | 6660 | +1119 |

**固定开销（Stage 0~2）占 68%**，而其中：
- TilingData 从 GM 拷到栈 + 首次 `MOVEMASK` 约 3100 cycles
- 10 次 `InitBuffer` 约 1500 cycles
- `BuildMask` 的倍增复制约 400 cycles

指令级画像（Stage 4，完整流程）：**SCALAR 管线独占 72.3%**，MTE2 只有 21 cycles。
说明这个算子根本不是带宽瓶颈 —— 数据量小到 MTE2 早已搬完，时间全耗在 Scalar 侧
的循环控制、Buffer 簿记与屏障等待上。

---

## 3. 优化项

### 3.1 去掉对角掩码（省 ~660 cycles）——**已被 §7.2 取代**

> 本节结论在真机数据出来后**已回退**：靠「填充是元素值重复」来省掉掩码，
> 等于把正确性押在 `DataCopyPad` 的实现细节上；而真机上真正该做的是把 4B 小块
> 换成 32B 整块搬运，这时块内就是同一行的真实元素，必须用掩码挑。
> 见 §7.2。下面保留原始推导，仅作为当时的实测记录。

原实现假设 `DataCopyPad` 在 32B datablock 内的填充字节内容不可依赖，
于是构造掩码 `[1,0,0,...,0]` 逐块屏蔽。实测发现**填充是「元素值重复」**：

```
256×385 float，关掉掩码后结果 = -168，而 golden = -21
-168 / -21 = 8 = ELEMS_PER_BLOCK（fp32 下一个 32B block 装 8 个 float）
```

所以整段归约得到的和恰好是真实和的 `ELEMS_PER_BLOCK` 倍，最后统一
`× 1/ELEMS_PER_BLOCK` 即可还原。这一步删掉了：
- `BuildMask()` 整套倍增复制逻辑
- 每轮循环一次全量 `Mul`
- `maskBuf` / `patBuf` 两块 UB

### 3.2 UB Buffer 合并，`InitBuffer` 10 次 → 2 次（省 ~900 cycles）

手册 3.8.3.3 指出 `TPipe` 的 `InitBuffer` 是纯 Scalar 侧簿记开销。原实现对 8 块
`TBuf` + 1 个 `TQue` 各调一次，共 10 次。改为：

- **一次** `InitBuffer(arenaBuf, ...)`，内部按 float 下标切出
  scalar / accum / final / out / cross / work 各槽位（全部 32B 对齐）
- **一次** `InitBuffer(inputQueueX, ...)`
- 工作区按需分配：fp32 就地归约，只有 half/bf16 才分配 FP32 Cast 工作区

### 3.3 归约改用「二分累加 + WholeReduceSum」（省 ~200 cycles）

手册 3.8.6.3「选择低延迟指令，优化归约操作性能」给出的实测数据：
`ReduceSum` 高层的延迟最高，先把数据折半、剩余 ≤ 一条 Repeat 时再用
`WholeReduceSum` 收尾可以显著降低延迟。

实现要点（**三个坑，都是踩出来的**）：

```cpp
    SetMaskCount();
    int64_t nb = n / ELEMS_PER_BLOCK_F;    // 折半以 32B block 为单位
    while (nb > 8) {
        const int64_t half = nb / 2;              // 等分的前半 block 数
        const int64_t cnt = half * ELEMS_PER_BLOCK_F;
        int64_t off = 0;
        while (off < cnt) {
            int64_t chunk = cnt - off;
            if (chunk > REDUCE_CHUNK) {
                chunk = REDUCE_CHUNK;
            }
            SetVectorMask<uint8_t, MaskMode::COUNTER>(0, static_cast<uint32_t>(chunk));
            Add<float, false>(src[off], src[off], src[cnt + off], MASK_PLACEHOLDER, 1, params);
            off += chunk;
        }
        // 奇数 block：把最后一个块并入第 0 块，避免丢元素
        if ((nb & 1) != 0) {
            SetVectorMask<uint8_t, MaskMode::COUNTER>(0, static_cast<uint32_t>(ELEMS_PER_BLOCK_F));
            Add<float, false>(src[0], src[0], src[(nb - 1) * ELEMS_PER_BLOCK_F], MASK_PLACEHOLDER, 1, params);
        }
        nb = half;
    }
```

| 坑 | 症状 | 原因 |
|---|---|---|
| 单条指令 mask 计数上限 | `ubFactor=1024` 必错，`512` 全对 | 折半首步要处理 4096 个元素，实测上限在 4095。按 `REDUCE_CHUNK = 8*511 = 4088` 分块下发 |
| 折半必须「等分」 | 结果偏小，且只有部分 diagLen 出错 | 按 `floor(cur/2)` 折半时，`[cnt, half)` 那段既不在目的区也不在源区，被静默丢掉。改成按 32B block 等分 + 奇数尾块并回第 0 块 |
| 就地相加的区间重叠 | 结果错乱 | 目的区 `[0,cnt)` 必须与源区 `[cnt, cnt+cnt)` 不重叠，等分即可保证 |

> 第二个坑最隐蔽：`diagLen=896` 时折半链上全是 2 的倍数，碰巧通过；
> 而 800 / 900 / 1024 全错。只测单一 shape 会漏掉这类 bug。

### 3.4 跨核汇总改向量归约

原来 0 号核用 `for` 循环逐个 `GetValue()` 累加（每次都是一次 DCache miss）。
改为把全部槽位读回后用同一个 `ReduceBinary` 归约。因为每个槽位只有第 0 个
float 有值、其余由 `Duplicate` 置 0，直接整段归约即可，累加顺序由归约本身
固定，结果仍然确定可复现。

### 3.5 核数启发式（省 800~20000 cycles，取决于 shape）

`SyncAll<true>` 走 FFTS 硬件同步（查 `dav_c220/kernel_operator_sync_impl.h`：
`ffts_cross_core_sync` + `wait_flag_dev`，已是最廉价的变体），但仍有约
**800 cycles 固定成本**。而单核搬运+归约的耗时随元素数增长很慢，于是存在拐点。

实测（同一 build 内）：

| diagLen | 1 核 | 16 核 | 结论 |
|---:|---:|---:|---|
| 32 | **4074** | 4829 (4核) | 单核快 |
| 128 | **4538** | 4891 (8核) | 单核快 |
| 160 | **4898** | 4923 | 单核快 |
| 192 | 4936 | **4920** | 打平（阈值） |
| 224 | 5267 | **4935** | 多核快 |
| 256 | 5423 | **5040** | 多核快 |
| 4096 | 27370 | **6970** (24核) | 多核大幅领先 |

据此 Tiling 策略：

```cpp
    const int64_t mcMinDiag = ReadEnvInt("TRACE_MC_MIN", MULTICORE_MIN_DIAG);
    int64_t blockDim = 1;
    if (diagLen >= mcMinDiag && diagLen > 0) {
        blockDim = CeilDiv(diagLen, MIN_DIAG_ELEMS_PER_CORE);
        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
        if (blockDim > diagLen) {
            blockDim = diagLen;
        }
    }
```

`MULTICORE_MIN_DIAG = 192`，`MIN_DIAG_ELEMS_PER_CORE = 16`。

---

## 4. 试过但**没**采纳的改动

记录下来，避免以后重复踩。

### 4.1 去掉单轮循环的累加器 —— 回退

单轮循环（`count_ <= ubFactor_`）时累加器是多余的，可以直接取归约结果，
省掉一次 `Duplicate` + `Add`。逻辑上更少工作，实测却出现：

| 版本 | 256×385 float | 256×385 half |
|---|---:|---:|
| 基准（保留累加器） | 5040 | **4799** |
| 去掉累加器（用 `bool` 成员） | 4717 | 5169 |
| 去掉累加器（用成员函数） | 4988 | 5163 |

half 一致退化 ~7%。原因是新增成员/分支改变了对象内存布局与代码布局，
ICache 命中变差，收益被抵消。**回退，保留更简单可验证的版本。**

### 4.2 保留分阶段探针 —— 不从 kernel 移除

探针本身（`if (stage_ == N) return;`）看似零成本，但把 `ReduceBinary` 等
多条路径都编译进 kernel 后，默认路径实测从 6010 涨到 6936（**+15%**）。
所以探针只保留在 Host 侧（环境变量控制核数/shape），kernel 内不留任何分支。

---

## 5. 测量方法论

### 5.1 周期数在同一 build 内完全可复现

同一源码、同一 build 重复跑三次：

```
stage=4 opts=0 : 6936 / 6936 / 6936
stage=4 opts=5 : 6405 / 6405 / 6405
```

跨容器（重新编译但源码不变）也一致。

### 5.2 跨 build 有 ±7% 的代码布局漂移

**这是本项目的关键陷阱。** 只改了一条用不到的分支，默认路径就能变化 15%（见 4.2）。
因此：

- 同一 build 内比较 → 可信（如 4.1 的表格）
- 不同 build 之间比较 → 必须把两版源码**各自完整编译并各跑一遍**，
  不能拿历史数字对比

本文第 1 节的 A/B 表格即按此方法得到：分别 checkout 两版源码，
在同一个容器镜像、同一台机器、同一 harness 下各跑一遍。

### 5.3 复现方式

```bash
# 全部默认用例
docker run --rm -v "$PWD:/workspace" ascendc-cpu-dev:cann92 \
    bash /workspace/ops/trace/tests/bench_910b.sh

# 指定用例：shape / dtype / 强制核数
CASES="m=256,n=385 m=1024,n=1025,cores=8 m=256,n=385,dtype=half" \
docker run --rm -v "$PWD:/workspace" -e CASES ascendc-cpu-dev:cann92 \
    bash /workspace/ops/trace/tests/bench_910b.sh
```

Host 侧可用的环境变量探针：

| 变量 | 作用 |
|---|---|
| `TRACE_CORES` | 强制指定核数（1~coreNum），用于验证核数策略 |
| `TRACE_MC_MIN` | 覆盖启用多核的 diagLen 阈值 |
| `TRACE_M` / `TRACE_N` | 覆盖输入 shape |
| `TRACE_DTYPE` | `float` / `half` |

---

## 6. 剩余空间

256×385 的 5040 cycles 中约 3000 属于框架入口/出口，**不是 kernel 逻辑能动的**：

| 项 | 周期 | 说明 |
|---|---:|---|
| 首次 `MOVEMASK` | 346 | 框架初始化 |
| `DCCI` | 356 | Cache 一致性，`GlobalTensor::SetValue` 触发 |
| 栈读写（`STI_XN_IMM` 等） | ~600 | TilingData 解包、局部变量溢出 |
| `InitBuffer` + arena 设置 | ~700 | 已从 10 次降到 2 次，剩余是 TPipe 固定成本 |

还可尝试的方向（估计合计 5% 左右，已接近噪声水平）：
1. 把 scalar 槽位塞进输入队列尾部，`InitBuffer` 再降到 1 次
2. `WriteOut` 避开 `SetValue`，改用 `DataCopy` 直接写 GM 标量
3. 极端情况下用 `TRACE_MC_MIN` 按实测拐点做更细的分段查表

---

## 7. 真机数据与 4 倍差距（2026-11-10）

在真实 NPU 上用判题环境跑出来的结果：

| 测试点 | 结果 | 输出误差 | 用时 | 该点最优用时 |
|---|---|---|---:|---:|
| 1~6 | Pass | 0.00% | 5.34 ~ 6.20 µs | 1.41 ~ 1.80 µs |
| 7 | **Runtime Error** | 100% | — | 1.90 µs |
| 8~12 | Skipped | — | — | 2.52 ~ 3.90 µs |

两个独立问题，必须分开看。

### 7.1 测试点 7 崩溃：`SyncAll` 硬同步死锁（已修）

日志关键行：

```
[WARN] <ReplayOnce> Kernel run on device 1 No. 1 time failed.
[ERROR] aclrtSynchronizeStreamWithTimeout(stream, 3000) failed , ret=507015
Task Duration(us): 0.000000, Block Dim: 16
```

`ret=507015` 是 3 秒超时，即 kernel 根本没跑完——**卡死**，不是算错。
对照手册 6.2.3.7.2.3 `SyncAll` 的约束说明：

> 使用该接口进行多核控制时，算子调用时指定的逻辑 numBlocks 必须保证不大于
> 实际运行该算子的 AI 处理器核数，否则框架进行多轮调度时会插入异常同步，
> **导致 Kernel"卡死"**。
>
> 使用该接口时，建议开启 batchmode 模式，使算子独占全部所需核资源，否则可能
> 因满足以下条件导致**死锁**：多流并发（≥2 条执行流）、≥2 个算子并发执行、
> 所有并发算子的核数总和超过物理核数、≥2 个并发算子使用了核间同步功能。

现象完全吻合：**`Block Dim: 1` 的 6 个点全过，唯一 `Block Dim: 16` 的点 7 死锁**。
单核路径不进 `SyncAll`，所以不受影响；一旦多核就踩约束。

三处修复（缺一不可）：

1. `trace_tiling.cpp`：`context->SetScheduleMode(1)` 开 batchmode，独占核资源。
2. `trace.h`：参与同步的核数改用运行时 `GetBlockNum()`，不再单纯信 Tiling 下发的
   `usedCoreNum`——两者一旦不一致，就会出现「部分核等、部分核不到」的经典死锁。
3. workspace 按 `MAX_SLOTS = 64` 预留（固定 2048B），而不是 `blockDim * 32B`。
   这样即使框架上核数多于预期，写回也不会越界踩坏别的内存。

### 7.2 5.3~6.2 µs vs 1.41 µs：CAModel 没有建模 DMA

这不是同一口径下的差距，先看清楚差在哪：

- 1.41 µs @1650 MHz ≈ **2327 cycles**。而我们 CAModel 实测「几乎空 kernel」的入口
  就是 ~3100 cycles，说明**真机任务的基础开销本身就接近 1.4 µs**——最优实现
  在这个 shape 上几乎没花额外时间。
- CAModel 对 256×385 报 5040 cycles，其中 MTE2（搬入）只算 **21 cycles**。
  真机上要按 `diagLen` 个 4 字节跨 1540B 步长去取数（128×129 是 128 个块），
  每个块都要独立命中一个 32B sector，绝不可能只用 21 cycles。

**结论：CAModel 对 MTE2 的建模严重失真，任何以 GM 访存为主的优化都不能用
CAModel 判断收益。** §1~§6 的结论只对「指令条数 / 标量簿记 / 向量流水」有效。

针对这一点做了搬运方式改造（`TRACE_COPY` 可切）：

| 方式 | blockLen | 说明 |
|---|---|---|
| `TRACE_COPY=0`（默认） | 32B | 从对角元素本身起始整块取满一个 datablock，纯 strided 拷贝，DMA 侧不做「4B 展开成 32B 并复制填充」的工作 |
| `TRACE_COPY=1` | sizeof(T) | 原来的 4B + `isPad`，块数相同但每块更小 |

两条路径在 UB 里布局完全一致（第 r 个对角元素都在第 r 个 32B block 的起始处），
所以可以**前半段走 32B、末尾少量行走 4B**——末尾那几行是为了避免 32B 读越过
输入张量边界。掩码 `[1,0,0,...,0]` 精确挑出块首元素，因此顺带**不再依赖
`DataCopyPad` 的填充内容**（之前实测 fp32 下填充是「元素值重复」，属实现细节，
不该依赖），也不用再乘 `1/ELEMS_PER_BLOCK` 还原，数值上更干净。

代价是 CAModel 口径多了掩码构建 + 一次 `Mul`（128×129 从 4538 涨到 5325 cycles，
约 +17%），但 CAModel 根本不计算这笔 DMA 收益是否为正——**这个 A/B 只能在真机上做**：

```bash
SHAPES="128x129 256x385" MODES="0 1" bash ops/trace/tests/bench_real.sh
```

`bench_real.sh` 用 `msprof op --aic-metrics=PipeUtilization` 直接给出
`aiv_mte2_ratio` / `aiv_vec_ratio` / `aiv_scalar_ratio`，把 4 倍差距归因到具体管线；
如果 `mte2_ratio` 高，继续往搬运方向做；如果 `scalar_ratio` 高，就得回头砍簿记。

### 7.3 顺带修掉的环境问题

`tests/ut` 的 golden 生成原来 `import torch`，只为用 `torch.trace`。镜像里
torch 的 pyc 一旦与解释器不匹配（`bad marshal data`），整个 UT 链路直接跑不起来。
已改为纯 numpy（`np.trace`，同样 FP32 累加），UT 不再依赖 torch。

---

## 8. 改动文件

| 文件 | 改动 |
|---|---|
| `op_kernel/trace.h` | UB 合并、二分归约、向量化跨核汇总；搬运改为 32B 整块 + 掩码；核数取 `GetBlockNum()`；workspace 按 `MAX_SLOTS` 预留 |
| `op_kernel/trace_tiling_data.h` | 移除性能探针字段，新增 `copyMode`（真机 A/B 用） |
| `op_host/trace_tiling.cpp` | 核数启发式、UB 预算 320→192 B/元素、`SetScheduleMode(1)` 开 batchmode、workspace 固定上限 |
| `examples/test_aclnn_trace.cpp` | 支持 `TRACE_DTYPE=half` 及对应回读 |
| `tests/ut/op_host/test_trace_tiling.cpp` | tiling 期望值同步新策略，补阈值覆盖用例 |
| `tests/ut/op_kernel/trace_data/gen_data.py` | golden 去掉 torch 依赖，改纯 numpy |
| `tests/bench_910b.sh` | 新增，取代 `measure_baseline.sh` / `sweep_opts.sh`；支持 `copy=` 切换搬运方式 |
| `tests/bench_real.sh` | 新增，真机 `msprof op` 测耗时 + 管线归因，支持搬运方式 A/B |
