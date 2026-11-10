# Trace 算子性能优化记录

本文记录 Trace 算子在 Ascend 910B1 上的性能分析过程与优化结果。
所有数字均为**实测**（CAModel 周期级仿真），不是估算。

- 目标平台：Ascend 910B1（`dav_2201`），CANN 9.2.0
- 测量工具：`tests/bench_910b.sh`（CAModel + `profile_aiv_log0.toml`）
- 参考文档：`docs/CANN_9.0.0_AscendC_算子开发指南.pdf` 第 3.7、3.8 章

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

### 3.1 去掉对角掩码（省 ~660 cycles）

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

## 7. 改动文件

| 文件 | 改动 |
|---|---|
| `op_kernel/trace.h` | UB 合并、去掩码、二分归约、向量化跨核汇总 |
| `op_kernel/trace_tiling_data.h` | 移除全部性能探针字段 |
| `op_host/trace_tiling.cpp` | 核数启发式、UB 预算 320→192 B/元素 |
| `examples/test_aclnn_trace.cpp` | 支持 `TRACE_DTYPE=half` 及对应回读 |
| `tests/ut/op_host/test_trace_tiling.cpp` | tiling 期望值同步新策略，补阈值覆盖用例 |
| `tests/bench_910b.sh` | 新增，取代 `measure_baseline.sh` / `sweep_opts.sh` |
