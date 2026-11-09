/*!
 * \file trace.h
 * \brief Trace 算子 kernel 类定义
 *
 * 功能：out[0] = sum_{i=0}^{min(M,N)-1} self[i][i]，内部以 FP32 累加，最后转回输入 dtype。
 *
 * 实现要点（对应赛题要求）：
 *   1. 跨步收集：主对角线在 GM 上步长为 (N+1) 个元素。用 DataCopyPad 的
 *      blockCount/blockLen/srcStride 描述多块跳跃搬运，一次 MTE2 搬完一段对角线，
 *      不做逐元素标量读取，也不整矩阵搬运。
 *   2. UB 布局与掩码：GM->UB 的 DataCopyPad 以 32B datablock 为最小写入粒度，
 *      第 i 个对角元素落在 UB 的第 i*ELEMS_PER_BLOCK 个槽位，块内其余字节的填充内容
 *      由硬件实现决定（可能填 pad 值，也可能重复元素值）。为不依赖该行为，
 *      先用固定掩码 [1,0,0,...,0] 逐块屏蔽，只保留每个 block 的首个元素再整段归约。
 *   3. FP32 累加：half/bf16 输入先 Cast 到 FP32 再参与 Mul/ReduceSum，累加器始终 FP32。
 *   4. 多核并行 + 确定性：沿对角线切分，各核把部分和用 MTE3 写入 workspace 的独立
 *      32B 槽位（DataCopy 保证写回可见），SyncAll 后由 0 号核用 MTE2 读回，
 *      按 blockIdx 升序串行累加。累加顺序固定，结果可复现。
 *   5. Double Buffer：输入队列深度 BUFFER_NUM=2，搬运与计算形成 ping-pong 流水。
 */

#ifndef TRACE_H
#define TRACE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "trace_tiling_data.h"
#include "trace_tiling_key.h"

namespace NsTrace {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;               // 输入队列深度（double buffer）
constexpr int64_t BLOCK_BYTES = 32;             // UB 一个 datablock 的字节数
constexpr int64_t SLOT_FLOATS = 8;              // 跨核部分和的槽位宽度（32B / sizeof(float)）
constexpr int64_t MAX_BLOCK_COUNT = 4095;       // DataCopyPad blockCount 上限

template <typename T>
class Trace {
public:
    __aicore__ inline Trace(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const TraceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // 每个 32B datablock 能容纳的 T 元素个数
    static constexpr int64_t ELEMS_PER_BLOCK = BLOCK_BYTES / (int64_t)sizeof(T);

    __aicore__ inline void BuildMask();
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void WriteOut(float total);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TBuf<TPosition::VECCALC> maskBuf;     // 对角掩码 [1,0,0,...]
    TBuf<TPosition::VECCALC> patBuf;      // 32B 掩码模式块 [1,0,0,...,0]
    TBuf<TPosition::VECCALC> workBuf;     // FP32 工作区（搬入数据转 FP32 后按掩码过滤）
    TBuf<TPosition::VECCALC> reduceBuf;   // ReduceSum 中间缓冲
    TBuf<TPosition::VECCALC> scalarBuf;   // 单段归约结果
    TBuf<TPosition::VECCALC> accumBuf;    // 本核 FP32 累加器
    TBuf<TPosition::VECCALC> finalBuf;    // 最终 float 值
    TBuf<TPosition::VECCALC> outBuf;      // 最终 T 类型输出值
    TBuf<TPosition::VECCALC> crossBuf;    // 跨核部分和的中转 UB 区

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    GlobalTensor<float> workspaceGM;

    int64_t cols_ = 0;       // 矩阵列数 N（决定对角步长 N+1）
    int64_t start_ = 0;      // 本核负责的第一个对角元素下标
    int64_t count_ = 0;      // 本核负责的对角元素个数
    int64_t ubFactor_ = 1;   // 单次循环处理的元素个数
    int64_t usedCoreNum_ = 1;
};

template <typename T>
__aicore__ inline void Trace<T>::Init(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                      const TraceTilingData* tilingData)
{
    const int64_t rows = tilingData->rows;
    cols_ = tilingData->cols;
    usedCoreNum_ = tilingData->usedCoreNum > 0 ? tilingData->usedCoreNum : 1;

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    start_ = blockIdx * tilingData->blockFactor;
    int64_t end = start_ + tilingData->blockFactor;
    if (end > tilingData->diagLen) {
        end = tilingData->diagLen;
    }
    count_ = (end > start_) ? (end - start_) : 0;

    ubFactor_ = tilingData->ubFactor;
    if (ubFactor_ <= 0) {
        ubFactor_ = 1;
    }
    if (ubFactor_ > MAX_BLOCK_COUNT) {
        ubFactor_ = MAX_BLOCK_COUNT;
    }

    if (GetBlockIdx() == 0) {
        printf("[TILING] rows=%lld cols=%lld diag=%lld bf=%lld ub=%lld cores=%lld\n",
               (long long)rows, (long long)cols_, (long long)tilingData->diagLen,
               (long long)tilingData->blockFactor, (long long)tilingData->ubFactor, (long long)usedCoreNum_);
    }
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), rows * cols_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), 1);
    // 每个核独占一个 32B 槽位，避免相邻核写同一 cache line 造成伪共享
    workspaceGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), usedCoreNum_ * SLOT_FLOATS);

    // 小缓冲在任何情况下都要初始化：即使本核没有对角元素，也必须参与 SyncAll
    pipe.InitBuffer(scalarBuf, BLOCK_BYTES);
    pipe.InitBuffer(accumBuf, BLOCK_BYTES);
    pipe.InitBuffer(finalBuf, BLOCK_BYTES);
    pipe.InitBuffer(outBuf, BLOCK_BYTES);
    pipe.InitBuffer(patBuf, BLOCK_BYTES);
    pipe.InitBuffer(crossBuf, usedCoreNum_ * SLOT_FLOATS * (int64_t)sizeof(float));

    if (count_ <= 0) {
        return;   // 无数据可搬，但仍会进入 Process 参与跨核同步
    }

    // 每个对角元素在 UB 中固定占一个 32B datablock；
    // 转成 FP32 并按元素计数后，稠密数组占用 slots * 4 字节
    const int64_t denseBytes = ubFactor_ * (BLOCK_BYTES * 4 / (int64_t)sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubFactor_ * BLOCK_BYTES);
    pipe.InitBuffer(maskBuf, denseBytes);
    pipe.InitBuffer(workBuf, denseBytes);
    pipe.InitBuffer(reduceBuf, denseBytes);

    BuildMask();
}

/*!
 * \brief 构造重复掩码 [1,0,0,...,0]（每 ELEMS_PER_BLOCK 个元素一个 1）
 *
 * 只依赖掩码本身，不依赖 DataCopyPad 在 UB 中如何填充 block 的空闲字节。
 *
 * 构造方式（倍增复制，O(log ubFactor) 条搬运指令，无标量循环）：
 *   1. 先写首个 32B 模式块 [1,0,0,0,0,0,0,0]；
 *   2. 已初始化的 len 个块整体复制到 [len, 2*len) 位置，len 翻倍，直到覆盖 ubFactor。
 * 相比逐块 SetValue（ubFactor 次标量写），这样避免把 SCALAR 管线打满。
 */
template <typename T>
__aicore__ inline void Trace<T>::BuildMask()
{
    LocalTensor<float> maskLocal = maskBuf.Get<float>();

    // 每个对角元素在 UB 中占一个 32B datablock；转 FP32 后对应 ELEMS_PER_BLOCK 个 float。
    // 因此掩码的重复周期是 ELEMS_PER_BLOCK（fp32 为 8，fp16/bf16 为 16），
    // 对应 patBlocks 个 32B datablock。
    const int64_t patFloats = ELEMS_PER_BLOCK;
    const int64_t patBlocks = patFloats / SLOT_FLOATS;

    // 首个模式单元：[1, 0, 0, ..., 0]
    Duplicate(maskLocal, 0.0f, static_cast<int32_t>(patFloats));
    PipeBarrier<PIPE_ALL>();
    maskLocal.SetValue(0, 1.0f);
    PipeBarrier<PIPE_ALL>();

    int64_t len = 1;   // 已初始化好的模式单元个数
    while (len < ubFactor_) {
        int64_t n = ubFactor_ - len;
        if (n > len) {
            n = len;
        }
        // srcStride = dstStride = 0 表示源/目的各自连续，即把前 len 个单元整体搬到 len 之后
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(n * patBlocks);
        copyParams.blockLen = 1;   // 单位：32B datablock
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopy(maskLocal[len * patFloats], maskLocal, copyParams);
        PipeBarrier<PIPE_ALL>();
        len += n;
    }
}

template <typename T>
__aicore__ inline void Trace<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    // 第 (start_ + offset + i) 行的对角元素位于 GM 下标 (start_+offset+i)*(cols_+1)。
    // blockLen = 单元素字节数；srcStride = 相邻数据块之间跳过的字节数 = N 个元素。
    DataCopyExtParams extParams;
    extParams.blockCount = static_cast<uint16_t>(currentNum);
    extParams.blockLen = static_cast<uint32_t>(sizeof(T));
    extParams.srcStride = static_cast<uint32_t>(cols_ * static_cast<int64_t>(sizeof(T)));
    extParams.dstStride = 0; // UB 侧逐 datablock 排列

    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad(xLocal, inputGMX[(start_ + offset) * (cols_ + 1)], extParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Trace<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> maskLocal = maskBuf.Get<float>();
    LocalTensor<float> accumLocal = accumBuf.Get<float>();
    LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();

    // UB 中每个对角元素占一个 32B datablock
    const int64_t slots = currentNum * ELEMS_PER_BLOCK;

    // 1) 统一转 FP32，并按掩码只保留每个 block 的首元素（不依赖 padding 内容）
    if constexpr (sizeof(T) == sizeof(float)) {
        Mul(workLocal, xLocal, maskLocal, static_cast<int32_t>(slots));
    } else {
        Cast(workLocal, xLocal, RoundMode::CAST_NONE, static_cast<int32_t>(slots));
        Mul(workLocal, workLocal, maskLocal, static_cast<int32_t>(slots));
    }
    // 2) 整段归约到 FP32 标量
    ReduceSum<float>(scalarLocal, workLocal, reduceLocal, static_cast<int32_t>(slots));
    // 3) 段间累加（FP32）
    Add(accumLocal, accumLocal, scalarLocal, 1);

    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Trace<T>::WriteOut(float total)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        outputGMY.SetValue(0, total);
    } else {
        LocalTensor<float> finalLocal = finalBuf.Get<float>();
        LocalTensor<T> yLocal = outBuf.Get<T>();
        finalLocal.SetValue(0, total);
        PipeBarrier<PIPE_ALL>();
        Cast(yLocal, finalLocal, RoundMode::CAST_RINT, 1);
        PipeBarrier<PIPE_ALL>();
        outputGMY.SetValue(0, yLocal.GetValue(0));
    }
}

template <typename T>
__aicore__ inline void Trace<T>::Process()
{
    float localSum = 0.0f;
    if (count_ > 0) {
        LocalTensor<float> accumLocal = accumBuf.Get<float>();
        Duplicate(accumLocal, 0.0f, 1);

        int64_t done = 0;
        while (done < count_) {
            int64_t currentNum = count_ - done;
            if (currentNum > ubFactor_) {
                currentNum = ubFactor_;
            }
            CopyIn(done, currentNum);
            Compute(currentNum);
            done += currentNum;
        }
        localSum = accumLocal.GetValue(0);
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    LocalTensor<float> crossLocal = crossBuf.Get<float>();

    if (usedCoreNum_ <= 1) {
        if (blockIdx == 0) {
            WriteOut(localSum);
        }
        return;
    }

    // ---- 多核汇总 ----
    // 用 MTE3 写回，保证别的核能通过 MTE2 读到最新值；槽位按 32B 对齐互不重叠。
    Duplicate(crossLocal, 0.0f, static_cast<int32_t>(SLOT_FLOATS));
    crossLocal.SetValue(0, localSum);
    PipeBarrier<PIPE_ALL>();
    DataCopy(workspaceGM[blockIdx * SLOT_FLOATS], crossLocal, SLOT_FLOATS);
    PipeBarrier<PIPE_ALL>();

    SyncAll<true>();

    if (blockIdx == 0) {
        // 0 号核按 blockIdx 升序串行累加，累加顺序固定（确定性）
        DataCopy(crossLocal, workspaceGM[0], usedCoreNum_ * SLOT_FLOATS);
        PipeBarrier<PIPE_ALL>();
        float total = 0.0f;
        for (int64_t i = 0; i < usedCoreNum_; ++i) {
            total += crossLocal.GetValue(i * SLOT_FLOATS);
        }
        WriteOut(total);
    }
}

} // namespace NsTrace
#endif // TRACE_H
