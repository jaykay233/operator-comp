/*!
 * \file trace.h
 * \brief Trace 算子 kernel 类定义（优化版）
 *
 * 功能：out[0] = sum_{i=0}^{min(M,N)-1} self[i][i]，内部以 FP32 累加，最后转回输入 dtype。
 *
 * ---------------------------------------------------------------------------
 * 设计要点
 * ---------------------------------------------------------------------------
 * 1. 跨步收集：主对角线在 GM 上步长为 (N+1) 个元素。用一条 DataCopyPad 的
 *    blockCount/blockLen/srcStride 描述整段对角线，不做逐元素标量读取。
 *
 * 2. UB 布局与 padding 语义（实测结论）：
 *    DataCopyPad 以 32B datablock 为最小粒度，第 i 个对角元素落在 UB 的第
 *    i*ELEMS_PER_BLOCK 个 float 槽位。实测该 datablock 内其余字节是**元素值重复**
 *    填充（fp32 下 8 个 float 全等于该元素）。因此不需要逐块掩码：
 *    对整段做归约得到的和恰好是真实和的 ELEMS_PER_BLOCK 倍，最后按
 *    1/ELEMS_PER_BLOCK 还原即可。这一步省掉了整段 mask 构造与一次全量 Mul。
 *
 * 3. 归约：采用手册 3.8.6.3「二分累加 + WholeReduceSum」方案 —— 先用 Add 把数据
 *    反复折半，剩余元素数不超过一条 Repeat（float 为 64 个）时用一次
 *    WholeReduceSum 收尾，避免 ReduceSum 接口内部的多指令组合与 Scalar 簿记。
 *    折半在原张量上就地完成，不需要额外工作区。
 *
 * 4. 多核 + 确定性：按核沿对角线切分；各核把部分和写入 workspace 的独立 32B 槽位，
 *    SyncAll 后 0 号核按 blockIdx 升序归约，累加顺序固定、结果可复现。
 *    实测跨核屏障开销可达 1200 cycles，而本算子在小 shape 下是延迟瓶颈
 *    （计算耗时几乎与元素数无关），因此 Tiling 默认只用 1 个核，仅在
 *    diagLen 很大时才启用多核（见 trace_tiling.cpp 的 TRACE_MULTICORE_MIN_DIAG）。
 *
 * 5. UB 按需分配：标量/中转区仅 32 float + 跨核槽位；只有 half/bf16 需要一块
 *    FP32 工作区（Cast 目标）。InitBuffer 从 10 次降到 2 次。
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
constexpr int64_t ONE_REPEAT_FLOAT = 64;        // 一条 Repeat 的 float 个数（256B / 4B）
constexpr int64_t REDUCE_CHUNK = 8 * 511;       // 折半时单条指令处理的元素上限（规避 mask 计数上限）
constexpr int64_t ELEMS_PER_BLOCK_F = 8;        // 一个 32B block 的 float 个数（折半粒度）

template <typename T>
class Trace {
public:
    __aicore__ inline Trace(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const TraceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // 每个 32B datablock 能容纳的 T 元素个数
    static constexpr int64_t ELEMS_PER_BLOCK = BLOCK_BYTES / (int64_t)sizeof(T);
    // half/bf16 需要一块 FP32 工作区做 Cast；float 可以就地归约
    static constexpr bool NEED_FLOAT_BUF = (sizeof(T) != sizeof(float));

    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void ReduceBinary(const LocalTensor<float>& dst, const LocalTensor<float>& src, int64_t n);
    __aicore__ inline void WriteOut(float total);

    // arena 内的 float 下标（单位：float，均按 32B = 8 float 对齐）
    __aicore__ inline int64_t OffScalar() const { return 0; }
    __aicore__ inline int64_t OffAccum() const { return 8; }
    __aicore__ inline int64_t OffFinal() const { return 16; }
    __aicore__ inline int64_t OffOut() const { return 24; }
    __aicore__ inline int64_t OffCross() const { return 32; }
    __aicore__ inline int64_t OffWork() const { return 32 + usedCoreNum_ * SLOT_FLOATS; }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    // 所有 VECCALC Buffer 合并成一个 arena，只调一次 InitBuffer
    TBuf<TPosition::VECCALC> arenaBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    GlobalTensor<float> workspaceGM;

    int64_t cols_ = 0;       // 矩阵列数 N（决定对角步长 N+1）
    int64_t start_ = 0;      // 本核负责的第一个对角元素下标
    int64_t count_ = 0;      // 本核负责的对角元素个数
    int64_t ubFactor_ = 1;   // 单次循环处理的元素个数
    int64_t slots_ = 0;      // 单次循环在 UB 中的稠密 float 个数
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
    slots_ = ubFactor_ * ELEMS_PER_BLOCK;

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), rows * cols_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), 1);
    // 每个核独占一个 32B 槽位，避免相邻核写同一 cache line 造成伪共享
    workspaceGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), usedCoreNum_ * SLOT_FLOATS);

    // 一次 InitBuffer 覆盖全部 VECCALC 需求：
    //   小缓冲 32 float（scalar/accum/final/out 各 8）+ 跨核槽位 [+ half/bf16 的 FP32 工作区]
    const int64_t arenaFloats = OffWork() + (NEED_FLOAT_BUF ? slots_ : 0);
    pipe.InitBuffer(arenaBuf, arenaFloats * (int64_t)sizeof(float));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubFactor_ * BLOCK_BYTES);

    LocalTensor<float> arena = arenaBuf.Get<float>();
    // 累加器清零（即使本核没有对角元素，也要参与后续跨核同步）
    Duplicate(arena[OffAccum()], 0.0f, static_cast<int32_t>(SLOT_FLOATS));
    PipeBarrier<PIPE_V>();
}

/*!
 * \brief 二分累加 + WholeReduceSum（手册 3.8.6.3）
 *
 * 先用 Add 把数据反复折半，剩余元素数 ≤ 一条 Repeat（float 为 64 个）时用一次
 * WholeReduceSum 收尾。全程就地完成，不需要额外工作区。
 *
 * 三个实现约束（都实测踩过）：
 *   1. 必须按 32B block（8 个 float）为粒度折叠，且折半只能是「等分」：
 *      把 [0,half) 与 [half,2*half) 相加。若只折 floor(cur/2) 个元素，
 *      中间那段元素既不在目的区也不在源区，会被静默丢掉 —— 表现为结果偏小。
 *   2. block 数为奇数时，最后一块单独并入第 0 块，然后再继续折半。
 *   3. 单条向量指令的 mask 计数有上限（实测 4096 越界、4095 以内安全），
 *      因此 Add 按 REDUCE_CHUNK = 8*511 = 4088 个元素分块下发。
 */
template <typename T>
__aicore__ inline void Trace<T>::ReduceBinary(const LocalTensor<float>& dst, const LocalTensor<float>& src, int64_t n)
{
    BinaryRepeatParams params;
    params.dstBlkStride = 1;
    params.src0BlkStride = 1;
    params.src1BlkStride = 1;
    params.dstRepStride = 8;
    params.src0RepStride = 8;
    params.src1RepStride = 8;

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
    // 收尾：一次 WholeReduceSum 把 1 个 Repeat 内的元素归约成 1 个
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, static_cast<uint32_t>(nb * ELEMS_PER_BLOCK_F));
    WholeReduceSum<float, false>(dst, src, MASK_PLACEHOLDER, 1, 1, 1, 8);
    ResetMask();
    SetMaskNorm();
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
    LocalTensor<float> arena = arenaBuf.Get<float>();
    LocalTensor<float> accumLocal = arena[OffAccum()];
    LocalTensor<float> scalarLocal = arena[OffScalar()];

    const int64_t slots = currentNum * ELEMS_PER_BLOCK;

    LocalTensor<float> srcLocal;
    if constexpr (NEED_FLOAT_BUF) {
        // half/bf16：先统一转 FP32（padding 同样是元素值重复，转换后仍然重复）
        srcLocal = arena[OffWork()];
        Cast(srcLocal, xLocal, RoundMode::CAST_NONE, static_cast<int32_t>(slots));
    } else {
        // float：就地归约，不需要额外工作区
        srcLocal = xLocal;
    }

    ReduceBinary(scalarLocal, srcLocal, slots);
    // 段间累加（FP32）
    Add(accumLocal, accumLocal, scalarLocal, 1);

    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Trace<T>::WriteOut(float total)
{
    LocalTensor<float> arena = arenaBuf.Get<float>();
    if constexpr (sizeof(T) == sizeof(float)) {
        outputGMY.SetValue(0, total);
    } else {
        LocalTensor<float> finalLocal = arena[OffFinal()];
        // out 槽位按字节偏移换算成 T 单位
        LocalTensor<T> yLocal = arenaBuf.Get<T>()[OffOut() * (int64_t)sizeof(float) / (int64_t)sizeof(T)];
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
        LocalTensor<float> accumLocal = arenaBuf.Get<float>()[OffAccum()];

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
        // padding 是元素值重复 → 归约结果被放大 ELEMS_PER_BLOCK 倍，此处一次性还原
        localSum = accumLocal.GetValue(0) / static_cast<float>(ELEMS_PER_BLOCK);
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());

    if (usedCoreNum_ <= 1) {
        if (blockIdx == 0) {
            WriteOut(localSum);
        }
        return;
    }

    // ---- 多核汇总 ----
    // 用 MTE3 写回，保证别的核能通过 MTE2 读到最新值；槽位按 32B 对齐互不重叠。
    LocalTensor<float> crossLocal = arenaBuf.Get<float>()[OffCross()];
    Duplicate(crossLocal, 0.0f, static_cast<int32_t>(SLOT_FLOATS));
    crossLocal.SetValue(0, localSum);
    PipeBarrier<PIPE_ALL>();
    DataCopy(workspaceGM[blockIdx * SLOT_FLOATS], crossLocal, SLOT_FLOATS);
    PipeBarrier<PIPE_ALL>();

    SyncAll<true>();

    if (blockIdx == 0) {
        // 0 号核读回全部槽位：每个槽位只有第 0 个 float 有值、其余为 0，
        // 因此可以直接向量归约（累加顺序由归约本身固定，结果确定）。
        DataCopy(crossLocal, workspaceGM[0], usedCoreNum_ * SLOT_FLOATS);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> scalarLocal = arenaBuf.Get<float>()[OffScalar()];
        ReduceBinary(scalarLocal, crossLocal, usedCoreNum_ * SLOT_FLOATS);
        WriteOut(scalarLocal.GetValue(0));
    }
}

} // namespace NsTrace
#endif // TRACE_H
