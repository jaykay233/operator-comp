/*!
 * \file trace.h
 * \brief Trace 算子 kernel 类定义
 *
 * 功能：out[0] = sum_{i=0}^{min(M,N)-1} self[i][i]，内部以 FP32 累加，最后转回输入 dtype。
 *
 * ---------------------------------------------------------------------------
 * 设计要点
 * ---------------------------------------------------------------------------
 * 1. 跨步收集：主对角线在 GM 上步长为 (N+1) 个元素。用一条 DataCopyPad 的
 *    blockCount/blockLen/srcStride 描述整段对角线，不做逐元素标量读取。
 *
 * 2. 两种搬入方式（由 cols 决定，见 CopyIn）：
 *    - 32B 整块：blockLen = 32B，从对角元素本身起始，一次取满一个 datablock。
 *      第 r 个 block 的第 0 个元素就是该行的对角元素，块内其余 7 个是同行的
 *      后续元素，靠固定掩码 [1,0,0,...,0] 屏蔽。这是纯 strided 拷贝，
 *      DMA 侧不需要做「4B 展开成 32B 并复制填充」的工作。
 *    - 4B + padding：cols 太小时相邻行间隔不足 32B，无法整块取，回退到
 *      blockLen = sizeof(T) 加 isPad。此时 padding 内容由硬件决定
 *      （实测 fp32 下是元素值重复），同样由掩码屏蔽，因此**不依赖该行为**。
 *
 *    两种方式在 UB 中的布局完全一致（第 r 个对角元素位于第 r 个 32B block 的
 *    起始处），所以同一段对角线可以前半段用 32B、末尾少量行用 4B 混着搬，
 *    末尾那几行是为了避免 32B 读越过输入张量边界。
 *
 * 3. 掩码：周期为 ELEMS_PER_BLOCK 的 [1,0,0,...,0]。构造方式是「写入 1 个模式
 *    单元 + 倍增复制」，O(log ubFactor) 条搬运指令，没有 O(ubFactor) 的标量循环。
 *    由掩码精确挑出对角元素，因此不需要再按 1/ELEMS_PER_BLOCK 还原，
 *    数值上也更干净。
 *
 * 4. FP32 累加：half/bf16 输入先 Cast 到 FP32 再参与 Mul/归约，累加器始终 FP32。
 *
 * 5. 归约：二分累加 + WholeReduceSum（手册 2.5.2.3.2 / 3.8.6.3）：
 *    先把数据反复折半，剩余不超过一个 Repeat 时用一次 WholeReduceSum 收尾。
 *
 * 6. 多核 + 确定性：沿对角线切分，各核把部分和写入 workspace 的独立 32B 槽位，
 *    SyncAll 后 0 号核按 blockIdx 升序归约，累加顺序固定、结果可复现。
 *
 *    【重要】SyncAll 是硬同步，手册 6.2.3.7.2.3 的约束：
 *      - 逻辑 numBlocks 必须不大于实际可运行的 AI Core 数，否则框架分多轮调度
 *        会插入异常同步，Kernel 直接卡死（真机表现为 3s 超时、ret=507015）；
 *      - 必须开 batchmode 独占核资源，否则多流/多算子并发时会死锁。
 *    因此：
 *      - 核数以运行时 GetBlockNum() 为准（而不是信 Tiling 下发的值）；
 *      - Host 侧 SetScheduleMode(1) 开启 batchmode；
 *      - workspace 按 kMaxSlots 预留，任何实际上核数都不会越界。
 *
 * 7. UB：所有 VECCALC Buffer 合并成一个 arena，InitBuffer 只调 2 次
 *    （手册 3.8.3.3：InitBuffer 是纯 Scalar 簿记开销）。
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
constexpr int64_t REDUCE_CHUNK = 8 * 511;       // 折半时单条指令处理的元素上限
constexpr int64_t ELEMS_PER_BLOCK_F = 8;        // 一个 32B block 的 float 个数
// 跨核槽位最大个数：workspace 按此预留，保证任何实际上核数都不会越界写。
// 910B1 的 AIV 核数远小于该值（≤ 48）。
constexpr int64_t MAX_SLOTS = 64;

template <typename T>
class Trace {
public:
    __aicore__ inline Trace(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const TraceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // 每个 32B datablock 能容纳的 T 元素个数
    static constexpr int64_t ELEMS_PER_BLOCK = BLOCK_BYTES / (int64_t)sizeof(T);
    // half/bf16 需要一块 FP32 工作区做 Cast；float 直接就地参与运算
    static constexpr bool NEED_CAST = (sizeof(T) != sizeof(float));

    __aicore__ inline void BuildMask();
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void ReduceBinary(const LocalTensor<float>& dst, const LocalTensor<float>& src, int64_t n);
    __aicore__ inline void WriteOut(const LocalTensor<float>& src);

    // arena 内的 float 下标（单位：float，均按 32B = 8 float 对齐）
    __aicore__ inline int64_t OffScalar() const { return 0; }
    __aicore__ inline int64_t OffAccum() const { return 8; }
    __aicore__ inline int64_t OffFinal() const { return 16; }
    __aicore__ inline int64_t OffOut() const { return 24; }
    __aicore__ inline int64_t OffCross() const { return 32; }
    __aicore__ inline int64_t OffMask() const { return OffCross() + MAX_SLOTS * SLOT_FLOATS; }
    __aicore__ inline int64_t OffWork() const { return OffMask() + slots_; }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    // 所有 VECCALC Buffer 合并成一个 arena，只调一次 InitBuffer
    TBuf<TPosition::VECCALC> arenaBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    GlobalTensor<float> workspaceGM;

    int64_t rows_ = 0;       // 矩阵行数 M
    int64_t cols_ = 0;       // 矩阵列数 N（决定对角步长 N+1）
    int64_t start_ = 0;      // 本核负责的第一个对角元素下标
    int64_t count_ = 0;      // 本核负责的对角元素个数
    int64_t ubFactor_ = 1;   // 单次循环处理的元素个数
    int64_t slots_ = 0;      // 单次循环在 UB 中的稠密 float 个数
    int64_t usedCoreNum_ = 1;
    bool useBlock32_ = false;  // 能否用 32B 整块搬（相邻行间隔 ≥ 32B）
};

template <typename T>
__aicore__ inline void Trace<T>::Init(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                      const TraceTilingData* tilingData)
{
    rows_ = tilingData->rows;
    cols_ = tilingData->cols;

    // 核数以运行时为准：手册 6.2.3.7.2.3 要求参与 SyncAll 的核数与逻辑 numBlocks
    // 严格一致，否则会卡死。这里直接用框架给出的实际核数，并夹到槽位上限内，
    // 保证 workspace 写回不会越界。
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockNum <= 0) {
        blockNum = tilingData->usedCoreNum > 0 ? tilingData->usedCoreNum : 1;
    }
    usedCoreNum_ = (blockNum > MAX_SLOTS) ? MAX_SLOTS : blockNum;

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

    // 相邻两行的对角元素间隔 (N+1) 个元素。该间隔 ≥ 32B 时才能整块搬，
    // 否则 srcStride 会下溢（DataCopyExtParams 的 stride 是无符号字节数）。
    // copyMode = 1 可强制走 4B+padding 路径，仅用于真机 A/B 对比。
    useBlock32_ = (tilingData->copyMode == 0) &&
                  ((cols_ + 1) * static_cast<int64_t>(sizeof(T)) >= BLOCK_BYTES);

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), rows_ * cols_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), 1);
    // workspace 按 MAX_SLOTS 预留：每个核独占一个 32B 槽位，避免伪共享
    workspaceGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), MAX_SLOTS * SLOT_FLOATS);

    // 一次 InitBuffer 覆盖全部 VECCALC 需求：
    //   小缓冲 32 float（scalar/accum/final/out 各 8）+ 跨核槽位 + 掩码 + 工作区
    const int64_t arenaFloats = OffWork() + slots_;
    pipe.InitBuffer(arenaBuf, arenaFloats * (int64_t)sizeof(float));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubFactor_ * BLOCK_BYTES);

    LocalTensor<float> arena = arenaBuf.Get<float>();
    // 累加器清零（即使本核没有对角元素，也要参与后续跨核同步）
    Duplicate(arena[OffAccum()], 0.0f, static_cast<int32_t>(SLOT_FLOATS));
    PipeBarrier<PIPE_V>();

    if (count_ > 0) {
        BuildMask();
    }
}

/*!
 * \brief 构造重复掩码 [1,0,0,...,0]（每 ELEMS_PER_BLOCK 个 float 一个 1）
 *
 * 第 r 个对角元素无论用 32B 整块还是 4B+padding 搬入，都落在第 r 个 32B block 的
 * 起始处；块内其余位置要么是同行后续元素（32B 路径），要么是硬件填充（4B 路径），
 * 都必须屏蔽掉。掩码只依赖「每块首元素」这一布局事实，不依赖填充内容。
 *
 * 构造方式（倍增复制）：先写首个模式单元，再把已初始化的部分整体复制到后面，
 * 长度翻倍，直到覆盖 ubFactor。避免 ubFactor 次标量 SetValue 打满 SCALAR 管线。
 */
template <typename T>
__aicore__ inline void Trace<T>::BuildMask()
{
    LocalTensor<float> maskLocal = arenaBuf.Get<float>()[OffMask()];

    // 掩码在 float 域构造，重复周期为 ELEMS_PER_BLOCK 个 float
    const int64_t patFloats = ELEMS_PER_BLOCK;
    const int64_t patBlocks = patFloats / SLOT_FLOATS;   // 一个模式单元占几个 32B block

    Duplicate(maskLocal, 0.0f, static_cast<int32_t>(patFloats));
    PipeBarrier<PIPE_V>();
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

    // 第 (start_ + offset + r) 行的对角元素位于 GM 下标 (start_+offset+r)*(cols_+1)。
    const int64_t elem0 = (start_ + offset) * (cols_ + 1);
    const int64_t rowStep = cols_ + 1;

    // 32B 整块搬能覆盖的块数。必须保证最后一块的 32B 不越过输入张量末尾：
    //   elem_last * sizeof(T) + 32 <= rows*cols*sizeof(T)
    // 末尾最多有 BLOCK_BYTES/sizeof(T)-1 行会越界，这些行回退到 4B+padding。
    int64_t n32 = 0;
    if (useBlock32_) {
        const int64_t totalElems = rows_ * cols_;
        const int64_t limit = totalElems - (BLOCK_BYTES / (int64_t)sizeof(T));
        const int64_t elemLast = elem0 + (currentNum - 1) * rowStep;
        if (elemLast <= limit) {
            n32 = currentNum;
        } else if (elem0 <= limit) {
            n32 = (limit - elem0) / rowStep + 1;
        }
    }

    if (n32 > 0) {
        // 纯 strided 拷贝：每块 32B，块间隔为下一行对角元素，
        // srcStride 是「当前块结束到下一块开始」的字节数。
        DataCopyExtParams p32;
        p32.blockCount = static_cast<uint16_t>(n32);
        p32.blockLen = static_cast<uint32_t>(BLOCK_BYTES);
        p32.srcStride = static_cast<uint32_t>(rowStep * static_cast<int64_t>(sizeof(T)) - BLOCK_BYTES);
        p32.dstStride = 0;   // UB 侧逐 datablock 连续排列

        DataCopyPadExtParams<T> noPad;
        noPad.isPad = false;
        noPad.leftPadding = 0;
        noPad.rightPadding = 0;
        noPad.paddingValue = 0;

        DataCopyPad(xLocal, inputGMX[elem0], p32, noPad);
    }

    if (n32 < currentNum) {
        // 末尾少量行：单元素 + padding。padding 内容由硬件决定，会被掩码屏蔽。
        DataCopyExtParams pPad;
        pPad.blockCount = static_cast<uint16_t>(currentNum - n32);
        pPad.blockLen = static_cast<uint32_t>(sizeof(T));
        pPad.srcStride = static_cast<uint32_t>(cols_ * static_cast<int64_t>(sizeof(T)));
        pPad.dstStride = 0;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        DataCopyPad(xLocal[n32 * ELEMS_PER_BLOCK], inputGMX[elem0 + n32 * rowStep], pPad, padParams);
    }

    inputQueueX.EnQue(xLocal);
}

/*!
 * \brief 二分累加 + WholeReduceSum（手册 2.5.2.3.2「如何使用归约计算 API」）
 *
 * 手册给的实测对比（30000 个 float）：二分累加 172 cycles vs WholeReduceSum
 * 单指令 242 cycles。做法是先用 Add 反复折半，剩余不超过一个 Repeat 时收尾。
 *
 * 三个实现约束（都实测踩过）：
 *   1. 必须按 32B block（8 个 float）为粒度折叠，且折半只能是「等分」：
 *      把 [0,half) 与 [half,2*half) 相加。若只折 floor(cur/2) 个元素，
 *      中间那段元素既不在目的区也不在源区，会被静默丢掉 —— 表现为结果偏小。
 *   2. block 数为奇数时，最后一块单独并入第 0 块，然后再继续折半。
 *   3. 单条向量指令的 mask 计数有实测上限（4096 越界、4095 以内安全），
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
__aicore__ inline void Trace<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<float> arena = arenaBuf.Get<float>();
    LocalTensor<float> accumLocal = arena[OffAccum()];
    LocalTensor<float> scalarLocal = arena[OffScalar()];
    LocalTensor<float> workLocal = arena[OffWork()];
    LocalTensor<float> maskLocal = arena[OffMask()];

    const int64_t slots = currentNum * ELEMS_PER_BLOCK;

    // 统一转 FP32 后按掩码挑出每个 block 的首元素（即对角元素）
    if constexpr (NEED_CAST) {
        Cast(workLocal, xLocal, RoundMode::CAST_NONE, static_cast<int32_t>(slots));
        Mul(workLocal, workLocal, maskLocal, static_cast<int32_t>(slots));
    } else {
        Mul(workLocal, xLocal, maskLocal, static_cast<int32_t>(slots));
    }

    ReduceBinary(scalarLocal, workLocal, slots);
    // 段间累加（FP32）
    Add(accumLocal, accumLocal, scalarLocal, 1);

    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Trace<T>::WriteOut(const LocalTensor<float>& src)
{
    LocalTensor<float> arena = arenaBuf.Get<float>();
    LocalTensor<float> finalLocal = arena[OffFinal()];
    // out 槽位按字节偏移换算成 T 单位
    LocalTensor<T> yLocal = arenaBuf.Get<T>()[OffOut() * (int64_t)sizeof(float) / (int64_t)sizeof(T)];

    if constexpr (NEED_CAST) {
        finalLocal.SetValue(0, src.GetValue(0));
        PipeBarrier<PIPE_ALL>();
        Cast(yLocal, finalLocal, RoundMode::CAST_RINT, 1);
        PipeBarrier<PIPE_ALL>();
        outputGMY.SetValue(0, yLocal.GetValue(0));
    } else {
        outputGMY.SetValue(0, src.GetValue(0));
    }
}

template <typename T>
__aicore__ inline void Trace<T>::Process()
{
    LocalTensor<float> arena = arenaBuf.Get<float>();
    LocalTensor<float> accumLocal = arena[OffAccum()];

    if (count_ > 0) {
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
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    LocalTensor<float> scalarLocal = arena[OffScalar()];

    if (usedCoreNum_ <= 1) {
        if (blockIdx == 0) {
            ReduceBinary(scalarLocal, accumLocal, SLOT_FLOATS);
            WriteOut(scalarLocal);
        }
        return;
    }

    // ---- 多核汇总 ----
    // 用 MTE3 写回，保证别的核能通过 MTE2 读到最新值；槽位按 32B 对齐互不重叠。
    // 越界保护：理论上 blockIdx < usedCoreNum_ 恒成立，这里再夹一次以防调度异常。
    LocalTensor<float> crossLocal = arena[OffCross()];
    const int64_t slot = (blockIdx < MAX_SLOTS) ? blockIdx : (MAX_SLOTS - 1);
    Duplicate(crossLocal, 0.0f, static_cast<int32_t>(SLOT_FLOATS));
    crossLocal.SetValue(0, accumLocal.GetValue(0));
    PipeBarrier<PIPE_ALL>();
    DataCopy(workspaceGM[slot * SLOT_FLOATS], crossLocal, SLOT_FLOATS);

    // SyncAll 内部自带 PipeBarrier<PIPE_ALL>，会等待 MTE3 搬完，无需额外屏障
    SyncAll<true>();

    if (blockIdx == 0) {
        // 0 号核读回全部槽位：每个槽位只有第 0 个 float 有值、其余为 0，
        // 因此可以直接整段归约（累加顺序由归约本身固定，结果确定）。
        DataCopy(crossLocal, workspaceGM[0], usedCoreNum_ * SLOT_FLOATS);
        PipeBarrier<PIPE_ALL>();
        ReduceBinary(scalarLocal, crossLocal, usedCoreNum_ * SLOT_FLOATS);
        WriteOut(scalarLocal);
    }
}

} // namespace NsTrace
#endif // TRACE_H
