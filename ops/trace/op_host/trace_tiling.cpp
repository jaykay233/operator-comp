/*!
 * \file trace_tiling.cpp
 * \brief Trace 算子 Tiling 实现
 *
 * 切分策略：
 *   1. 沿主对角线切分，每个核负责连续的一段对角元素。
 *   2. 核间负载均衡：blockFactor = ceil(diagLen / blockDim)，尾核处理剩余元素。
 *   3. UB 容量约束：每个对角元素在 UB 中占一个 32B datablock（输入 double buffer），
 *      half/bf16 另需一块等宽的 FP32 工作区；按 UB_BYTES_PER_DIAG_ELEM 预算反推
 *      单次循环元素数。
 *   4. workspace 用于多核部分和汇总（blockDim 个 32B 槽位）。
 *
 * 核数策略（实测数据，见 tests/bench_910b.sh）：
 *   跨核汇总的 SyncAll 屏障在本算子上有约 800 cycles 的固定成本。而单核搬运+归约
 *   的耗时随元素数增长很慢（128 元素 4538 cycles、256 元素 5423 cycles），
 *   所以小 shape 直接单核更快。实测拐点：
 *     diagLen=128 : 单核 4538  <  8 核 4891
 *     diagLen=192 : 单核 4936  ≈ 16 核 4920   （打平）
 *     diagLen=224 : 单核 5267  > 16 核 4935
 *     diagLen=256 : 单核 5423  > 16 核 5040
 *     diagLen=4096: 单核 27370 > 24 核 6970
 *   因此 diagLen < MULTICORE_MIN_DIAG 时单核，否则按「每核至少 MIN_DIAG_ELEMS_PER_CORE
 *   个元素」开核并把核数吃满。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/trace_tiling_data.h"
#include "../op_kernel/trace_tiling_key.h"

#include <cstdlib>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t BLOCK_BYTES = 32;

// 每个对角元素占用的 UB 预算：
//   输入 double buffer 64B + （half/bf16）FP32 工作区 64B + 余量
constexpr int64_t UB_BYTES_PER_DIAG_ELEM = 192;
// 每核至少处理的元素数，低于该粒度不再加核
constexpr int64_t MIN_DIAG_ELEMS_PER_CORE = 16;
// DataCopyPad 的 blockCount 上限
constexpr int64_t MAX_BLOCK_COUNT = 4095;
// 跨核部分和槽位宽度（32B，与 kernel 侧 SLOT_FLOATS 一致）
constexpr int64_t SLOT_BYTES = 32;
// 跨核槽位最大个数（与 kernel 侧 MAX_SLOTS 一致）：workspace 按此预留
constexpr int64_t MAX_SLOTS = 64;
// 启用多核的最小对角线长度（低于此值单核更快，实测见文件头）
constexpr int64_t MULTICORE_MIN_DIAG = 192;

// ---------------------------------------------------------------------------
// 性能探针（仅 Host 侧，不增加 Device 代码）：
//   TRACE_CORES  ：强制指定核数（1~coreNum），用于核数扫描
//   TRACE_MC_MIN ：覆盖启用多核的 diagLen 阈值
// ---------------------------------------------------------------------------
static int64_t ReadEnvInt(const char* name, int64_t defaultValue)
{
    const char* v = std::getenv(name);
    if (v == nullptr) {
        return defaultValue;
    }
    long long parsed = std::strtoll(v, nullptr, 10);
    return (parsed > 0) ? static_cast<int64_t>(parsed) : defaultValue;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TraceTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 输入 shape：self 为二维矩阵 (M, N)
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inShape = inputShapePtr->GetStorageShape();

    int64_t rows = 1;
    int64_t cols = 1;
    const size_t dimNum = inShape.GetDimNum();
    if (dimNum >= 2) {
        rows = inShape.GetDim(dimNum - 2);
        cols = inShape.GetDim(dimNum - 1);
    } else if (dimNum == 1) {
        cols = inShape.GetDim(0);
    }
    if (rows < 0) {
        rows = 0;
    }
    if (cols < 0) {
        cols = 0;
    }
    const int64_t diagLen = (rows < cols) ? rows : cols;

    // 数据类型（决定 tilingKey 与 UB 对齐粒度）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dType = inputDesc->GetDataType();
    int64_t dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(dType));
    if (dtypeSize <= 0) {
        dtypeSize = TYPE_SIZE;
    }
    int64_t alignElems = static_cast<int64_t>(GetUbBlockSize(context)) / dtypeSize;
    if (alignElems < 1) {
        alignElems = 1;
    }

    TraceTilingData* tiling = context->GetTilingData<TraceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- 多核切分 ----
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
    if (blockDim < 1) {
        blockDim = 1;
    }
    // 性能探针：TRACE_CORES 强制指定核数（1~coreNum）
    const int64_t forceCores = ReadEnvInt("TRACE_CORES", 0);
    if (forceCores > 0) {
        blockDim = (forceCores < coreNum) ? forceCores : coreNum;
        if (diagLen > 0 && blockDim > diagLen) {
            blockDim = diagLen;
        }
        if (blockDim < 1) {
            blockDim = 1;
        }
    }

    const int64_t blockFactor = CeilDiv(diagLen, blockDim);

    // ---- 单次 UB 循环元素数 ----
    int64_t ubFactor = static_cast<int64_t>(ubSize) / UB_BYTES_PER_DIAG_ELEM;
    ubFactor = FloorAlign(ubFactor, alignElems);
    if (ubFactor < alignElems) {
        ubFactor = alignElems;
    }
    if (ubFactor > MAX_BLOCK_COUNT) {
        ubFactor = MAX_BLOCK_COUNT;
    }
    if (blockFactor < ubFactor) {
        ubFactor = blockFactor;
    }
    if (ubFactor < 1) {
        ubFactor = 1;
    }

    tiling->rows = rows;
    tiling->cols = cols;
    tiling->diagLen = diagLen;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->usedCoreNum = blockDim;
    // 搬运方式探针：TRACE_COPY=1 强制走 4B+padding，用于真机 A/B 对比
    tiling->copyMode = static_cast<uint8_t>(ReadEnvInt("TRACE_COPY", 0) == 1 ? 1 : 0);
    context->GetRawTilingData()->SetDataSize(sizeof(TraceTilingData));

    // ---- workspace：每核一个 32B 槽位存部分和（DataCopy 要求 32B 对齐） ----
    // 按 MAX_SLOTS 预留而不是按 blockDim：手册 6.2.3.7.2.3 的 SyncAll 约束下，
    // 实际上核数必须不大于逻辑 numBlocks，预留到上限可避免任何调度差异导致越界写。
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(MAX_SLOTS) * SLOT_BYTES;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 【真机必需】开启 batchmode，让算子在执行期间独占所需核资源。
    // 手册 6.2.3.7.2.3 SyncAll 约束：不开 batchmode 时，多流并发 / ≥2 算子并发 /
    // 并发核数总和超过物理核数 的任一情况都会让核间同步死锁，
    // 真机表现为 kernel 卡住 3s 超时（aclrtSynchronizeStreamWithTimeout ret=507015）。
    context->SetScheduleMode(1);

    // 根据输入 dtype 选择 tilingKey（half/bf16 -> 0，其余 -> 1）
    uint64_t tilingKey;
    if (dType == ge::DT_FLOAT16 || dType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(TRACE_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TRACE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTrace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TraceCompileInfo {};

IMPL_OP_OPTILING(Trace).Tiling(TraceTilingFunc).TilingParse<TraceCompileInfo>(TilingParseForTrace);

} // namespace optiling
