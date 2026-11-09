/*!
 * \file trace_tiling.cpp
 * \brief Trace 算子 Tiling 实现
 *
 * 切分策略：
 *   1. 沿主对角线切分，每个核负责连续的一段对角元素，实现多核并行归约。
 *   2. 核间负载均衡：blockFactor = ceil(diagLen / blockDim)，尾核处理剩余元素。
 *   3. UB 容量约束：每个对角元素在 UB 中占一个 32B datablock，转 FP32 后还需掩码、
 *      工作区与归约临时区，按 UB_BYTES_PER_DIAG_ELEM 的预算反推单次循环元素数。
 *   4. workspace 用于多核部分和汇总（blockDim 个 float），0 号核按索引升序累加。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/trace_tiling_data.h"
#include "../op_kernel/trace_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t BLOCK_BYTES = 32;

// 每个对角元素占用的 UB 预算（与 kernel 侧 UB_BYTES_PER_DIAG_ELEM 保持一致）：
//   输入 double buffer 64B + 掩码 32~64B + FP32 工作区 32~64B + 归约临时区 32~64B
constexpr int64_t UB_BYTES_PER_DIAG_ELEM = 320;
// 每核至少处理的元素数，低于该粒度不再切核（避免启动/同步开销占比过高）
constexpr int64_t MIN_DIAG_ELEMS_PER_CORE = 64;
// DataCopyPad 的 blockCount 上限
constexpr int64_t MAX_BLOCK_COUNT = 4095;
// 跨核部分和槽位宽度（32B，与 kernel 侧 SLOT_FLOATS 一致）
constexpr int64_t SLOT_BYTES = 32;

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
    int64_t blockDim = 1;
    if (diagLen > 0) {
        blockDim = CeilDiv(diagLen, MIN_DIAG_ELEMS_PER_CORE);
        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
        if (blockDim > diagLen) {
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
    context->GetRawTilingData()->SetDataSize(sizeof(TraceTilingData));

    // ---- workspace：每核一个 32B 槽位存部分和（DataCopy 要求 32B 对齐） ----
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(blockDim) * SLOT_BYTES;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

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
