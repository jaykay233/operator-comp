/*!
 * \file trace.cpp
 * \brief Trace 算子 kernel 入口
 */

#include "trace.h"

enum class TraceTilingKey : uint32_t
{
    TILING_KEY_TRACE_MODE_0 = 0,  // half
    TILING_KEY_TRACE_MODE_1 = 1,  // float
};

template <uint32_t schMode>
__global__ __aicore__ void trace(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(ASCENDC_CPU_DEBUG) && (ASCENDC_CPU_DEBUG == 1)
    // CPU 孪生调试（tikicpulib）：tiling 指针直接指向宿主结构体，
    // 此处不使用 GET_TILING_DATA_WITH_STRUCT（该宏在 UT 编译下会退化为默认构造的空结构）。
    const TraceTilingData* tilingData = reinterpret_cast<const TraceTilingData*>(tiling);
#else
    REGISTER_TILING_DEFAULT(TraceTilingData);
    GET_TILING_DATA_WITH_STRUCT(TraceTilingData, tilingDataStruct, tiling);
    const TraceTilingData* tilingData = &tilingDataStruct;
#endif

    if constexpr (schMode == static_cast<uint32_t>(TraceTilingKey::TILING_KEY_TRACE_MODE_0)) {
        NsTrace::Trace<half> op;
        op.Init(self, out, workspace, tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(TraceTilingKey::TILING_KEY_TRACE_MODE_1)) {
        NsTrace::Trace<float> op;
        op.Init(self, out, workspace, tilingData);
        op.Process();
    }
}
