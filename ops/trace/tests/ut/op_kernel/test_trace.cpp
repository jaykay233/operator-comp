/*!
 * \file test_trace.cpp
 * \brief Trace 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "trace_tiling.h"
#include "../../../op_kernel/trace.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class TraceKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "TraceKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "TraceKernelTest TearDown" << endl;
    }
};

TEST_F(TraceKernelTest, test_kernel_run)
{
    // 1x1 fp32 用例：与 tests/ut/op_kernel/trace_data/gen_data.py 生成的 golden 保持一致
    constexpr int64_t M = 1, N = 1;
    constexpr size_t tilingDataSize = sizeof(TraceTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t selfByteSize = M * N * 4;
    constexpr size_t outByteSize = 1 * 4;
    std::vector<float> selfHost(1, 1);
    std::vector<float> outHost(1, 0);
    
    
    uint8_t* self = (uint8_t*)AscendC::GmAlloc(selfByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(numBlocks * 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(self, selfHost.data(), selfByteSize);
    memset(workspace, 0, numBlocks * 32);
    
    // tilingData 字段见 op_kernel/trace_tiling_data.h
    TraceTilingData* tilingData = reinterpret_cast<TraceTilingData*>(tiling);
    tilingData->rows = M;
    tilingData->cols = N;
    tilingData->diagLen = M;
    tilingData->blockFactor = M;
    tilingData->ubFactor = M;
    tilingData->usedCoreNum = numBlocks;
    
    // tilingKey 与 op_host/trace_tiling.cpp 一致（fp32 -> schMode 1）
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((trace<1>), numBlocks, self, out, workspace, tiling);
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(outHost.data(), out, outByteSize);
    { std::ofstream _ofs("float32_output_trace_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(outHost.data()), outByteSize); }
    
    AscendC::GmFree(self);
    AscendC::GmFree(out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
