/*!
 * \file test_trace_verify.cpp
 * \brief 临时验证：多 shape / 多核 / fp16 正确性（本地验证用，非交付件）
 */

#include "trace_tiling.h"
#include "../../../op_kernel/trace.cpp"

#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static float Golden(const std::vector<float>& a, int64_t M, int64_t N)
{
    int64_t d = M < N ? M : N;
    float s = 0.0f;
    for (int64_t i = 0; i < d; ++i) {
        s += a[i * N + i];
    }
    return s;
}

static void RunCase(int64_t M, int64_t N, uint32_t numBlocks, int64_t ubFactor, float tol, bool useHalf)
{
    const int64_t total = M * N;
    const int64_t diagLen = M < N ? M : N;
    const int64_t blockFactor = (diagLen + (int64_t)numBlocks - 1) / (int64_t)numBlocks;

    std::vector<float> host(total);
    for (int64_t i = 0; i < total; ++i) {
        host[i] = static_cast<float>((i * 37 % 97) - 48) * 0.5f;
    }
    const float expect = Golden(host, M, N);

    const size_t elemSize = useHalf ? 2 : 4;
    const size_t selfBytes = total * elemSize;
    const size_t wsBytes = numBlocks * 32 + 64;  // 每核一个 32B 槽位

    uint8_t* self = (uint8_t*)AscendC::GmAlloc(selfBytes);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(elemSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(wsBytes);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(TraceTilingData));

    // 输入转 dtype
    for (int64_t i = 0; i < total; ++i) {
        if (useHalf) {
            uint32_t bits; memcpy(&bits, &host[i], 4);
            uint32_t sign = (bits >> 16) & 0x8000;
            int32_t exp = ((int32_t)((bits >> 23) & 0xff)) - 127 + 15;
            if (exp <= 0) { uint16_t h = (uint16_t)sign; memcpy(self + i * 2, &h, 2); continue; }
            if (exp >= 31) { uint16_t h = (uint16_t)(sign | 0x7c00); memcpy(self + i * 2, &h, 2); continue; }
            uint32_t mant = (bits >> 13) & 0x3ff;
            uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
            memcpy(self + i * 2, &h, 2);
        } else {
            memcpy(self + i * 4, &host[i], 4);
        }
    }
    memset(out, 0, elemSize);
    memset(workspace, 0, wsBytes);

    TraceTilingData* td = reinterpret_cast<TraceTilingData*>(tiling);
    td->rows = M;
    td->cols = N;
    td->diagLen = diagLen;
    td->blockFactor = blockFactor;
    td->ubFactor = ubFactor;
    td->usedCoreNum = (int64_t)numBlocks;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if (useHalf) {
        ICPU_SET_TILING_KEY(0);
        ICPU_RUN_KF((trace<0>), numBlocks, self, out, workspace, tiling);
    } else {
        ICPU_SET_TILING_KEY(1);
        ICPU_RUN_KF((trace<1>), numBlocks, self, out, workspace, tiling);
    }

    float got;
    if (useHalf) {
        uint16_t h; memcpy(&h, out, 2);
        uint32_t sign = (uint32_t)(h & 0x8000) << 16;
        uint32_t exp = (h >> 10) & 0x1f;
        uint32_t mant = h & 0x3ff;
        if (exp == 0) { got = 0.0f; }
        else {
            uint32_t bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
            memcpy(&got, &bits, 4);
        }
    } else {
        memcpy(&got, out, 4);
    }

    std::cout << "[CASE] M=" << M << " N=" << N << " blocks=" << numBlocks
              << " ubFactor=" << ubFactor << " dtype=" << (useHalf ? "fp16" : "fp32")
              << " got=" << got << " expect=" << expect << std::endl;
    EXPECT_NEAR(got, expect, tol * std::max(1.0f, std::fabs(expect)));

    AscendC::GmFree(self);
    AscendC::GmFree(out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(TraceVerify, fp32_shapes_single_block)
{
    RunCase(1, 1, 1, 1, 1e-5f, false);
    RunCase(4, 5, 1, 8, 1e-5f, false);
    RunCase(5, 4, 1, 8, 1e-5f, false);
    RunCase(33, 17, 1, 8, 1e-5f, false);
    RunCase(33, 17, 1, 64, 1e-5f, false);
    RunCase(128, 128, 1, 16, 1e-5f, false);
    RunCase(200, 129, 1, 7, 1e-5f, false);
}

TEST(TraceVerify, fp32_multi_block)
{
    RunCase(128, 128, 4, 16, 1e-5f, false);
    RunCase(513, 257, 8, 40, 1e-5f, false);
}

TEST(TraceVerify, fp16_shapes)
{
    RunCase(4, 5, 1, 8, 1e-2f, true);
    RunCase(64, 64, 1, 16, 1e-2f, true);
    RunCase(64, 64, 4, 16, 1e-2f, true);
}
