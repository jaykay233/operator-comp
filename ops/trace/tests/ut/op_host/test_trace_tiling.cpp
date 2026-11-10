#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "trace_tiling_data.h"

namespace TraceUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Trace";

struct TraceTestParam {
    std::string caseName;
    std::initializer_list<int64_t> selfShape;
    ge::DataType selfDtype;
    ge::Format selfFormat;
    std::initializer_list<int64_t> outShape;
    ge::DataType outDtype;
    ge::Format outFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// 期望值说明：
//   expectTilingKey:  fp32 输入 -> schMode 1
//   expectTilingData: rows cols diagLen blockFactor ubFactor usedCoreNum
//   expectWorkspaces: workspace = blockDim * 32B 槽位
static TraceTestParam testCases[] = {
    // 1x1：最小边界，单核单元素
    {"trace_0", {1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "1 1 1 1 1 1 ", {32}, 64, 262144, 4096},
    // 128x129：diagLen=128 < MULTICORE_MIN_DIAG(192)，单核整段处理
    {"trace_1", {128, 129}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "128 129 128 128 128 1 ", {32}, 64, 262144, 4096},
    // 256x385：diagLen=256 ≥ 192，按每核 16 个元素切 16 核
    {"trace_2", {256, 385}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "256 385 256 16 16 16 ", {512}, 64, 262144, 4096},
    // 512x513：diagLen=512，每核 16 个元素 -> 32 核，未超过 maxAIVNum(64)
    {"trace_3", {512, 513}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "512 513 512 16 16 32 ", {1024}, 64, 262144, 4096},
};

class TraceTilingTest : public testing::TestWithParam<TraceTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "TraceTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "TraceTilingTest TearDown." << std::endl;
    }
};

struct TraceCompileInfo {} compileInfo;

static void TestOneParamCase(const TraceTestParam &param)
{
    gert::StorageShape selfShape = {param.selfShape, param.selfShape};
    gert::StorageShape outShape = {param.outShape, param.outShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{selfShape, param.selfDtype, param.selfFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{outShape, param.outDtype, param.outFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(TraceTilingTest, tiling_test)
{
    const TraceTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    TraceTilingTests,
    TraceTilingTest,
    testing::ValuesIn(testCases));

}
