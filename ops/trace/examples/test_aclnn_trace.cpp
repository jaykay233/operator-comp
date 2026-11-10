#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "aclnn_trace.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}


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

// half / bf16 -> float，用于回读校验
static float HalfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= 0x3FFu;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &f, sizeof(float));
    return r;
}

static float BFloat16ToFloat(uint16_t b) {
    uint32_t f = (uint32_t)b << 16;
    float r;
    memcpy(&r, &f, sizeof(float));
    return r;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto elemCount = GetShapeSize(shape);
    int64_t elemSize = sizeof(T);
    switch (dataType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_INT16:
        case aclDataType::ACL_UINT16:
            elemSize = 2;
            break;
        case aclDataType::ACL_INT8:
        case aclDataType::ACL_UINT8:
        case aclDataType::ACL_BOOL:
            elemSize = 1;
            break;
        case aclDataType::ACL_INT64:
        case aclDataType::ACL_UINT64:
        case aclDataType::ACL_DOUBLE:
            elemSize = 8;
            break;
        default:
            break;
    }
    auto size = elemCount * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<uint8_t> convBuf(size);
    if (dataType == aclDataType::ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t h = FloatToHalf(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &h, 2);
        }
    } else if (dataType == aclDataType::ACL_BF16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t b = FloatToBFloat16(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &b, 2);
        }
    } else if (dataType == aclDataType::ACL_DOUBLE) {
        for (int64_t i = 0; i < elemCount; i++) {
            double d = static_cast<double>(hostData[i]);
            memcpy(convBuf.data() + i * 8, &d, 8);
        }
    } else {
        auto copySize = std::min((int64_t)(elemCount * sizeof(T)), size);
        memcpy(convBuf.data(), hostData.data(), copySize);
    }
    ret = aclrtMemcpy(*deviceAddr, size, convBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main(int argc, char** argv)
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 构造输入 tensor（默认用非对齐的矩形 shape，覆盖 stride 跨步与非对齐尾块）
    // 形状可通过命令行参数或环境变量 TRACE_M / TRACE_N 指定
    int64_t dimM = (argc > 2) ? std::atoll(argv[1]) : 256;
    int64_t dimN = (argc > 2) ? std::atoll(argv[2]) : 385;
    if (const char* envM = std::getenv("TRACE_M")) {
        dimM = std::atoll(envM);
    }
    if (const char* envN = std::getenv("TRACE_N")) {
        dimN = std::atoll(envN);
    }
    aclTensor* self = nullptr;
    void* selfDeviceAddr = nullptr;
    std::vector<int64_t> selfShape = {dimM, dimN};
    const int64_t elemCount = selfShape[0] * selfShape[1];
    std::vector<float> selfHostData(elemCount);
    for (int64_t i = 0; i < elemCount; ++i) {
        selfHostData[i] = static_cast<float>((i * 37 % 101) - 50) * 0.25f;
    }
    // 输入 dtype 可通过环境变量 TRACE_DTYPE 指定（float / half / bf16），默认 float
    aclDataType selfDtype = aclDataType::ACL_FLOAT;
    if (const char* envDtype = std::getenv("TRACE_DTYPE")) {
        const std::string dt(envDtype);
        if (dt == "half" || dt == "fp16") {
            selfDtype = aclDataType::ACL_FLOAT16;
        } else if (dt == "bf16") {
            selfDtype = aclDataType::ACL_BF16;
        }
    }
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, selfDtype, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出 tensor
    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {1};
    std::vector<float> outHostData(1, 0);
    // 输出 dtype 与输入一致（算子定义只支持 half/float 两种组合）
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, selfDtype, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    

    // 调用 aclnnTraceGetWorkspaceSize 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnTraceGetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnTraceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnTrace 第二段接口
    ret = aclnnTrace(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnTrace failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 回读结果并与 CPU 参考实现对比（按输出 dtype 解释，输出与输入同 dtype）
    float outHost = 0.0f;
    if (selfDtype == aclDataType::ACL_FLOAT16) {
        uint16_t h = 0;
        ret = aclrtMemcpy(&h, sizeof(h), outDeviceAddr, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out failed. ERROR: %d\n", ret); return ret);
        outHost = HalfToFloat(h);
    } else if (selfDtype == aclDataType::ACL_BF16) {
        uint16_t b = 0;
        ret = aclrtMemcpy(&b, sizeof(b), outDeviceAddr, sizeof(b), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out failed. ERROR: %d\n", ret); return ret);
        outHost = BFloat16ToFloat(b);
    } else {
        ret = aclrtMemcpy(&outHost, sizeof(float), outDeviceAddr, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out failed. ERROR: %d\n", ret); return ret);
    }

    const int64_t diagLen = std::min(selfShape[0], selfShape[1]);
    double golden = 0.0;
    for (int64_t i = 0; i < diagLen; ++i) {
        golden += static_cast<double>(selfHostData[i * selfShape[1] + i]);
    }
    double absErr = std::fabs(static_cast<double>(outHost) - golden);
    double relErr = absErr / std::max(1.0, std::fabs(golden));
    LOG_PRINT("[RESULT] trace(%lld x %lld) = %.6f, golden = %.6f, absErr = %.3e, relErr = %.3e\n",
              (long long)selfShape[0], (long long)selfShape[1], (double)outHost, golden, absErr, relErr);
    LOG_PRINT("[CHECK] %s\n", relErr < 1e-5 ? "PASS" : "FAIL");

    // 释放资源
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
