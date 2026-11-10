/*!
 * \file trace_tiling_data.h
 * \brief Trace 算子 TilingData 结构
 *
 * 注意（手册 3.8.3.2「限制 TilingData 结构大小」）：TilingData 会从 GM 拷贝到
 * AI Core 栈空间，期间是 us 级开销，小 shape 下占比明显。因此只保留必要字段。
 * NumBlocks 不在此传递 —— 它已通过 SetBlockDim 下发，Kernel 侧可直接 GetBlockNum。
 */

#ifndef _TRACE_TILING_DATA_H_
#define _TRACE_TILING_DATA_H_

struct TraceTilingData {
    int64_t rows = 0;          // 输入矩阵行数 M
    int64_t cols = 0;          // 输入矩阵列数 N
    int64_t diagLen = 0;       // 主对角线元素个数 min(M, N)
    int64_t blockFactor = 0;   // 每个核负责的对角元素个数（核间切分粒度）
    int64_t ubFactor = 0;      // 单次 UB 循环处理的对角元素个数
    int64_t usedCoreNum = 0;   // 实际使用的核数（用于核间归约）
};
#endif
