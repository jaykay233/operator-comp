/*!
 * \file trace_tiling_data.h
 * \brief Trace 算子 TilingData 结构
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
