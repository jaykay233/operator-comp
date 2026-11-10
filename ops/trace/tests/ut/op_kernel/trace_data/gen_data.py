#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16


def impl(self):
    """Trace算子golden实现

    参数:
        self: (M, N) numpy 数组，输入二维矩阵

    返回:
        out: (1,) 主对角线元素之和（非方阵取 min(M, N) 个对角元），
             dtype 与 self 一致（内部 FP32 累加）。

    说明：这里刻意只用 numpy，不引入 torch。golden 只需要「取对角 + 求和」，
    torch.trace 与 numpy 结果一致，但 torch 会把 UT 数据生成链路绑到一个
    很重的可选依赖上（镜像里一旦 pyc/版本不匹配，整个 UT 直接跑不起来）。
    """
    # 统一转 float32 累加（fp16 精度不足，内部用 FP32），与 kernel 侧一致
    orig_dtype = self.dtype
    x32 = np.asarray(self, dtype=np.float32)

    # 主对角线取数 + 求和（非方阵取 min(M, N) 个对角元），输出 shape (1,)
    out = np.trace(x32).reshape(1)

    # 转回原始 dtype（与 JSON output_desc 一致）
    return out.astype(orig_dtype)


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "float32"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]
    
    # 生成输入数据
    input_self = np.ones((1, 1)).astype(d_type_dict["float32"])

    
    # 计算 golden 数据
    golden = impl(input_self)
    
    # 保存数据到文件
    input_self.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_trace_self.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_trace.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_trace.bin")
    
    print(f"生成完成: dtype={d_type}")
