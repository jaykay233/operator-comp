#!/bin/bash
# 在容器内跑通「910B CAModel 周期仿真」的验证脚本
set -e

export ASCEND_OPP_PATH=/usr/local/Ascend/cann/opp
cd /workspace/ops/trace

echo "### 1. 安装自定义算子包"
bash build/custom_opp_ubuntu_x86_64.run --quiet 2>&1 | tail -2
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/opp/vendors/trace_custom/op_api/lib:$LD_LIBRARY_PATH

echo "### 2. 编译 aclnn 调用示例"
cd /workspace/ops/trace/examples
rm -rf build
mkdir -p build
cd build
cmake .. 2>&1 | tail -5
make -j16 2>&1 | tail -10
echo "--- 产物 ---"
find . -name "test_aclnn_trace" -o -name "*.so" | head

echo "### 3. 用 Ascend910B1 仿真器运行"
BIN=$(find . -name test_aclnn_trace -type f | head -1)
echo "BIN=$BIN"
rm -rf /tmp/out
msprof op simulator --soc-version=Ascend910B1 --application="$PWD/$BIN" --output=/tmp/out 2>&1 | tail -30
