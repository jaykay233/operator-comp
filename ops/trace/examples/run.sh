#!/bin/bash
# trace 算子调用示例执行脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "========================================"
echo "trace 算子调用示例"
echo "========================================"

if [ -z "$ASCEND_HOME_PATH" ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann
fi

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake ..
make -j$(nproc)

echo "执行调用示例..."
cd bin
./test_aclnn_trace

echo "========================================"
echo "执行完成"
echo "========================================"
