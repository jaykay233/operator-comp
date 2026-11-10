#!/bin/bash
# ===========================================================================
# verify_all.sh —— ascendc-cpu-dev:cann92 镜像内的完整验收脚本
#
# 在容器内执行（推荐挂载仓库到 /workspace）：
#   docker run --rm -v "$PWD:/workspace" ascendc-cpu-dev:cann92 \
#       bash /workspace/ops/trace/tests/verify_all.sh
#
# 覆盖四层验证：
#   0. CANN / CAModel 环境自检
#   1. 算子工程编译（默认 ascend910b）
#   2. UT（tikicpulib CPU 逐指令仿真）—— 精度验证主通道
#   3. aclnn 示例编译 + 910B cycle 级仿真（探测，已知 msopprof 507033 缺陷不阻塞）
#
# 常用：
#   bash verify_all.sh              # 全量验收
#   SKIP_SIM=1 bash verify_all.sh   # 跳过最后的仿真探测，省时间
#
# 日常只跑 UT（最快，秒级）：
#   cd /workspace/ops/trace && bash build.sh -u
# ===========================================================================
set -o pipefail

TRACE_DIR=/workspace/ops/trace
SOC=${SOC:-ascend910b}
SIM_SOC=${SIM_SOC:-Ascend910B1}

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
FAILED=()
PASSED=()

step()  { echo; echo "${BOLD}========== $* ==========${NC}"; }
ok()    { echo "${GREEN}[PASS]${NC} $*"; PASSED+=("$*"); }
fail()  { echo "${RED}[FAIL]${NC} $*"; FAILED+=("$*"); }
warn()  { echo "${YELLOW}[WARN]${NC} $*"; }

cd "$TRACE_DIR" || { echo "找不到 $TRACE_DIR"; exit 1; }

# ---------------------------------------------------------------------------
step "0. 环境自检"
# ---------------------------------------------------------------------------
echo "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH:-<空>}"
echo "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH:-<空>}"

if [ -n "$ASCEND_HOME_PATH" ] && [ -d "$ASCEND_HOME_PATH" ]; then
    ok "ASCEND_HOME_PATH 存在（$(readlink -f "$ASCEND_HOME_PATH")）"
else
    fail "ASCEND_HOME_PATH 未设置或不存在"
fi

# host 架构目录
ARCH_DIR=""
for d in "$ASCEND_HOME_PATH"/*-linux; do
    [ -d "$d" ] && ARCH_DIR="$d" && break
done
if [ -n "$ARCH_DIR" ]; then
    ok "host 架构目录：$ARCH_DIR"
else
    fail "未找到 host 架构目录（x86_64-linux / aarch64-linux）"
fi

# CAModel 模型
if [ -f "$ASCEND_HOME_PATH/x86_64-linux/simulator/$SIM_SOC/lib/libruntime_camodel.so" ] \
   || [ -f "$ARCH_DIR/simulator/$SIM_SOC/lib/libruntime_camodel.so" ]; then
    ok "$SIM_SOC 的 CAModel 运行时存在"
else
    fail "$SIM_SOC 的 CAModel 运行时缺失"
fi

# aclnn 依赖
for lib in libopapi.so libnnopbase.so libascendcl.so; do
    if [ -f "$ASCEND_HOME_PATH/lib64/$lib" ]; then
        ok "$lib 存在"
    else
        fail "$lib 缺失（需要 ops 算子包）"
    fi
done

# ---------------------------------------------------------------------------
step "1. 编译算子工程（--soc=$SOC）"
# ---------------------------------------------------------------------------
rm -rf build build_out
if bash build.sh -j"$(nproc)" --soc="$SOC" > /tmp/build_all.log 2>&1; then
    ok "算子包编译成功"
    tail -4 /tmp/build_all.log | sed 's/^/    /'
else
    fail "算子包编译失败（详见 /tmp/build_all.log）"
    grep -E "error:|Error [0-9]" /tmp/build_all.log | head -5 | sed 's/^/    /'
fi
PKG=$(ls build/custom_opp_*.run 2>/dev/null | head -1)

# ---------------------------------------------------------------------------
step "2. UT（tikicpulib CPU 逐指令仿真）"
# ---------------------------------------------------------------------------
if bash build.sh -u > /tmp/ut.log 2>&1; then
    ok "UT 全部通过"
    grep -E "\[  PASSED  \]|所有测试通过|PASSED! MERE" /tmp/ut.log | sed 's/^/    /'
else
    fail "UT 失败（详见 /tmp/ut.log）"
    grep -E "\[  FAILED  \]|error:|失败的测试" /tmp/ut.log | head -8 | sed 's/^/    /'
fi

# ---------------------------------------------------------------------------
step "3. aclnn 示例 + $SIM_SOC cycle 级仿真"
# ---------------------------------------------------------------------------
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}"

if [ -n "$PKG" ]; then
    if bash "$PKG" --quiet > /tmp/install_opp.log 2>&1; then
        ok "自定义算子包安装成功"
    else
        fail "自定义算子包安装失败（详见 /tmp/install_opp.log）"
        tail -3 /tmp/install_opp.log | sed 's/^/    /'
    fi
else
    fail "找不到 custom_opp 包，跳过安装"
fi

VENDOR_LIB="$ASCEND_OPP_PATH/vendors/trace_custom/op_api/lib"
[ -d "$VENDOR_LIB" ] && export LD_LIBRARY_PATH="$VENDOR_LIB:$LD_LIBRARY_PATH"

cd "$TRACE_DIR/examples" || exit 1
rm -rf build && mkdir -p build && cd build
if cmake .. > /tmp/ex_cmake.log 2>&1 && make -j"$(nproc)" > /tmp/ex_make.log 2>&1; then
    ok "aclnn 示例编译成功"
else
    fail "aclnn 示例编译失败"
    grep -E "error|Error|未找到" /tmp/ex_cmake.log /tmp/ex_make.log | head -6 | sed 's/^/    /'
fi

APP=$(find . -name test_aclnn_trace -type f | head -1)
if [ -z "$APP" ]; then
    fail "找不到 test_aclnn_trace 可执行文件"
elif [ "${SKIP_SIM:-0}" = "1" ]; then
    warn "已按 SKIP_SIM=1 跳过 $SIM_SOC cycle 仿真"
else
    # 已知问题：msprof op simulator 在 CANN 9.2.0-beta.1 上报
    #   aclrtSetDevice failed. ERROR: 507033 (ACL_ERROR_RT_DEV_SETUP_ERROR)
    # 官方 issue 在 910B / 950PR 上均可复现，属工具侧缺陷，不影响精度验证。
    # https://gitcode.com/Ascend/msopprof/issues/159
    # 因此这里只做「探测 + 记录」，不作为验收失败项。
    rm -rf /tmp/sim_out
    msprof op simulator --soc-version="$SIM_SOC" \
        --application="$PWD/$APP" --output=/tmp/sim_out > /tmp/sim.log 2>&1

    if grep -q "Profiling running finished. All task success" /tmp/sim.log \
       && ! grep -q "507033" /tmp/sim.log; then
        ok "$SIM_SOC cycle 仿真执行成功"
        grep -E "Running simulation task|All task success" /tmp/sim.log | sed 's/^/    /'
        echo "    产物目录：$(find /tmp/sim_out -maxdepth 1 -type d | tail -1)"
        echo "    下一步：msprof op --output=/tmp/sim_out --kernel-name=<kernel>"
    elif grep -q "507033" /tmp/sim.log; then
        warn "$SIM_SOC cycle 仿真命中 msopprof 已知缺陷 507033（不影响编译/UT/精度验证）"
        echo "    aclrtSetDevice failed. ERROR: 507033 = ACL_ERROR_RT_DEV_SETUP_ERROR"
        echo "    参见 https://gitcode.com/Ascend/msopprof/issues/159"
        echo "    详见 /tmp/sim.log"
    else
        warn "$SIM_SOC 仿真未产出 profiling 数据，详见 /tmp/sim.log"
        grep -E "ERROR|WARN" /tmp/sim.log | head -5 | sed 's/^/    /'
    fi
fi

# ---------------------------------------------------------------------------
step "验收汇总"
# ---------------------------------------------------------------------------
echo "${GREEN}通过 ${#PASSED[@]} 项${NC}"
for p in "${PASSED[@]}"; do echo "  + $p"; done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo
    echo "${RED}失败 ${#FAILED[@]} 项${NC}"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi

echo
echo "${GREEN}${BOLD}全部验收项通过${NC}"
