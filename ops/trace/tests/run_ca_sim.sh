#!/bin/bash
# ===========================================================================
# run_ca_sim.sh —— 周期级（cycle-accurate）仿真一键脚本
#
# 用法（容器内）：
#   docker run --rm -v "$PWD:/workspace" ascendc-cpu-dev:cann92 \
#       bash /workspace/ops/trace/tests/run_ca_sim.sh
#
#   OUT=/workspace/ca_out bash .../run_ca_sim.sh   # 指定产物目录
#
# 背景（重要）：
#   CANN 9.2.0-beta.1 的 simulator 里，只有 dav_3510（= Ascend950PR 系列）
#   带 camodel/ 目录（libSoC.so / libEslTop.so / libHBMSim.so 等真实硅模型，
#   93MB）；而 dav_2201（= Ascend910B1/B2/B3/B4、910_93xx）只有 lib/ 里的
#   PV/CA 运行时垫片，没有硅模型，因此：
#
#     msprof op simulator --soc-version=Ascend910B1  → aclrtSetDevice 507033
#
#   这不是配置问题，是模型没随包发布（8.2.RC1 同样如此）。
#   所以本脚本走 "kernel 编 ascend950 + Ascend950PR CA 模型" 这条路，
#   拿到真实的周期级数据（kernel_total_clocks / 各 pipe 利用率 / IPC）。
#
#   验证证据：
#     ls $ASCEND_HOME_PATH/x86_64-linux/simulator/dav_3510/camodel/libSoC.so   # 有
#     ls $ASCEND_HOME_PATH/x86_64-linux/simulator/dav_2201/camodel             # 无
#
# 产物：
#   $OUT/npusim_*/report/SUMMARY.md          —— 汇总表（各 core 的 IPC 报告）
#   $OUT/npusim_*/report/index.html          —— 可视化总览（浏览器打开）
#   $OUT/npusim_*/report/results/kernel_0_reports/summary.json  —— 原始指标
#   $OUT/npusim_*/report/results/kernel_0_reports/core_0/trace_core0.json
#                                            —— Chrome/Perfetto 可打开的 trace
# ===========================================================================
set -o pipefail

TRACE_DIR=${TRACE_DIR:-/workspace/ops/trace}
SOC=${SOC:-ascend950}            # kernel 编译目标（cycle 仿真必须 950）
SIM_SOC=${SIM_SOC:-Ascend950}    # npusim 的 --soc-version
OUT=${OUT:-/tmp/ca_sim}
JOBS=${JOBS:-$(nproc)}

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
step() { echo; echo "${BOLD}========== $* ==========${NC}"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
die()  { echo "${RED}[FAIL]${NC} $*"; exit 1; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }

CANN=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
cd "$TRACE_DIR" || die "找不到 $TRACE_DIR"

# ---------------------------------------------------------------------------
step "0. 检查 CA 模型是否真的存在"
# ---------------------------------------------------------------------------
CA_DIR=$(ls -d "$CANN"/*-linux/simulator/dav_3510/camodel 2>/dev/null | head -1)
[ -n "$CA_DIR" ] || die "找不到 dav_3510/camodel（CA 模型缺失，无法做周期仿真）"
for f in libSoC.so libEslTop.so libHBMSim.so libascend_hal.so libruntime.so; do
    [ -f "$CA_DIR/$f" ] || die "CA 模型不完整，缺少 $f"
done
ok "CA 模型：$CA_DIR"

if [ -d "$CANN/x86_64-linux/simulator/dav_2201/camodel" ]; then
    ok "910B（dav_2201）也带 CA 模型，可用 --soc-version=Ascend910B1"
else
    warn "910B（dav_2201）无 camodel/ → 910B 周期仿真在本 CANN 版本不可用"
    warn "将使用 $SIM_SOC 的 CA 模型（kernel 编译目标 $SOC）"
fi

# npusim 报告需要 plotly
python3 -c "import plotly" 2>/dev/null || {
    warn "缺 plotly，尝试安装"
    python3 -m pip install --no-cache-dir -q plotly pandas \
        -i https://repo.huaweicloud.com/repository/pypi/simple || warn "plotly 安装失败，报告会退化为 json"
}

# ---------------------------------------------------------------------------
step "1. 编译算子（--soc=$SOC）+ 安装自定义算子包"
# ---------------------------------------------------------------------------
rm -rf build build_out
bash build.sh -j"$JOBS" --soc="$SOC" > /tmp/ca_build.log 2>&1 \
    || { tail -20 /tmp/ca_build.log; die "编译失败（详见 /tmp/ca_build.log）"; }
ok "编译成功"
grep -E "Kernel binary:" /tmp/ca_build.log | tail -1 | sed 's/^/    /'

PKG=$(ls build/custom_opp_*.run 2>/dev/null | head -1)
[ -n "$PKG" ] || die "找不到 custom_opp 包"
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-$CANN/opp}
bash "$PKG" --quiet > /tmp/ca_install.log 2>&1 || die "算子包安装失败"
ok "自定义算子包已安装"

VENDOR_LIB="$ASCEND_OPP_PATH/vendors/trace_custom/op_api/lib"
[ -d "$VENDOR_LIB" ] || die "找不到 $VENDOR_LIB"
export LD_LIBRARY_PATH="$VENDOR_LIB:$CANN/lib64:$LD_LIBRARY_PATH"

# ---------------------------------------------------------------------------
step "2. 编译 aclnn 调用示例"
# ---------------------------------------------------------------------------
cd "$TRACE_DIR/examples" || die "找不到 examples"
rm -rf build && mkdir -p build && cd build
cmake .. > /tmp/ca_cmake.log 2>&1 && make -j"$JOBS" > /tmp/ca_make.log 2>&1 \
    || { grep -E "error|Error" /tmp/ca_cmake.log /tmp/ca_make.log | head -6; die "示例编译失败"; }
APP=$(find . -name test_aclnn_trace -type f | head -1)
[ -n "$APP" ] || die "找不到 test_aclnn_trace"
ok "示例：$TRACE_DIR/examples/build/$APP"

# ---------------------------------------------------------------------------
step "3. npusim record（CA 模型上真跑一遍）"
# ---------------------------------------------------------------------------
OBJ=$(ls "$TRACE_DIR"/build/op_kernel/ascendc_kernels/binary/"$SOC"/*/*.o 2>/dev/null | head -1)
rm -rf "$OUT" && mkdir -p "$OUT"

REC_ARGS=(-s "$SIM_SOC" -g -o "$OUT")
[ -n "$OBJ" ] && REC_ARGS+=(-f "$OBJ")
echo "    npusim record ${REC_ARGS[*]} -- $APP"

npusim record "${REC_ARGS[@]}" -- "$PWD/$APP" > /tmp/ca_npusim.log 2>&1
RC=$?
grep -E "Simulation SUCCESS|Simulation FAILED|USER_APP|hardware sim|Simulation" /tmp/ca_npusim.log | tail -5 | sed 's/^/    /'
[ $RC -eq 0 ] || { grep -vE "DRVSTUB|CORE_WRAPPER|TmSim" /tmp/ca_npusim.log | tail -15; die "npusim record 失败"; }

RUN_DIR=$(ls -dt "$OUT"/npusim_* 2>/dev/null | head -1)
[ -n "$RUN_DIR" ] || die "没有产出 npusim 目录"
ok "仿真产物：$RUN_DIR"

# ---------------------------------------------------------------------------
step "4. npusim report（生成流水/IPC 报告）"
# ---------------------------------------------------------------------------
if [ ! -f "$RUN_DIR/report/SUMMARY.md" ]; then
    npusim report -e "$RUN_DIR" > /tmp/ca_report.log 2>&1 \
        || warn "report 失败（详见 /tmp/ca_report.log）"
fi
SUMMARY="$RUN_DIR/report/results/kernel_0_reports/summary.json"
[ -f "$SUMMARY" ] || { warn "没有 summary.json，report 可能未生成"; }

# ---------------------------------------------------------------------------
step "5. 周期数据"
# ---------------------------------------------------------------------------
if [ -f "$SUMMARY" ]; then
    python3 - "$SUMMARY" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ki = d.get("kernel_info", {})
tp = d.get("top_level_diagnosis", {})
print(f"  kernel_total_clocks        : {ki.get('kernel_total_clocks')}")
print(f"  instructions_executed      : {ki.get('kernel_instructions_executed')}")
print(f"  ai_core_active             : {ki.get('ai_core_active')}")
print(f"  dominant_pipeline          : {tp.get('dominant_pipeline')} "
      f"({tp.get('dominant_pipeline_util')})")
print(f"  imbalance_ratio            : {tp.get('imbalance_ratio')}")
print()
print("  pipe 利用率:")
for name, v in (d.get("pipe_utilization", {}).get("pipeline_util_summary", {}) or {}).items():
    if isinstance(v, dict):
        print(f"    {name:<14} mean={v.get('mean')}  max={v.get('max')}")
    else:
        print(f"    {name:<14} {v}")

print()
print("  其它:")
for key in ("cache", "bandwidth", "pipeline_overlap"):
    sub = d.get(key) or {}
    for k, v in sub.items():
        if isinstance(v, (int, float, str)):
            print(f"    {key}.{k} = {v}")
PY
else
    grep -E "cycle" /tmp/ca_npusim.log | grep -v DRVSTUB | tail -3 | sed 's/^/    /'
fi

echo
echo "${GREEN}${BOLD}周期仿真完成${NC}"
echo "  汇总：  $RUN_DIR/report/SUMMARY.md"
echo "  可视化：$RUN_DIR/report/index.html"
echo "  trace： $RUN_DIR/report/results/kernel_0_reports/core_0/trace_core0.json"
echo "          （chrome://tracing 或 https://ui.perfetto.dev 打开）"
