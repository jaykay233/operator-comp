#!/bin/bash
# ===========================================================================
# run_910b_ca.sh —— Ascend 910B1 周期级（cycle-accurate）仿真一键脚本
#
# 用法（容器内）：
#   docker run --rm -v "$PWD:/workspace" ascendc-cpu-dev:cann92 \
#       bash /workspace/ops/trace/tests/run_910b_ca.sh
#
#   OUT=/workspace/ca910b bash .../run_910b_ca.sh    # 指定日志目录
#   REPEAT=5  bash .../run_910b_ca.sh                # 跑 5 次看抖动
#
# ---------------------------------------------------------------------------
# 为什么默认路径跑不通（已彻底定位）
# ---------------------------------------------------------------------------
# 1) 910B 的 simulator 目录（dav_2201）里**没有 camodel/ 子目录**，
#    只有 lib/ 下的运行时垫片：
#        libruntime_camodel.so  libnpu_drv_camodel.so  libstars.so  libmodel_top.so
#    而 950（dav_3510）有完整 camodel/（48 个 .so，93MB 真硅模型）。
#    所以  msprof op simulator --soc-version=Ascend910B1  必然报
#        aclrtSetDevice failed. ERROR: 507033 (ACL_ERROR_RT_DEV_SETUP_ERROR)
#
# 2) 真正缺的是 **libascend_hal.so**。CANN 没为 910B 提供这个驱动壳，
#    而 910B 的 libnpu_drv.so 本身就导出了全部 114 个 hal* 符号，
#    所以只要  ln -sf libnpu_drv.so libascend_hal.so  就能补上。
#
# 3) msprof op simulator 这条路即使补了 HAL 也走不通：
#    它的 LD_PRELOAD 注入库 libmsopprof_injection.so 自身有缺陷，
#    单独 preload 就会让 aclrtSetDevice 报 507033
#    （官方 issue: gitcode.com/Ascend/msopprof/issues/159，
#      910B / 950PR 均复现，CANN 9.2.0-beta.1）。
#
#    → 因此本脚本绕过 msprof，直接跑 CAModel，用 CAMODEL_LOG_PATH
#      让模型自己吐出逐核指令级 dump 和周期统计。
#
# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------
#   $OUT/log_ca/profile_aiv_log0.toml   AIV 每个 block 的 start/end/duration cycle
#   $OUT/log_ca/profile_aic_log0.toml   AIC 同上（cube 算子才有内容）
#   $OUT/log_ca/core<N>.<type>.instr_log.dump         逐条指令
#   $OUT/log_ca/core<N>.<type>.instr_popped_log.dump  指令出队时刻
#   $OUT/log_ca/core<N>.<type>.ccu_log.dump           指令流水占用
#   $OUT/log_ca/core<N>.<type>.dcache_log.dump        数据缓存访问
#   $OUT/log_ca/core<N>.<type>.ifu.icache_log.dump    指令缓存未命中
# ===========================================================================
set -o pipefail

TRACE_DIR=${TRACE_DIR:-/workspace/ops/trace}
SOC=${SOC:-ascend910b}
SIM_SOC=${SIM_SOC:-Ascend910B1}
OUT=${OUT:-/workspace/ca910b}
JOBS=${JOBS:-$(nproc)}
REPEAT=${REPEAT:-1}

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
step() { echo; echo "${BOLD}========== $* ==========${NC}"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
die()  { echo "${RED}[FAIL]${NC} $*"; exit 1; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }

CANN=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
ARCH_DIR=$(ls -d "$CANN"/*-linux 2>/dev/null | head -1)
[ -n "$ARCH_DIR" ] || die "找不到 CANN host 架构目录"

# 910B1 -> dav_2201
SOC_DIR=$ARCH_DIR/simulator/dav_2201
[ -d "$SOC_DIR/lib" ] || die "找不到 $SOC_DIR/lib"
LIB=$SOC_DIR/lib
CA=$SOC_DIR/camodel

cd "$TRACE_DIR" || die "找不到 $TRACE_DIR"

# ---------------------------------------------------------------------------
step "0. 准备 910B CAModel 目录（补 libascend_hal.so）"
# ---------------------------------------------------------------------------
mkdir -p "$CA"
cp -f "$LIB"/*.so "$LIB"/*.json "$CA"/ 2>/dev/null || true
ln -sf "$CA/libnpu_drv.so" "$CA/libascend_hal.so"
ln -sf "$CA/libruntime_camodel.so" "$CA/libruntime.so"
[ -e "$CA/libascend_hal.so" ] || die "libascend_hal.so 软链创建失败"
ok "CAModel 目录：$CA"
ok "libascend_hal.so -> libnpu_drv.so（导出 $(nm -D --defined-only "$CA/libnpu_drv.so" 2>/dev/null | grep -c ' hal') 个 hal* 符号）"

# npusim 的 SoC 表里没有 910B，补进去（npusim 只认 Ascend950/Ascend950DT）
SOCPY=$CANN/python/site-packages/cannsim/core/public/soc_info.py
if [ -f "$SOCPY" ] && ! grep -q "\"$SIM_SOC\"" "$SOCPY"; then
    python3 - "$SOCPY" "$SIM_SOC" <<'PYEOF'
import sys
p, soc = sys.argv[1], sys.argv[2]
s = open(p).read()
s = s.replace('SOC_INFO: Dict[str, SocSpec] = {\n',
              f'SOC_INFO: Dict[str, SocSpec] = {{\n    "{soc}": {{\n        "max_cores": 24,\n    }},\n')
open(p, 'w').write(s)
PYEOF
    ok "已把 $SIM_SOC 注册进 npusim SOC_INFO（max_cores=24）"
fi

# ---------------------------------------------------------------------------
step "1. 编译算子（--soc=$SOC）+ 安装自定义算子包"
# ---------------------------------------------------------------------------
rm -rf build build_out
bash build.sh -j"$JOBS" --soc="$SOC" > /tmp/ca910b_build.log 2>&1 \
    || { tail -20 /tmp/ca910b_build.log; die "编译失败"; }
ok "编译成功"
grep -E "Kernel binary:" /tmp/ca910b_build.log | tail -1 | sed 's/^/    /'
KERNEL_OBJ=$(grep -E "Kernel binary:" /tmp/ca910b_build.log | tail -1 | sed 's/.*Kernel binary: //')

PKG=$(ls build/custom_opp_*.run 2>/dev/null | head -1)
[ -n "$PKG" ] || die "找不到 custom_opp 包"
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-$CANN/opp}
bash "$PKG" --quiet > /tmp/ca910b_install.log 2>&1 || die "算子包安装失败"
ok "自定义算子包已安装"

VENDOR_LIB="$ASCEND_OPP_PATH/vendors/trace_custom/op_api/lib"
VENDOR_INC="$ASCEND_OPP_PATH/vendors/trace_custom/op_api/include"
[ -d "$VENDOR_LIB" ] || die "找不到 $VENDOR_LIB"

# ---------------------------------------------------------------------------
step "2. 编译 aclnn 调用示例（必须 old ABI，CANN 库是这个 ABI）"
# ---------------------------------------------------------------------------
cd "$TRACE_DIR/examples" || die "找不到 examples"
rm -rf build && mkdir -p build && cd build
cmake .. > /tmp/ca910b_cmake.log 2>&1 && make -j"$JOBS" > /tmp/ca910b_make.log 2>&1
APP=$(find . -name test_aclnn_trace -type f | head -1)
if [ -z "$APP" ]; then
    warn "CMake 路径没出产物，改用 old-ABI 手编"
    g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -O0 -g \
        -I"$CANN/include" -I"$VENDOR_INC" -I"$TRACE_DIR/build/autogen" \
        "$TRACE_DIR/examples/test_aclnn_trace.cpp" -o ./bin/test_aclnn_trace \
        -L"$VENDOR_LIB" -lcust_opapi -L"$CANN/lib64" -lascendcl -lnnopbase -lopapi \
        > /tmp/ca910b_app.log 2>&1 \
        || { tail -10 /tmp/ca910b_app.log; die "示例编译失败"; }
    APP=./bin/test_aclnn_trace
fi
APP=$(realpath "$APP")
ok "示例：$APP"

# ---------------------------------------------------------------------------
step "3. 在 910B1 CAModel 上运行 $REPEAT 次"
# ---------------------------------------------------------------------------
rm -rf "$OUT"
ALL_TICKS=()
for i in $(seq 1 "$REPEAT"); do
    LOGD=$OUT/run$i/log_ca
    mkdir -p "$LOGD"
    ( cd "$CA" && env LD_LIBRARY_PATH="$CA:$VENDOR_LIB:$CANN/lib64" \
        CAMODEL_SOC_VERSION="$SIM_SOC" \
        CAMODEL_LOG_PATH="$LOGD" STARS_LOG_PATH="$LOGD" \
        timeout 600 "$APP" ) > "$OUT/run$i/app.log" 2>&1
    RC=$?
    TICK=$(grep -oE "Total tick: [0-9]+" "$OUT/run$i/app.log" | tail -1 | awk '{print $3}')
    CHECK=$(grep -oE "\[CHECK\] (PASS|FAIL)" "$OUT/run$i/app.log" | tail -1)
    [ $RC -eq 0 ] || { tail -15 "$OUT/run$i/app.log"; die "第 $i 次运行失败"; }
    ALL_TICKS+=("${TICK:-0}")
    echo "    第 $i 次：$CHECK  Total tick=${TICK:-?}"
done
ok "全部运行完成"

# ---------------------------------------------------------------------------
step "4. 周期数据（profile_aiv/aic_log0.toml）"
# ---------------------------------------------------------------------------
python3 - "$OUT" "$REPEAT" <<'PY'
import re, sys, statistics
from pathlib import Path

out = Path(sys.argv[1])
repeat = int(sys.argv[2])

def parse_toml(path):
    blocks, cur = [], {}
    if not path.exists():
        return blocks
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('['):
            if cur:
                blocks.append(cur)
            cur = {'name': line.strip('[]')}
        elif '=' in line:
            k, v = line.split('=', 1)
            cur[k.strip()] = v.strip()
    if cur:
        blocks.append(cur)
    return blocks

for i in range(1, repeat + 1):
    logd = out / f"run{i}" / "log_ca"
    aiv = parse_toml(logd / "profile_aiv_log0.toml")
    aic = parse_toml(logd / "profile_aic_log0.toml")
    blocks = aiv + aic
    print(f"--- 第 {i} 次")
    if not blocks:
        print("    （没有 profile 数据）")
        continue

    def ival(b, k):
        try:
            return int(b.get(k, 0))
        except ValueError:
            return 0

    spans = [(ival(b, 'start_cycle'), ival(b, 'end_cycle')) for b in blocks]
    durs = [ival(b, 'duration') for b in blocks]
    lo = min(s for s, _ in spans)
    hi = max(e for _, e in spans)
    print(f"    参与核数           : {len(blocks)}")
    print(f"    kernel 总跨度      : {hi - lo} cycles")
    print(f"    单核 duration      : min={min(durs)}  max={max(durs)}  "
          f"mean={statistics.mean(durs):.1f}  median={statistics.median(durs)}")
    print(f"    最长核 / 最短核    : {max(durs) / min(durs):.3f}x  (负载均衡)")
    print()
    print(f"    {'core':>4}  {'start':>7}  {'end':>7}  {'dur':>7}")
    for b in sorted(blocks, key=lambda x: ival(x, 'core_id')):
        print(f"    {ival(b, 'core_id'):>4}  {ival(b, 'start_cycle'):>7}  "
              f"{ival(b, 'end_cycle'):>7}  {ival(b, 'duration'):>7}")
    print()

# 抖动
if repeat > 1:
    totals = []
    for i in range(1, repeat + 1):
        log = out / f"run{i}" / "app.log"
        m = re.search(r"Total tick: (\d+)", log.read_text())
        if m:
            totals.append(int(m.group(1)))
    if len(totals) > 1:
        print("--- Total tick 抖动")
        print(f"    {totals}")
        print(f"    mean={statistics.mean(totals):.1f}  "
              f"stdev={statistics.pstdev(totals):.1f}  "
              f"相对抖动={statistics.pstdev(totals) / statistics.mean(totals) * 100:.2f}%")
PY

step "汇总"
echo "  周期统计 : $OUT/run1/log_ca/profile_aiv_log0.toml"
echo "  指令流水 : $OUT/run1/log_ca/core0.veccore0.instr_log.dump"
echo "  完整日志 : $OUT/run1/app.log"
[ -n "$KERNEL_OBJ" ] && echo "  kernel   : $KERNEL_OBJ"
echo
echo "${GREEN}${BOLD}910B1 周期仿真完成${NC}"
