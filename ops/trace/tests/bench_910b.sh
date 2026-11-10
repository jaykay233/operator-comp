#!/bin/bash
# ===========================================================================
# bench_910b.sh —— 在 Ascend910B1 CAModel 上测 Trace 算子的 kernel cycle 数
#
# 一次编译，多组运行，给出对比表 + 正确性校验：
#   * shape   : m= / n=
#   * dtype   : dtype=float|half（bf16 未在算子原型中声明，暂不支持）
#   * 核数    : cores=N 强制指定核数（用于验证 Tiling 的核数策略）
#
# 用法（容器内）：
#   docker run --rm -v "$PWD:/workspace" ascendc-cpu-dev:cann92 \
#       bash /workspace/ops/trace/tests/bench_910b.sh
#
#   OUT=/workspace/ca910b_bench bash .../bench_910b.sh
#   CASES="m=256,n=385 m=1024,n=1025,cores=8" bash .../bench_910b.sh
#
# 说明：kernel cycle 数在同一源码 + 同一 build 下完全可复现（重复跑三次结果一致），
# 但**跨 build 会有约 ±10% 的代码布局漂移**（ICache 命中差异）。因此比较不同实现时
# 必须把两版源码分别编译、各自跑一遍，不能拿不同 build 的数字直接对比。
# ===========================================================================
set -o pipefail

TRACE_DIR=${TRACE_DIR:-/workspace/ops/trace}
SOC=${SOC:-ascend910b}
SIM_SOC=${SIM_SOC:-Ascend910B1}
OUT=${OUT:-/workspace/ca910b_bench}
JOBS=${JOBS:-$(nproc)}

# 默认对比组：不同规模 + 两种 dtype，覆盖「单核区 / 多核区」两侧
CASES=${CASES:-"m=1,n=1 m=128,n=129 m=256,n=385 m=1024,n=1025 m=4096,n=4097 m=256,n=385,dtype=half m=4096,n=4097,dtype=half"}

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
step() { echo; echo "${BOLD}========== $* ==========${NC}"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
die()  { echo "${RED}[FAIL]${NC} $*"; exit 1; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }

CANN=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
ARCH_DIR=$(ls -d "$CANN"/*-linux 2>/dev/null | head -1)
[ -n "$ARCH_DIR" ] || die "找不到 CANN host 架构目录"
SOC_DIR=$ARCH_DIR/simulator/dav_2201
[ -d "$SOC_DIR/lib" ] || die "找不到 $SOC_DIR/lib"
LIB=$SOC_DIR/lib
CA=$SOC_DIR/camodel

cd "$TRACE_DIR" || die "找不到 $TRACE_DIR"

# ---------------------------------------------------------------------------
step "0. 准备 910B CAModel 目录"
# ---------------------------------------------------------------------------
mkdir -p "$CA"
cp -f "$LIB"/*.so "$LIB"/*.json "$CA"/ 2>/dev/null || true
ln -sf "$CA/libnpu_drv.so" "$CA/libascend_hal.so"
ln -sf "$CA/libruntime_camodel.so" "$CA/libruntime.so"
[ -e "$CA/libascend_hal.so" ] || die "libascend_hal.so 软链创建失败"
ok "CAModel 目录：$CA"

# ---------------------------------------------------------------------------
step "1. 编译算子 + 安装算子包"
# ---------------------------------------------------------------------------
rm -rf build build_out
bash build.sh -j"$JOBS" --soc="$SOC" > /tmp/bench_build.log 2>&1 \
    || { tail -40 /tmp/bench_build.log; die "编译失败"; }
ok "编译成功"

PKG=$(ls build/custom_opp_*.run 2>/dev/null | head -1)
[ -n "$PKG" ] || die "找不到 custom_opp 包"
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-$CANN/opp}
bash "$PKG" --quiet > /tmp/bench_install.log 2>&1 || die "算子包安装失败"
ok "算子包已安装"

VENDOR_LIB="$ASCEND_OPP_PATH/vendors/trace_custom/op_api/lib"
VENDOR_INC="$ASCEND_OPP_PATH/vendors/trace_custom/op_api/include"

# ---------------------------------------------------------------------------
step "2. 编译 aclnn 示例"
# ---------------------------------------------------------------------------
cd "$TRACE_DIR/examples" || die "找不到 examples"
rm -rf build && mkdir -p build && cd build
cmake .. > /tmp/bench_cmake.log 2>&1 && make -j"$JOBS" > /tmp/bench_make.log 2>&1
APP=$(find . -name test_aclnn_trace -type f | head -1)
[ -n "$APP" ] || { mkdir -p ./bin; g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -O0 -g \
    -I"$CANN/include" -I"$VENDOR_INC" -I"$TRACE_DIR/build/autogen" \
    "$TRACE_DIR/examples/test_aclnn_trace.cpp" -o ./bin/test_aclnn_trace \
    -L"$VENDOR_LIB" -lcust_opapi -L"$CANN/lib64" -lascendcl -lnnopbase -lopapi 2>/dev/null; \
    APP=./bin/test_aclnn_trace; }
APP=$(realpath "$APP" 2>/dev/null)
ok "示例：$APP"

# ---------------------------------------------------------------------------
step "3. 逐组运行"
# ---------------------------------------------------------------------------
rm -rf "$OUT"
i=0
for c in $CASES; do
    i=$((i + 1))
    co=$(echo "$c" | sed -n 's/.*cores=\([0-9]*\).*/\1/p')
    mm=$(echo "$c" | sed -n 's/.*m=\([0-9]*\).*/\1/p')
    nn=$(echo "$c" | sed -n 's/.*n=\([0-9]*\).*/\1/p')
    dt=$(echo "$c" | sed -n 's/.*dtype=\([a-z0-9]*\).*/\1/p')
    co=${co:-0}; mm=${mm:-256}; nn=${nn:-385}; dt=${dt:-float}
    TAG="c${i}_n${co}_${dt}_${mm}x${nn}"
    LOGD=$OUT/$TAG/log_ca
    mkdir -p "$LOGD"
    ( cd "$CA" && env LD_LIBRARY_PATH="$CA:$VENDOR_LIB:$CANN/lib64" \
        CAMODEL_SOC_VERSION="$SIM_SOC" \
        CAMODEL_LOG_PATH="$LOGD" STARS_LOG_PATH="$LOGD" \
        TRACE_CORES="$co" TRACE_M="$mm" TRACE_N="$nn" TRACE_DTYPE="$dt" \
        timeout 600 "$APP" ) > "$OUT/$TAG/app.log" 2>&1
    RC=$?
    CHECK=$(grep -oE "\[CHECK\] (PASS|FAIL)" "$OUT/$TAG/app.log" | tail -1)
    [ $RC -eq 0 ] || { tail -15 "$OUT/$TAG/app.log"; die "case $c 运行失败"; }
    echo "    [$i] $dt ${mm}x${nn} cores=${co:-默认} : ${CHECK:-?}"
done
ok "全部运行完成"

# ---------------------------------------------------------------------------
step "4. 结果表"
# ---------------------------------------------------------------------------
python3 - "$OUT" "$CASES" <<'PY'
import sys
from pathlib import Path

out = Path(sys.argv[1])
cases = sys.argv[2].split()

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

def ival(b, k):
    try:
        return int(b.get(k, 0))
    except ValueError:
        return 0

def span_of(d):
    bs = parse_toml(d / "profile_aiv_log0.toml") + parse_toml(d / "profile_aic_log0.toml")
    if not bs:
        return None, None
    lo = min(ival(b, 'start_cycle') for b in bs)
    hi = max(ival(b, 'end_cycle') for b in bs)
    dmax = max(ival(b, 'duration') for b in bs)
    return hi - lo, dmax

rows = []
for i, c in enumerate(cases, 1):
    co = '0'; mm = '256'; nn = '385'; dt = 'float'
    for part in c.split(','):
        if part.startswith('cores='): co = part[6:]
        elif part.startswith('m='): mm = part[2:]
        elif part.startswith('n='): nn = part[2:]
        elif part.startswith('dtype='): dt = part[6:]
    d = out / f"c{i}_n{co}_{dt}_{mm}x{nn}"
    span, dmax = span_of(d / "log_ca")
    log = (d / "app.log")
    txt = log.read_text() if log.exists() else ""
    ok = "[CHECK] PASS" in txt
    rows.append((i, co, mm, nn, dt, span, dmax, ok))

print(f"    {'#':>2}  {'dtype':>5}  {'shape':>11}  {'cores':>6}  {'kernel跨度':>10}  {'单核最长':>8}  {'正确性':>6}")
for i, co, mm, nn, dt, span, dmax, ok in rows:
    s = f"{span}" if span is not None else "无数据"
    dm = f"{dmax}" if dmax is not None else "-"
    shown = co if co != '0' else '默认'
    print(f"    {i:>2}  {dt:>5}  {mm+'x'+nn:>11}  {shown:>6}  {s:>10}  {dm:>8}  {'PASS' if ok else 'FAIL':>6}")
print()

fails = [r for r in rows if not r[7]]
if fails:
    print(f"    !! 有 {len(fails)} 组正确性失败：")
    for r in fails:
        print(f"       #{r[0]} {r[4]} {r[2]}x{r[3]} cores={r[1]}")
else:
    print(f"    全部 {len(rows)} 组正确性 PASS")
PY

step "汇总"
echo "  逐组日志 : $OUT/c*/app.log"
echo "  周期来源 : $OUT/c*/log_ca/profile_aiv_log0.toml"
echo
echo "${GREEN}${BOLD}基准测试完成${NC}"
