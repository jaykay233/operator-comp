#!/bin/bash
# ===========================================================================
# bench_real.sh —— 在**真机**（Ascend NPU）上测 Trace 算子耗时并做管线归因
#
# 与 bench_910b.sh 的区别：
#   bench_910b.sh 跑 CAModel 周期仿真，只能看指令和 cycle；
#   本脚本跑真机 + msprof op，能拿到 Task Duration(us) 与各管线的占用比例 ——
#   这是判断「到底卡在 MTE2 搬入、VEC 计算还是 SCALAR 簿记」的唯一可靠数据。
#
# 为什么需要它：CAModel 对 MTE2 的建模严重失真（整个 kernel 的 MTE2 只报 21 cycles），
# 因此凡是以 GM 访存为主的优化都不能用 CAModel 判断收益，必须真机 A/B。
#
# 用法（在装有 NPU 的机器上，算子包已安装或先跑 build.sh + install）：
#   bash ops/trace/tests/bench_real.sh
#
#   # 指定 shape 列表
#   SHAPES="128x129 256x385" bash ops/trace/tests/bench_real.sh
#
#   # 只跑 32B 与 4B 两种搬运方式的对比
#   MODES="0 1" SHAPES="256x385" bash ops/trace/tests/bench_real.sh
#
# 环境变量：
#   SHAPES  : 空格分隔的 MxN 列表
#   DTYPES  : float half（默认两个都跑）
#   MODES   : 0 = 32B 整块搬运（默认），1 = 4B+padding
#   METRICS : msprof 的 --aic-metrics 取值，默认 PipeUtilization
#             （想同时看访存明细可设 "PipeUtilization Memory"）
# ===========================================================================
set -o pipefail

TRACE_DIR=${TRACE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${OUT:-$TRACE_DIR/log_real}
SHAPES=${SHAPES:-"128x129 256x385 1024x1025"}
DTYPES=${DTYPES:-"float half"}
MODES=${MODES:-"0"}
METRICS=${METRICS:-"PipeUtilization"}
WARMUP=${WARMUP:-5}

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
step() { echo; echo "${BOLD}========== $* ==========${NC}"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
die()  { echo "${RED}[FAIL]${NC} $*"; exit 1; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }

CANN=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
export LD_LIBRARY_PATH="$CANN/lib64:$LD_LIBRARY_PATH"

# 可执行文件：优先用 APP，否则按常见产物路径依次探测
if [ -z "$APP" ]; then
    for c in "$TRACE_DIR/examples/build/bin/test_aclnn_trace" \
             "$TRACE_DIR/examples/build/test_aclnn_trace" \
             "$TRACE_DIR/examples/build/bin/aclnn_trace"; do
        [ -x "$c" ] && APP="$c" && break
    done
fi
[ -n "$APP" ] && [ -x "$APP" ] || die "找不到示例可执行文件（在 ops/trace/examples 下跑 bash run.sh 编译，或设 APP=/abs/path）"
command -v msprof > /dev/null || die "找不到 msprof（需 CANN toolkit 且已 source set_env.sh）"
PYBIN=$(command -v python3 || true)
[ -n "$PYBIN" ] || die "找不到 python3（脚本用它解析 op_summary.csv；没有就手动看 csv）"

# 算子包必须已安装，否则 aclnn 调用会直接失败
if [ -z "$ASCEND_OPP_PATH" ]; then
    warn "ASCEND_OPP_PATH 为空，确认算子包已 install（bash build.sh && ./custom_opp_*.run）"
fi

echo "  可执行文件 : $APP"
echo "  msprof     : $(command -v msprof)"
echo "  采集指标   : $METRICS"

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
step "1. 逐组合真机采集"
# ---------------------------------------------------------------------------
for mode in $MODES; do
    for dt in $DTYPES; do
        for shape in $SHAPES; do
            m=${shape%x*}
            n=${shape#*x}
            tag="copy${mode}_${dt}_${m}x${n}"
            d="$OUT/$tag"
            rm -rf "$d"
            mkdir -p "$d"
            # shape/dtype/copyMode 通过环境变量传给算子（msprof 继承父进程环境），
            # 必须在调用 msprof 之前 export。
            export TRACE_M="$m"
            export TRACE_N="$n"
            export TRACE_DTYPE="$dt"
            export TRACE_COPY="$mode"
            # 采集管线占比：这是判断瓶颈在哪个 pipe 的关键
            msprof op \
                --output="$d" \
                --application="$APP" \
                --aic-metrics=$METRICS \
                --warm-up="$WARMUP" \
                --launch-count=1 \
                --launch-skip-before-match=0 \
                > "$d/msprof.log" 2>&1
            rc=$?
            if [ $rc -ne 0 ]; then
                warn "$tag : msprof 返回 $rc，见 $d/msprof.log"
            fi
            echo "    [$tag] 采集完成"
        done
    done
done

unset TRACE_M TRACE_N TRACE_DTYPE TRACE_COPY

# ---------------------------------------------------------------------------
step "2. 解析结果"
# ---------------------------------------------------------------------------
python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path

out = Path(sys.argv[1])
KEY = ("Duration", "ratio", "time(us)", "Name", "Type", "Block Dim")

def find_summary(d):
    """msprof op 的产物名各版本不完全一致，按内容找而不是按文件名猜。"""
    cands = []
    for pat in ("op_summary*.csv", "*summary*.csv", "*.csv"):
        cands = sorted(d.rglob(pat))
        for c in cands:
            try:
                head = c.open(encoding="utf-8", errors="replace").readline()
            except OSError:
                continue
            if "Duration" in head or "Op Name" in head:
                return c
        if cands:
            break
    return None

rows = []
for d in sorted(out.iterdir()):
    if not d.is_dir():
        continue
    csvp = find_summary(d)
    if csvp is None:
        rows.append((d.name, None, {}))
        continue
    with open(csvp, newline="", encoding="utf-8", errors="replace") as f:
        recs = list(csv.DictReader(f))
    if not recs:
        rows.append((d.name, None, {}))
        continue
    # 同一 shape 可能有多个 kernel（half/float），取 Task Duration 最大的那条
    def dur(r):
        for k, v in r.items():
            if k and "Task Duration" in k:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0
    best = max(recs, key=dur)
    keep = {k: v for k, v in best.items() if k and any(t in k for t in KEY)}
    rows.append((d.name, dur(best), keep))

if not rows:
    print("    未找到任何采集结果，检查 $OUT/*/msprof.log")
    sys.exit(0)

missing = [n for n, dur, _ in rows if dur is None]
if missing:
    print(f"    {len(missing)} 组没有采集到数据：{', '.join(missing)}")

# 汇总：耗时 + 各 pipe 占比
cols = None
for _, dur, kv in rows:
    if kv:
        cols = list(kv.keys())
        break

print(f"    {'case':>28}  {'Task(us)':>9}")
for name, dur, kv in rows:
    print(f"    {name:>28}  {dur if dur is not None else '-':>9}")

if cols:
    print()
    print("    管线占比 / 明细：")
    # 只挑对定位瓶颈有用的列
    want = [c for c in cols if any(t in c for t in
            ("vec_ratio", "scalar_ratio", "mte2_ratio", "mte3_ratio",
             "vec_time", "scalar_time", "mte2_time", "mte3_time",
             "Block Dim"))]
    want = want[:12]
    print("    " + f"{'case':>28}  " + "  ".join(f"{c.replace('(us)',''):>12}" for c in want))
    for name, dur, kv in rows:
        if not kv:
            continue
        print("    " + f"{name:>28}  " + "  ".join(f"{kv.get(c,'-'):>12}" for c in want))

# A/B 对比（copy0 vs copy1，同 shape/dtype）
print()
groups = {}
for name, dur, kv in rows:
    if not name.startswith("copy"):
        continue
    mode = name.split("_")[0].replace("copy", "")
    rest = name.split("_", 1)[1]
    groups.setdefault(rest, {})[mode] = dur

done = False
for rest, m in sorted(groups.items()):
    if "0" in m and "1" in m and m["0"] and m["1"]:
        d0, d1 = m["0"], m["1"]
        print(f"    {rest:>24}: 32B整块 {d0:.3f}us  4B+padding {d1:.3f}us  "
              f"-> 32B 快 {(d1-d0)*100.0/d1:+.1f}%")
        done = True
if not done:
    print("    想对比搬运方式，用 MODES=\"0 1\" 再跑一次")
PY

step "汇总"
echo "  采集目录 : $OUT"
echo
echo "${GREEN}${BOLD}真机基准完成${NC}"
