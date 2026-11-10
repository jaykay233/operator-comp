#!/usr/bin/env python3
# ===========================================================================
# parse_ca_dump.py —— 从 910B CAModel 的 dump 重建 pipe 占用 / 停顿归因
#
# 背景：为什么不用 msprof
# ---------------------------------------------------------------------------
# msprof 在这台机器上有两种用法，都不可用：
#   1) msprof op（真机采集）：需要真实 NPU + 驱动，无卡机器不可能
#   2) msprof op simulator（cycle 仿真）：它的 LD_PRELOAD 注入库
#      libmsopprof_injection.so 自身有缺陷，单独 preload 就使
#      aclrtSetDevice 报 507033（官方 msopprof issue #159，
#      CANN 9.2.0-beta.1 在 910B / 950PR 均复现）；
#      且它会覆盖子进程 LD_LIBRARY_PATH，导致 app 缺 libcust_opapi.so 而
#      退出 127，日志表现为 "Profiling results are empty"。
#
# 因此 run_910b_ca.sh 直接跑 CAModel，用 CAMODEL_LOG_PATH 让模型自己吐 dump。
# 本脚本把这些 dump 换算成 msprof PipeUtilization 那一类指标。
#
# dump 里有什么、能算出什么
# ---------------------------------------------------------------------------
#   instr_popped_log.dump   [cycle] (PC) PIPE: 指令 —— 每条指令的**发射(issue)时刻**
#                           → 精确得到：每 pipe 的指令数、发射槽占用（1 cycle/发射）
#   ccu_log.dump            逐 cycle 的译码/发射/停顿事件
#                           → 精确得到：每个停顿 cycle 的原因
#                             (RAW / wait_flag / do issue failed / D1-I1 failed ...)
#   profile_aiv_log0.toml   每个 block 的 start/end/duration cycle
#   dcache/icache_log.dump  访存与取指明细（本脚本只用于标注，不计入占比）
#
# 局限（务必知道，不然会得出错误结论）
# ---------------------------------------------------------------------------
#   * CAModel 对 MTE2/MTE3 的建模严重失真：搬运指令发完几 cycle 就 SET_FLAG
#     （PERF.md 记的「整个 kernel MTE2 只报 21 cycles」就是这个原因）。
#     所以本脚本给出的 MTE2/MTE3 占比是“模型口径”，不能当真机带宽用。
#   * VEC 没有 retire 事件，占用用「发射 → 下一条 VEC 发射 / 下一条 BAR」的
#     区间估计；区间内若被 flag 卡住会略高估（dump 无 completion 事件，无法再细）。
#   * SCALAR 采用发射槽口径（1 cycle/发射），LD 的 dcache 延迟体现为
#     「RAW 停顿」而不是 SCALAR 占用 —— 这与真机 scalar_ratio 口径不同，
#     但对定位瓶颈更有用。
#
# 用法
# ---------------------------------------------------------------------------
#   # 单个 log_ca 目录
#   python3 parse_ca_dump.py ca910b_final/c3_n0_float_256x385/log_ca
#
#   # 整个基准目录（递归找所有 profile_aiv_log0.toml）
#   python3 parse_ca_dump.py ca910b_final
#
#   # 只看 0 号核，并输出 JSON
#   python3 parse_ca_dump.py ca910b_final --core 0 --json
# ===========================================================================
import argparse
import bisect
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
INSTR_RE = re.compile(
    r'\[(\d+)\] \(PC: (0x[0-9a-fA-F]+)\) ([A-Z0-9_]+)\s+:\s+\(Binary: 0x[0-9a-fA-F]+\) (\S+)'
)
CCU_RE = re.compile(r'^\[(?:info|debug)\] (\d+): (.*)$')

# 停顿归因：按优先级排列（同一窗口内出现多种关键词时取优先级最高的）
STALL_KINDS = [
    ('cross_core', ('wait_core_rely', 'core_rely')),
    ('wait_flag', ('wait_flag hazard', 'waiting_v syn hazard', 'wait_v')),
    ('raw', ('RAW hazard', 'wait_long hazard')),
    ('ldst', ('ldst scalar failed',)),
    ('decode', ('D1 to I1 failed',)),
    ('issue_block', ('do issue failed',)),
]

ICACHE_MISS_RE = re.compile(r'^\[(?:info|debug)\] (\d+): .*status is MISS')
DCACHE_MISS_RE = re.compile(r'^\[(?:info|debug)\] (\d+): lookup_tag is MISS')

PIPE_ORDER = ['SCALAR', 'VEC', 'MTE2', 'MTE3', 'FIX']
# 数据搬运类（CAModel 里会被 SET_FLAG 快速“完成”）
MOVE_PIPES = {'MTE2', 'MTE3'}


def parse_instr(path):
    """[cycle] (PC) PIPE: NAME  ->  [(cycle, pipe, name, pc)]"""
    out = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = INSTR_RE.search(line)
            if m:
                out.append((int(m.group(1)), m.group(3), m.group(4), m.group(2)))
    return out


def parse_ccu(path):
    """[(cycle, text)]，按 cycle 排序（同 cycle 保持原序）"""
    out = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = CCU_RE.match(line)
            if m:
                out.append((int(m.group(1)), m.group(2)))
    out.sort(key=lambda x: x[0])
    return out


def parse_miss_cycles(path, regex):
    """icache/dcache 的 MISS 时刻列表（升序）"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = regex.match(line)
            if m:
                out.append(int(m.group(1)))
    out.sort()
    return out


def parse_profile_toml(path):
    """[block] 段 -> {'block0': {block_id, core_id, start_cycle, end_cycle, duration}}"""
    if not os.path.exists(path):
        return {}
    txt = open(path, 'r', errors='replace').read()
    blocks = {}
    for chunk in re.split(r'(?=^\[)', txt, flags=re.M):
        fields = dict(re.findall(r'(\w+) = (\d+)', chunk))
        if 'block_id' not in fields:
            continue
        b = {k: int(v) for k, v in fields.items()}
        blocks[b['block_id']] = b
    return blocks


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def classify_window(ccu, lo, hi, icache=None, dcache=None, next_instr=None):
    """对 [lo, hi) 内的 ccu 行做停顿归因，返回 (kind, 命中行数)

    只统计“明确表达了停顿原因”的行；decode/issue 之类的过程性日志不参与投票，
    否则它们淹没有效信号（一个 600 cycle 的窗口能刷 600 行 decode 日志）。
    ccu 完全没有事件时，退化为：icache/dcache miss → 同步栅栏 → 静默。
    """
    i = bisect.bisect_left(ccu, (lo, ''))
    j = bisect.bisect_left(ccu, (hi, ''))
    counts = Counter()
    for _, text in ccu[i:j]:
        for kind, keys in STALL_KINDS:
            if any(k in text for k in keys):
                counts[kind] += 1
                break
    if counts:
        order = {k: idx for idx, (k, _) in enumerate(STALL_KINDS)}
        kind, hits = max(counts.items(), key=lambda kv: (kv[1], -order.get(kv[0], 99)))
        # 证据太弱（整个窗口只有 1 行命中）时不下结论，交给后面的证据链
        if hits > 1:
            return kind, hits

    if icache is not None and bisect.bisect_left(icache, lo) < bisect.bisect_left(icache, hi):
        return 'icache', 0
    if dcache is not None and bisect.bisect_left(dcache, lo) < bisect.bisect_left(dcache, hi):
        return 'dcache', 0
    if next_instr and next_instr.startswith('BAR'):
        return 'barrier', 0
    return 'silent', 0


def pipe_busy(ins, pipe, bars):
    """
    估算某 pipe 的占用 cycle：
      * VEC / 其他计算 pipe：发射 → 同一 pipe 下一次发射或下一个 BAR，取先到者
      * MTE2 / MTE3 的数据搬运指令：发射 → 紧跟的 SET_FLAG（CAModel 口径）
      * SET_FLAG / WAIT_FLAG / BAR 这类同步伪指令不计入搬运占用
    """
    spec = [x for x in ins if x[1] == pipe]
    if not spec:
        return 0

    def next_drain(cycle, nxt):
        cut = [b for b in bars if cycle <= b < nxt]
        return min(cut) if cut else nxt

    busy = 0
    if pipe in MOVE_PIPES:
        for idx, (c, _, name, _) in enumerate(spec):
            if name in ('SET_FLAG', 'WAIT_FLAG'):
                continue
            # 找紧跟的 SET_FLAG（同一 pipe 内的下一条即可）
            end = spec[idx + 1][0] if idx + 1 < len(spec) else c
            busy += max(0, end - c)
    else:
        for idx in range(len(spec)):
            a = spec[idx][0]
            b = spec[idx + 1][0] if idx + 1 < len(spec) else ins[-1][0]
            busy += max(0, next_drain(a, b) - a)
    return busy


def analyze_dump(dump_dir, core_filter=None):
    """返回 (blocks_meta, rows) —— rows 每个 (dump 文件) 一行指标"""
    aiv = parse_profile_toml(os.path.join(dump_dir, 'profile_aiv_log0.toml'))
    aic = parse_profile_toml(os.path.join(dump_dir, 'profile_aic_log0.toml'))
    blocks = {}
    blocks.update(aic)
    for k, v in aiv.items():
        blocks.setdefault(k + 100000, v)  # AIV block 号与 AIC 分开，避免撞号

    dumps = []
    for p in sorted(glob.glob(os.path.join(dump_dir, 'core*.cubecore*.instr_popped_log.dump')) +
                    glob.glob(os.path.join(dump_dir, 'core*.veccore*.instr_popped_log.dump'))):
        ins = parse_instr(p)
        if ins:
            dumps.append((p, ins))

    # dump -> block 映射：数量一致时按 (首指令时刻, core, veccore) 与 block 起始 cycle
    # 排序配对（CAModel 每个 block 的首指令相对 block start 的 prologue 基本一致）；
    # 否则退化为“时间区间包含”，再退化为“起始 cycle 最近”。
    blist = sorted(blocks.values(), key=lambda b: b['start_cycle'])
    mapping = {}
    if len(dumps) == len(blist):
        def key(item):
            p, ins = item
            m = re.search(r'core(\d+)\.(\w+)core(\d+)', os.path.basename(p))
            return (ins[0][0], int(m.group(1)) if m else 0,
                    int(m.group(3)) if m else 0)
        for item, blk in zip(sorted(dumps, key=key), blist):
            mapping[item[0]] = blk
    else:
        for p, ins in dumps:
            lo, hi = ins[0][0], ins[-1][0]
            cand = [b for b in blist if b['start_cycle'] <= lo <= hi <= b['end_cycle']]
            if len(cand) == 1:
                mapping[p] = cand[0]
            elif cand:
                mapping[p] = min(cand, key=lambda b: abs(b['start_cycle'] - lo))
            elif blist:
                mapping[p] = min(blist, key=lambda b: abs(b['start_cycle'] - lo))

    rows = []
    for p, ins in dumps:
        base = os.path.basename(p)
        m = re.search(r'core(\d+)\.(\w+core)(\d+)', base)
        core = int(m.group(1)) if m else -1
        unit = (m.group(2) + m.group(3)) if m else base
        if core_filter is not None and core != core_filter:
            continue
        blk = mapping.get(p)
        if blk is None:
            continue

        bars = [c for c, pipe, name, _ in ins if pipe == 'ALL' and name.startswith('BAR')]
        span = blk['duration']
        first, last = ins[0][0], ins[-1][0]
        prologue = max(0, first - blk['start_cycle'])
        epilogue = max(0, blk['end_cycle'] - last)

        # 停顿归因：相邻两次发射之间的空档
        ccu = parse_ccu(os.path.join(dump_dir, f"core{core}.{unit}.ccu_log.dump"))
        icache = parse_miss_cycles(
            os.path.join(dump_dir, f"core{core}.{unit}.ifu.icache_log.dump"), ICACHE_MISS_RE)
        dcache = parse_miss_cycles(
            os.path.join(dump_dir, f"core{core}.{unit}.dcache_log.dump"), DCACHE_MISS_RE)
        stalls = Counter()
        for i in range(len(ins) - 1):
            a, b = ins[i][0], ins[i + 1][0]
            if b > a:
                kind, _ = classify_window(ccu, a, b, icache, dcache, ins[i + 1][2])
                stalls[kind] += b - a
        window = max(0, last - first)
        accounted = sum(stalls.values())
        if window > accounted:
            stalls['other'] += window - accounted

        pipes = {}
        for pipe in PIPE_ORDER:
            n = sum(1 for x in ins if x[1] == pipe)
            if not n:
                continue
            busy = pipe_busy(ins, pipe, bars)
            issue_cycles = len({x[0] for x in ins if x[1] == pipe})
            pipes[pipe] = {'n': n, 'busy': busy, 'issue_cycles': issue_cycles}

        rows.append({
            'dump': base,
            'name': f"core{core}/{'vc' if unit.startswith('veccore') else 'cc'}{unit[-1]}",
            'core': core, 'unit': unit,
            'block_id': blk['block_id'], 'span': span,
            'first': first, 'last': last,
            'prologue': prologue, 'epilogue': epilogue,
            'n_instr': len(ins), 'pipes': pipes,
            'stalls': dict(stalls), 'window': window,
        })
    return blocks, sorted(rows, key=lambda r: (r['core'], r['unit']))


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------
STALL_LABEL = {
    'cross_core': '跨核同步',
    'wait_flag': '等异步 flag',
    'raw': 'RAW 标量依赖',
    'ldst': '标量访存重试',
    'decode': '译码阻塞',
    'issue_block': '下游队列占用',
    'icache': 'ICache miss',
    'dcache': 'DCache miss',
    'barrier': '同步栅栏',
    'silent': '无事件(等管道)',
    'other': '其他/未归因',
}
STALL_ORDER = ['raw', 'wait_flag', 'issue_block', 'cross_core', 'dcache', 'icache',
               'decode', 'ldst', 'barrier', 'silent', 'other']

RED = '\033[31m'; GREEN = '\033[32m'; YELLOW = '\033[33m'; BOLD = '\033[1m'; NC = '\033[0m'


def pct(v, total):
    return (100.0 * v / total) if total else 0.0


def print_case(name, rows, verbose=True):
    if not rows:
        print(f'  {YELLOW}[WARN]{NC} {name}: 没有可解析的 dump')
        return
    span_max = max(r['span'] for r in rows)
    total = len(rows)
    nblk = len({r['block_id'] for r in rows})
    print(f'\n{BOLD}========== {name} =========={NC}')
    print(f'  veccore 数 {total}   block 数 {nblk}   kernel 跨度(max) {span_max} cycles')

    if verbose:
        hdr = f"  {'core':>10} {'blk':>4} {'span':>6} "
        hdr += ' '.join(f'{p:>16}' for p in ('SCALAR', 'VEC', 'MTE2', 'MTE3'))
        print(hdr)
        print("  口径: SCALAR=发射槽占用(1 cycle/发射)  VEC/MTE=队列占用(发射→下一条/同步)")
        print("         MTE2/MTE3 是 CAModel 口径，不代表真机带宽")
    for r in rows:
        if not verbose:
            continue
        cells = []
        for pipe in ('SCALAR', 'VEC', 'MTE2', 'MTE3'):
            d = r['pipes'].get(pipe)
            if not d:
                cells.append(f"{'-':>16}")
            elif pipe == 'SCALAR':
                cells.append(f"{d['issue_cycles']:>7} {pct(d['issue_cycles'], r['span']):>6.1f}%")
            else:
                cells.append(f"{d['busy']:>7} {pct(d['busy'], r['span']):>6.1f}%")
        print(f"  {r['name']:>10} {r['block_id']:>4} {r['span']:>6} " + ' '.join(cells))

    # 汇总（各 pipe 平均占比）
    agg = defaultdict(list)
    for r in rows:
        for pipe, d in r['pipes'].items():
            agg[pipe].append(pct(d['issue_cycles'] if pipe == 'SCALAR' else d['busy'], r['span']))
    print('\n  各 pipe 平均占比（分母 = 各自 block duration）:')
    for pipe in [p for p in PIPE_ORDER if p in agg]:
        vals = agg[pipe]
        note = '  ← CAModel 失真，别当真机带宽' if pipe in MOVE_PIPES else ''
        print(f'    {pipe:<7} {sum(vals)/len(vals):>6.1f}%  (min {min(vals):.1f} / max {max(vals):.1f}){note}')

    # 停顿归因（取最慢的核做代表，因为优化看的是长尾）
    worst = max(rows, key=lambda r: r['span'])
    span = worst['span']
    print(f"\n  停顿/开销归因（最慢核 {worst['name']} / block{worst['block_id']}，{span} cycles）:")
    items = [('prologue', worst['prologue'], '模型启动 → 首指令')]
    for k in STALL_ORDER:
        v = worst['stalls'].get(k, 0)
        if not v:
            continue
        items.append((k, v, STALL_LABEL.get(k, k)))
    items.append(('epilogue', worst['epilogue'], '末指令 → block 结束'))
    for k, v, label in items:
        bar = '#' * max(0, int(round(pct(v, span) / 2)))
        print(f'    {label:<14} {v:>6} {pct(v, span):>6.1f}%  {bar}')

    # 最慢核与平均核的差距
    if len(rows) > 1:
        avg = sum(r['span'] for r in rows) / len(rows)
        print(f"\n  负载均衡: 最慢 {span} / 平均 {avg:.0f} / 最快 {min(r['span'] for r in rows)}"
              f"  → 最长核慢 {pct(span - avg, avg):+.1f}%")


def main():
    ap = argparse.ArgumentParser(
        description='把 910B CAModel dump 解析成 pipe 占用/停顿归因（msprof 的替代品）',
        epilog='msprof 在这台机器上不可用：真机模式没卡；仿真模式的 libmsopprof_injection.so '
               '会让 aclrtSetDevice 报 507033（msopprof issue #159）。'
               'run_910b_ca.sh 直接跑 CAModel，本脚本负责把它吐的 dump 换算成指标。')
    ap.add_argument('path', help='log_ca 目录，或包含多个 case 的基准目录')
    ap.add_argument('--core', type=int, default=None, help='只看某个 core')
    ap.add_argument('--json', action='store_true', help='输出 JSON')
    ap.add_argument('--quiet', action='store_true', help='只打印汇总，不打印逐核表')
    args = ap.parse_args()

    root = args.path
    if os.path.isfile(os.path.join(root, 'profile_aiv_log0.toml')) or \
       os.path.isfile(os.path.join(root, 'profile_aic_log0.toml')):
        dirs = [root]
    else:
        dirs = sorted({os.path.dirname(p) for p in
                       glob.glob(os.path.join(root, '**', 'profile_a*_log0.toml'), recursive=True)})
    if not dirs:
        print(f'{RED}[FAIL]{NC} 在 {root} 下没找到 profile_a*_log0.toml', file=sys.stderr)
        return 2

    results = {}
    for d in dirs:
        name = os.path.basename(os.path.dirname(d)) or d
        _, rows = analyze_dump(d, args.core)
        results[name] = rows

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for name, rows in results.items():
        print_case(name, rows, verbose=not args.quiet)
    return 0


if __name__ == '__main__':
    sys.exit(main())
