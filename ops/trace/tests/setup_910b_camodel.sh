#!/bin/bash
# 把 910B1 (dav_2201) 伪装成 npusim 能识别的 camodel 目录
set -eux
C=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
S=$C/x86_64-linux/simulator/dav_2201/lib
CA=$C/x86_64-linux/simulator/dav_2201/camodel

mkdir -p "$CA"
cp -f "$S"/*.so "$S"/*.json "$CA"/ 2>/dev/null || true
# 关键 1：CANN 没给 910B 提供 libascend_hal.so，它其实就是 libnpu_drv.so 的别名
ln -sf "$CA/libnpu_drv.so" "$CA/libascend_hal.so"
# 关键 2：runtime 软链
ln -sf "$CA/libruntime_camodel.so" "$CA/libruntime.so"

# 关键 3：把 Ascend910B1 注册进 npusim 的 SoC 表
PY=$C/python/site-packages/cannsim/core/public/soc_info.py
grep -q '"Ascend910B1"' "$PY" || python3 - "$PY" <<'PYEOF'
import sys, re
p = sys.argv[1]
s = open(p).read()
s = s.replace(
    'SOC_INFO: Dict[str, SocSpec] = {\n',
    'SOC_INFO: Dict[str, SocSpec] = {\n    "Ascend910B1": {\n        "max_cores": 24,\n    },\n'
)
open(p, 'w').write(s)
PYEOF

python3 -c "from cannsim.core.public.soc_info import SOC_INFO; print('SOC_INFO:', SOC_INFO)"
ls -l "$CA/libascend_hal.so" "$CA/libruntime.so"
echo "[OK] 910B1 camodel 目录就绪: $CA"
