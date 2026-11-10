# ---------------------------------------------------------------------------
# ascendc-cpu-dev:cann92
#
# 用途：AscendC 自定义算子（ops/trace）在 **CPU 侧** 的开发 / 编译 / UT / 仿真环境：
#   - CANN toolkit 9.2.0-beta.1（x86_64，编译器 + tikicpulib + 头文件 + 库）
#   - CANN ops 算子包 9.2.0-beta.1（libopapi / libnnopbase / 910B 内置算子）
#   - CAModel 周期精确模型，含 910B：
#       * tikicpulib           → 逐指令 CPU 仿真，跑 UT
#       * msprof op simulator  → cycle 级性能仿真，--soc-version=Ascend910B1
#       * npusim               → 新工具，目前仅支持 Ascend950/Ascend950DT
#   - gcc / cmake / make / python3 等构建依赖
#   - numpy / ml_dtypes / torch(CPU) —— UT 的 gen_data.py / compare_data.py 需要
#
# 说明：
#   ARM（aarch64）包与本镜像不兼容。本构建机为 x86_64，故使用 x86_64 包；
#   如需 ARM 镜像，换用 Ascend-cann-toolkit_9.2.0-beta.1_linux-aarch64.run 并在
#   ARM 主机（或 qemu-user 交叉环境）上构建。
#
# 构建：
#   mkdir -p cann-pkgs
#   cp Ascend-cann-toolkit_9.2.0-beta.1_linux-x86_64.run cann-pkgs/
#   cp Ascend-cann-910b-ops_9.2.0-beta.1_linux-x86_64.run  cann-pkgs/
#   docker build -t ascendc-cpu-dev:cann92 .
#
# 运行：
#   docker run --rm -it -v "$PWD:/workspace" ascendc-cpu-dev:cann92
#   # 容器内：
#   cd /workspace/ops/trace && bash build.sh          # 编译算子包
#   cd /workspace/ops/trace && bash build.sh -u       # 跑 UT（tikicpulib）
#   # cycles 仿真（910B）：
#   msprof op simulator --soc-version=Ascend910B1 \
#       --application=./your_aclnn_app --output=./sim_out
# ---------------------------------------------------------------------------
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# CANN run 包文件名与安装位置
ARG CANN_RUN=Ascend-cann-toolkit_9.2.0-beta.1_linux-x86_64.run
# ops 算子包：提供 libopapi.so / libnnopbase.so 与 910B 内置算子，
# aclnn 应用与「算子级性能仿真」都依赖它（toolkit 不含）
ARG CANN_OPS_RUN=Ascend-cann-910b-ops_9.2.0-beta.1_linux-x86_64.run
ARG CANN_INSTALL_PATH=/usr/local/Ascend

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai

# ---------------------------------------------------------------------------
# 1. 系统依赖
#    build-essential / cmake / python3 是 CANN 与算子工程的硬依赖；
#    gdb + vim 等便于调试；libgomp1 供 tikicpulib 多线程模拟使用。
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        wget \
        ca-certificates \
        python3 \
        python3-dev \
        python3-pip \
        gawk \
        unzip \
        zip \
        xz-utils \
        file \
        pkg-config \
        dos2unix \
        rsync \
        openssh-client \
        bash-completion \
        libgomp1 \
        libssl-dev \
        zlib1g-dev \
        nlohmann-json3-dev \
        gdb \
        vim \
        less \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Python 依赖（UT 数据生成 / 精度比对）
#    torch 走 aliyun 的 CPU 版 wheel 镜像，避免拉取带 CUDA 依赖的巨包。
# ---------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir --upgrade pip \
        -i https://repo.huaweicloud.com/repository/pypi/simple \
    && python3 -m pip install --no-cache-dir \
        numpy ml_dtypes pytest \
        -i https://repo.huaweicloud.com/repository/pypi/simple \
    && python3 -m pip install --no-cache-dir \
        --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu \
        -i https://repo.huaweicloud.com/repository/pypi/simple \
        torch \
    && python3 -c "import numpy, ml_dtypes, torch; print('torch', torch.__version__, 'numpy', numpy.__version__)"

# npusim report 渲染 html/svg 需要 plotly + pandas。
# 单独成层：CANN 安装层很贵（约 5 分钟），不能被这种小改动击穿缓存。
RUN python3 -m pip install --no-cache-dir plotly pandas \
        -i https://repo.huaweicloud.com/repository/pypi/simple \
    && python3 -c "import plotly, pandas; print('[OK] plotly', plotly.__version__, 'pandas', pandas.__version__)"

# ---------------------------------------------------------------------------
# 3. 安装 CANN toolkit + ops 算子包
#    --install --quiet：非交互安装；--install-path 显式指定，避免默认装到 $HOME。
#    ops 包必须在 toolkit 之后装（它往 toolkit 目录里补算子库）。
# ---------------------------------------------------------------------------
COPY cann-pkgs/${CANN_RUN} /tmp/cann-pkg/${CANN_RUN}
COPY cann-pkgs/${CANN_OPS_RUN} /tmp/cann-pkg/${CANN_OPS_RUN}

RUN set -eux; \
    chmod +x /tmp/cann-pkg/*.run; \
    cd /tmp/cann-pkg; \
    ./${CANN_RUN} --install --quiet --install-path=${CANN_INSTALL_PATH}; \
    ./${CANN_OPS_RUN} --install --quiet --install-path=${CANN_INSTALL_PATH}; \
    rm -rf /tmp/cann-pkg; \
    ls -l ${CANN_INSTALL_PATH}; \
    ls -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest/*-linux

# ---------------------------------------------------------------------------
# 4. 环境变量
#
# 说明：CANN 9.x 的安装布局是 /usr/local/Ascend/cann-<version>，
#       并保留稳定软链 /usr/local/Ascend/cann 与 <base>/ascend-toolkit/latest。
#       这里统一用 cann（软链），版本升级时无需改 Dockerfile。
#
# CMAKE_PREFIX_PATH 必须显式设置，否则算子工程里的
#       find_package(ASC REQUIRED)        -> lib64/cmake/asc-config.cmake
#       find_package(tikicpulib REQUIRED) -> toolkit/tools/tikicpulib/lib/cmake
# 都会找不到（官方 set_env.sh 也是这么设的）。
# ---------------------------------------------------------------------------
ENV ASCEND_INSTALL_PATH=/usr/local/Ascend \
    ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann \
    ASCEND_HOME_PATH=/usr/local/Ascend/cann \
    CMAKE_PREFIX_PATH=/usr/local/Ascend/cann/lib64/cmake:/usr/local/Ascend/cann/toolkit/tools/tikicpulib/lib/cmake \
    LD_LIBRARY_PATH=/usr/local/Ascend/cann/lib64:/usr/local/Ascend/cann/lib64/plugin/opskernel:/usr/local/Ascend/cann/lib64/plugin/nnengine:/usr/local/Ascend/cann/x86_64-linux/lib64:/usr/local/Ascend/cann/devlib \
    PYTHONPATH=/usr/local/Ascend/cann/python/site-packages \
    PATH=/usr/local/Ascend/cann/bin:/usr/local/Ascend/cann/tools/ccec_compiler/bin:/usr/local/Ascend/cann/tools/profiler/bin:/usr/local/Ascend/cann/tools/msobjdump:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN printf '%s\n' \
        '# CANN toolkit 环境（官方 set_env.sh，登录 shell 生效）' \
        '[ -f /usr/local/Ascend/cann/set_env.sh ] && source /usr/local/Ascend/cann/set_env.sh' \
        > /etc/profile.d/ascend-toolkit.sh \
    && chmod +x /etc/profile.d/ascend-toolkit.sh

# 自检：确认架构目录 / 编译器 / tikicpulib 都在位（失败则 build 立即报错）
RUN test -d "${ASCEND_HOME_PATH}/x86_64-linux/include" \
    && test -d "${ASCEND_HOME_PATH}/compiler/ascendc/include" \
    && test -f "${ASCEND_HOME_PATH}/lib64/libmmpa.so" \
    && test -f "${ASCEND_HOME_PATH}/lib64/cmake/asc-config.cmake" \
    && test -f "${ASCEND_HOME_PATH}/toolkit/tools/tikicpulib/lib/cmake/tikicpulib-config.cmake" \
    && echo "[OK] CANN toolkit ready: $(readlink -f ${ASCEND_HOME_PATH})"

# ---------------------------------------------------------------------------
# 5. CAModel（Cycle-Accurate Model）自检
#
# CAModel 由 toolkit 内的子包 cann-simulator 提供。安装时 get_install_mode
# 因检测不到 NPU 驱动而判定为 local（开发/编译）环境，cann-simulator 按
# "--full&--pylocal" 安装，因此上面的 --install 已经把它装好了。
#
# 安装后布局：
#   ${ASCEND_HOME_PATH}/x86_64-linux/simulator/<SoC>/lib/      性能仿真（PV）模型
#   ${ASCEND_HOME_PATH}/x86_64-linux/simulator/dav_*/camodel/  周期精确（CA）模型
#   ${ASCEND_HOME_PATH}/python/site-packages/cannsim           Python 驱动
#   ${ASCEND_HOME_PATH}/bin/{cannsim,npusim}                   命令行入口
#
# 【重要结论 —— 决定了周期仿真怎么跑】
#
#   dav_3510（= Ascend950PR 系列）目录里带 camodel/，是完整硅模型
#   （libSoC.so / libEslTop.so / libHBMSim.so / libdramsim.so / libascend_hal.so
#     / camodel_v100.json 等，48 个 .so，93MB）。
#
#   dav_2201（= Ascend910B1/B2/B3/B4、910_93xx）**没有 camodel/ 子目录**，
#   模型在 lib/ 下，是另一代架构：
#       libruntime_camodel.so + libnpu_drv_camodel.so + libstars.so + libmodel_top.so
#   跑起来会打印 "PEM MODEL ... Model Init Success"，并输出 Total tick。
#
#   ★ 910B 默认跑不起来的原因：缺 libascend_hal.so
#     CANN 没给 910B 提供这个 HAL 壳，而 dav_2201/lib/libnpu_drv.so 自己就
#     导出了全部 114 个 hal* 符号。补一个软链即可：
#         ln -sf libnpu_drv.so libascend_hal.so
#     补完之后 910B1 的 CA 模型可以直接运行，并吐出逐核周期数据。
#
#   ★ msprof op simulator 这条路走不通（与 HAL 无关）
#     它的 LD_PRELOAD 注入库 libmsopprof_injection.so 自身有缺陷：单独
#     preload 就会让 aclrtSetDevice 报 507033，与 910B/950PR 无关。
#     官方 issue: gitcode.com/Ascend/msopprof/issues/159（CANN 9.2.0-beta.1）
#
#   ★ npusim 的 SoC 表只注册了 Ascend950/Ascend950DT，所以下面顺手把
#     Ascend910B1 补进去（cannsim/core/public/soc_info.py 的 SOC_INFO）。
#
#   可用的两个一键脚本：
#     ops/trace/tests/run_910b_ca.sh   ← 910B1 周期仿真（推荐，duration 可复现）
#     ops/trace/tests/run_ca_sim.sh    ← 950 CA + npusim report（有流水/IPC 报告）
# ---------------------------------------------------------------------------
RUN set -eux; \
    SIM=${ASCEND_HOME_PATH}/x86_64-linux/simulator; \
    test -d "${SIM}" \
        || { echo "[ERROR] simulator 目录缺失，CAModel 未安装"; exit 1; }; \
    test -f "${SIM}/dav_3510/camodel/libSoC.so" \
        && test -f "${SIM}/dav_3510/camodel/libEslTop.so" \
        && test -f "${SIM}/dav_3510/camodel/libascend_hal.so" \
        && test -f "${SIM}/dav_3510/camodel/camodel_v100.json" \
        || { echo "[ERROR] Ascend950PR CA 模型不完整"; exit 1; }; \
    echo "[OK] CA 模型就绪：dav_3510 (Ascend950PR)"; \
    test -f "${SIM}/dav_2201/lib/libruntime_camodel.so" \
        && test -f "${SIM}/dav_2201/lib/libnpu_drv_camodel.so" \
        && test -f "${SIM}/dav_2201/lib/libstars.so" \
        && test -f "${SIM}/dav_2201/lib/libmodel_top.so" \
        || { echo "[ERROR] Ascend910B CA 模型不完整"; exit 1; }; \
    echo "[OK] CA 模型就绪：dav_2201 (Ascend910B1)"; \
    if ! python3 -c "import cannsim" 2>/dev/null; then \
        echo "[WARN] cannsim 未进入 site-packages，手动补装"; \
        pip3 install --no-deps --force-reinstall --disable-pip-version-check \
            "${SIM}/bin/cannsim-0.1.0-py3-none-any.whl"; \
    fi; \
    python3 -c "import cannsim; print('[OK] cannsim ->', cannsim.__file__)"; \
    command -v npusim >/dev/null && echo "[OK] npusim 可用"

# ---------------------------------------------------------------------------
# 5b. 910B CAModel 开箱可用化
#
#    (1) 建 camodel/ 目录并补 libascend_hal.so -> libnpu_drv.so
#        （CANN 没给 910B 提供 HAL 壳；这是 910B CA 仿真的唯一硬性缺口）
#    (2) 把 Ascend910B1 注册进 npusim 的 SOC_INFO，让 npusim 也认 910B
#
#    做完后 910B1 的 CA 模型可以直接跑出周期数据：
#        [CHECK] PASS
#        [INFO] Total tick: 13992
#      profile_aiv_log0.toml 里每个 block 的 duration 可复现（逐次一致）。
# ---------------------------------------------------------------------------
RUN set -eux; \
    SIM=${ASCEND_HOME_PATH}/x86_64-linux/simulator; \
    LIB=${SIM}/dav_2201/lib; \
    CA=${SIM}/dav_2201/camodel; \
    mkdir -p "${CA}"; \
    cp -f "${LIB}"/*.so "${LIB}"/*.json "${CA}"/; \
    ln -sf "${CA}/libnpu_drv.so" "${CA}/libascend_hal.so"; \
    ln -sf "${CA}/libruntime_camodel.so" "${CA}/libruntime.so"; \
    test -e "${CA}/libascend_hal.so"; \
    echo "[OK] 910B camodel/ 就绪，libascend_hal.so -> libnpu_drv.so"; \
    SOCPY=${ASCEND_HOME_PATH}/python/site-packages/cannsim/core/public/soc_info.py; \
    if ! grep -q '"Ascend910B1"' "${SOCPY}"; then \
        sed -i '/^SOC_INFO: Dict\[str, SocSpec\] = {$/a\    "Ascend910B1": {\n        "max_cores": 24,\n    },' "${SOCPY}"; \
    fi; \
    python3 -c "from cannsim.core.public.soc_info import SOC_INFO; assert 'Ascend910B1' in SOC_INFO, SOC_INFO; print('[OK] npusim SOC_INFO 已含 Ascend910B1 ->', SOC_INFO['Ascend910B1'])"

# ---------------------------------------------------------------------------
# 6. ops 算子包自检：libopapi / libnnopbase 是 aclnn 应用与算子仿真的前提
# ---------------------------------------------------------------------------
RUN test -f "${ASCEND_HOME_PATH}/lib64/libopapi.so" \
    && test -f "${ASCEND_HOME_PATH}/lib64/libnnopbase.so" \
    && test -d "${ASCEND_HOME_PATH}/opp/built-in/op_impl" \
    && echo "[OK] ops 算子包就绪（opapi/nnopbase/内置算子）"

WORKDIR /workspace

CMD ["/bin/bash"]
