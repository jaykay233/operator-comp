#
# ascend_host_arch.cmake
#
# 目的：统一解析 CANN 安装目录下的“host 架构子目录”名称。
#
# CANN toolkit 的 host 侧头文件 / 库放在 <CANN_HOME>/<arch>-linux 下，
# 其中 <arch> 随安装包架构变化：
#     x86_64 安装包 -> ${CANN_HOME}/x86_64-linux
#     aarch64 安装包 -> ${CANN_HOME}/aarch64-linux
#
# 之前各 CMakeLists 里直接写死 aarch64-linux，导致 x86_64 环境下找不到头文件和库。
# 这里按“主机架构推断 + 目录实际探测”解析一次，各 CMakeLists 统一使用
# ${ASCEND_HOST_ARCH_DIR}。
#
# 输入（可选）：
#   CANN_HOME            —— 若未定义，则回退取环境变量 ASCEND_HOME_PATH
#   ASCEND_HOST_ARCH_DIR —— 若已定义且目录存在，则直接复用，不再重复探测
#
# 输出：
#   ASCEND_HOST_ARCH_DIR —— 如 /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux
#                           探测不到时为空字符串（由调用方决定如何处理）
#

if(DEFINED ASCEND_HOST_ARCH_DIR AND NOT "${ASCEND_HOST_ARCH_DIR}" STREQUAL ""
   AND EXISTS "${ASCEND_HOST_ARCH_DIR}")
    return()
endif()

if(NOT DEFINED CANN_HOME OR "${CANN_HOME}" STREQUAL "")
    if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(CANN_HOME "$ENV{ASCEND_HOME_PATH}")
    endif()
endif()

if("${CANN_HOME}" STREQUAL "")
    set(ASCEND_HOST_ARCH_DIR "")
    message(WARNING "ascend_host_arch: CANN_HOME/ASCEND_HOME_PATH 均未设置，无法定位 CANN 安装目录")
    return()
endif()

# 1) 按主机处理器架构推断首选目录名
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|armv8.*)$")
    set(_ascend_arch_dir_name "aarch64-linux")
else()
    set(_ascend_arch_dir_name "x86_64-linux")
endif()

# 2) 目录不存在时，扫描 <CANN_HOME>/*-linux 兜底（例如交叉安装）
set(ASCEND_HOST_ARCH_DIR "")
if(EXISTS "${CANN_HOME}/${_ascend_arch_dir_name}")
    set(ASCEND_HOST_ARCH_DIR "${CANN_HOME}/${_ascend_arch_dir_name}")
else()
    file(GLOB _ascend_arch_candidates "${CANN_HOME}/*-linux")
    foreach(_ascend_cand ${_ascend_arch_candidates})
        if(IS_DIRECTORY "${_ascend_cand}" AND EXISTS "${_ascend_cand}/include")
            set(ASCEND_HOST_ARCH_DIR "${_ascend_cand}")
            break()
        endif()
    endforeach()
endif()

if("${ASCEND_HOST_ARCH_DIR}" STREQUAL "")
    message(WARNING
        "ascend_host_arch: 未在 '${CANN_HOME}' 下找到 host 架构目录"
        "（期望 ${_ascend_arch_dir_name}，或任意 *-linux），"
        "请确认 CANN toolkit 安装完整、ASCEND_HOME_PATH 指向正确")
else()
    message(STATUS "ascend_host_arch: ASCEND_HOME_PATH=${CANN_HOME}")
    message(STATUS "ascend_host_arch: ASCEND_HOST_ARCH_DIR=${ASCEND_HOST_ARCH_DIR}")
endif()

# ---------------------------------------------------------------------------
# 附带解析 tikicpulib 的 cmake 配置目录
#
# 该路径随 CANN 版本变化过：
#   CANN 8.x        -> <CANN_HOME>/tools/tikicpulib/lib/cmake
#   CANN 9.x        -> <CANN_HOME>/toolkit/tools/tikicpulib/lib/cmake
# 两个位置都探一遍，避免再次踩硬编码的坑。
# ---------------------------------------------------------------------------
if(DEFINED TIKICPULIB_CMAKE_DIR AND NOT "${TIKICPULIB_CMAKE_DIR}" STREQUAL ""
   AND EXISTS "${TIKICPULIB_CMAKE_DIR}/tikicpulib-config.cmake")
    return()
endif()

set(TIKICPULIB_CMAKE_DIR "")
foreach(_tikicpulib_cand
        "${CANN_HOME}/toolkit/tools/tikicpulib/lib/cmake"
        "${CANN_HOME}/tools/tikicpulib/lib/cmake")
    if(EXISTS "${_tikicpulib_cand}/tikicpulib-config.cmake")
        set(TIKICPULIB_CMAKE_DIR "${_tikicpulib_cand}")
        break()
    endif()
endforeach()

if("${TIKICPULIB_CMAKE_DIR}" STREQUAL "")
    message(WARNING "ascend_host_arch: 未找到 tikicpulib-config.cmake，CPU 侧 kernel UT 可能无法链接")
else()
    message(STATUS "ascend_host_arch: TIKICPULIB_CMAKE_DIR=${TIKICPULIB_CMAKE_DIR}")
endif()
