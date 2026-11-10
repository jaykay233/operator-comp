set(OP_NAME trace)
set(UT_EXE ${OP_NAME}_op_kernel_ut)

message(STATUS "Building op_kernel UT for: ${OP_NAME}")

if(DEFINED ENV{ASCEND_HOME_PATH} AND EXISTS "$ENV{ASCEND_HOME_PATH}")
    set(CANN_HOME "$ENV{ASCEND_HOME_PATH}")
elseif(EXISTS "/usr/local/Ascend/cann-8.5.0")
    set(CANN_HOME "/usr/local/Ascend/cann-8.5.0")
else()
    message(FATAL_ERROR "Cannot find CANN installation")
endif()

message(STATUS "Using CANN_HOME: ${CANN_HOME}")

# 解析 CANN host 架构目录（x86_64-linux / aarch64-linux），统一用 ${ASCEND_HOST_ARCH_DIR}
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/ascend_host_arch.cmake)

set(SOC_VERSION "Ascend910B1")

set(tikicpulib_DIR ${TIKICPULIB_CMAKE_DIR})
find_package(tikicpulib REQUIRED)

set(OP_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel)

set(UT_TEST_SRCS
    test_${OP_NAME}.cpp
    test_${OP_NAME}_verify.cpp
)

set(UT_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OP_KERNEL_DIR}
    ${UT_COMMON_INCLUDE_DIRS}
    ${CANN_HOME}/toolkit/tools/tikicpulib/lib/include
    ${CANN_HOME}/compiler/ascendc/include/highlevel_api
    ${CANN_HOME}/compiler/ascendc/include/basic_api/impl
    ${CANN_HOME}/compiler/ascendc/include/basic_api
    ${CANN_HOME}/compiler/ascendc/include
    ${CANN_HOME}/compiler/ascendc/include/ascendc/host_api/tiling
    ${CANN_HOME}/compiler/ascendc/include/ascendc/host_api
    ${CANN_HOME}/compiler/ascendc/include/ascendc
    ${ASCEND_HOST_ARCH_DIR}/include
    ${ASCEND_HOST_ARCH_DIR}/asc/include/basic_api
    ${ASCEND_HOST_ARCH_DIR}/asc/include
    ${ASCEND_HOST_ARCH_DIR}/asc/include/interface
    ${ASCEND_HOST_ARCH_DIR}/asc/impl/basic_api
    ${ASCEND_HOST_ARCH_DIR}/asc
    ${CANN_HOME}/compiler/tikcpp/tikcfw
    ${CANN_HOME}/compiler/tikcpp/tikcfw/impl
    ${CANN_HOME}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOST_ARCH_DIR}/include/graph
    ${ASCEND_HOST_ARCH_DIR}/ascendc/include/highlevel_api
    ${ASCEND_HOST_ARCH_DIR}/asc/impl/basic_api/utils
)

add_executable(${UT_EXE}
    ${UT_TEST_SRCS}
)

target_include_directories(${UT_EXE} PRIVATE
    ${UT_INCLUDE_DIRS}
)

target_compile_definitions(${UT_EXE} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    __aicore__=
    __NPU_TILING__=1
    ASCENDC_CPU_DEBUG=1
)

target_compile_options(${UT_EXE} PRIVATE
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -Wno-unknown-pragmas
    -fno-access-control
    -std=c++17
)

target_link_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_LIB_DIRS}
    ${CANN_HOME}/lib64
)

target_link_libraries(${UT_EXE} PRIVATE
    gtest_main
    gtest
    ${CANN_HOME}/lib64/libmmpa.so
    tikicpulib::${SOC_VERSION}
    dl
    pthread
)

set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;${CANN_HOME}/lib64;${CANN_HOME}/toolkit/tools/tikicpulib/lib;${CANN_HOME}/toolkit/tools/tikicpulib/lib/${SOC_VERSION};${ASCEND_HOST_ARCH_DIR}/simulator/${SOC_VERSION}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

message(STATUS "  Kernel UT executable: ${UT_EXE}")
message(STATUS "  CANN Home: ${CANN_HOME}")
message(STATUS "  SOC Version: ${SOC_VERSION}")
