cmake_minimum_required(VERSION 3.16.0)
project(trace_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")

# 解析 CANN host 架构目录（x86_64-linux / aarch64-linux），统一用 ${ASCEND_HOST_ARCH_DIR}
set(CANN_HOME "$ENV{ASCEND_HOME_PATH}")
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/ascend_host_arch.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)

# DTYPE 宏定义（根据算子原型和测试用例生成）
add_definitions(
    -DDTYPE_SELF=float
    -DDTYPE_OUT=float
)

set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    ${ASCEND_HOST_ARCH_DIR}/include
    ${ASCEND_HOST_ARCH_DIR}/include/base
    ${ASCEND_HOST_ARCH_DIR}/include/base/context_builder
    ${ASCEND_HOST_ARCH_DIR}/asc/include
    ${ASCEND_HOST_ARCH_DIR}/asc/include/tiling
    ${ASCEND_HOST_ARCH_DIR}/pkg_inc
    ${ASCEND_HOST_ARCH_DIR}/pkg_inc/op_common
    ${ASCEND_HOST_ARCH_DIR}/pkg_inc/base
    ${ASCEND_HOST_ARCH_DIR}/pkg_inc/exe_graph
    ${ASCEND_HOST_ARCH_DIR}/pkg_inc/graph
)

set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOST_ARCH_DIR}/lib64
)

add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(op_host)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../op_api)
    add_subdirectory(op_api)
endif()
add_subdirectory(op_kernel)
