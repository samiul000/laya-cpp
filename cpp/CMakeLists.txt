cmake_minimum_required(VERSION 3.20)
project(laya_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime from environment variable set in Docker
set(ORT_HOME "$ENV{ORT_HOME}")
if(NOT ORT_HOME)
    message(FATAL_ERROR "ORT_HOME environment variable is not set!")
endif()

include_directories(
    include
    ${ORT_HOME}/include
)

link_directories(${ORT_HOME}/lib)

# 1. Build CLI & Benchmark Runner
add_executable(laya_cli
    src/laya.cpp
    src/main.cpp
)
target_link_libraries(laya_cli PRIVATE onnxruntime)

# Embed RPATH so the executable finds libonnxruntime.so without configuration
set_target_properties(laya_cli PROPERTIES
    BUILD_RPATH "${ORT_HOME}/lib"
    INSTALL_RPATH "${ORT_HOME}/lib"
)

# 3. Enable CTest
enable_testing()
add_test(NAME SanityTest COMMAND laya_cli "/workspace/models/laya.onnx" 2)