cmake_minimum_required(VERSION 3.16)
project(lstm_gaze_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization and SIMD flags based on architecture
if(MSVC)
    add_compile_options(/O2 /W3)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-O3 -Wall)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        add_compile_options(-mavx2)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # 64-bit ARM (Raspberry Pi OS 64-bit) enables NEON automatically with -O3
        # Optional: fine-tune for native CPU if compiling directly on the Pi
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l|armhf")
        # 32-bit ARM (Raspberry Pi OS 32-bit)
        add_compile_options(-mfpu=neon-vfpv4 -mfloat-abi=hard)
    endif()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

add_executable(benchmark_RTdetection_lstm benchmark_RTdetection_lstm.cpp)
target_include_directories(benchmark_RTdetection_lstm PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(benchmark_RTdetection_lstm PRIVATE ${OpenCV_LIBS})
