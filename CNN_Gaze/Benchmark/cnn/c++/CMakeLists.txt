cmake_minimum_required(VERSION 3.16)
project(cnn_gaze_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable aggressive CPU optimizations (AVX2 / O3)
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W3)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -mavx2 -Wall)
endif()

# ── Dependencies ─────────────────────────────────────────────────────────────
# OpenCV (core, imgproc, highgui, videoio, dnn)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(benchmark_RTdetection benchmark_RTdetection.cpp)

target_include_directories(benchmark_RTdetection PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(benchmark_RTdetection PRIVATE
    ${OpenCV_LIBS}
)
