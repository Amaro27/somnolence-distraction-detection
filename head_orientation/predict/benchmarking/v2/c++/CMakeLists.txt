cmake_minimum_required(VERSION 3.16)
project(head_pose_benchmark_v2 LANGUAGES CXX)

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
# 1. OpenCV (core, imgproc, highgui, videoio)
find_package(OpenCV REQUIRED)

# 2. Dlib (image_processing, frontal_face_detector)
find_package(dlib REQUIRED)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(benchmark_head_pose benchmark_head_pose.cpp)

target_include_directories(benchmark_head_pose PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${dlib_INCLUDE_DIRS}
)

target_link_libraries(benchmark_head_pose PRIVATE
    ${OpenCV_LIBS}
    dlib::dlib
)
