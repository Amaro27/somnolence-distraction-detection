cmake_minimum_required(VERSION 3.16)
project(modular_distraction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization and SIMD flags based on architecture
if(MSVC)
    add_compile_options(/O2 /W3)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-O3 -Wall -Wextra)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        add_compile_options(-mavx2)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # 64-bit ARM (Raspberry Pi OS 64-bit) enables NEON automatically with -O3
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l|armhf")
        add_compile_options(-mfpu=neon-vfpv4 -mfloat-abi=hard)
    endif()
endif()

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio dnn)
find_package(dlib REQUIRED)

set(MODULAR_DISTRACTION_SOURCES
    src/CameraCapture.cpp
    src/Preprocessor.cpp
    src/GazeDetector.cpp
    src/AlertLogic.cpp
    src/AlertManager.cpp
    src/ConsoleController.cpp
    src/FaceLandmarkDetector.cpp
    src/FaceGeometry.cpp
    src/HeadOrientationDetector.cpp
    src/EarDetector.cpp
    src/MarDetector.cpp
    src/AudioManager.cpp
    src/Calibration.cpp
    src/SoundPlayer.cpp
    src/PipelineCoordinator.cpp
    src/main.cpp
)

add_executable(modular_distraction ${MODULAR_DISTRACTION_SOURCES})

target_include_directories(modular_distraction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(modular_distraction PRIVATE
    ${OpenCV_LIBS}
    dlib::dlib
    Threads::Threads
)

if(WIN32)
    target_link_libraries(modular_distraction PRIVATE winmm)
endif()

# Ensure input/ directory is mirrored to build output directory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/input)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()
