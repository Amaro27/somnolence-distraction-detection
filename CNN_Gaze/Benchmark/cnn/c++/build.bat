@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo Building C++ CNN Gaze Benchmark (LightGazeCNN)
echo =======================================================

set BUILD_DIR=%~dp0build
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

set CMAKE_OPTS=-DCMAKE_BUILD_TYPE=Release

:: Auto-detect MSYS2 UCRT64 toolchain and libraries
if exist "C:\msys64\ucrt64" (
    echo [INFO] Detected MSYS2 UCRT64 environment.
    set PATH=C:\msys64\ucrt64\bin;!PATH!
    set CMAKE_OPTS=!CMAKE_OPTS! -DCMAKE_PREFIX_PATH="C:/msys64/ucrt64"
    
    where.exe ninja >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set CMAKE_OPTS=!CMAKE_OPTS! -G Ninja
    ) else (
        where.exe mingw32-make >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            set CMAKE_OPTS=!CMAKE_OPTS! -G "MinGW Makefiles"
        )
    )
)

echo [INFO] Running CMake configuration: cmake .. %CMAKE_OPTS%
cmake .. %CMAKE_OPTS%
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] CMake configuration failed.
    echo Please ensure OpenCV with DNN module is installed.
    exit /b %ERRORLEVEL%
)

echo [INFO] Compiling benchmark_RTdetection executable...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Compilation failed.
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================
echo [SUCCESS] Build finished successfully!
echo Executable located at: %BUILD_DIR%\benchmark_RTdetection.exe
echo =======================================================
