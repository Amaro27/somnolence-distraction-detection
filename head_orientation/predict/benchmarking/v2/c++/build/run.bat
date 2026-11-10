@echo off
setlocal
set PATH=C:\msys64\ucrt64\bin;%PATH%
"%~dp0benchmark_head_pose.exe" %*
