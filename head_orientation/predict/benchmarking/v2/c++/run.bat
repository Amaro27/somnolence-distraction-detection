@echo off
setlocal
set PATH=C:\msys64\ucrt64\bin;%PATH%
"%~dp0build\benchmark_head_pose.exe" %*
