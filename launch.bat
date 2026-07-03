@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  vLLM Windows Server Launcher
echo  =============================
echo.

REM ============================================================
REM  Check Python installation
REM ============================================================
set "NEEDS_INSTALL=0"
if not exist "%~dp0python\python.exe" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\.torch-installed" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\.vllm-installed" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Include\Python.h" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\libs\python313.lib" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\torch" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\triton" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\vllm" set "NEEDS_INSTALL=1"

if "!NEEDS_INSTALL!"=="1" (
    echo  Python install is missing or incomplete. Running installer...
    echo.
    call "%~dp0install.bat"
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo  Installation failed. Please check errors above.
        pause
        exit /b 1
    )
    echo.
)

REM ============================================================
REM  Configure environment
REM ============================================================
set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0python\Library\bin;%PATH%"
if not defined CUDA_DEVICE_ORDER set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "TRITON_NVIDIA_DIR=%~dp0python\Lib\site-packages\triton\backends\nvidia"
if exist "%TRITON_NVIDIA_DIR%\bin\ptxas.exe" (
    set "CUDA_PATH=%TRITON_NVIDIA_DIR%"
    set "CUDA_HOME=%TRITON_NVIDIA_DIR%"
    set "PATH=%TRITON_NVIDIA_DIR%\bin;%PATH%"
)
set "VLLM_HOST_IP=127.0.0.1"

REM Suppress tokenizer parallelism warning
set "TOKENIZERS_PARALLELISM=false"

REM ============================================================
REM  Launch vLLM server
REM ============================================================
REM All arguments are forwarded to vllm_launcher.py.
REM If no --model is passed, the interactive model selector activates.

"%~dp0python\python.exe" "%~dp0vllm_launcher.py" %*

if !ERRORLEVEL! NEQ 0 (
    set "SERVER_EXIT=!ERRORLEVEL!"
    echo.
    echo  Server exited with error code !SERVER_EXIT!
    pause
    exit /b !SERVER_EXIT!
)

endlocal
exit /b 0
