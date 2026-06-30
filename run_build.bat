@echo off
:: =====================================================================
:: vLLM v0.23.0 Windows automated build runner
:: Sets up VS 2022 + venv + env vars, then calls build.bat.
:: Edit the paths below to match your install.
:: =====================================================================

setlocal enabledelayedexpansion

:: ----- Edit me ---------------------------------------------------------
set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "VENV_PATH=%~dp0venv"
:: CUDA 12.8 is the first toolkit with Blackwell (sm_120) support. 12.6 cannot
:: target sm_120; CUDA 13.x compiles but crashes MSVC -- do not use it.
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
:: Compute capability for your GPU: RTX 30xx=8.6, RTX 40xx=8.9, RTX 50xx=12.0
:: Multi-arch covers all three (drop arches you don't need for a faster build).
set "TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0"
:: Use 2, not more: the heavy 3-arch CUDA TUs intermittently crash cl.exe
:: (0xC000001D) under higher parallelism. Also do NOT enable sccache (same).
set "MAX_JOBS=2"
:: vLLM 0.23.0's optional Rust frontend needs protoc. Download protoc for
:: Windows and either edit this path or set PROTOC before running.
if not defined PROTOC set "PROTOC=%~dp0tools\protoc\bin\protoc.exe"
:: ----------------------------------------------------------------------

if not exist "%VS_VCVARS%" (
    echo [ERROR] vcvars64.bat not found at:
    echo         %VS_VCVARS%
    echo         Edit VS_VCVARS in this script.
    exit /b 1
)

call "%VS_VCVARS%"
if %ERRORLEVEL% neq 0 (
    echo Failed to initialize MSVC environment
    exit /b 1
)
echo MSVC environment initialized

if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    echo [WARN] No venv found at %VENV_PATH%, using system Python.
)

set "VLLM_TARGET_DEVICE=cuda"
set "CMAKE_BUILD_TYPE=Release"
set "VLLM_DISABLE_SCCACHE=1"
set "SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0"

if defined PROTOC if not exist "%PROTOC%" (
    echo [WARN] PROTOC not found at %PROTOC%
    echo        Rust frontend build will be skipped unless PROTOC points to protoc.exe.
    set "PROTOC="
)

call "%~dp0build.bat"

endlocal
