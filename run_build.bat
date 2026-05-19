@echo off
:: =====================================================================
:: vLLM v0.21.0 Windows automated build runner
:: Sets up VS 2022 + venv + env vars, then calls build.bat.
:: Edit the paths below to match your install.
:: =====================================================================

setlocal enabledelayedexpansion

:: ----- Edit me ---------------------------------------------------------
set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "VENV_PATH=%~dp0venv"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
:: Compute capability for your GPU: RTX 30xx=8.6, RTX 40xx=8.9, RTX 50xx=12.0
set "TORCH_CUDA_ARCH_LIST=8.6"
:: Parallel compile jobs (4 is safe with 32 GB RAM, 8 with 64 GB)
set "MAX_JOBS=4"
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
set "SETUPTOOLS_SCM_PRETEND_VERSION=0.21.0"

call "%~dp0build.bat"

endlocal
