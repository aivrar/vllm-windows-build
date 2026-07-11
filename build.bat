@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: vLLM v0.24.0 Windows Build Script
:: Compiles vLLM from patched source with MSVC + CUDA + Ninja
:: ============================================================

echo.
echo  vLLM v0.24.0 Windows Build
echo  ==========================
echo.

:: -----------------------------------------------------------
:: 1. Check prerequisites
:: -----------------------------------------------------------

where cl.exe >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] cl.exe not found. Run this from a Visual Studio Developer Command Prompt
    echo         or run vcvars64.bat first:
    echo         "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    exit /b 1
)

if not defined CUDA_HOME (
    echo [ERROR] CUDA_HOME is not set. Point it at your CUDA toolkit, e.g.:
    echo         set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
    exit /b 1
)

if not exist "%CUDA_HOME%\bin\nvcc.exe" (
    echo [ERROR] nvcc.exe not found at %CUDA_HOME%\bin\nvcc.exe
    echo         Make sure CUDA_HOME points to a CUDA 12.8 install (12.8 is the
    echo         first toolkit with Blackwell sm_120; CUDA 13.x crashes MSVC).
    exit /b 1
)

:: -----------------------------------------------------------
:: 2. Configuration (edit these for your system)
:: -----------------------------------------------------------

:: Compute capability — multi-arch covers all (drop ones you don't need):
::   RTX 30xx = 8.6, RTX 40xx = 8.9, RTX 50xx = 12.0
if not defined TORCH_CUDA_ARCH_LIST set TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0

:: Parallel compile jobs. Keep at 2: higher parallelism intermittently crashes
:: MSVC cl.exe (0xC000001D) on the heavy 3-arch CUDA TUs. Do not enable sccache.
if not defined MAX_JOBS set MAX_JOBS=2

set VLLM_TARGET_DEVICE=cuda
set SETUPTOOLS_SCM_PRETEND_VERSION=0.24.0

:: -----------------------------------------------------------
:: 3. Locate vllm source
:: -----------------------------------------------------------

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%vllm-source\setup.py" (
    set "VLLM_SRC=%SCRIPT_DIR%vllm-source"
) else if exist "%SCRIPT_DIR%setup.py" (
    set "VLLM_SRC=%SCRIPT_DIR%"
) else (
    echo [ERROR] Cannot find vLLM source. Clone it into vllm-source\ next to this script:
    echo         git clone https://github.com/vllm-project/vllm.git vllm-source
    echo         cd vllm-source ^&^& git checkout v0.24.0
    exit /b 1
)

:: -----------------------------------------------------------
:: 4. Apply patch
:: -----------------------------------------------------------

if exist "%SCRIPT_DIR%vllm-windows-v8.patch" (
    pushd "%VLLM_SRC%"
    git diff --quiet HEAD 2>nul
    if !ERRORLEVEL! equ 0 (
        echo Applying vllm-windows-v8.patch...
        git apply "%SCRIPT_DIR%vllm-windows-v8.patch"
        if !ERRORLEVEL! neq 0 (
            echo [WARN] Patch may already be applied or has conflicts. Continuing anyway.
        )
    ) else (
        echo Source already has local changes, skipping patch apply.
    )
    popd
) else (
    echo [WARN] vllm-windows-v8.patch not found next to build.bat
)

if not defined PROTOC (
    if exist "%SCRIPT_DIR%tools\protoc\bin\protoc.exe" (
        set "PROTOC=%SCRIPT_DIR%tools\protoc\bin\protoc.exe"
    )
)

if not defined PROTOC (
    echo [WARN] PROTOC is not set. vLLM 0.24.0's optional Rust frontend needs protoc.
    echo        Install protoc and set PROTOC=...\protoc.exe to build vllm-rs.exe.
)

:: -----------------------------------------------------------
:: 5. Build
:: -----------------------------------------------------------

echo.
echo Configuration:
echo   CUDA_HOME              = %CUDA_HOME%
echo   TORCH_CUDA_ARCH_LIST   = %TORCH_CUDA_ARCH_LIST%
echo   MAX_JOBS               = %MAX_JOBS%
echo   Source                 = %VLLM_SRC%
if defined PROTOC echo   PROTOC                 = %PROTOC%
echo.
echo Starting build (this can take several hours for 8.6;8.9;12.0)...
echo.

cd /d "%VLLM_SRC%"
pip install -e . --no-build-isolation -v 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed. Check output above for errors.
    exit /b 1
)

:: -----------------------------------------------------------
:: 6. Post-build: copy flash-attn Python wrappers
:: -----------------------------------------------------------

if exist ".deps\vllm-flash-attn-src\vllm_flash_attn\__init__.py" (
    echo Copying flash-attn Python wrappers...
    xcopy /E /Y /Q ".deps\vllm-flash-attn-src\vllm_flash_attn\*.py" "vllm\vllm_flash_attn\" >nul 2>&1
    if exist ".deps\vllm-flash-attn-src\vllm_flash_attn\layers" (
        xcopy /E /Y /Q ".deps\vllm-flash-attn-src\vllm_flash_attn\layers\*" "vllm\vllm_flash_attn\layers\" >nul 2>&1
    )
    if exist ".deps\vllm-flash-attn-src\vllm_flash_attn\ops" (
        xcopy /E /Y /Q ".deps\vllm-flash-attn-src\vllm_flash_attn\ops\*" "vllm\vllm_flash_attn\ops\" >nul 2>&1
    )
)

echo.
echo Build complete!
echo.
echo Required environment variables for running vLLM on Windows:
echo   set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo   set VLLM_HOST_IP=127.0.0.1
echo.

endlocal
