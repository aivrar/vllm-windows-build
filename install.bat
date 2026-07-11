@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ============================================================
echo          vLLM v0.24.0 Windows Installer
echo     Portable Python 3.13.11 + PyTorch 2.11.0 (cu128) + vLLM 0.24.0
echo  ============================================================
echo.

REM ============================================================
REM  Version Configuration
REM ============================================================
set "PYTHON_VERSION=3.13.11"
set "PYTHON_URL=https://www.python.org/ftp/python/3.13.11/python-3.13.11-embed-amd64.zip"
set "PYTHON_DEV_URL=https://www.nuget.org/api/v2/package/python/3.13.11"
set "PYTHON_PTH_FILE=python313._pth"
set "PYTHON_PTH_ZIP=python313.zip"
set "PYTHON_LIB_NAME=python313.lib"
set "TRITON_NVIDIA_DIR=%~dp0python\Lib\site-packages\triton\backends\nvidia"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"

REM Pre-built vLLM wheel (auto-downloaded into dist-v8\ if not present locally)
set "WHEEL_NAME=vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl"
set "WHEEL_URL=https://github.com/aivrar/vllm-windows-build/releases/download/v0.24.0-win-cu128/vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl"
set "WHEEL_SHA256=4A76CDE2F36689A76A6F8AB7C4EE9B4C47AEFC194479C085619F9072C563B7DA"

set "STAGES_TOTAL=5"
if not defined CUDA_DEVICE_ORDER set "CUDA_DEVICE_ORDER=PCI_BUS_ID"

echo  Components to install:
echo    - Python %PYTHON_VERSION% (embedded distribution + headers/libs)
echo    - pip (package manager)
echo    - PyTorch 2.11.0+cu128 + triton-windows (CUDA GPU acceleration)
echo    - vLLM 0.24.0 wheel (pre-built Windows binary)
echo    - Verification
echo.

REM ============================================================
REM  STAGE 1: Download and Extract Python Embedded
REM ============================================================
echo [1/%STAGES_TOTAL%] Python %PYTHON_VERSION% embedded...
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>nul
    if !ERRORLEVEL! NEQ 0 (
        echo          FAILED: Existing python\ is not Python 3.13.
        echo          Delete python\ and rerun install.bat to install Python %PYTHON_VERSION%.
        goto :fail
    )
    echo          SKIP - already installed
    goto :pythonDevFiles
)
echo          Downloading from python.org...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%TEMP%\vllm-python-embed.zip'"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not download Python %PYTHON_VERSION%
    echo          URL: %PYTHON_URL%
    goto :fail
)
echo          Extracting to python\ ...
if not exist "%~dp0python" mkdir "%~dp0python"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -Path '%TEMP%\vllm-python-embed.zip' -DestinationPath '%~dp0python' -Force"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not extract Python archive
    goto :fail
)
del "%TEMP%\vllm-python-embed.zip" 2>nul
if not exist "%~dp0python\python.exe" (
    echo          FAILED: python.exe not found after extraction
    goto :fail
)
echo          OK

:pythonDevFiles
REM Triton JIT compiles a small CUDA driver helper at runtime and needs
REM Python.h plus python313.lib. The embeddable Python zip does not ship them.
echo          Checking Python headers/libs for Triton...
if exist "%~dp0python\Include\Python.h" if exist "%~dp0python\libs\%PYTHON_LIB_NAME%" (
    echo          OK - Python development files present
    goto :stage2
)
echo          Downloading Python headers/libs from NuGet...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "$pkg = Join-Path $env:TEMP 'vllm-python-dev.nupkg';" ^
    "$zip = Join-Path $env:TEMP 'vllm-python-dev.zip';" ^
    "$out = Join-Path $env:TEMP 'vllm-python-dev';" ^
    "Remove-Item $pkg, $zip -Force -ErrorAction SilentlyContinue;" ^
    "Remove-Item $out -Recurse -Force -ErrorAction SilentlyContinue;" ^
    "Invoke-WebRequest -Uri '%PYTHON_DEV_URL%' -OutFile $pkg;" ^
    "Copy-Item $pkg $zip -Force;" ^
    "Expand-Archive -Path $zip -DestinationPath $out -Force;" ^
    "Copy-Item -Path (Join-Path $out 'tools\include') -Destination '%~dp0python\Include' -Recurse -Force;" ^
    "Copy-Item -Path (Join-Path $out 'tools\libs') -Destination '%~dp0python\libs' -Recurse -Force"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not install Python headers/libs.
    goto :fail
)
if not exist "%~dp0python\Include\Python.h" (
    echo          FAILED: Python.h not found after installing development files.
    goto :fail
)
if not exist "%~dp0python\libs\%PYTHON_LIB_NAME%" (
    echo          FAILED: %PYTHON_LIB_NAME% not found after installing development files.
    goto :fail
)
echo          OK

:stage2
REM ============================================================
REM  STAGE 2: Configure Python for site-packages + Install pip
REM ============================================================
echo [2/%STAGES_TOTAL%] Python configuration + pip...

REM Write python313._pth to enable site-packages and import site.
echo          Configuring %PYTHON_PTH_FILE%...
(
    echo %PYTHON_PTH_ZIP%
    echo .
    echo Lib
    echo Lib\site-packages
    echo DLLs
    echo import site
) > "%~dp0python\%PYTHON_PTH_FILE%"

REM Create required directories
if not exist "%~dp0python\Lib\site-packages" mkdir "%~dp0python\Lib\site-packages"
if not exist "%~dp0python\Scripts" mkdir "%~dp0python\Scripts"

if exist "%~dp0python\Scripts\pip.exe" (
    echo          OK - pip already installed
    goto :stage3
)

REM Download and run get-pip.py
echo          Downloading get-pip.py...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile '%TEMP%\get-pip.py'"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not download get-pip.py
    goto :fail
)
echo          Installing pip...
"%~dp0python\python.exe" "%TEMP%\get-pip.py" --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: pip installation error
    goto :fail
)
del "%TEMP%\get-pip.py" 2>nul
if not exist "%~dp0python\Scripts\pip.exe" (
    echo          FAILED: pip.exe not found after installation
    goto :fail
)
echo          OK

:stage3
REM ============================================================
REM  STAGE 3: Install PyTorch 2.11.0+cu128 + triton-windows
REM ============================================================
echo [3/%STAGES_TOTAL%] PyTorch 2.11.0+cu128 + triton-windows (~2.5 GB download)...
if exist "%~dp0python\.torch-installed" (
    "%~dp0python\python.exe" -c "import torch, triton" >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        echo          SKIP - already installed ^(delete python\.torch-installed to force^)
        goto :stage4
    )
    echo          Existing marker found, but torch/triton import failed - reinstalling...
    del "%~dp0python\.torch-installed" 2>nul
)
echo          Installing PyTorch 2.11.0 from pytorch.org...
"%~dp0python\python.exe" -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url %TORCH_INDEX% --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: PyTorch installation error - check output above
    goto :fail
)
echo          Installing triton-windows 3.6.0...
"%~dp0python\python.exe" -m pip install "triton-windows==3.6.0.post26" --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: triton-windows installation error
    goto :fail
)
echo %DATE% %TIME% > "%~dp0python\.torch-installed"
echo          OK

:stage4
REM ============================================================
REM  STAGE 4: Install vLLM Wheel + Dependencies
REM ============================================================
echo [4/%STAGES_TOTAL%] vLLM wheel + dependencies...
if exist "%~dp0python\.vllm-installed" (
    "%~dp0python\python.exe" -c "import vllm, llguidance, xgrammar; from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb; assert vllm.__version__ == '0.24.0+cu128'" >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        echo          SKIP - already installed ^(delete python\.vllm-installed to force^)
        goto :stage5
    )
    echo          Existing marker found, but vLLM dependencies import failed - reinstalling...
    del "%~dp0python\.vllm-installed" 2>nul
)

REM Only accept the current fixed v0.24.0 wheel. Older local wheels are not
REM compatible substitutes and the original dist-v7 artifact omitted required
REM FlashAttention Python modules.
set "WHEEL_FILE="
for %%f in ("%~dp0dist-v8\%WHEEL_NAME%") do if exist "%%~f" set "WHEEL_FILE=%%~f"
REM No local wheel found - auto-download the latest (cu128) from GitHub Releases
if "!WHEEL_FILE!"=="" (
    echo          No local wheel found - downloading from GitHub Releases ^(~319 MB^)...
    if not exist "%~dp0dist-v8" mkdir "%~dp0dist-v8"
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
        "$ProgressPreference = 'SilentlyContinue';" ^
        "Invoke-WebRequest -Uri '%WHEEL_URL%' -OutFile '%~dp0dist-v8\%WHEEL_NAME%'"
    if !ERRORLEVEL! NEQ 0 (
        echo          FAILED: Could not download the vLLM wheel.
        echo          URL: %WHEEL_URL%
        echo          Download it manually and place it in: %~dp0dist-v8\
        goto :fail
    )
    REM Sanity check: the real wheel is ~319 MB; reject a truncated/HTML error page
    for %%f in ("%~dp0dist-v8\%WHEEL_NAME%") do set "WHEEL_SIZE=%%~zf"
    if !WHEEL_SIZE! LSS 100000000 (
        echo          FAILED: Downloaded wheel is only !WHEEL_SIZE! bytes ^(expected ~319 MB^).
        echo          The download was likely incomplete or blocked. Delete it and retry,
        echo          or download manually from:
        echo            https://github.com/aivrar/vllm-windows-build/releases
        del "%~dp0dist-v8\%WHEEL_NAME%" 2>nul
        goto :fail
    )
    set "WHEEL_FILE=%~dp0dist-v8\%WHEEL_NAME%"
    echo          Downloaded OK ^(!WHEEL_SIZE! bytes^)
)
echo          Found wheel: !WHEEL_FILE!
set "WHEEL_HASH="
for /f "usebackq delims=" %%h in (`powershell -NoProfile -Command "(Get-FileHash -LiteralPath '!WHEEL_FILE!' -Algorithm SHA256).Hash"`) do set "WHEEL_HASH=%%h"
if /I not "!WHEEL_HASH!"=="%WHEEL_SHA256%" (
    echo          FAILED: Wheel SHA256 does not match the fixed release artifact.
    echo          Expected: %WHEEL_SHA256%
    echo          Actual:   !WHEEL_HASH!
    echo          Delete the stale or incomplete wheel and rerun install.bat.
    goto :fail
)
echo          SHA256 verified
echo          Installing corrected vLLM wheel...
"%~dp0python\python.exe" -m pip install "!WHEEL_FILE!" --force-reinstall --no-deps --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: vLLM installation error - check output above
    goto :fail
)
echo          Resolving vLLM dependencies...
"%~dp0python\python.exe" -m pip install "!WHEEL_FILE!" --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: vLLM dependency installation error - check output above
    goto :fail
)
REM llguidance and xgrammar (structured-output backends) are gated in vLLM's
REM requirements on platform_machine=="x86_64", but Windows reports "AMD64",
REM so pip silently skips them and vLLM then fails to import. Install them
REM explicitly here.
echo          Installing structured-output backends (llguidance, xgrammar)...
"%~dp0python\python.exe" -m pip install "llguidance>=1.7.0,<1.8.0" "xgrammar>=0.2.0,<1.0.0" --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          WARNING: llguidance/xgrammar install failed - guided decoding may be unavailable
)
echo %DATE% %TIME% > "%~dp0python\.vllm-installed"
echo          OK

:stage5
REM ============================================================
REM  STAGE 5: Verify Installation
REM ============================================================
echo [5/%STAGES_TOTAL%] Verification...
if exist "%TRITON_NVIDIA_DIR%\bin\ptxas.exe" (
    set "CUDA_PATH=%TRITON_NVIDIA_DIR%"
    set "CUDA_HOME=%TRITON_NVIDIA_DIR%"
    set "PATH=%TRITON_NVIDIA_DIR%\bin;%PATH%"
    echo          Using bundled Triton CUDA toolkit: %TRITON_NVIDIA_DIR%
)
"%~dp0python\python.exe" -c "import vllm; from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb; assert vllm.__version__ == '0.24.0+cu128'; print(f'  vLLM {vllm.__version__} and FlashAttention rotary loaded successfully')"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: vLLM import failed - check the install output above.
    goto :fail
)
echo          Checking Triton CUDA runtime...
"%~dp0python\python.exe" -c "import triton.runtime.driver as d; drv=getattr(d, 'active', None); drv=drv if drv is not None else d.driver.active; target=drv.get_current_target(); print(f'  Triton CUDA target: {target.backend} {target.arch}')"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Triton CUDA runtime check failed.
    echo          Make sure an NVIDIA driver is installed and rerun install.bat.
    echo          If you see Python.h or python313.lib errors, delete python\ and rerun install.bat.
    goto :fail
)
echo          OK

echo.
echo  ============================================================
echo                   Installation Complete
echo  ============================================================
echo.
echo  To start vLLM:
echo    launch.bat                                  (interactive model selector)
echo    launch.bat --model path\to\model            (direct launch)
echo    launch.bat --model path\to\model --port 8000
echo.
echo  Or manually:
echo    python\python.exe vllm_launcher.py --model path\to\model
echo.
endlocal
exit /b 0

:fail
endlocal
exit /b 1
