@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ============================================================
echo          vLLM v0.21.0 Windows Installer
echo     Portable Python 3.10.11 + PyTorch 2.11.0 + vLLM 0.21.0
echo  ============================================================
echo.

REM ============================================================
REM  Version Configuration
REM ============================================================
set "PYTHON_VERSION=3.10.11"
set "PYTHON_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
set "PYTHON_PTH_FILE=python310._pth"
set "PYTHON_PTH_ZIP=python310.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"

set "STAGES_TOTAL=5"

echo  Components to install:
echo    - Python %PYTHON_VERSION% (embedded distribution)
echo    - pip (package manager)
echo    - PyTorch 2.11.0+cu126 + triton-windows (CUDA GPU acceleration)
echo    - vLLM 0.21.0 wheel (pre-built Windows binary)
echo    - Verification
echo.

REM ============================================================
REM  STAGE 1: Download and Extract Python Embedded
REM ============================================================
echo [1/%STAGES_TOTAL%] Python %PYTHON_VERSION% embedded...
if exist "%~dp0python\python.exe" (
    echo          SKIP - already installed
    goto :stage2
)
echo          Downloading from python.org...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%TEMP%\vllm-python-embed.zip'"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not download Python %PYTHON_VERSION%
    echo          URL: %PYTHON_URL%
    exit /b 1
)
echo          Extracting to python\ ...
if not exist "%~dp0python" mkdir "%~dp0python"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -Path '%TEMP%\vllm-python-embed.zip' -DestinationPath '%~dp0python' -Force"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not extract Python archive
    exit /b 1
)
del "%TEMP%\vllm-python-embed.zip" 2>nul
if not exist "%~dp0python\python.exe" (
    echo          FAILED: python.exe not found after extraction
    exit /b 1
)
echo          OK

:stage2
REM ============================================================
REM  STAGE 2: Configure Python for site-packages + Install pip
REM ============================================================
echo [2/%STAGES_TOTAL%] Python configuration + pip...
if exist "%~dp0python\Scripts\pip.exe" (
    echo          SKIP - pip already installed
    goto :stage3
)

REM Write python310._pth to enable site-packages and import site
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

REM Download and run get-pip.py
echo          Downloading get-pip.py...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile '%TEMP%\get-pip.py'"
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: Could not download get-pip.py
    exit /b 1
)
echo          Installing pip...
"%~dp0python\python.exe" "%TEMP%\get-pip.py" --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: pip installation error
    exit /b 1
)
del "%TEMP%\get-pip.py" 2>nul
if not exist "%~dp0python\Scripts\pip.exe" (
    echo          FAILED: pip.exe not found after installation
    exit /b 1
)
echo          OK

:stage3
REM ============================================================
REM  STAGE 3: Install PyTorch 2.10.0+cu126 + triton-windows
REM ============================================================
echo [3/%STAGES_TOTAL%] PyTorch 2.10.0+cu126 + triton-windows (~2.5 GB download)...
if exist "%~dp0python\.torch-installed" (
    echo          SKIP - already installed ^(delete python\.torch-installed to force^)
    goto :stage4
)
echo          Installing PyTorch 2.11.0 from pytorch.org...
"%~dp0python\python.exe" -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url %TORCH_INDEX% --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: PyTorch installation error - check output above
    exit /b 1
)
echo          Installing triton-windows 3.6.0...
"%~dp0python\python.exe" -m pip install "triton-windows==3.6.0.post26" --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: triton-windows installation error
    exit /b 1
)
echo %DATE% %TIME% > "%~dp0python\.torch-installed"
echo          OK

:stage4
REM ============================================================
REM  STAGE 4: Install vLLM Wheel + Dependencies
REM ============================================================
echo [4/%STAGES_TOTAL%] vLLM wheel + dependencies...
if exist "%~dp0python\.vllm-installed" (
    echo          SKIP - already installed ^(delete python\.vllm-installed to force^)
    goto :stage5
)

REM Find the v0.21.0 wheel first, then fall back to older releases
set "WHEEL_FILE="
for %%f in ("%~dp0dist-v5\vllm-0.21.0*.whl") do set "WHEEL_FILE=%%f"
if "!WHEEL_FILE!"=="" (
    for %%f in ("%~dp0dist-v4\vllm-0.19.1*.whl") do set "WHEEL_FILE=%%f"
)
if "!WHEEL_FILE!"=="" (
    for %%f in ("%~dp0dist-v3\vllm-0.19.0*.whl") do set "WHEEL_FILE=%%f"
)
if "!WHEEL_FILE!"=="" (
    for %%f in ("%~dp0dist-v2\vllm-*.whl") do set "WHEEL_FILE=%%f"
)
if "!WHEEL_FILE!"=="" (
    for %%f in ("%~dp0dist\vllm-*.whl") do set "WHEEL_FILE=%%f"
)
if "!WHEEL_FILE!"=="" (
    echo          FAILED: No vllm wheel found in dist-v5\, dist-v4\, dist-v3\, dist-v2\, or dist\
    echo          Download from:
    echo            https://github.com/aivrar/vllm-windows-build/releases
    echo          Or build it yourself: python build_wheel.py
    exit /b 1
)
echo          Found wheel: !WHEEL_FILE!
echo          Installing vLLM and dependencies...
"%~dp0python\python.exe" -m pip install "!WHEEL_FILE!" --no-warn-script-location
if !ERRORLEVEL! NEQ 0 (
    echo          FAILED: vLLM installation error - check output above
    exit /b 1
)
echo %DATE% %TIME% > "%~dp0python\.vllm-installed"
echo          OK

:stage5
REM ============================================================
REM  STAGE 5: Verify Installation
REM ============================================================
echo [5/%STAGES_TOTAL%] Verification...
"%~dp0python\python.exe" -c "import vllm; print(f'  vLLM {vllm.__version__} loaded successfully')"
if !ERRORLEVEL! NEQ 0 (
    echo          WARNING: vLLM import failed - some dependencies may be missing
    echo          Try running: python\python.exe -m pip install -r requirements.txt
) else (
    echo          OK
)

echo.
echo  ============================================================
echo                   Installation Complete!
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
