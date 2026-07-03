# Install vLLM v0.24.0 on Windows

Two paths:

- **Install the pre-built wheel**: no compiler needed; recommended for most users.
- **Build from source**: requires Visual Studio 2022, CUDA 12.8, Python 3.13, and patience.

`install.bat` handles the wheel path end to end. `build.bat` and
`run_build.bat` handle the source build path.

## Install The Wheel

### Requirements

| Component | Version | Notes |
|---|---|---|
| Windows | 10 / 11 x64 | Tested on Windows 10 Pro 22H2 |
| GPU | NVIDIA SM 8.0+ | RTX 30/40/50, A100, H100 |
| Driver | R570+ | Required for RTX 50-series / Blackwell |
| Python | 3.13.x | `install.bat` uses embedded Python 3.13.11 plus headers/libs for Triton |
| PyTorch | 2.11.0+cu128 | CUDA 12.8 runtime from PyTorch wheels |
| Triton | triton-windows 3.6.0.post26 | Installed by `install.bat` |
| Disk | 5 GB+ | Python, PyTorch, Triton, and wheel |

You do not need CUDA Toolkit or Visual Studio to install the pre-built wheel.

### Automated Install

```bat
install.bat
```

The installer downloads embedded Python, adds the Python headers/libs
needed by Triton's runtime compiler, installs PyTorch cu128,
triton-windows, the vLLM wheel, structured-output backends, and verifies
both `import vllm` and Triton's CUDA runtime driver path.

It caches state in:

- `python\.torch-installed`
- `python\.vllm-installed`

Delete those files to force reinstall.

Rerunning `install.bat` also repairs an existing portable Python 3.13
directory if `Include\Python.h` or `libs\python313.lib` is missing.
`launch.bat` performs the same repair check before starting the server.
If `python\` is from an older major/minor Python version, delete
`python\` and rerun the installer.

### Manual Install

```bat
py -3.13 -m venv venv
venv\Scripts\activate

pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 ^
    --index-url https://download.pytorch.org/whl/cu128

pip install triton-windows==3.6.0.post26
pip install "llguidance>=1.7.0,<1.8.0" "xgrammar>=0.2.0,<1.0.0"
pip install git+https://github.com/aivrar/multi-turboquant.git
pip install dist-v7\vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl
```

Or download the wheel from the latest GitHub release:

```text
https://github.com/aivrar/vllm-windows-build/releases/tag/v0.24.0-win-cu128
```

### Verify

```bat
python -c "import vllm; print(vllm.__version__)"
vllm --help
vllm serve --help
```

Expected version:

```text
0.24.0+cu128
```

## Build From Source

### Requirements

| Component | Version |
|---|---|
| Visual Studio | VS 2022 with C++ workload |
| CUDA Toolkit | 12.8 |
| Python | 3.13.x |
| PyTorch | 2.11.0+cu128 |
| Ninja | Available in the venv or on PATH |
| Rust | Current stable MSVC toolchain |
| protoc | Required for v0.24 Rust frontend/tool parser |
| RAM | 32 GB minimum, 64 GB recommended |
| Disk | 30 GB+ |

### Source Tree

```bat
git clone https://github.com/vllm-project/vllm.git vllm-source
cd vllm-source
git checkout v0.24.0
git apply ..\vllm-windows-v7.patch
cd ..
```

### Build

Edit `run_build.bat` for your paths, then run it from a normal command prompt:

```bat
run_build.bat
```

Important defaults:

```bat
set TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0
set MAX_JOBS=2
set VLLM_DISABLE_SCCACHE=1
set SETUPTOOLS_SCM_PRETEND_VERSION=0.24.0
```

Keep `MAX_JOBS=2`. Higher parallelism has repeatedly produced MSVC
compiler crashes on the heavy multi-arch CUDA translation units.

### Build A Wheel From An Already-Built Tree

After the editable install succeeds and `vllm.egg-info` exists:

```bat
python assemble_wheel_cu128_v0.24.0.py
```

Output:

```text
dist-v7\vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl
```

## Runtime Environment

These are recommended before running models:

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set VLLM_HOST_IP=127.0.0.1
```

For multi-GPU systems, set the CUDA device ordering explicitly:

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
```

For more, see [troubleshooting.md](troubleshooting.md).
