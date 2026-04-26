# Install vLLM v0.19.0 on Windows

Two paths: **(A)** install the pre-built wheel (no C++ toolchain needed,
~3 min), **(B)** build from source (requires VS 2022 + CUDA 12.6,
~30-45 min). Pick A unless you need to modify the patches.

> **TL;DR** — `install.bat` does (A) end-to-end, `build.bat` does (B).

---

## (A) Install the pre-built wheel

### Requirements

| Component | Version | Notes |
|---|---|---|
| Windows | 10 / 11 (x64) | Tested on Windows 10 Pro 22H2 |
| GPU | NVIDIA, SM 8.0+ | RTX 3060/3070/3080/3090, A4000/A5000/A6000, RTX 40-series |
| Driver | R545+ | CUDA 12.6 runtime is bundled with the wheel via PyTorch |
| RAM | 32 GB recommended | 16 GB works for <14B models |
| Disk | ~5 GB free | wheel + Python + dependencies |

You do **not** need a CUDA toolkit, Visual Studio, or any compiler — the
wheel ships pre-built `_C.pyd` extensions.

### Option 1 — automated installer

```bat
install.bat
```

This downloads Python 3.10.11 (embedded), installs PyTorch 2.10.0+cu126,
installs the vLLM wheel, and runs a smoke import. The whole thing is
self-contained in the script directory; nothing touches your system
Python.

The installer caches state in `python\.torch-installed` and
`python\.vllm-installed`. Delete those files to force a re-install.

### Option 2 — manual install in your own venv

```bat
python -m venv venv
venv\Scripts\activate
pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
pip install triton-windows==3.6.0.post26
pip install dist-v3\vllm-0.19.0+cu126-cp310-cp310-win_amd64.whl
```

Or download the wheel directly from
[Releases](https://github.com/aivrar/vllm-windows-build/releases/latest).

### Verify

```bat
python -c "import vllm; print(vllm.__version__)"
```

Should print `0.19.0+cu126`. If you see a DLL load error, see
[Troubleshooting](troubleshooting.md).

---

## (B) Build from source

### Requirements

| Component | Version |
|---|---|
| Visual Studio 2022 | Community (or higher) with C++ workload |
| CUDA Toolkit | 12.6 (CUDA 12.8+ would skip QuTLASS but should work) |
| Python | 3.10.x (3.11/3.12 untested) |
| RAM | 32 GB minimum, 64 GB recommended |
| Disk | ~25 GB free |
| Time | 30-45 minutes |

### Prepare the source tree

```bat
git clone https://github.com/vllm-project/vllm.git vllm-source
cd vllm-source
git checkout v0.19.0
cd ..
```

### Apply the Windows patch

The patch (`vllm-windows-v3.patch`) modifies 33 files: build system,
CUDA kernels, runtime Python, plus a new file
`vllm/v1/attention/ops/multi_turboquant_kv.py` for the TurboQuant
integration.

```bat
cd vllm-source
git apply ..\vllm-windows-v3.patch
cd ..
```

### Build

Edit the paths at the top of `run_build.bat` to match your install:

```bat
set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "VENV_PATH=%~dp0venv"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "TORCH_CUDA_ARCH_LIST=8.6"
set "MAX_JOBS=4"
```

`TORCH_CUDA_ARCH_LIST` should match your GPU:
- RTX 30-series → `8.6`
- RTX 40-series → `8.9`
- RTX 50-series → `12.0`
- A100 → `8.0`
- H100 → `9.0`

Then run:

```bat
run_build.bat
```

The build compiles 140 CUDA targets in two phases:

1. **CMake configure** (~5-10 min) — fetches CUTLASS and triton_kernels via FetchContent.
2. **Compile** (~25-35 min with MAX_JOBS=4 on a 16-core CPU).

### Build a redistributable wheel

After the editable install succeeds, you can package the compiled
binaries into a `.whl`:

```bat
python build_wheel.py
```

This produces `dist-v3\vllm-0.19.0+cu126-cp310-cp310-win_amd64.whl`.

---

## Post-install: Windows runtime env vars

Whether you install the wheel or build from source, set these env vars
before importing vLLM in your own scripts:

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set VLLM_HOST_IP=127.0.0.1
```

`expandable_segments` is **required** if your Windows pagefile is small
or disabled — without it, PyTorch's allocator hits fragmentation and
crashes mid-run.

For more, see [Troubleshooting → "CUDA out of memory" with free GPU](troubleshooting.md#oom-with-free-gpu).
