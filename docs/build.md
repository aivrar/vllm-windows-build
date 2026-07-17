# Build From Source

This page documents the v0.24.0 Windows build flow and how to iterate on
`vllm-windows-v8.patch`.

For install-only usage, see [install.md](install.md).

## Patch Scope

`vllm-windows-v8.patch` is a unified diff against upstream
`vllm-project/vllm` tag `v0.24.0` (`ee0da84ab`).

Main categories:

| Area | Purpose |
|---|---|
| Build system | Allow CUDA builds on Windows, force CUDA 12.8 paths, apply CUTLASS patches, skip Linux-only optional extensions |
| CUDA kernels | MSVC compatibility for GCC-only syntax, `__int128_t`, `__builtin_clz`, macro/preprocessor issues, and generated selector depth |
| Runtime Python | Windows multiprocessing/network/event-loop fixes, safetensors reader, FakeProcessGroup, API server fallbacks |
| Rust artifacts | Build and package `vllm-rs.exe` and `_rust_tool_parser.pyd` |
| Multi-TurboQuant | Carry the 6 local KV-cache compression methods alongside upstream TurboQuant variants |

## Required Toolchain

| Component | Version |
|---|---|
| Visual Studio | 2022 Community or newer, C++ workload |
| CUDA Toolkit | 12.8 |
| Python | 3.13.x |
| PyTorch | 2.11.0+cu128 |
| Triton | triton-windows 3.6.0.post26 |
| Rust | MSVC stable toolchain |
| protoc | Required for Rust frontend/tool parser |
| Generator | Ninja |

Recommended environment:

```bat
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0
set VLLM_TARGET_DEVICE=cuda
set CMAKE_BUILD_TYPE=Release
set VLLM_DISABLE_SCCACHE=1
set SETUPTOOLS_SCM_PRETEND_VERSION=0.24.0
set MAX_JOBS=2
set PROTOC=C:\path\to\protoc.exe
```

## Build Phases

### 1. Patch Upstream

```bat
git clone https://github.com/vllm-project/vllm.git vllm-source
cd vllm-source
git checkout v0.24.0
git apply ..\vllm-windows-v8.patch
cd ..
```

### 2. Configure And Build

`python -m pip install -e . --no-build-isolation -v` drives CMake/Ninja through
`setup.py`.

`build.bat` accepts either a clean upstream v0.24.0 tree or a tree with the
complete v8 patch already applied. It stops on a partial/conflicting patch and
verifies the Python, PyTorch, CUDA, `protoc`, native, Rust, FlashAttention, and
third-party payload contract before reporting success.

Expected native artifacts in `vllm-source\vllm\` after a successful build:

```text
_C_stable_libtorch.pyd
_moe_C_stable_libtorch.pyd
_rust_tool_parser.pyd
cumem_allocator.pyd
spinloop.pyd
vllm_flash_attn\_vllm_fa2_C.pyd
vllm-rs.exe
third_party\triton_kernels\...
third_party\fmha_sm100\...
```

Intentionally absent on Windows:

```text
_qutlass_C.pyd
_deep_gemm_C.pyd
cooperative_topk op
```

Those paths are optional in this build. vLLM falls back when they are not
available.

### 3. Generate Metadata

If `vllm.egg-info` was not left in the source tree by the editable build,
generate it before assembling a wheel:

```bat
set VLLM_TARGET_DEVICE=cuda
set SETUPTOOLS_SCM_PRETEND_VERSION=0.24.0
python setup.py egg_info
```

Confirm `vllm.egg-info\PKG-INFO` contains:

```text
Version: 0.24.0+cu128
```

### 4. Assemble Wheel

```bat
python assemble_wheel_cu128_v0.24.0.py
```

Output:

```text
dist-v8\vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl
```

### 5. Smoke Test The Wheel

Validate archive completeness and RECORD before installing:

```bat
python tests\test_wheel_contents.py dist-v8\vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl
```

The assembler and wheel-content test also verify that request-seed generation
explicitly uses NumPy `int64`, preventing the Windows 32-bit C `long` overflow
reported in issue #10.

Install the wheel from outside the source tree:

```bat
python -m pip install --force-reinstall --no-deps dist-v8\vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl
```

Run:

```bat
python -c "import vllm; print(vllm.__version__)"
vllm --help
vllm serve --help
```

For the issue #7 Qwen3-VL/FlashAttention regression, install the wheel into
an isolated target and run:

```bat
python -m pip install --no-deps --target %TEMP%\vllm-wheel-test dist-v8\vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl
python tests\test_issue7_flash_attn.py --package-root %TEMP%\vllm-wheel-test
```

Required Rust frontend check:

```bat
set VLLM_USE_RUST_FRONTEND=1
python -c "from pathlib import Path; import vllm.envs as e; print(e.VLLM_RUST_FRONTEND_PATH); assert Path(e.VLLM_RUST_FRONTEND_PATH).exists()"
```

## Iterating

If you change Python files, rerun the smoke tests.

If you change CUDA/C++ files:

```bat
python -m pip install -e . --no-build-isolation -v
```

If you change `setup.py` or CMake files, clear the temp build directory
or start from a fresh build temp before rebuilding.

## Regenerating The Patch

From the patched vLLM source tree:

```bat
git diff --binary v0.24.0..HEAD --output=..\vllm-windows-v8.patch -- .
```

Validate against a clean upstream worktree:

```bat
git worktree add --detach ..\patch-check-v0.24.0 v0.24.0
git -C ..\patch-check-v0.24.0 apply --check ..\vllm-windows-v8.patch
```

Also run:

```bat
git diff --check v0.24.0..HEAD -- . ":!cutlass-windows.patch" ":!vllm-flash-attn-cutlass-windows.patch"
```
