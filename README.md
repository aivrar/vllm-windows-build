# vllm-windows-build

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue)
![vLLM: v0.24.0](https://img.shields.io/badge/vLLM-v0.24.0-orange)
![CUDA: 12.8](https://img.shields.io/badge/CUDA-12.8-76B900)
![Python: 3.13](https://img.shields.io/badge/Python-3.13-3776AB)
![PyTorch: 2.11](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C)
![Triton: 3.6](https://img.shields.io/badge/Triton-3.6-red)
![GPU: Blackwell sm_120](https://img.shields.io/badge/GPU-Ampere%20%E2%86%92%20Blackwell-76B900)
![Multi-TurboQuant](https://img.shields.io/badge/Multi--TurboQuant-6%20methods-purple)
![+ Upstream TurboQuant](https://img.shields.io/badge/+%20Upstream%20TurboQuant-4%20variants-purple)

**Native Windows build of vLLM 0.24.0 - no WSL, no Docker, no Linux VM.**

> **Latest build (cu128 / Python 3.13 / Blackwell):** updated to
> **vLLM 0.24.0** with RTX 30-/40-/50-series kernels
> (`TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0`), PyTorch 2.11.0+cu128, CUDA 12.8,
> the Windows OpenAI API-server fixes carried forward, the Rust frontend
> binary (`vllm-rs.exe`), and the new Rust tool parser extension packaged
> on Windows. See [What's new in v0.24.0](#whats-new-in-v0240).

Ships with **10 KV cache compression methods**: the 6 Multi-TurboQuant
methods (`isoquant`/`planarquant`/`turboquant25/35`) plus the 4 new
upstream TurboQuant variants that landed in v0.19.2rc0 (`turboquant_k8v4`,
`turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`).

vLLM is the most popular open-source LLM serving engine, but it
officially only supports Linux. This repo provides a **pre-built wheel**
(just download and install) plus a complete patchset for compiling vLLM
v0.24.0 natively on Windows with full CUDA acceleration, Triton support,
and Multi-TurboQuant integration.

## Releases

| Release | vLLM | PyTorch | Triton | KV compression | Download |
|---|---|---|---|---|---|
| **v0.24.0-win-cu128 (latest)** | 0.24.0 | 2.11.0+cu128 | 3.6.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8 - **Python 3.13, Blackwell sm_120, Rust frontend + Rust tool parser included** | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.24.0-win-cu128) |
| v0.23.0-win-cu128 | 0.23.0 | 2.11.0+cu128 | 3.6.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8 - **Python 3.13, Blackwell sm_120, Rust frontend included** | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.23.0-win-cu128) |
| v0.21.0-win-cu128 | 0.21.0 | 2.11.0+cu128 | 3.6.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8 — **Python 3.13, Blackwell sm_120** | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.21.0-win-cu128) |
| v0.21.0-win | 0.21.0 | 2.11.0+cu126 | 3.6.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8 (Python 3.10) | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.21.0-win) |
| v0.19.1-win | 0.19.1 | 2.10.0+cu126 | 3.6.0 | Multi-TurboQuant (6 methods) + fp8 | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.19.1-win) |
| v0.19.0-win | 0.19.0 | 2.10.0+cu126 | 3.6.0 | Multi-TurboQuant (6 methods) + fp8 | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.19.0-win) |
| v0.17.1-win | 0.17.1 | 2.10.0+cu126 | 3.6.0 | TurboQuant (2 recipes) | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.17.1-win) |
| v0.14.2-win | 0.14.2 | 2.9.1+cu126 | n/a | fp8 only | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.14.2-win) |

### What's new in v0.24.0

- **vLLM v0.24.0 base** - carries forward the Windows CUDA/MSVC fixes from
  v0.23.0 and adds the upstream v0.24 engine, model, serving, parser, and
  security fixes.
- **Rust tool parser packaged on Windows** - v0.24 adds a PyO3
  `_rust_tool_parser.pyd` beside `vllm-rs.exe`; the wheel now includes and
  smoke-tests both Rust artifacts.
- **Windows-only skips for Linux-only v0.24 extensions** - QuTLASS,
  cooperative TopK, and DeepGEMM are skipped on Windows. Their callers are
  guarded or fall back to existing paths, and the expected missing QuTLASS
  warning is suppressed on Windows.
- **v0.24 third-party CUDA helper packages included** - the wheel carries
  the generated `vllm.third_party.triton_kernels` and `fmha_sm100` files
  copied by the build.
- **Qwen3-VL FlashAttention packaging fixed** - the rebuilt wheel includes
  the generated rotary, Triton rotary, and CuteDSL Python modules that the
  upstream editable-build copy step dropped on Windows. The v8 patch makes
  that copy path platform-independent, and the assembler now rejects an
  incomplete FlashAttention payload.
- **Windows request sampling fixed** - NumPy now generates the full-range
  internal request seed explicitly as `int64`, avoiding the 32-bit C `long`
  default that caused issue #10's follow-up error on 64-bit Windows.
- **Smoke tested from the final wheel** - installed the assembled wheel,
  imported `vllm`, the stable libtorch CUDA extensions, FA2, `spinloop`,
  `cumem_allocator`, `_rust_tool_parser`, OpenAI API server / DP supervisor
  modules, ran `vllm --help` and `vllm serve --help`, verified
  `VLLM_USE_RUST_FRONTEND=1` resolves `vllm-rs.exe`, and verified the
  intentionally skipped DeepGEMM/cooperative-TopK paths report unavailable.
- **Portable installer repairs Triton runtime compilation support** -
  `install.bat` now adds `Python.h` and `python313.lib` to the embedded
  Python tree, which Triton needs when it JIT-compiles CUDA helpers for
  models such as Qwen3.5. `launch.bat` runs the same repair check before
  starting the server, and both scripts pin Triton to its bundled CUDA
  helper toolkit when present. `launch.bat` also no longer sets the
  removed `VLLM_ATTENTION_BACKEND` environment variable.
- **Installer integrity and repair hardened** - Python, NuGet, and bootstrap
  downloads plus both project release wheels are pinned by exact size and SHA-256.
  Wheels download to a temporary file, stale/truncated local wheels are
  replaced automatically, and the install marker records the verified wheel
  hash only after CUDA, Rust, Qwen3.5/Qwen3-VL, and FlashAttention checks pass.
- **Older Windows PowerShell bootstrap fixed** - pre-Python verification and
  extraction use direct .NET APIs, not `Get-FileHash` or `Expand-Archive`.
  Windows PowerShell 3 or newer is required for the downloader.
- **Concurrent launcher requests fixed** - one dispatcher now owns
  `engine.step()` and routes outputs by request ID. Streaming and
  non-streaming requests can no longer consume each other's engine output.

### What's new in v0.23.0

- **vLLM v0.23.0 base** - carries forward the Windows CUDA/MSVC fixes from
  v0.21.0 and adds the upstream v0.22/v0.23 bug fixes and frontend work.
- **Rust frontend builds on Windows** - added `protoc` support to the build
  flow, made the Rust managed-engine process handling platform-aware, gated
  Unix-only listener/signal paths, disabled mimalloc on Windows to avoid an
  MSVC CRT link mismatch, and fixed `VLLM_RUST_FRONTEND_PATH=auto` to resolve
  `vllm-rs.exe`.
- **Wheel packaging fixed for `uv`** - the v0.23.0 wheel is assembled with
  proper CSV `RECORD` generation, so comma-containing fused-MoE config
  filenames install correctly with `uv`.
- **Smoke tested** - installed with `uv`, imported `vllm`, `_C`,
  `_C_stable_libtorch`, `_moe_C`, `spinloop`, `cumem_allocator`, FA2, the
  OpenAI API server / DP supervisor import surface, `vllm --help`, and
  `vllm serve --help`; also verified `VLLM_USE_RUST_FRONTEND=1` resolves the
  packaged `vllm-rs.exe`.

### What's new (cu128 / Python 3.13 / Blackwell)

This is a rebuild of the same vLLM 0.21.0 source for **RTX 50-series
(Blackwell)** plus a set of Windows API-server fixes. Thanks to
[@Dhrhciebcy](https://github.com/aivrar/vllm-windows-build/issues/4) for
the report that surfaced both the Blackwell gap and the API-server bug.

- **Blackwell (sm_120) support** — built with `TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0`
  on **CUDA 12.8 + PyTorch 2.11.0+cu128 + Python 3.13**, so the wheel carries
  sm_86 / sm_89 / **sm_120** kernels (verified with `cuobjdump`). The older
  `v0.21.0-win` wheel (cu126, sm_86 only) fails on a 5090 with
  `no kernel image is available for execution on the device` — that's a
  compute-capability gap, not a Python-version problem.
- **The OpenAI API server now works on Windows.** Previously only the
  in-process `LLM()` path worked; `vllm serve` / `api_server` crashed. Four
  Windows-only bugs fixed: (1) bare `import uvloop` (Unix-only) in six
  server/entrypoint modules → falls back to `asyncio`; (2) `wait_for_engine_startup()`
  registered process *sentinels* (Windows HANDLEs, not sockets) with a
  `zmq.Poller` → `not a socket`, now skipped on win32 with exit-code
  liveness checks; (3) pyzmq needs `loop.add_reader`, absent from the
  Windows Proactor loop → set `WindowsSelectorEventLoopPolicy` (no tornado);
  (4) `loop.add_signal_handler` is `NotImplementedError` on Windows → falls
  back to `signal.signal`. **winloop is no longer needed.**
- **Two Blackwell-only kernels are skipped on Windows** (they don't compile
  under MSVC and aren't usable here anyway): **QuTLASS** (NVFP4/MXFP4
  microscaling quant — uses GCC inline-PTX `asm`) and the **MiniMax**
  multi-GPU all-reduce RMS fusion (needs real multi-GPU comm; Windows uses
  `FakeProcessGroup`). Their vLLM callers are `hasattr`-guarded, so FP4 and
  MiniMax just degrade gracefully. Everything mainstream — FP16/BF16, AWQ,
  GPTQ/Marlin, FP8, and all 10 KV-cache compression methods — is unaffected.
- **Dependency note:** vLLM gates `llguidance` and `xgrammar` on
  `platform_machine == "x86_64"`, but Windows reports `AMD64`, so pip
  silently skips them and vLLM then fails to import. `install.bat` installs
  them explicitly; if installing manually, run
  `pip install "llguidance>=1.7.0,<1.8.0" "xgrammar>=0.2.0,<1.0.0"`.

### What's new in v0.21.0

- **vLLM v0.21.0 base** — 1,157 upstream commits since v0.19.1, including
  the new native TurboQuant attention backend (PR #38479), DeepGEMM
  extension, fastsafetensors prefetch helpers, and v1 engine maturity.
- **PyTorch 2.11.0 + CUDA 12.6** (was 2.10.0). New compiler flags needed
  for MSVC: `/Usmall` to dodge the `rpcndr.h` macro that collides with
  PyTorch's new `bool small` parameter name, and `/Zc:__cplusplus` so
  CUTLASS's `is_unsigned_v` (C++17) actually sees the standard `__cplusplus`
  value.
- **Upstream TurboQuant coexists with Multi-TurboQuant** — the patch
  registers our 6 method names alongside upstream's 4 in `CacheDType`.
  Backend dispatch in `vllm/platforms/cuda.py` routes `turboquant_*` to
  the new `TurboQuantBackend`; ours stay on the existing `TritonAttention`
  backend with the dispatch hooks from the v4 patch.
- **CUTLASS 4.4.2 (vendored + vllm-flash-attn submodule) is now patched
  inline** — two MSVC fixes (`memsetDevice` host/device mismatch, four
  `static constexpr dim3 get_block_shape()` violations). The patches
  ship as `cutlass-windows.patch` and `vllm-flash-attn-cutlass-windows.patch`
  inside `vllm-source/`; `CMakeLists.txt` applies them automatically after
  `FetchContent_MakeAvailable`, so no manual intervention.
- **flashinfer is now silently skipped on Windows** — upstream defaults
  `VLLM_USE_FLASHINFER_SAMPLER=True`, which then unconditionally `import
  flashinfer` (no Windows wheel). The patch flips the default to `False`
  on `win32` so the Triton fallback is used transparently.
- **Smoke-tested end-to-end on RTX 3090, Qwen3-14B-AWQ-4bit** with both
  `kv_cache_dtype=auto` (9.7 tok/s) and `turboquant35` (0.73 tok/s,
  consistent with v0.19.x).

### Carried over from v0.19.x

- **Multi-TurboQuant integration**: 6 KV cache compression methods
  (`isoquant3`, `isoquant4`, `planarquant3`, `planarquant4`,
  `turboquant25`, `turboquant35`) with real uint8 packed storage —
  **2× more KV cache tokens** at the same `gpu_memory_utilization`.
- **Custom Windows safetensors reader**: numpy memory-mapping +
  chunked GPU streaming. Loads a 14B model in seconds and works on
  systems with the Windows pagefile disabled.
- **All 140 CUDA targets compile clean** with MSVC 2022 + CUDA 12.6 +
  Ninja. 36 source files patched + 3 new files (the TQ dispatch helper
  and the two CUTLASS patches).
- **Tests included**: end-to-end validation suite that proves each
  TQ method actually compresses (not a placebo) and each one produces
  unique output from FP16.

### Real numbers

Single 24 GB RTX 3090, Qwen3-14B AWQ-4bit, `gpu_memory_utilization=0.5`:

| KV dtype | Cache tokens | Concurrency @ 512 | vs FP16 |
|---|---|---|---|
| `auto` (fp16) | 16,336 | 31.91× | 1.00× |
| `isoquant3`/`4`, `planarquant3`/`4`, `turboquant25`/`35` | **32,672** | **63.94×** | **2.00×** |

Full benchmarks → [docs/benchmarks.md](docs/benchmarks.md)

---

## Quick Start

### Option A — Pre-built wheel (no compiler needed)

Download
**[vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.24.0-win-cu128)**
and `multi_turboquant-0.1.0-py3-none-any.whl` from the Releases page, then:

| Artifact | SHA-256 |
|---|---|
| `vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl` | `41E930FBCF994E4FD7E5CB1585F8D277AF3FFDBA0AEE7F5DDE5822DD90E6FBA7` |
| `multi_turboquant-0.1.0-py3-none-any.whl` | `5B310E05904B588539D9A8E3374DFA6C160F025F9C2099BA5C7877C79B2FA149` |

```batch
:: Create a Python 3.13 venv
py -3.13 -m venv venv
venv\Scripts\activate

:: Install PyTorch 2.11.0 with CUDA 12.8 (cu128 = Blackwell support)
pip install torch==2.11.0 ^
    --index-url https://download.pytorch.org/whl/cu128

:: Install Triton for Windows
pip install triton-windows==3.6.0.post26

:: Install the pre-built vLLM wheel
pip install vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl

:: Structured-output backends vLLM gates on x86_64 (Windows = AMD64, so pip
:: skips them and vLLM won't import without these)
pip install "llguidance>=1.7.0,<1.8.0" "xgrammar>=0.2.0,<1.0.0"

:: Install Multi-TurboQuant for the 6 KV cache compression methods
pip install multi_turboquant-0.1.0-py3-none-any.whl
```

Or just run **`install.bat`** for a fully self-contained, one-click portable
Python install — it downloads Python 3.13, PyTorch cu128, and the vLLM wheel
itself (no manual download or folder creation needed). If you already have the
`.whl` locally, drop it in `dist-v8\` next to `install.bat` and the script uses
that instead of downloading.

Fresh installs use Python 3.13.14. Rerunning `install.bat` repairs an existing
portable Python 3.13 install if its dependencies, native/Rust payloads,
FlashAttention modules, marker hash, headers, or import checks are incomplete;
`launch.bat` checks the same release contract before it starts the server.
The installer also downloads the pinned `multi_turboquant-0.1.0` release wheel;
Git is not required for the portable path.

### Option B — Build from source

Requires Visual Studio 2022 (Community is fine), CUDA 12.8, and a Python 3.13
venv. Building all three arches (8.6;8.9;12.0) takes ~3-4 h at `MAX_JOBS=2`
(the CUDA compile dominates; see notes below). Use `MAX_JOBS=2` and **do not
enable sccache** — both cause intermittent MSVC `cl.exe` crashes (0xC000001D)
on the heavy multi-arch CUDA kernels.

```batch
git clone https://github.com/vllm-project/vllm.git vllm-source
cd vllm-source && git checkout v0.24.0 && cd ..
git apply vllm-windows-v8.patch --directory vllm-source
build.bat
```

The patch also drops `cutlass-windows.patch` and
`vllm-flash-attn-cutlass-windows.patch` into `vllm-source/`. The build's
CMakeLists.txt applies them automatically to the FetchContent-managed
`.deps/cutlass-src/` and `.deps/vllm-flash-attn-src/csrc/cutlass` after
the first configure, so you don't need a separate step.

For the v0.24 Rust frontend, install `protoc` and set
`PROTOC=C:\path\to\protoc.exe` before running the build if it is not already
on PATH.

Full instructions, including all the env vars and prerequisites:
**→ [docs/install.md](docs/install.md)**

---

## Hello world

```python
import os
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

# CUDA + torch DLL search paths
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
os.add_dll_directory(r"C:\path\to\venv\Lib\site-packages\torch\lib")

# Both uvloop and flashinfer fallbacks are baked into the wheel.
# Multi-GPU host? Don't forget CUDA_DEVICE_ORDER + CUDA_VISIBLE_DEVICES
# so vLLM lands on the GPU you actually want.

from vllm import LLM, SamplingParams

llm = LLM(
    model=r"E:\models\Qwen2.5-0.5B-Instruct",
    dtype="float16",
    kv_cache_dtype="auto",        # Fast FP16 baseline
    max_model_len=512,
    gpu_memory_utilization=0.5,
)

outputs = llm.generate(
    ["Explain CUDA streams in three sentences:"],
    SamplingParams(temperature=0.0, max_tokens=32, seed=0),
)
print(outputs[0].outputs[0].text)
```

`auto` is deliberate here: it establishes normal baseline performance. The
first request can include one-time JIT or CUDA-graph setup, so benchmark a
second request in the same process. Add `enforce_eager=True` only when
diagnosing graph/compile compatibility; it disables those optimizations.

For OpenAI-compatible HTTP serving and more usage patterns:
**→ [docs/usage.md](docs/usage.md)**

---

## KV cache compression: 10 methods (6 ours + 4 upstream)

vLLM v0.24.0 on Windows ships with integrated support for **ten** KV cache
compression dtypes. The four `turboquant_*` entries are the new upstream
TurboQuant attention backend (PR #38479, landed in v0.19.2rc0); the six
others come from our [Multi-TurboQuant](https://github.com/aivrar/multi-turboquant)
library and run on the patched `TritonAttention` backend.

| Method | Bits | Family | Calibration | Use case |
|---|---|---|---|---|
| `turboquant_k8v4` | 8.25 / 4.25 | upstream | none | Mixed-precision K/V |
| `turboquant_4bit_nc` | 4.25 | upstream | none | Upstream default |
| `turboquant_k3v4_nc` | 3.25 / 4.25 | upstream | none | More aggressive K |
| `turboquant_3bit_nc` | 3.25 | upstream | none | Most aggressive upstream |
| `isoquant4` | 4.25 | quaternion 4D rotation | none | Quality-first local TQ; offline/memory-first use |
| `planarquant4` | 4.25 | Givens 2D rotation | none | Same memory, simpler transform |
| `isoquant3` | 3.25 | quaternion 4D rotation | none | More aggressive |
| `planarquant3` | 3.25 | Givens 2D rotation | none | More aggressive |
| `turboquant35` | 3.25 | WHT + MSE codebook + QJL | runtime | Calibrated outliers |
| `turboquant25` | 2.25 | WHT + MSE codebook + QJL | runtime | Most compression |

Just pass the method name as `kv_cache_dtype` when constructing an
`LLM` (or `--kv-cache-dtype` to `vllm serve`). Upstream `turboquant_*`
names are routed by `vllm/platforms/cuda.py` to the new
`TurboQuantBackend` (separate cache layout + Triton encode/decode);
ours stay on `TritonAttention` with the dispatch hooks from the v4
patch.

**Trade-off (ours)**: throughput drops ~30-300× with our 6 methods enabled
because the encode/decode runs in PyTorch (no fused Triton kernel yet).
Memory savings are real, throughput cost is the price. Best for
offline / long-context / batch workloads. The upstream variants use
fused Triton kernels and don't pay this cost.  See
**[docs/turboquant.md](docs/turboquant.md)** for the full picture.

---

## What's in the patch

`vllm-windows-v8.patch` is a unified diff against `vllm-project/vllm`
at tag `v0.24.0`. It touches the Windows build/runtime/Rust frontend
surface plus **3 new files** (the TQ
dispatch helper plus two CUTLASS-vendor patches):

- **Build system**: CMakeLists, cmake/utils, setup.py, requirements/cuda.txt
  (with `/Usmall` + `/Zc:__cplusplus` for MSVC, Linux-only CUDA deps
  commented out, auto-apply of cutlass-windows patches, Windows skips for
  QuTLASS, cooperative TopK, and DeepGEMM)
- **CUDA kernels**: MSVC compatibility for keyword operators,
  designated initializers, `__builtin_clz`, variable templates with
  attributes, nested constexpr lambdas, deeply nested `else if`,
  `__attribute__((aligned))`, `std::isinf`, `__int128_t`, the new
  `persistent_topk.cuh` `__forceinline` swap, `fused_silu_mul_block_quant.cu`
  `quant_type_max_v<T>()` call-syntax, and the `topk_softplus_sqrt_kernels.cu`
  preprocessor-in-macro-arg refactor
- **Runtime Python**: `fcntl` → `msvcrt`, ZMQ IPC → TCP, fork →
  spawn, NCCL → FakeProcessGroup, custom safetensors reader for small
  pagefile systems, `uvloop` fallback, `VLLM_USE_FLASHINFER_SAMPLER`
  default-False on Windows, Windows Rust artifact lookup, and optional
  QuTLASS warning suppression
- **Multi-TurboQuant integration** (4 + 1 new): 6 new `CacheDType`
  literals, dtype mapping, attention backend dispatch, plus the new
  `vllm/v1/attention/ops/multi_turboquant_kv.py` (295 lines)
- **CUTLASS patches** (2 new files): `cutlass-windows.patch` (5 files
  in CUTLASS 4.4.2: `cuda_host_adapter.hpp` + 4 SM100/SM103 headers
  with `static constexpr dim3` violations) and
  `vllm-flash-attn-cutlass-windows.patch` (5 files in the vendored
  CUTLASS submodule under vllm-flash-attn).

Full per-file breakdown → **[PATCHES.md](PATCHES.md)**

All changes are guarded by `#ifdef _MSC_VER`, `sys.platform == "win32"`,
`if(MSVC ...)`, or similar conditionals. **Zero impact on Linux builds.**

---

## Documentation

| Page | Topic |
|---|---|
| [docs/install.md](docs/install.md) | Install the wheel or build from source |
| [docs/usage.md](docs/usage.md) | Python embedding + HTTP server |
| [docs/turboquant.md](docs/turboquant.md) | Multi-TurboQuant deep dive |
| [docs/benchmarks.md](docs/benchmarks.md) | Real numbers, all 6 methods |
| [docs/build.md](docs/build.md) | Patch internals + iterating on builds |
| [docs/architecture.md](docs/architecture.md) | How the integration works |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common errors + fixes |
| [tests/README.md](tests/README.md) | End-to-end test scripts |

---

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 21H2 (x64) | Windows 10 22H2 / Windows 11 |
| GPU | NVIDIA SM 8.0+ (RTX 30/40/50, A100, H100) | RTX 3090 / 4090 / A6000 |
| VRAM | 12 GB | 24 GB |
| RAM | 16 GB | 32+ GB |
| CUDA driver | R570+ (Blackwell needs R570+) | latest |
| Python | 3.13.x | 3.13.14 |
| Compiler (build only) | VS 2022 Community + Win 10 SDK | Same |
| CUDA Toolkit (build only) | 12.8 (first toolkit with sm_120) | 12.8 |

For build-from-source, you also need a **Windows pagefile** (system
managed is fine). Without it, large allocations during compilation can
fail. See [docs/troubleshooting.md → OSError 1455](docs/troubleshooting.md#oserror-1455).

---

## Tested with

- RTX 3090 (24 GB, SM 8.6, driver 596.36) - v0.24.0 build, wheel install, native import, Rust frontend, and CLI smoke tests
- Qwen2.5-0.5B-Instruct (smoke test), Qwen3-14B-abliterated-AWQ-4bit
- Qwen3.5-9B-abliterated-GPTQ-4bit (text-only)
- Windows 10 Pro 22H2
- Visual Studio 2022 Community 17.13 (MSVC 14.43)
- CUDA Toolkit 12.8
- Python 3.13.14
- RTX 50-series (Blackwell sm_120): kernels compiled & verified via `cuobjdump`; runtime confirmation pending community hardware

### v0.21.0 smoke test (RTX 3090, Qwen3-14B-abliterated-AWQ-4bit)

`kv_cache_dtype=auto` (FlashAttention 2): **20 tokens in 2.06 s,
9.7 tok/s** with `max_model_len=512`, `gpu_memory_utilization=0.92`.
First model load completes in ~24 s after the safetensors cache warms.

`kv_cache_dtype=turboquant35` (Triton attention + Multi-TurboQuant
PyTorch-fallback encode/decode): **20 tokens in 27.39 s, 0.73 tok/s** —
in line with the v0.19.x figure (0.92 tok/s for 5 tokens). All other
Multi-TurboQuant methods (`isoquant3/4`, `planarquant3/4`,
`turboquant25`) should behave the same as in v0.19.x; rerun
`tests/test_tq_real.py` for a full sweep.

### v0.19.1 historical reference

Older Multi-TurboQuant timings on the same hardware (5 decoded tokens,
`gpu_memory_utilization=0.5`):

| Method | Preset | Time (5 tok) | Output tok/s | Status |
|---|---|---:|---:|---|
| `isoquant3` | no_calibration_symmetric | 41.5s | 0.12 | PASS |
| `isoquant4` | no_calibration_quality | 53.0s | 0.09 | PASS |
| `planarquant3` | k_only_planar | 40.5s | 0.12 | PASS |
| `planarquant4` | k_only_planar | 53.0s | 0.09 | PASS |
| `turboquant25` | max_compression | 6.7s | **0.74** | PASS |
| `turboquant35` | speed | 5.4s | **0.92** | PASS |

`turboquant25/35` are ~8× faster than the iso/planar family on the
PyTorch-fallback path. Reproduce with:

```bat
set TQ_METHOD=isoquant3
%VLLM_PYTHON% tests\test_tq_diag.py
```

---

## Limitations

- **Single GPU only.** NCCL doesn't ship with PyTorch on Windows; the
  patch wires up `FakeProcessGroup` for single-rank operation. Multi-GPU
  needs separate vLLM instances + external load balancing.
- **No FlashInfer.** No Windows wheel. The patch defaults
  `VLLM_USE_FLASHINFER_SAMPLER=False` on `win32` so vLLM falls back to
  the Triton sampler transparently.
- **No FlashAttention 3, no FlashAttention 4 (CuteDSL).** FA3 has
  MSVC-incompatible PTX macros, FA4 needs `nvidia-cutlass-dsl` (no
  Windows wheel). FlashAttention 2 works fine.
- **No fastsafetensors.** Linux-only (`io_uring`). The patched
  `weight_utils.py` keeps the in-tree numpy-mmap + chunked-GPU-stream
  reader from v0.19.x for the safetensors path.
- **No DeepGEMM, no Quack, no Tilelang, no TokenSpeed-MLA, no NIXL.**
  None ship Windows wheels; CMake skips DeepGEMM automatically when the
  target arch is below SM 9.0+.
- **Our 6 Multi-TurboQuant methods are still on the PyTorch-fallback
  encode/decode.** Memory savings real, throughput cost real
  (`turboquant35` ≈ 0.73 tok/s on Qwen3-14B). The upstream
  `turboquant_*` variants don't pay this cost — they use the fused
  Triton store/decode kernels that landed in PR #38479.
- **Triton JIT cold-start latency.** First inference with Triton kernels
  (e.g. Qwen3.5 GDN layers) takes ~1-2 minutes for compilation.

---

## Credits

| | |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | The original engine |
| [PyTorch](https://github.com/pytorch/pytorch) | Tensor library + CUDA bindings |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | NVIDIA |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FA2 kernels |
| [triton-windows](https://github.com/triton-lang/triton-windows) | Triton compiler ported to Windows |
| [Multi-TurboQuant](https://github.com/aivrar/multi-turboquant) | KV cache compression methods (ours) |
| [Upstream TurboQuant](https://github.com/vllm-project/vllm/pull/38479) | TurboQuant attention backend (vLLM PR #38479) |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | GEMM kernels (CUTLASS 4.4.2 with MSVC patches) |
| [TurboQuant paper](https://arxiv.org/abs/2501.06725) | Walsh-Hadamard quantization |

Built with the help of [Claude](https://claude.ai).

---

## License

MIT. See [LICENSE](LICENSE).
