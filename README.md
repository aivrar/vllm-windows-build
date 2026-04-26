# vllm-windows-build

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue)
![vLLM: v0.19.1](https://img.shields.io/badge/vLLM-v0.19.1-orange)
![CUDA: 12.6](https://img.shields.io/badge/CUDA-12.6-76B900)
![Python: 3.10](https://img.shields.io/badge/Python-3.10-3776AB)
![Triton: 3.6](https://img.shields.io/badge/Triton-3.6-red)
![Multi-TurboQuant](https://img.shields.io/badge/Multi--TurboQuant-6%20methods-purple)

**Native Windows build of vLLM 0.19.1 — no WSL, no Docker, no Linux VM.**
Now with the full **Multi-TurboQuant** KV cache compression suite (6
methods, real packed-uint8 storage, **2× cache capacity**) and a custom
safetensors reader that loads models **29× faster** on Windows.

vLLM is the most popular open-source LLM serving engine, but it
officially only supports Linux. This repo provides a **pre-built wheel**
(just download and install) plus a complete patchset for compiling vLLM
v0.19.1 natively on Windows with full CUDA acceleration, Triton support,
and Multi-TurboQuant integration.

## Releases

| Release | vLLM | PyTorch | Triton | KV compression | Download |
|---|---|---|---|---|---|
| **v0.19.1-win (latest)** | 0.19.1 | 2.10.0+cu126 | 3.6.0 | Multi-TurboQuant (6 methods) + fp8 | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.19.1-win) |
| v0.19.0-win | 0.19.0 | 2.10.0+cu126 | 3.6.0 | Multi-TurboQuant (6 methods) + fp8 | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.19.0-win) |
| v0.17.1-win | 0.17.1 | 2.10.0+cu126 | 3.6.0 | TurboQuant (2 recipes) | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.17.1-win) |
| v0.14.2-win | 0.14.2 | 2.9.1+cu126 | n/a | fp8 only | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.14.2-win) |

### What's new in v0.19.1

- **vLLM v0.19.1 base** — upstream point release (CI fixes, pinned
  `nixl-cu{12,13}`, Jina ColBERT rotary recomputation for transformers v5).
- **`uvloop` fallback baked into the wheel** — upstream added an
  unconditional `import uvloop` in `vllm/v1/utils.py`; the patch now
  wraps it in `try/except ImportError → asyncio`, so user code no
  longer needs the `sys.modules.setdefault("uvloop", ...)` stub.
- **All 6 TQ methods re-verified on RTX 3090** end-to-end. See the
  [test results](#tested-with) section.
- **`tests/test_tq_diag.py`** added — faulthandler-guarded diagnostic
  that distinguishes genuine hangs from slow (but terminating)
  PyTorch-fallback decodes.

### Carried over from v0.19.0

- **Multi-TurboQuant integration**: 6 KV cache compression methods
  (`isoquant3`, `isoquant4`, `planarquant3`, `planarquant4`,
  `turboquant25`, `turboquant35`) with real uint8 packed storage —
  **2× more KV cache tokens** at the same `gpu_memory_utilization`.
- **Custom Windows safetensors reader**: numpy memory-mapping +
  chunked GPU streaming. Loads a 14B model in **6.5 seconds** vs 189
  seconds with the upstream mmap path. Works on systems with the
  Windows pagefile disabled.
- **All 140 CUDA targets compile clean** with MSVC 2022 + CUDA 12.6 +
  Ninja. 34 source files patched + 1 new file.
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
**[vllm-0.19.1+cu126-cp310-cp310-win_amd64.whl](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.19.1-win)**
from the Releases page, then:

```batch
:: Create a Python 3.10 venv
py -3.10 -m venv venv
venv\Scripts\activate

:: Install PyTorch 2.10.0 with CUDA 12.6
pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 ^
    --index-url https://download.pytorch.org/whl/cu126

:: Install Triton for Windows
pip install triton-windows==3.6.0.post26

:: Install the pre-built vLLM wheel
pip install vllm-0.19.1+cu126-cp310-cp310-win_amd64.whl

:: Install Multi-TurboQuant for the 6 KV cache compression methods
pip install git+https://github.com/aivrar/multi-turboquant.git
```

Or run `install.bat` for a fully self-contained portable Python install.

### Option B — Build from source

Requires Visual Studio 2022 (Community is fine), CUDA 12.6, ~30-45 min.

```batch
git clone https://github.com/vllm-project/vllm.git vllm-source
cd vllm-source && git checkout v0.19.1 && cd ..
git apply vllm-windows-v4.patch --directory vllm-source
build.bat
```

Full instructions, including all the env vars and prerequisites:
**→ [docs/install.md](docs/install.md)**

---

## Hello world

```python
import os
# Required on Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CUDA + torch DLL search paths
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
os.add_dll_directory(r"C:\path\to\venv\Lib\site-packages\torch\lib")

# uvloop stub no longer needed as of v0.19.1 — the patch wraps
# `import uvloop` in a try/except ImportError with an asyncio fallback.

from vllm import LLM, SamplingParams

llm = LLM(
    model=r"E:\models\Qwen3-14B-AWQ-4bit",
    dtype="float16",
    kv_cache_dtype="isoquant4",   # 2× KV cache capacity, near-FP16 quality
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
)

outputs = llm.generate(
    ["Explain CUDA streams in three sentences:"],
    SamplingParams(temperature=0.7, max_tokens=200),
)
print(outputs[0].outputs[0].text)
```

For OpenAI-compatible HTTP serving and more usage patterns:
**→ [docs/usage.md](docs/usage.md)**

---

## Multi-TurboQuant: 6 KV cache compression methods

vLLM v0.19.1 on Windows ships with integrated support for six KV cache
compression methods from the [Multi-TurboQuant](https://github.com/aivrar/multi-turboquant)
library:

| Method | Bits | Family | Calibration | Use case |
|---|---|---|---|---|
| `isoquant4` | 4.25 | quaternion 4D rotation | none | **Recommended default** |
| `planarquant4` | 4.25 | Givens 2D rotation | none | Same memory, simpler transform |
| `isoquant3` | 3.25 | quaternion 4D rotation | none | More aggressive |
| `planarquant3` | 3.25 | Givens 2D rotation | none | More aggressive |
| `turboquant35` | 3.25 | WHT + MSE codebook + QJL | runtime | Calibrated outliers |
| `turboquant25` | 2.25 | WHT + MSE codebook + QJL | runtime | Most compression |

Just pass the method name as `kv_cache_dtype` when constructing an
`LLM` (or `--kv-cache-dtype` to `vllm serve`). The cache will
automatically use uint8 packed storage and the attention backend will
decode active blocks on each forward pass.

**Trade-off**: throughput drops ~30-300× with TQ enabled because the
encode/decode currently runs in PyTorch (no fused Triton kernel yet).
Memory savings are real, throughput cost is the price. Best for
offline / long-context / batch workloads. See
**[docs/turboquant.md](docs/turboquant.md)** for the full picture.

---

## What's in the patch

`vllm-windows-v4.patch` is a unified diff against `vllm-project/vllm`
at tag `v0.19.1`. It touches **34 files** + **1 new file**:

- **Build system** (4): CMakeLists, cmake/utils, setup.py, requirements/cuda.txt
- **CUDA kernels** (16): MSVC compatibility for keyword operators,
  designated initializers, `__builtin_clz`, variable templates with
  attributes, nested constexpr lambdas, deeply nested `else if`,
  `__attribute__((aligned))`, `std::isinf`, `__int128_t`
- **Runtime Python** (8): `fcntl` → `msvcrt`, ZMQ IPC → TCP, fork →
  spawn, NCCL → FakeProcessGroup, custom safetensors reader for small
  pagefile systems
- **Multi-TurboQuant integration** (4 + 1 new): 6 new `CacheDType`
  literals, dtype mapping, attention backend dispatch, plus the new
  `vllm/v1/attention/ops/multi_turboquant_kv.py` (295 lines)

Full per-file breakdown → **[docs/build.md](docs/build.md#whats-in-the-patch)**

All changes are guarded by `#ifdef _MSC_VER`, `sys.platform == "win32"`,
or similar conditionals. **Zero impact on Linux builds.**

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
| CUDA driver | R545+ | latest |
| Python | 3.10.x | 3.10.11 |
| Compiler (build only) | VS 2022 Community + Win 10 SDK | Same |
| CUDA Toolkit (build only) | 12.6 | 12.6 |

For build-from-source, you also need a **Windows pagefile** (system
managed is fine). Without it, large allocations during compilation can
fail. See [docs/troubleshooting.md → OSError 1455](docs/troubleshooting.md#oserror-1455).

---

## Tested with

- RTX 3090 (24 GB, SM 8.6, driver 591.86)
- Qwen3-14B-abliterated-AWQ-4bit
- Qwen3.5-9B-abliterated-GPTQ-4bit (text-only)
- Windows 10 Pro 22H2
- Visual Studio 2022 Community 17.13
- CUDA Toolkit 12.6
- Python 3.10.11

### v0.19.1 end-to-end test run (RTX 3090, Qwen3-14B AWQ-4bit)

Smoke test (FlashAttention 2 backend, `kv_cache_dtype=auto`): **933 ms
for 16 tokens**, ~17 tok/s.

All six TurboQuant methods (Triton attention backend, PyTorch-fallback
encode/decode on Windows). Timings are for 5 decoded tokens with
`max_model_len=512`, `gpu_memory_utilization=0.5`:

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
- **No FlashInfer.** No Windows wheel.
- **No FlashAttention 3.** FA3 has MSVC-incompatible PTX macros.
  FlashAttention 2 works fine.
- **TQ throughput is unoptimized.** Encode/decode runs in
  PyTorch-vectorised mode. Memory savings are real, throughput cost is
  the trade-off until a fused Triton kernel lands.
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
| [Multi-TurboQuant](https://github.com/aivrar/multi-turboquant) | KV cache compression methods |
| [TurboQuant paper](https://arxiv.org/abs/2501.06725) | Walsh-Hadamard quantization |

Built with the help of [Claude](https://claude.ai).

---

## License

MIT. See [LICENSE](LICENSE).
