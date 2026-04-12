# Build from Source

This page covers what each phase of the v0.19.0 Windows build actually
does, what to look for if it breaks, and how to iterate when modifying
the patches.

For a quick "I just want to build it" walk-through, see [install.md](install.md#b-build-from-source).

---

## What's in the patch

`vllm-windows-v3.patch` is a unified diff against `vllm-project/vllm`
at tag `v0.19.0`. It touches **33 files** across four categories:

### 1. Build system (4 files)

| File | Purpose |
|---|---|
| `CMakeLists.txt` | Force `CUDA_HOME` toolkit, MSVC `/Zc:preprocessor`, quote python paths, link `CUDA::cublas` |
| `cmake/utils.cmake` | Quote executable paths in `file(REAL_PATH)` |
| `setup.py` | Allow Windows CUDA builds, backslash → forward-slash, find Ninja in venv Scripts |
| `requirements/cuda.txt` | Comment out flashinfer, nvidia-cudnn-frontend, cutlass-dsl, quack-kernels (no Windows wheels) |

### 2. CUDA kernels (16 files) — MSVC compatibility

MSVC's host compiler is stricter than gcc/clang in a few specific ways
nvcc relies on. Each fix is the smallest change that compiles cleanly:

| Pattern | Files | Fix |
|---|---|---|
| `__builtin_clz` (gcc-only) | `csrc/core/math.hpp` | Portable bit-twiddling |
| `__attribute((aligned))` | `csrc/moe/grouped_topk_kernels.cu` | `__align__(N)` instead |
| `__asm__ __volatile__` | `csrc/quantization/awq/gemm_kernels.cu` | `asm volatile` (MSVC PTX syntax) |
| Designated initializers `{.x = N}` in CUDA | `csrc/quantization/activation_kernels.cu`, `csrc/topk.cu` | Positional init `{N}` |
| `__int128_t` / `__int64_t` (gcc-only) | `csrc/quantization/activation_kernels.cu` | `int4` / `int64_t` |
| `or` / `and` / `not` keywords | `csrc/quantization/gptq_allspark/...`, `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu` | `\|\|` / `&&` / `!` |
| `std::isinf` in device code | `csrc/attention/merge_attn_states.cu` | `isinf` (CUDA built-in) |
| Variable templates with `__device__` | `csrc/quantization/utils.cuh` | Function templates |
| Nested `BOOL_SWITCH` lambdas | `csrc/mamba/mamba_ssm/selective_scan_fwd.cu` | Explicit template dispatch |
| Deeply nested `else if` (C1061) | `csrc/quantization/marlin/generate_kernels.py`, `csrc/moe/marlin_moe_wna16/generate_kernels.py` | Flat `if` chain |
| `__builtin_clz` / `ssize_t` | `csrc/cumem_allocator.cpp` | `<BaseTsd.h>`, `SSIZE_T` |
| Missing `uint` typedef | several `.cu` files | `typedef unsigned int uint;` under `#ifdef _MSC_VER` |
| `subprocess.call(["rm", ...])` | both `generate_kernels.py` files | `os.remove(...)` |

### 3. Runtime Python (8 files) — Windows platform fixes

| File | Fix |
|---|---|
| `vllm/distributed/parallel_state.py` | PyTorch Windows builds have no Gloo TCP/UV — fall back to `FakeProcessGroup` + `FileStore` |
| `vllm/utils/network_utils.py` | ZMQ IPC sockets unavailable — use `tcp://127.0.0.1` instead |
| `vllm/utils/system_utils.py` | Force `spawn` multiprocessing (no `fork` on Windows) |
| `vllm/v1/engine/core_client.py` | Force `InprocClient` (multiprocess ZMQ fails with spawn) |
| `vllm/entrypoints/openai/api_server.py` | Guard `SO_REUSEPORT` |
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | `fcntl.flock` → `msvcrt.locking` |
| `vllm/model_executor/model_loader/weight_utils.py` | Custom safetensors reader (numpy mmap → chunked GPU streaming) for systems with small/disabled pagefile |
| `vllm/v1/attention/backends/triton_attn.py` | Wire Multi-TurboQuant encode/decode dispatch |

### 4. Multi-TurboQuant integration (4 files + 1 new)

| File | Change |
|---|---|
| `vllm/config/cache.py` | Add 6 new `CacheDType` literals: `turboquant25`, `turboquant35`, `isoquant3`, `isoquant4`, `planarquant3`, `planarquant4` |
| `vllm/utils/torch_utils.py` | Map all 6 TQ dtypes to `torch.uint8` (storage type) |
| `vllm/v1/attention/backends/triton_attn.py` | Hook `do_kv_cache_update` and `forward` to dispatch through the TQ helper |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | Pass through TQ dtypes (handled upstream) |
| **`vllm/v1/attention/ops/multi_turboquant_kv.py` (NEW)** | The encode/decode bridge — gathers active blocks, calls `multi_turboquant.{iso,planar,turbo}_quant.encode/decode`, remaps `block_table` for the compact temporary cache |

The new file is included in the patch (it shows up as a `--- /dev/null`
add) so a single `git apply vllm-windows-v3.patch` does everything.

---

## Build phases

### Phase 1: CMake configure (~5-10 min)

Triggered by `pip install -e .`, runs `cmake -G Ninja vllm-source`.

What happens:
1. Detect MSVC, CUDA toolkit version, Python, torch, GPU arch list
2. `FetchContent` clones CUTLASS, FlashAttention, triton_kernels into `.deps/`
3. Generate Marlin kernel sources (`generate_kernels.py` produces ~250 `.cu` files)
4. Write `build.ninja`

**What to watch for:**
- `Found CUDA: ... (found version "12.6")` — confirms it's not picking up CUDA 13.x via VS MSBuild integration
- `Build files have been written to: ...build-temp/Release` — configure done

If CMake errors out with "generator: Visual Studio 17 2022 does not match
the generator used previously: Ninja", clean `.deps/`:

```bat
rmdir /s /q vllm-source\.deps
```

### Phase 2: Compile (~25-35 min)

Runs `cmake --build .` which invokes Ninja.

Targets:
1. `cumem_allocator` (~1 min) — small CXX
2. `_C` (~10 min) — main CUDA extension, 30+ kernels
3. `_moe_C` (~10 min) — Marlin MOE templates, lots of variants
4. `_C_stable_libtorch` (~30 sec)
5. `_vllm_fa2_C` (~10 min) — FlashAttention 2 with bundled CUTLASS

**What to watch for:**
- `[N/140] Building CUDA object ...` — progress
- `error MSB3721` — compiler crashed (usually OOM, lower `MAX_JOBS`)
- `catastrophic error: out of memory` — same; lower `MAX_JOBS`
- `cl : error C2018: unknown character 'or' / 'and' / 'not'` — missing operator-keyword fix; check the patch applied cleanly

### Phase 3: Editable install (~30 sec)

`pip install -e .` writes a `__editable__.vllm-0.19.0+cu126.pth` finder
into the venv site-packages that points back at `vllm-source/vllm/`.
The compiled `.pyd` files live in `vllm-source/vllm/` next to the
Python sources.

---

## Iterating on the patch

If you change a `.cu` file in `vllm-source/`, just re-run `pip install
-e . --no-build-isolation`. Ninja only recompiles what changed (~10-30s
per kernel).

If you change `setup.py` or `CMakeLists.txt`, you usually need to clear
the build temp:

```bat
del /s /q "%TEMP%\tmp*.build-temp"
```

If you change anything in `vllm-source/vllm/*.py`, the editable install
picks it up automatically — just re-run your test.

If you change `vllm/v1/attention/ops/multi_turboquant_kv.py` (the TQ
helper), no rebuild needed — it's pure Python.

---

## Regenerating the patch

After modifying files in `vllm-source/`:

```bat
cd vllm-source
git add -N vllm/v1/attention/ops/multi_turboquant_kv.py
git diff > ..\vllm-windows-v3.patch
```

The `git add -N` step is needed to make the new
`multi_turboquant_kv.py` show up in the diff (it's untracked).

---

## Build environment cheat-sheet

| Variable | Value | Why |
|---|---|---|
| `CUDA_HOME` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6` | Forces CMake to ignore other CUDA installs |
| `TORCH_CUDA_ARCH_LIST` | `8.6` (RTX 30xx) | Compiles only for your GPU; saves ~30% build time |
| `VLLM_TARGET_DEVICE` | `cuda` | Required on Windows since vLLM defaults to "empty" |
| `MAX_JOBS` | `4` (32 GB) or `8` (64 GB) | Higher uses more RAM during compile |
| `SETUPTOOLS_SCM_PRETEND_VERSION` | `0.19.0` | Stops setuptools-scm from reading git tags |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Required at runtime if Windows pagefile is small |
