# Patch Reference

Detailed breakdown of every change in `vllm-windows-v3.patch`,
organized by category. This is the v0.19.0 patchset; the older
v0.17.1 patch (`vllm-windows-v2.patch`) and v0.14.x patch
(`vllm-windows.patch`) are kept in the repo for legacy installs.

For build internals (phases, iterating on the patch, regenerating it),
see [docs/build.md](docs/build.md).

## Build environment

| | |
|---|---|
| Base | vLLM v0.19.0 (commit `2a69949bd`) |
| Compiler | MSVC 19.43.34810 (Visual Studio 2022 Community 17.13) |
| CUDA | 12.6 |
| Python | 3.10.11 |
| PyTorch | 2.10.0+cu126 |
| Triton | triton-windows 3.6.0.post26 |
| Generator | Ninja |
| GPU | RTX 3090 (sm_86) — patches built and tested for SM 8.0+ |

## Diff stats

```
33 files changed, 491 insertions(+), 135 deletions(-)
+ 1 new file: vllm/v1/attention/ops/multi_turboquant_kv.py (295 lines)
```

## Files modified

| Category | File | Hunks | Purpose |
|---|---|---|---|
| Build system | `CMakeLists.txt` | 5 | Force CUDA toolkit, MSVC flags, link `CUDA::cublas` |
| Build system | `cmake/utils.cmake` | 1 | Quote paths in `file(REAL_PATH)` |
| Build system | `setup.py` | 5 | Allow Win CUDA, path conversion, find Ninja in venv |
| Build system | `requirements/cuda.txt` | 1 | Comment out flashinfer / cutlass-dsl / quack-kernels |
| CUDA kernel | `csrc/attention/merge_attn_states.cu` | 3 | `uint` typedef, `std::isinf` → `isinf` |
| CUDA kernel | `csrc/core/math.hpp` | 1 | `__builtin_clz` → portable bit-twiddle |
| CUDA kernel | `csrc/cumem_allocator.cpp` | 1 | `<BaseTsd.h>`, `SSIZE_T`/`ssize_t` |
| CUDA kernel | `csrc/fused_qknorm_rope_kernel.cu` | 1 | `uint` typedef |
| CUDA kernel | `csrc/mamba/mamba_ssm/selective_scan_fwd.cu` | 3 | Replace nested `BOOL_SWITCH` lambdas with explicit dispatch |
| CUDA kernel | `csrc/moe/grouped_topk_kernels.cu` | 4 | `__attribute((aligned))` → `__align__` |
| CUDA kernel | `csrc/moe/marlin_moe_wna16/generate_kernels.py` | 2 | Flat `if` chain (avoid C1061), `os.remove` |
| CUDA kernel | `csrc/quantization/activation_kernels.cu` | 8 | designated init, `int4` for `__int128_t`, `int64_t` |
| CUDA kernel | `csrc/quantization/awq/gemm_kernels.cu` | 2 | `__asm__ __volatile__` → `asm volatile` |
| CUDA kernel | `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu` | 1 | `and` → `&&` |
| CUDA kernel | `csrc/quantization/fused_kernels/layernorm_utils.cuh` | 1 | `quant_type_max_v<T>` → `quant_type_max_v<T>()` |
| CUDA kernel | `csrc/quantization/fused_kernels/quant_conversions.cuh` | 1 | same |
| CUDA kernel | `csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu` | 1 | `or` → `\|\|` |
| CUDA kernel | `csrc/quantization/marlin/generate_kernels.py` | 1 | Flat `if` chain, `os.remove` |
| CUDA kernel | `csrc/quantization/utils.cuh` | 2 | Variable template → function template |
| CUDA kernel | `csrc/quantization/w8a8/fp8/common.cu` | 1 | `quant_type_max_v` call site fix |
| CUDA kernel | `csrc/quantization/w8a8/fp8/common.cuh` | 1 | same |
| CUDA kernel | `csrc/topk.cu` | 3 | designated initializer → positional |
| Runtime Python | `vllm/distributed/parallel_state.py` | 2 | `FakeProcessGroup` + `FileStore` instead of Gloo |
| Runtime Python | `vllm/entrypoints/openai/api_server.py` | 1 | Guard `SO_REUSEPORT` |
| Runtime Python | `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | 2 | `fcntl.flock` → `msvcrt.locking` |
| Runtime Python | `vllm/model_executor/model_loader/weight_utils.py` | 8 | Custom safetensors reader (numpy mmap + chunked GPU stream) |
| Runtime Python | `vllm/utils/network_utils.py` | 1 | ZMQ IPC → `tcp://127.0.0.1` |
| Runtime Python | `vllm/utils/system_utils.py` | 1 | Force `spawn` multiprocessing |
| Runtime Python | `vllm/v1/engine/core_client.py` | 1 | Force `InprocClient` |
| TQ integration | `vllm/config/cache.py` | 1 | Add 6 TQ entries to `CacheDType` literal |
| TQ integration | `vllm/utils/torch_utils.py` | 2 | Map TQ dtypes to `torch.uint8` |
| TQ integration | `vllm/v1/attention/backends/triton_attn.py` | 4 | Hook `do_kv_cache_update` and `forward` |
| TQ integration | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | 1 | (compatible passthrough) |
| **TQ integration (NEW)** | `vllm/v1/attention/ops/multi_turboquant_kv.py` | new | 295-line dispatch helper |

---

## Categorized notes

### CMakeLists.txt — CUDA toolkit forcing

When multiple CUDA toolkits are installed, CMake (or VS MSBuild
integration) can pick up the wrong one. Patch forces
`CUDA_TOOLKIT_ROOT_DIR`, `CUDAToolkit_ROOT`, and `CUDA_BIN_PATH` from
`$ENV{CUDA_HOME}` before `find_package(Torch)`.

Also adds an MSVC-specific block:
- `_CRT_DECLARE_NONSTDC_NAMES=1` for POSIX compatibility
- `USE_CUDA` define (activates a CCCL workaround in PyTorch's
  `compiled_autograd.h`)
- `/Zc:preprocessor` for correct variadic macro handling — required by
  the nested `BOOL_SWITCH` macros that survived the kernel patch

### setup.py — find Ninja, allow CUDA on Windows

Default is to set `VLLM_TARGET_DEVICE=empty` on non-Linux. Patch adds
an explicit override path: if `os.getenv("VLLM_TARGET_DEVICE") ==
"cuda"`, allow the build. Also fixes `is_ninja_available()` to look in
the venv `Scripts\` directory (not just `PATH`) so the build picks up
`pip install ninja` automatically.

### MSVC keyword operators (`and`, `or`, `not`)

C++ accepts these as alternative tokens in code, but MSVC doesn't
enable them by default in CUDA files. The patch replaces every
instance with `&&`, `||`, `!` in three files. There's no other way —
even `/Zc:alternateTokens` doesn't enable them in nvcc.

### MSVC `__builtin_clz`

GCC/Clang have `__builtin_clz` (count leading zeros). MSVC uses
`_BitScanReverse` or `__lzcnt`. The patch replaces it with portable
bit-twiddling code in `csrc/core/math.hpp`:

```cpp
uint32_t v = num - 1;
v |= v >> 1; v |= v >> 2; v |= v >> 4;
v |= v >> 8; v |= v >> 16;
return v + 1;
```

### MSVC variable template + `__device__`

`csrc/quantization/utils.cuh` uses a variable template
`quant_type_max_v<T>` annotated with `MAYBE_HOST_DEVICE`. MSVC's nvcc
can't apply `__host__`/`__device__` attributes to variable templates.
The patch converts it to a function template with the same signature
(`quant_type_max_v<T>()` at call sites).

### MSVC nested constexpr lambdas

`csrc/mamba/mamba_ssm/selective_scan_fwd.cu` uses nested `BOOL_SWITCH`
macros that each generate a lambda. The compile-time `bool` template
parameters get captured in lambdas, but MSVC doesn't propagate
`constexpr` through lambda captures the way gcc does.

The patch replaces the nested lambdas with an explicit 8-way `if/else`
dispatch tree calling a helper function template
`selective_scan_fwd_dispatch<..., kIsEvenLen, kHasZ, kVarlen>()` with
all booleans as real template parameters.

### MSVC C1061 — blocks nested too deeply

`csrc/quantization/marlin/generate_kernels.py` and the MOE variant
generate C++ kernel selectors with 700+ branches using `if/else if`
chains. MSVC hits `C1061: compiler limit: blocks nested too deeply`.

The patch generates flat `if` statements instead of `else if`. The
selectors are mutually exclusive by construction (each `if` checks a
unique combination of template parameters), so the flat-`if` form is
semantically equivalent to the chain.

### MSVC designated initializers

CUDA files use C99-style designated initializers like
`__nv_bfloat16_raw{.x = 17376}`. MSVC C++ mode doesn't accept these.
Patch converts to positional initialization `__nv_bfloat16_raw{17376}`.

### MSVC `__int128_t` / `__int64_t`

`__int128_t` doesn't exist in MSVC. The patch replaces it with `int4`
(CUDA's built-in 128-bit vector type) for shared-memory pointers in
`csrc/quantization/activation_kernels.cu`. `__int64_t` is replaced
with the standard `int64_t`.

### Distributed: FakeProcessGroup + FileStore

PyTorch Windows builds have `GLOO_HAVE_TRANSPORT_TCP=false` and
`GLOO_HAVE_TRANSPORT_UV=false`. All Gloo init methods (`file://`,
`env://`, `tcp://`, `GLOO_SOCKET_IFNAME`) fail with
`makeDeviceForHostname(): unsupported gloo device`.

The patch wires up `torch.testing._internal.distributed.fake_pg.FakeProcessGroup`
as the backend on Windows, with a `FileStore` for the rendezvous. This
works for single-rank single-GPU operation because vLLM only uses the
distributed API for rank/world_size bookkeeping when running on one GPU.

This is the biggest single Windows compatibility blocker the patch
solves.

### Custom safetensors reader (weight_utils.py)

The biggest *runtime* fix. Windows uses commit charge (RAM + pagefile)
for any process allocation. On systems with the pagefile disabled or
small, the upstream safetensors `safe_open` mmap path fails with
`OSError 1455 (The paging file is too small for this operation to
complete)` mid-load. Eager file reads and `torch.empty(nbytes)` paths
also fail because all need ~1.5 GB of contiguous committed memory for
the embedding tensor.

The patch adds `_windows_safetensors_iterator` which:
1. `numpy.memmap` the safetensors file (file-backed, no commit charge)
2. For tensors >256 MB, allocate the destination directly on the GPU
   and stream bytes in 64 MB chunks (avoids the CPU staging buffer)
3. For smaller tensors, return a zero-copy `torch.from_numpy` view
   over the mmap

Result: model loading **29× faster** (6.5s vs 189s on Qwen3-14B) and
works without a pagefile.

### TritonAttention Multi-TurboQuant dispatch

Two hooks added to `vllm/v1/attention/backends/triton_attn.py`:

1. **`do_kv_cache_update`**: if `kv_cache_dtype` is one of the 6 TQ
   methods, call `tq_write_kv_cache` (encode + scatter packed bytes
   to the uint8 cache). Otherwise fall through to standard
   `triton_reshape_and_cache_flash`.
2. **`forward`**: same check; if TQ, call `tq_decode_active_blocks`
   to gather and decode the active blocks into a compact fp16 cache,
   remap `block_table` to compact indices, then run the standard
   `unified_attention` Triton kernel on the compact cache.

The dispatch helper lives in
`vllm/v1/attention/ops/multi_turboquant_kv.py` (new file, 295 lines).
It imports from the `multi_turboquant` package at function-call time
to keep import overhead zero for non-TQ runs.

---

## What's NOT patched (known limitations)

- **NCCL** — No Windows support upstream. No multi-GPU tensor parallelism.
- **FlashInfer** — No Windows wheel.
- **FlashAttention 3** — Has MSVC-incompatible PTX macros. FA2 works fine.
- **`nvidia-cutlass-dsl`** — No Windows wheel. Used by FA4 / QuACK; FA2 path is fine.
- **Gloo distributed** — Worked around with `FakeProcessGroup`.
- **TQ throughput** — Encode/decode runs in PyTorch (no fused Triton kernel). Memory savings real, throughput cost is the trade-off.

---

## Stale patches in the repo

Older patches against earlier vLLM versions are kept for legacy
installs:

| Patch file | Base vLLM | Status |
|---|---|---|
| `vllm-windows-v3.patch` | v0.19.0 | **current** |
| `vllm-windows-v2.patch` | v0.17.1 | stale; still works for v0.17.1 builds |
| `vllm-windows.patch` | v0.14.1 | stale; for v0.14.2 legacy install |
