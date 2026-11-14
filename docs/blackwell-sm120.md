# Native SM120 (Blackwell) FP8/NVFP4 kernels - experimental add-on patch

> **Status: experimental and opt-in.** The release wheels do **not** include these kernels. The add-on
> changes nothing unless you set the environment switches below. It has been run on **one GPU, an
> RTX 5090 (SM 12.0)**, and has not been built or regression-tested for SM 7.5/8.6/8.9, with CUDA 13.0/13.1,
> or with an accuracy evaluation. See [Validation](#validation).

`vllm-windows-v12.patch` omits the native SM120 CUTLASS FP8 scaled-mm and SM12x NVFP4 kernels because their
generated host stubs do not compile with MSVC (see [issue #16](https://github.com/aivrar/vllm-windows-build/issues/16)).
This add-on re-enables them on top of v12 and records what is needed to make them compile.

## Files

| File | Purpose |
|---|---|
| `patches/vllm-windows-v12-sm120.patch` | Apply **after** `vllm-windows-v12.patch`. CMake opt-in switches, FlashAttention target tweak, source fixes, and a new `cutlass-windows-sm120.patch` at the source root. |
| `patches/cutlass-windows-sm120.patch` | The CUTLASS hunk on its own (identical to the file the patch above installs). Applied to the fetched CUTLASS by CMake. |
| `patches/make-cuda-shim.ps1` | Creates the `cuda.h` shim directory described below. |

## Why they were omitted

1. **`C2719: formal parameter with requested alignment of 128 won't be aligned`** in the generated
   `.cudafe1.stub.c`. CUTLASS kernels take a by-value `Params` struct that holds TMA descriptors, and
   `CUtensorMap` is declared `alignas(128)` in the CUDA 13.x `cuda.h` (per the maintainer's release build
   machine, 13.0 and 13.1 declare 128 as well; only 12.6 and 12.8 declare 64). MSVC cannot pass such a
   parameter by value. Compiling against a copy of `cuda.h` with `alignas(64)` fixes it.
2. **cute/CUTLASS fails to compile in C++20 mode** with CUDA 13.5.37 (`C3545: 'Ints': parameter pack expects a
   non-type template argument` in `cute/stride.hpp`, then a cascade of `C2976 cute::Layout`). The same errors
   appeared with MSVC 14.43 and 14.44. They disappear with `-std=c++17`. The release build (CUDA 13.0) compiles
   these files as C++20, so this override may not be needed there - it has not been tried.
3. **Small MSVC/UCRT issues** (harmless on other platforms, applied unconditionally): `FLOAT`/`INT`/`UINT8`
   clash with Windows typedefs in `nvfp4_experts_quant.cu`; UCRT `INFINITY` is `(float)1e300`, which CUDA 13.5
   rejects in a `constexpr float` (`grouped_topk_kernels.cu`); `NAN` is not visible inside CUTLASS
   `tensor_fill.h` (`nvfp4_blockwise_moe_kernel.cu`).

## Switches (all default OFF)

| Environment variable | Effect |
|---|---|
| `VLLM_WIN_NATIVE_SM120=1` | Builds the SM120 scaled-mm and SM12x NVFP4 sources (the two `NOT WIN32` guards in v12) and applies `cutlass-windows-sm120.patch` to CUTLASS. Requires the `cuda.h` shim with CUDA 13.x. |
| `VLLM_WIN_CXX17_CUTLASS=1` | Builds the CUTLASS-heavy sources (`csrc/quantization/{w8a8/cutlass,fp4,cutlass_w4a8}`, `csrc/sparse/cutlass`, `csrc/attention/mla`) and the `_vllm_fa2_C`/`_vllm_fa3_C` targets as C++17. Needed with CUDA 13.5. |
| `VLLM_CUDA_SHIM_DIR=<dir>` | Puts `<dir>` first on the nvcc include path (for the patched `cuda.h`). |

With none of them set, the generated build is the same as v12 plus the three unconditional source fixes above.

## Build

```bat
:: 1. Apply on top of upstream v0.33.0
git apply vllm-windows-v12.patch
git apply patches/vllm-windows-v12-sm120.patch

:: 2. cuda.h shim (CUDA 13.x)
powershell -File patches\make-cuda-shim.ps1 -Out C:\vllm-build\shim

:: 3. Build for Blackwell only (add the usual archs for a multi-arch wheel)
set VLLM_WIN_NATIVE_SM120=1
set VLLM_WIN_CXX17_CUTLASS=1
set VLLM_CUDA_SHIM_DIR=C:/vllm-build/shim
set TORCH_CUDA_ARCH_LIST=12.0a
set VLLM_TARGET_DEVICE=cuda
set CMAKE_BUILD_TYPE=Release
set VLLM_DISABLE_SCCACHE=1
set SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0
set MAX_JOBS=8
```

then continue with the normal [build workflow](build.md) (`tools\build_rust.py --release`, `setup.py build_ext --inplace -v`,
`uv build --wheel --no-build-isolation`). `TORCH_CUDA_ARCH_LIST` needs `12.0a` (not `12.0`) for the NVFP4 kernels. Do not start a
second build while ninja is still draining the first one in the same build directory.

## Validation

Tested configuration: Windows 11, RTX 5090 (SM 12.0), driver 621.24, AVX-512 enabled, CUDA 13.5.37, MSVC 19.44,
Python 3.13, PyTorch 2.14.0+cu130, triton-windows 3.8.0.post29, `TORCH_CUDA_ARCH_LIST=12.0a`, both switches on.
The build compiled all 82 CUDA/C++ objects and assembled the wheel. (That build used an equivalent, unconditional form of the
same edits; the opt-in form in this patch was verified with `git apply` on a clean v12 tree, against the real CUTLASS source,
and for the switch logic - see the pull request for any later full-build result.)

| Check | Result |
|---|---|
| `cutlass_scaled_mm` FP8, `scaled_fp4_quant`, `cutlass_scaled_mm_supports_fp8/block_fp8/fp4(120)` | run on the 5090, sane values, all `True` |
| `unsloth/Qwen3.8-27B-NVFP4` (FP8 + NVFP4 layers; the checkpoint from issue #16), MTP 3 draft tokens, fp8 KV cache | selects `CutlassFP8ScaledMMLinearKernel` and `CutlassNvFp4LinearKernel`; ~143 tok/s cold / ~146 warm single-stream at a 122,880-token context; tool calls work |
| Same checkpoint, 89,625-token prompt with a hidden code | code retrieved correctly; prefill ~3,050 tok/s; GPU memory stable (~29 GB) |
| `Inferact/Qwen3.8-27B-NVFP4` (pure NVFP4), MTP 2 tokens | ~88-102 tok/s cold / ~103 warm on the same flags as a native 0.27.x build that did ~74 / ~93 |

**Not tested:** builds with CUDA 13.0/13.1, a multi-arch (`7.5;8.6;8.9;12.0`) build, any non-Blackwell GPU, other SM 12.x cards,
accuracy against the Marlin path (smoke tests, a tool call and one needle prompt only), and the `-std=c++17` override outside CUDA 13.5.

## Runtime notes

* FlashInfer is not installed on Windows, but the `auto` GDN prefill setting selects it on SM12.x. Use `--gdn-prefill-backend triton`.
* With an fp8 KV cache vLLM only allows FA3 on SM90 / FA4 on SM100, so `TRITON_ATTN` is the only valid attention backend on SM120.
