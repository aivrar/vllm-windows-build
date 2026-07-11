# Architecture

How the v0.24.0 Windows build hangs together and what each piece owns.

## Repository Layout

```text
vllm-windows-build/
  README.md
  VLLM.md
  vllm-windows-v8.patch
  build.bat
  run_build.bat
  install.bat
  launch.bat
  assemble_wheel_cu128_v0.24.0.py
  vllm_launcher.py
  dist-v8/
    vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl
  docs/
  tests/
```

## Layer 1: Windows Compatibility Patch

`vllm-windows-v8.patch` is a unified diff against upstream
`vllm-project/vllm` tag `v0.24.0`.

Main categories:

- Build system: MSVC/CUDA flags, CUDA 12.8 paths, CUTLASS patching, and
  skips for Linux-only optional extensions, plus Windows-safe generated
  FlashAttention Python-file copy-back.
- CUDA kernels: MSVC compatibility for GCC-only syntax and generated
  selector depth.
- Runtime Python: Windows multiprocessing, event-loop, networking,
  safetensors, and FakeProcessGroup fallbacks.
- Rust artifacts: packaging for `vllm-rs.exe` and `_rust_tool_parser.pyd`.
- Multi-TurboQuant: six local KV-cache compression methods carried
  alongside the four upstream TurboQuant variants.

See [build.md](build.md) for the current build flow.

The wheel assembler overlays generated FlashAttention Python modules from
the fetched dependency when needed and refuses to produce an artifact if
the rotary or CuteDSL payload is incomplete. `tests/test_wheel_contents.py`
then validates every ZIP member against wheel RECORD before release.

## Layer 2: Portable Installer

`install.bat` provides the no-compiler path:

1. Downloads embedded Python 3.13.11.
2. Adds CPython development files (`Include\Python.h` and
   `libs\python313.lib`) from the Python NuGet package for Triton's
   runtime CUDA helper compilation.
3. Installs PyTorch 2.11.0+cu128 and `triton-windows`.
4. Installs the v0.24.0+cu128 wheel and structured-output backends.
5. Verifies both `import vllm` and Triton's CUDA driver path.

`launch.bat` checks for missing Python, package directories, marker
files, and Triton Python development files before starting the server.
If anything is incomplete, it reruns `install.bat`.

For wheel installs, both scripts prefer Triton's bundled CUDA helper
toolkit when it is present, avoiding accidental use of an incompatible
system `CUDA_PATH`.

## Layer 3: Serving Entry Points

There are two serving paths:

- `vllm_launcher.py`: a Windows-friendly OpenAI-compatible server that
  keeps the launch path explicit and supports chat, completions,
  embeddings, health checks, streaming, and parsed tool calls.
- `vllm serve`: upstream vLLM's CLI server, with the Windows event-loop
  and process-handling fixes carried in the patch.

`launch.bat` starts `vllm_launcher.py` and pins the portable environment.

## Layer 4: KV Cache Compression

The current build exposes ten KV-cache compression dtypes:

- Six local Multi-TurboQuant methods:
  `isoquant3`, `isoquant4`, `planarquant3`, `planarquant4`,
  `turboquant25`, `turboquant35`.
- Four upstream TurboQuant variants:
  `turboquant_k8v4`, `turboquant_4bit_nc`,
  `turboquant_k3v4_nc`, `turboquant_3bit_nc`.

The local methods are wired through
`vllm/v1/attention/ops/multi_turboquant_kv.py` and
`vllm/v1/attention/backends/triton_attn.py`.

The persistent KV cache is stored as `torch.uint8`, roughly halving cache
memory. Active blocks are decoded into a compact fp16 temporary cache for
the attention call. The memory savings are real; the local methods still
pay a throughput cost because encode/decode currently run through a
PyTorch fallback path.

## Layer 5: Windows Safetensors Reader

The patch keeps the custom Windows safetensors path for systems with a
small or disabled pagefile:

1. Memory-map safetensors files with `numpy.memmap`.
2. Stream large tensors to GPU in chunks.
3. Avoid large committed CPU staging buffers that trigger Windows
   `OSError 1455`.

## Future Work

- Fuse local Multi-TurboQuant encode/decode into Triton kernels.
- Shrink local TQ cache slots to the exact packed dimension instead of
  preserving the standard slot width.
- Add calibrated outlier indices for `turboquant25`/`turboquant35`.
- Improve multi-GPU support beyond the current single-GPU Windows path.
