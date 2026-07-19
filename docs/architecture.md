# Architecture

How the v0.25.1 Windows build hangs together and what each piece owns.

## Repository Layout

```text
vllm-windows-build/
  README.md
  VLLM.md
  vllm-windows-v9.patch
  build.bat
  run_build.bat
  install.bat
  launch.bat
  verify_artifact.py
  verify_bootstrap.ps1
  expand_zip.ps1
  verify_install.py
  engine_dispatcher.py
  assemble_wheel_cu128_v0.25.1.py
  vllm_launcher.py
  dist-v9/
    vllm-0.25.1+cu128-cp313-cp313-win_amd64.whl
  docs/
  tests/
```

## Layer 1: Windows Compatibility Patch

`vllm-windows-v9.patch` is a unified diff against upstream
`vllm-project/vllm` tag `v0.25.1`.

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
- KV offload: Windows-safe DMA, shared mmap, filesystem I/O/cache paths,
  native `fs_io_C.pyd`, CPU LRU/ARC, and tiered persistent storage.

See [build.md](build.md) for the current build flow.

The wheel assembler overlays generated FlashAttention Python modules from
the fetched dependency when needed and refuses to produce an artifact if
the rotary or CuteDSL payload is incomplete. `tests/test_wheel_contents.py`
then validates every ZIP member against wheel RECORD before release.

## Layer 2: Portable Installer

`install.bat` provides the no-compiler path:

1. Downloads embedded Python 3.13.14 and verifies its exact size and SHA-256.
2. Adds CPython development files (`Include\Python.h` and
   `libs\python313.lib`) from the Python NuGet package for Triton's
   runtime CUDA helper compilation.
3. Installs PyTorch 2.11.0+cu128 and `triton-windows`.
4. Verifies and installs the v0.25.1+cu128 wheel, the pinned Multi-TurboQuant
   wheel, and structured-output backends.
   Artifacts are downloaded through `.part` files and the install marker stores
   both release SHA-256 values only after the full runtime check succeeds.
5. Verifies both `import vllm` and Triton's CUDA driver path.

Before Python exists, `verify_bootstrap.ps1` computes SHA-256 through .NET and
`expand_zip.ps1` extracts archives through `System.IO.Compression`. This avoids
PowerShell module cmdlets that are absent in some Windows environments; the ZIP
helper also rejects entries that escape the destination directory.

`launch.bat` checks the exact marker hash plus Python, native/Rust,
FlashAttention, model-module, dependency, and Triton development files before
starting the server.
If anything is incomplete, it reruns `install.bat`.

For wheel installs, both scripts prefer Triton's bundled CUDA helper
toolkit when it is present, avoiding accidental use of an incompatible
system `CUDA_PATH`.

## Layer 3: Serving Entry Points

There are two serving paths:

- `vllm_launcher.py`: a Windows-friendly OpenAI-compatible server that
  keeps the launch path explicit and supports chat completions,
  embeddings, health checks, streaming, and parsed tool calls.
- `vllm serve`: upstream vLLM's CLI server, with the Windows event-loop
  and process-handling fixes carried in the patch.

`launch.bat` starts `vllm_launcher.py` and pins the portable environment.
`engine_dispatcher.py` gives one task exclusive ownership of `engine.step()`
and routes each output to its request queue so concurrent streams cannot steal
or discard another request's output. The installer adds the repository root to
`python313._pth` so embedded Python can import this launcher module explicitly.

## Layer 4: Prompt-KV Offload

`OffloadingConnector` remains the vLLM integration point. The launcher maps
its opt-in modes onto two native vLLM specs:

- `CPUOffloadingSpec`: a pinned-RAM primary tier with LRU or ARC eviction.
- `TieringOffloadingSpec`: the same RAM primary tier plus a filesystem
  secondary tier with independent read- and write-priority worker pools.

The GPU worker transfers blocks through CUDA DMA. On Windows, file-backed mmap
restores use the native batch-copy route because Triton cannot safely
dereference that registered host pointer for every payload shape. The scheduler
and GPU worker share a Windows `mmap.ACCESS_WRITE` mapping in the system temp
directory. Filesystem blocks use binary mode and a sanitized, hashed namespace
derived from the complete model/cache configuration.

Offload is disabled by default. Filesystem mode requires an explicit root,
sets no disk quota, and needs a fixed `PYTHONHASHSEED` before process startup
for cache filenames to remain reusable across restarts. `launch.bat` sets the
seed; users still own cache-directory capacity and cleanup.

## Layer 5: KV Cache Compression

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

## Layer 6: Windows Safetensors Reader

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
- Add an explicit filesystem byte quota/eviction policy before considering the
  storage tier a general-purpose default.
- Evaluate remote/distributed tiers separately; this release does not port
  LMCache P2P, NIXL, GDS, object-store, or remote cache features.
