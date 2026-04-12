# Architecture

How the v0.19.0 Windows build hangs together — what each piece does
and why it exists.

## Repository layout

```
vllm-windows-build/
├── README.md                 ← entry point
├── vllm-windows-v3.patch     ← the unified diff (33 files, ~1640 lines)
├── patches/
│   └── multi_turboquant_kv.py  ← reference copy of the new file in the patch
├── build.bat                 ← from-source build (apply patch + pip install -e .)
├── run_build.bat             ← convenience wrapper that sets VS / CUDA env vars
├── install.bat               ← pre-built wheel installer (Python embedded)
├── launch.bat                ← interactive model launcher
├── build_wheel.py            ← packages compiled vllm-source/ into a .whl
├── vllm_launcher.py          ← OpenAI-compatible HTTP server
├── dist-v3/
│   └── vllm-0.19.0+cu126-cp310-cp310-win_amd64.whl  ← release artifact
├── tests/                    ← end-to-end test scripts
│   ├── README.md
│   ├── test_v19.py
│   ├── test_tq_real.py
│   └── test_tq_thorough.py
└── docs/                     ← this directory
    ├── install.md
    ├── build.md
    ├── turboquant.md
    ├── benchmarks.md
    ├── troubleshooting.md
    └── architecture.md       ← (you are here)
```

## Three layers

### Layer 1: the Windows compatibility patch

`vllm-windows-v3.patch` is the *only* thing in this repo that touches
the upstream vLLM source. It exists because every version of vLLM
needs roughly the same set of fixes to compile on MSVC and run on
Windows runtime, and shipping it as a single diff against an upstream
tag makes the changes auditable and rebaseable.

Categories (more detail in [build.md](build.md#whats-in-the-patch)):

1. **Build system** — CMake CUDA toolkit forcing, MSVC preprocessor
   flags, Ninja generator detection
2. **CUDA kernels** — MSVC compatibility for things gcc accepts:
   keyword operators, designated initializers, `__builtin_clz`,
   variable templates with attributes, `__attribute__((aligned))`,
   nested constexpr lambdas, deeply nested `else if` chains
3. **Runtime Python** — `fcntl` → `msvcrt`, ZMQ IPC → TCP, fork →
   spawn, NCCL/Gloo → FakeProcessGroup with FileStore, `SO_REUSEPORT`
   guards
4. **Multi-TurboQuant integration** — adds 6 KV cache dtypes and the
   dispatch helper

### Layer 2: vllm_launcher.py + serve.py

vLLM v0.19.0 ships an OpenAI-compatible server (`vllm serve`), but on
Windows it depends on `uvloop` (Linux-only) and uses the multiprocess
ZMQ engine which is unreliable on Windows.

`vllm_launcher.py` is a thin Windows-friendly replacement:
- Stubs out `uvloop` before importing vLLM
- Uses `InprocClient` (single process, no ZMQ)
- Implements `/v1/chat/completions` (streaming + non-streaming) and
  `/v1/models`
- Supports tool calling via parsed `<tool_call>` tags

It's a single 30 KB file. You can swap it for the upstream `vllm
serve` once vLLM upstream supports Windows out of the box.

### Layer 3: Multi-TurboQuant KV cache compression

The TQ integration is the meat of this v0.19.0 release. The flow:

```
                        ┌──────────────────────────┐
   user code            │  LLM(kv_cache_dtype=     │
                        │       'isoquant3')       │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
   vllm config          │  CacheDType Literal      │
                        │  + STR_DTYPE_TO_TORCH    │
                        │    → torch.uint8         │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
   vllm kv allocation   │  AttentionSpec           │
                        │    dtype=uint8           │
                        │    page_size halved      │
                        │  → KV cache buffer       │
                        │    is 1/2 the size       │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
   per cache write     │  TritonAttentionImpl     │
                        │    .do_kv_cache_update   │
                        │      ↓                   │
                        │  multi_turboquant_kv.    │
                        │    tq_write_kv_cache     │
                        │      ↓                   │
                        │  multi_turboquant.       │
                        │    {iso/planar/turbo}    │
                        │    .encode()             │
                        │      → packed uint8      │
                        │      ↓                   │
                        │  scatter to cache slots  │
                        └──────────────────────────┘

                        ┌──────────────────────────┐
   per attention call  │  TritonAttentionImpl     │
                        │    .forward              │
                        │      ↓                   │
                        │  multi_turboquant_kv.    │
                        │    tq_decode_active_     │
                        │    blocks                │
                        │      ↓                   │
                        │  gather active blocks    │
                        │  decode to fp16          │
                        │  remap block_table       │
                        │      ↓                   │
                        │  unified_attention       │
                        │  (standard Triton kernel)│
                        │  on compact temp cache   │
                        └──────────────────────────┘
```

The persistent cache is small (uint8, half the size). The temporary
fp16 buffer for each attention call only contains the blocks
referenced by the current batch — typically 100x smaller than the
full cache.

### The custom safetensors reader

Not strictly part of the TQ integration, but a separate Windows fix
that ships with this build.

**Problem**: on Windows systems with the pagefile set to zero (which
is common on workstations with lots of RAM), `safetensors.safe_open`
fails with `OSError 1455` because Windows commit charge runs out
during the mmap. Subsequent fallbacks (eager file read, torch.empty
allocations) also fail because they all need ~1.5 GB of contiguous
committed memory for the embedding tensor.

**Fix**: a new `_windows_safetensors_iterator` in
`vllm/model_executor/model_loader/weight_utils.py` that:

1. Memory-maps the safetensors file via `numpy.memmap` — uses the file
   itself as backing storage, no commit charge
2. For tensors >256 MB, allocates the destination directly on the GPU
   and streams the bytes in 64 MB chunks (avoids the CPU staging
   buffer)
3. For smaller tensors, returns a zero-copy `torch.from_numpy` view
   over the mmap

The result: model loading **29× faster** (6.5s vs 189s) and works
without a pagefile.

## Why a separate `multi_turboquant_kv.py` file

The patch could have inlined this logic into `triton_attn.py`, but a
separate file:
- Keeps the diff against upstream small (one new file vs heavy edits
  to a 600-line existing file)
- Makes the dispatcher easy to swap out (e.g. when porting to vLLM
  v0.20.0+)
- Makes it obvious where the integration lives — anyone reading the
  code finds the entry point quickly

The file is referenced from `triton_attn.py` via two top-level imports
and is invoked from `do_kv_cache_update` and `forward`. The rest of
vLLM is untouched.

## Future work

1. **Fused Triton encode/decode kernel** — biggest performance win.
   The current PyTorch-vectorised path is correct but ~150× slower
   than fp16. A fused kernel would close most of the gap.
2. **Override `AttentionSpec.real_page_size_bytes` for TQ** — currently
   the cache uses `head_size` bytes per slot but only `packed_dim`
   (54-70 bytes for the 6 supported methods) carry data. Shrinking the
   cache to exactly `packed_dim` bytes per slot would take memory
   savings from 50% → 75-80%.
3. **Calibrated outlier indices for TurboQuant** — currently uses fixed
   "first N dims" which is the worst case. Plugging in the metadata
   from `multi_turboquant.calibration.generate_metadata` would
   improve quality without any kernel changes.
4. **Multi-GPU support** — single-GPU only for now. NCCL is unavailable
   on Windows so multi-GPU needs custom CPU coordination.
