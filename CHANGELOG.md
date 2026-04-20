# Changelog

## v0.19.1-win — 2026-04-19

Point release. Upstream vLLM bumped to 0.19.1, one new Windows patch,
all six Multi-TurboQuant methods re-verified on RTX 3090.

### New

- **vLLM v0.19.1 base** — upstream point release (CI fixes, minor
  bugfixes, pinned `nixl-cu{12,13}` versions, Jina ColBERT rotary
  recomputation for transformers v5).
- **`tests/test_tq_diag.py`** — faulthandler-guarded diagnostic that
  dumps stack traces if `generate()` stalls past 90s. Catches hangs vs
  genuinely slow runs (PyTorch-fallback decode is slow but terminates).

### Changed

- `vllm-windows-v4.patch` is now the active patch. 34 modified files +
  1 new file (`vllm/v1/attention/ops/multi_turboquant_kv.py`). 1656
  lines. The only functional delta vs v3 is a new hunk in
  `vllm/v1/utils.py` (see Fixed).
- `build.bat` now applies `vllm-windows-v4.patch` and sets
  `SETUPTOOLS_SCM_PRETEND_VERSION=0.19.1`.
- `README.md` version badges, tables, and examples updated to v0.19.1.

### Fixed

- **`vllm/v1/utils.py` unconditional `import uvloop`** — upstream added
  this import at module load. `uvloop` is Unix-only, so the import fails
  on Windows before the `Hello world` snippet's `sys.modules` stub gets
  a chance to run. The patch wraps the import in a `try/except
  ImportError` that aliases `asyncio` as `uvloop` (same `run()`
  signature). This means user code no longer needs to stub `uvloop` at
  all — the fallback is baked into the wheel.

### Verified (RTX 3090, Qwen3-14B-abliterated-AWQ-4bit)

All 6 TQ methods load the model, initialize the KV cache, and generate
coherent output end-to-end. Timings are for 5 decoded tokens with
`max_model_len=512`, `gpu_memory_utilization=0.5`:

| Method | Preset | Time (5 tok) | Output tok/s |
|---|---|---:|---:|
| `isoquant3` | no_calibration_symmetric | 41.5s | 0.12 |
| `isoquant4` | no_calibration_quality | 53.0s | 0.09 |
| `planarquant3` | k_only_planar | 40.5s | 0.12 |
| `planarquant4` | k_only_planar | 53.0s | 0.09 |
| `turboquant25` | max_compression | 6.7s | **0.74** |
| `turboquant35` | speed | 5.4s | **0.92** |

`turboquant25/35` are ~8× faster than the iso/planar methods on
Windows — their algorithm favours the PyTorch-fallback path. All
methods still pay the expected ~30-300× throughput cost vs FP16 until
a fused Triton kernel lands (tracked as a known limitation, not new in
v0.19.1).

### Known limitations

Unchanged from v0.19.0-win:

- TQ throughput penalty from PyTorch-fallback encode/decode.
- Single GPU only (NCCL still unavailable on Windows).
- No FlashAttention 3, no FlashInfer.

---

## v0.19.0-win — 2026-04-12

Major release. vLLM 0.19.0 base, full Multi-TurboQuant integration with
real packed-uint8 storage, custom Windows safetensors reader.

### New

- **vLLM v0.19.0 base** — Gemma 4 support, zero-bubble async scheduling,
  Model Runner V2 maturation, online MXFP8, batched chat completions
  endpoint, ViT full CUDA graphs.
- **Multi-TurboQuant integration** — six KV cache compression methods
  with real packed uint8 storage:
  - `isoquant3` (3.25-bit, no calibration, quaternion 4D rotation)
  - `isoquant4` (4.25-bit, no calibration, quaternion 4D rotation)
  - `planarquant3` (3.25-bit, no calibration, Givens 2D rotation)
  - `planarquant4` (4.25-bit, no calibration, Givens 2D rotation)
  - `turboquant25` (2.25-bit, runtime calibration, WHT + MSE + QJL)
  - `turboquant35` (3.25-bit, runtime calibration, WHT + MSE + QJL)

  All six methods deliver **2× more KV cache tokens** at the same
  `gpu_memory_utilization` (uint8 storage with `head_size` bytes per
  slot vs fp16 with `head_size * 2`). Each method's quantization noise
  is verified to actually affect inference output (not a placebo) via
  `tests/test_tq_real.py`.

- **Custom Windows safetensors reader** — `numpy.memmap` + chunked GPU
  streaming. Loads a 14B model in **6.5 seconds** vs ~189 seconds with
  the upstream mmap path. Works on systems with the Windows pagefile
  disabled.

- **End-to-end test suite** — `tests/test_v19.py` (smoke),
  `tests/test_tq_real.py` (correctness sweep), `tests/test_tq_thorough.py`
  (full benchmark with 8 batched prompts and quality checks).

- **Comprehensive docs** — `docs/install.md`, `docs/usage.md`,
  `docs/turboquant.md`, `docs/benchmarks.md`, `docs/build.md`,
  `docs/architecture.md`, `docs/troubleshooting.md`.

### Changed

- `vllm-windows-v3.patch` is now the active patch (vs v0.17.1's
  `vllm-windows-v2.patch`). 33 modified files + 1 new file
  (`vllm/v1/attention/ops/multi_turboquant_kv.py`, 295 lines).
- `build_wheel.py` now defaults to `--source-dir vllm-source --output-dir dist-v3`.
- `install.bat` now installs PyTorch 2.10.0 + triton-windows 3.6.0.post26.
- `build.bat` now applies `vllm-windows-v3.patch` and uses Ninja generator.

### Fixed

- New patch entries for v0.19.0 source changes:
  - `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu` —
    `and` keyword → `&&`
  - CUTLASS in vllm-flash-attn — guard SM100/103/120 includes behind
    CUDA 12.8+ (vendored fix in `.deps/`)
- Custom safetensors reader resolves `OSError 1455 (The paging file is
  too small for this operation to complete)` on systems with the
  pagefile disabled.

### Known limitations

- TQ throughput drops ~30-300× because encode/decode runs in
  PyTorch-vectorised mode (no fused Triton kernel yet). Memory savings
  are real, throughput cost is the trade-off. Suitable for offline /
  long-context / batch workloads. Online serving should stick with
  `auto` or `fp8`.
- Single GPU only (NCCL still unavailable on Windows; the patch wires
  up `FakeProcessGroup` for single-rank operation).

---

## v0.17.1-win — 2026-03-21

vLLM 0.17.1 base. First release with TurboQuant KV cache compression
(2 recipes) and Triton support via triton-windows.

- vLLM 0.17.1 base
- TurboQuant KV cache compression: `turboquant25` (~6.4× reduction)
  and `turboquant35` (~4.27× reduction)
- Triton kernel support via triton-windows 3.6.0
- Qwen 3.5 (DeltaNet hybrid) support
- 27 patched files

---

## v0.14.2-win — 2026-02-28

First pre-built wheel release.

- vLLM 0.14.2 base
- PyTorch 2.9.1+cu126
- Pre-built wheel + portable Python installer (`install.bat`)
- 26 patched files (no Triton, no TQ)
- Tested with Qwen 2.5, Llama 3.x, Phi-4, xLAM

---

## v0.14.1-win — 2026-02-11

Initial release. Source patches only (no pre-built wheel).
