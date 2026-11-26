# Usage

> **Release selection:** [the Windows releases](releases.md) by Python/CUDA version, installer, and source-build guidance.

How to actually run vLLM v0.34.0 on Windows after installing it. Three
modes covered: **(A)** Python embedding, **(B)** OpenAI-compatible HTTP
server via `vllm_launcher.py`, **(C)** the raw `vllm serve` upstream
CLI.

**Using an FP8/NVFP4 model on Blackwell?** 0.34.0 includes the issue #16
FP8 Marlin fallback fixes, but native FP4 acceleration is not available and
Blackwell hardware was not tested for 0.34.0. Read the
[Blackwell notes](troubleshooting.md#blackwell-fp8-nvfp4) if a model fails.

---

## (A) Python embedding

The simplest way — load a model and call `.generate()`.

```python
import os
# Keep single-rank Windows initialization on the loopback interface.
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

# Make sure CUDA + torch DLLs are findable
import torch  # Load the DLLs bundled with the installed CUDA-enabled PyTorch wheel.

from vllm import LLM, SamplingParams

llm = LLM(
    model=r"E:\models\Qwen2.5-0.5B-Instruct",
    dtype="float16",
    kv_cache_dtype="auto",          # Fast baseline; model dtype for KV cache
    max_model_len=512,
    gpu_memory_utilization=0.5,
)

params = SamplingParams(temperature=0.0, max_tokens=32, seed=0)
outputs = llm.generate(
    ["Explain quantum entanglement to a 10-year-old:"],
    params,
)
print(outputs[0].outputs[0].text)
```

Measure a second request in the same process so one-time kernel compilation or
CUDA-graph capture is not counted as steady-state generation. If graph capture
causes a compatibility problem, add `enforce_eager=True` temporarily; that
setting disables `torch.compile` and CUDA graphs and therefore lowers
throughput.

### Multi-GPU note

Single GPU only on Windows. NCCL doesn't ship with PyTorch on Windows
and the patch wires up `FakeProcessGroup` for single-rank operation.
For multi-GPU, run separate vLLM instances on different GPUs and
load-balance externally.

---

## (B) OpenAI-compatible HTTP server (vllm_launcher.py)

`vllm_launcher.py` is a Windows-friendly OpenAI-compatible server.
It's a single file, ~30 KB, that wraps the embedding API in FastAPI
endpoints.

### Quick start

```bat
launch.bat
```

Without arguments, `launch.bat` shows an interactive model picker that
scans `models\` next to the script. With arguments:

```bat
launch.bat --model E:\models\Qwen3-14B-AWQ-4bit --port 8000
```

`launch.bat` also checks the portable install and reruns `install.bat`
if Python, PyTorch, Triton, vLLM, or Triton's Python development files
are missing.

### Example with explicit overrides

```bat
python vllm_launcher.py ^
    --model E:\models\Qwen3-14B-AWQ-4bit ^
    --port 8000 ^
    --host 127.0.0.1 ^
    --gpu-memory-utilization 0.85 ^
    --max-model-len 2048 ^
    --max-num-seqs 4 ^
    --trust-remote-code
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | (required) | Path to a HuggingFace-format model directory |
| `--port` | 8100 | HTTP port |
| `--host` | 127.0.0.1 | Bind address |
| `--gpu-memory-utilization` | 0.6 (0.89 with `--turing-compat`) | Maximum fraction of GPU memory vLLM may use; raise it if weights fit but the KV budget is negative |
| `--max-model-len` | 8192 | Maximum context length; lower for small smoke tests |
| `--max-num-seqs` | 64 (1 with `--turing-compat`) | Concurrent request limit |
| `--max-num-batched-tokens` | auto (2048 with `--turing-compat`) | Tokens per forward pass |
| `--enforce-eager` | False | Debug/compatibility option; disables compilation and CUDA graphs |
| `--attention-backend` | (auto) | Force a backend such as `TRITON_ATTN`; useful on RTX 20xx/Turing |
| `--linear-backend` | (auto) | 0.29.0 and later: select a linear backend such as `marlin`; omit for automatic selection |
| `--num-scheduler-steps` | 1 | Legacy compatibility argument; values other than `1` are rejected |
| `--kv-cache-dtype` | (auto) | KV-cache storage dtype; use `float16` on Turing when auto selection fails |
| `--block-size` | (auto) | KV-cache block size; `16` or `32` are conservative compatibility choices |
| `--turing-compat` | False | RTX 20xx/SM 7.5 profile: `TRITON_ATTN`, float16 KV, block 32, eager mode, 0.89 GPU utilization, one sequence, and 2,048 batched tokens |
| `--gpu-id` | (none) | Which GPU to pin to; otherwise preserve the current CUDA visibility |
| `--enable-prefix-caching` | (not forced) | Explicitly enable common-prefix reuse; otherwise use vLLM's default |
| `--task` | "generate" | "generate" or "embed" |
| `--trust-remote-code` | False | Required for some models |
| `--cpu-offload-gb` | 0 | Model-weight CPU offload; unrelated to the KV-cache options below |
| `--kv-offload` | disabled | Opt-in prompt-KV mode: `cpu-lru`, `cpu-arc`, `fs-lru`, or `fs-arc` |
| `--kv-offload-cpu-gb` | 4.0 | Pinned-RAM capacity used only when KV offload is enabled |
| `--kv-offload-fs-root` | (none) | Required explicit directory for `fs-lru`/`fs-arc` |
| `--kv-offload-read-threads` | 4 | Read-priority threads for a filesystem tier |
| `--kv-offload-write-threads` | 4 | Write-priority threads for a filesystem tier |

### RTX 20xx / Turing (SM 7.5)

The Windows release wheel contains CUDA code for SM 7.5, but
some newer attention and CUDA-graph paths are not available on Turing. If the
integrated launcher reports an unsupported architecture, start with its
compatibility profile:

```bat
launch.bat --model cyankiwi/Qwen3.5-9B-AWQ-4bit --gpu-id 0 --turing-compat
```

The profile applies the settings confirmed by the issue #14 reporter on an
11-GB RTX 2080 Ti: `TRITON_ATTN`, `float16` KV cache, 32-token blocks,
`--enforce-eager`, 0.89 GPU utilization, one sequence, and a 2,048-token batch
cap. The default context remains 8,192. Every explicit command-line value
overrides the profile.

`--gpu-memory-utilization` is a budget ceiling, not simply extra allocation.
If the weights load but vLLM reports a negative or unavailable KV-cache budget,
raise it when the GPU has free VRAM. Reduce `--max-model-len`,
`--max-num-seqs`, and `--max-num-batched-tokens` to reduce demand. Lower GPU
utilization only when you need to reserve VRAM for another application.
`launch.bat` sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `--gpu-id` follows the
order shown by `nvidia-smi`.

Direct `.gguf` files are not accepted. Use a Hugging Face-format directory or
repository ID containing `config.json` and compatible Safetensors, AWQ, or GPTQ
weights. The launcher rejects `.gguf` paths before Transformers can misreport
the binary file as invalid UTF-8 JSON.

From 0.29.0 onward, the launcher also accepts `--linear-backend marlin`
to select Marlin explicitly. Omit this option to use automatic selection.
The old `VLLM_TEST_FORCE_FP8_MARLIN` workaround is not used by 0.29.0 or later.
Legacy `--num-scheduler-steps` values other than `1` are rejected before
model loading. See the [0.34.0 build record](v0.34.0-build.md) for current validation scope.

### Experimental KV offload

KV offload is disabled by default. It supplements vLLM's GPU prefix cache by
keeping evicted prompt KV blocks in system RAM, or in RAM backed by a larger
filesystem/NVMe tier. Enabling any mode also enables prefix caching.

CPU-only ARC example:

```bat
launch.bat --model E:\models\Qwen3-14B-AWQ-4bit ^
    --kv-offload cpu-arc ^
    --kv-offload-cpu-gb 8
```

Persistent RAM + filesystem LRU example:

```bat
launch.bat --model E:\models\Qwen3-14B-AWQ-4bit ^
    --kv-offload fs-lru ^
    --kv-offload-cpu-gb 4 ^
    --kv-offload-fs-root E:\vllm-kv-cache
```

`launch.bat` sets `PYTHONHASHSEED=0` before Python starts so identical prompts
can find the same filesystem entries after a restart. If invoking Python
directly, set it yourself first:

```bat
set PYTHONHASHSEED=0
python vllm_launcher.py --model E:\models\Qwen3-14B-AWQ-4bit --kv-offload fs-arc --kv-offload-fs-root E:\vllm-kv-cache
```

Important limits:

- The filesystem tier has no automatic byte quota or cleanup policy. Choose a
  dedicated directory, monitor its size, and delete it when you want to clear
  the persistent cache.
- `--kv-offload-cpu-gb` controls the KV cache's pinned-RAM tier.
  `--cpu-offload-gb` controls model weights; the two settings are independent.
- This release offloads prompt blocks only. Benefits depend on repeated long
  prefixes and available RAM/storage bandwidth; it does not make uncached
  generation faster.
- This is an experimental local vLLM tiering path, not the complete LMCache
  feature set. Remote/distributed cache, P2P, NIXL, GDS, and object-store modes
  are not supported by this Windows release.

See [lmcache-inspired-windows-kv-cache.md](lmcache-inspired-windows-kv-cache.md)
for the design history, validation evidence, and remaining roadmap.

### Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/models` | List loaded models (OpenAI-compatible) |
| POST | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| POST | `/v1/completions` | Legacy text completions |
| GET | `/health` | Liveness check |
| POST | `/shutdown` | Graceful shutdown |

### Example: chat completion

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "qwen3-14b",
        "messages": [
            {"role": "user", "content": "Explain CUDA streams in 3 sentences."},
        ],
        "temperature": 0.7,
        "max_tokens": 200,
    },
)
print(resp.json()["choices"][0]["message"]["content"])
```

### Streaming

Set `"stream": true` in the request body. The server returns
Server-Sent Events compatible with the OpenAI streaming format.

### Tool calling

`vllm_launcher.py` parses tool calls from model output in two formats:
- `<tool_call>{...}</tool_call>` tags (Qwen3 format)
- Bare JSON objects with `"name"` and `"arguments"` keys

The parsed tool calls are returned in the OpenAI `tool_calls` field.

### Multi-GPU multi-server

For multi-GPU systems, run one server per GPU on different ports:

```bat
start "vLLM GPU 0" python vllm_launcher.py --model M --gpu-id 0 --port 8000
start "vLLM GPU 1" python vllm_launcher.py --model M --gpu-id 1 --port 8001
```

Then load-balance with nginx or your own router.

---

## (C) Upstream `vllm serve`

vLLM v0.34.0 ships an OpenAI-compatible server out of the box at
`vllm serve`. It works on Windows after the patches are applied:

```bat
vllm serve E:\models\Qwen3-14B-AWQ-4bit ^
    --dtype float16 ^
    --kv-cache-dtype auto ^
    --max-model-len 2048 ^
    --gpu-memory-utilization 0.6
```

The `--kv-cache-dtype` flag accepts the 6 Multi-TurboQuant methods,
the 4 upstream TurboQuant variants, plus the standard `auto`, `fp8`,
`fp8_e4m3`, and `fp8_e5m2` values.

`vllm_launcher.py` (mode B) is generally the more reliable choice on
Windows because it keeps the launch path explicit and pins the portable
environment. `vllm serve` is useful when you want the upstream CLI
surface directly.

---

## Memory tuning

For a 24 GB GPU loading a 14B AWQ-4bit model:

| Setting | Recommended | Why |
|---|---|---|
| `gpu_memory_utilization` | 0.6-0.85 | Start lower, then raise when the workload needs more KV capacity |
| `max_model_len` | 2048-4096 | Larger contexts eat KV cache linearly |
| `max_num_seqs` | 64-128 | Higher = more parallelism, more KV cache |
| `enforce_eager` | False | Normal optimized path; enable only to diagnose graph/compile problems |
| `kv_cache_dtype` | `auto` | Fast baseline; use compressed KV only when capacity matters more than latency |

For longer contexts, drop `max_num_seqs` to give each sequence more KV
budget. For high concurrency, drop `max_model_len`.

If you see `No available memory for the cache blocks` after the weights load:

1. Raise `gpu_memory_utilization` if `nvidia-smi` shows free VRAM.
2. Drop `max_num_seqs`, `max_num_batched_tokens`, and `max_model_len`.
3. Close other GPU processes and ensure the Windows pagefile is enabled.
4. Use a smaller or more strongly quantized model if the weights consume most
   of the card.

For a CUDA out-of-memory error caused by competing applications, close those
applications or lower the vLLM budget. Compressed KV formats can extend context
capacity, but use them only when their documented throughput cost is acceptable.

---

## Picking a KV cache dtype

| Method | When to use |
|---|---|
| `auto` (fp16) | Default. Best speed, most memory. |
| `isoquant4` | Quality-first local TQ. Half the memory, but the current fallback is for offline/memory-first use. |
| `planarquant4` | Same as iso4, simpler transform. |
| `isoquant3` | Aggressive — 3.25 bits, visible quality loss. |
| `planarquant3` | Same as iso3. |
| `turboquant35` | TurboQuant balanced — calibrated outlier handling. |
| `turboquant25` | Most aggressive — 2.25 bits, only for offline batch. |

**Throughput note**: all 6 TQ methods currently run with PyTorch-only
encode/decode (no fused Triton kernel). Throughput drops ~30-300×
depending on the method. Memory savings are real, throughput cost is
the trade-off until the kernels get fused. See [turboquant.md](turboquant.md#throughput-cost).

## FlashInfer attention (optional, community-tested: +43% prefill at the same window)

The kit does not ship FlashInfer, and the default backends cover normal use.
There is, however, a community-tested option: on RTX 30xx it is the only way to
use an fp8 KV cache, and at a 131K window it takes the working configuration
from **828 / 76 to 1,182 / 82 tok/s**, that is **+43% prefill and +8% decode**.
It is not officially supported and not part of the tested kit.

[Astemiir](https://github.com/Astemiir) ported FlashInfer's compile-on-first-use
(JIT) path to MSVC, measured it carefully, and publishes ready-made wheels
([issue #20](https://github.com/aivrar/vllm-windows-build/issues/20); every
Windows blocker with its fix:
[SystemPanic/flashinfer-windows#14](https://github.com/SystemPanic/flashinfer-windows/issues/14)).
Thanks to Astemiir for the port, the measurements, and the corrected comparison.

**Install with his prebuilt wheels (easiest route).** Download the three wheels
from his
[v0.7.2.post1-win.1 release](https://github.com/Astemiir/flashinfer-windows/releases/tag/v0.7.2.post1-win.1).
They contain the MSVC fixes and 29 precompiled kernels, so nothing compiles on
first use and no CUDA toolkit or MSVC is needed:

```bat
python -m pip install flashinfer_python-0.7.2.post1+win.1-py3-none-any.whl ^
                     flashinfer_jit_cache-0.7.2.post1+win.1-py3-none-any.whl ^
                     flashinfer_jit_cache_sm80-0.7.2.post1+win.1-cp39-abi3-win_amd64.whl
```

These are his builds, not part of this project, and we have not verified the
binaries. Once FlashInfer is installed, vLLM may select it automatically. His
stated limits (full card list on the release page):
- **GPUs:** the kernels are built for sm_80 and also run on sm_86 and sm_89:
  A100/A30, RTX 30xx and 40xx, and the matching RTX workstation cards (A4000 to
  A6000, RTX 4000 to 6000 Ada) and data-center cards (A10, A40, L4, L40, L40S).
  They do not run on Hopper or Blackwell (H100/H200, B100/B200, RTX 50xx). On
  smaller cards, memory is the real limit: a 27B model at a 131K window needs
  24 GB.
- **CUDA 13.x only:** cu130, with driver 580 or newer.
- **A fixed kernel set:** head dims 128 and 256, bf16/fp16, fp8_e4m3 KV, and no
  sliding window or logits soft cap. Anything else falls back to compiling on
  your machine, which needs the toolkit and MSVC again.

**Build it yourself** only if you need another FlashInfer version, head
dimension or GPU architecture. That needs a separately installed CUDA toolkit
(13.3 in the tested setup; the kit ships only the 13.0 runtime), MSVC Build
Tools, ninja, and `flashinfer-python` 0.7.2.post1 with the Windows fixes. His
recipe and scripts are in
[docs/build-windows-msvc.md](https://github.com/Astemiir/flashinfer-windows/blob/flashinfer-windows-rel/docs/build-windows-msvc.md).

**Drafters.** MTP works normally with FlashInfer. Drafters that select
candidates through vLLM's vocab-parallel top-k (DFlash2, for example) call
FlashInfer's top-k kernel during warm-up. That kernel does not build under
MSVC, so the engine fails to start. The opt-out is `VLLM_USE_FLASHINFER_TOPK=0`,
which falls back to `torch.topk` at about 0.1 ms per drafter step. Both 0.34.0 wheels include the opt-out. For 0.33.0 source builds, apply the v12 FlashInfer top-k add-on.

On his RTX 3090, MTP was faster
than DFlash2 anyway (78.9 vs 71.4 tok/s decode at a 32K window) and used less
VRAM.

**What it gains.** These are one person's measurements: a 27B GPTQ model with
MTP on one RTX 3090, a 33K-token prompt and a 500-token answer, three runs each.
Run-to-run spread was about ±5%. Accuracy was not checked, only throughput.

| Setup | Window | Prefill | Decode |
|---|---:|---:|---:|
| `bfloat16` KV + `TRITON_ATTN` (kit default path) | 65,536 | ~1,038 tok/s | ~85 tok/s |
| `bfloat16` KV + `FLASHINFER` | 65,536 | ~1,225 tok/s | ~83 tok/s |
| `int8_per_token_head` KV + `TRITON_ATTN` | 131,072 | ~828 tok/s | ~76 tok/s |
| `fp8_e4m3` KV + `FLASHINFER` | 131,072 | ~1,182 tok/s | ~82 tok/s |

- **Backend alone** (same bf16 KV, same window): about **+18% prefill** and
  **no decode gain**. With MTP, FlashInfer rebuilds attention metadata every
  step.
- **At a 131K window on 24 GB** (last two rows): about **+43% prefill and +8%
  decode**. Both the backend and the KV format change here (int8 to fp8),
  because the stock backend has no fp8 KV on Ampere.
- **fp8 KV on RTX 30xx needs FlashInfer.** The Triton backend refuses fp8 KV
  below SM 8.9. fp8 KV saves memory, which is what lets a 131K window fit in
  24 GB, but it does not make decode faster.
- **With the third-party `vllm-quantpack` plugin** (Astemiir ships a patched
  build; not part of this kit), his measurements differ. Its int8 prefill hook brings plain Triton prefill to
  about 1,172 tok/s, close to FlashInfer's 1,182, and its int4 KV path decodes at
  about 85 tok/s. FlashInfer's remaining advantages there are fp8 KV on RTX 30xx
  and faster decode at very long fills (about +32% at 99K over int8). Details are
  in his [project notes](https://github.com/Astemiir/qwen38-27b-windows-vllm).
- The default path without FlashInfer is perfectly usable. FlashInfer is an
  optional prefill gain and an fp8 KV option, not a requirement.

---

## See also

- [install.md](install.md) — install or build first
- [troubleshooting.md](troubleshooting.md) — common errors
- [turboquant.md](turboquant.md) — how the compression methods work
- [benchmarks.md](benchmarks.md) — real numbers
