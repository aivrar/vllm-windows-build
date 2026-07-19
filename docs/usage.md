# Usage

How to actually run vLLM v0.25.1 on Windows after installing it. Three
modes covered: **(A)** Python embedding, **(B)** OpenAI-compatible HTTP
server via `vllm_launcher.py`, **(C)** the raw `vllm serve` upstream
CLI.

---

## (A) Python embedding

The simplest way — load a model and call `.generate()`.

```python
import os
# Keep single-rank Windows initialization on the loopback interface.
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

# Make sure CUDA + torch DLLs are findable
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
os.add_dll_directory(r"C:\path\to\venv\Lib\site-packages\torch\lib")

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
    --max-num-seqs 64 ^
    --trust-remote-code
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | (required) | Path to a HuggingFace-format model directory |
| `--port` | 8100 | HTTP port |
| `--host` | 127.0.0.1 | Bind address |
| `--gpu-memory-utilization` | 0.6 | Fraction of GPU memory to use |
| `--max-model-len` | 8192 | Maximum context length; lower for small smoke tests |
| `--max-num-seqs` | 64 | Concurrent request limit |
| `--max-num-batched-tokens` | (auto) | Tokens per forward pass |
| `--enforce-eager` | False | Debug/compatibility option; disables compilation and CUDA graphs |
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

vLLM v0.25.1 ships an OpenAI-compatible server out of the box at
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

If you see `OutOfMemoryError`:
1. Lower `gpu_memory_utilization` by 0.1
2. Drop `max_num_seqs` and `max_model_len`
3. Close other GPU processes and ensure the Windows pagefile is enabled
4. Switch to a compressed KV dtype only if its documented throughput cost is acceptable

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

---

## See also

- [install.md](install.md) — install or build first
- [troubleshooting.md](troubleshooting.md) — common errors
- [turboquant.md](turboquant.md) — how the compression methods work
- [benchmarks.md](benchmarks.md) — real numbers
