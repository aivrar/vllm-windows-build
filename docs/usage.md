# Usage

How to actually run vLLM v0.19.0 on Windows after installing it. Three
modes covered: **(A)** Python embedding, **(B)** OpenAI-compatible HTTP
server via `vllm_launcher.py`, **(C)** the raw `vllm serve` upstream
CLI.

---

## (A) Python embedding

The simplest way — load a model and call `.generate()`.

```python
import os
# Required env vars on Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

# Make sure CUDA + torch DLLs are findable
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
os.add_dll_directory(r"C:\path\to\venv\Lib\site-packages\torch\lib")

# Stub uvloop (not available on Windows)
import sys
sys.modules.setdefault("uvloop", type(sys)("uvloop"))

from vllm import LLM, SamplingParams

llm = LLM(
    model=r"E:\models\Qwen3-14B-AWQ-4bit",
    dtype="float16",
    kv_cache_dtype="auto",          # or one of: isoquant3, isoquant4,
                                    # planarquant3, planarquant4,
                                    # turboquant25, turboquant35
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=True,             # cudagraph capture is slow on Win
    trust_remote_code=True,
)

params = SamplingParams(temperature=0.7, max_tokens=200)
outputs = llm.generate(
    ["Explain quantum entanglement to a 10-year-old:"],
    params,
)
print(outputs[0].outputs[0].text)
```

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

### Full options

```bat
python vllm_launcher.py ^
    --model E:\models\Qwen3-14B-AWQ-4bit ^
    --port 8000 ^
    --host 127.0.0.1 ^
    --gpu-memory-utilization 0.85 ^
    --max-model-len 2048 ^
    --max-num-seqs 64 ^
    --enforce-eager ^
    --trust-remote-code
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | (required) | Path to a HuggingFace-format model directory |
| `--port` | 8000 | HTTP port |
| `--host` | 127.0.0.1 | Bind address |
| `--gpu-memory-utilization` | 0.85 | Fraction of GPU memory to use |
| `--max-model-len` | 2048 | Maximum context length |
| `--max-num-seqs` | 64 | Concurrent request limit |
| `--max-num-batched-tokens` | (auto) | Tokens per forward pass |
| `--enforce-eager` | False | Skip CUDAGraph capture (recommended on Windows) |
| `--gpu-id` | 0 | Which GPU to pin to (multi-GPU systems) |
| `--enable-prefix-caching` | True | Cache common prompt prefixes |
| `--task` | "generate" | "generate" or "embed" |
| `--trust-remote-code` | False | Required for some models |

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

vLLM v0.19.0 ships an OpenAI-compatible server out of the box at
`vllm serve`. It works on Windows after the patches are applied:

```bat
vllm serve E:\models\Qwen3-14B-AWQ-4bit ^
    --dtype float16 ^
    --kv-cache-dtype isoquant3 ^
    --max-model-len 2048 ^
    --gpu-memory-utilization 0.85 ^
    --enforce-eager
```

The `--kv-cache-dtype` flag accepts any of the 6 Multi-TurboQuant
methods plus the standard `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`.

`vllm_launcher.py` (mode B) is generally the more reliable choice on
Windows because it bypasses the multiprocess engine path. Try `vllm
serve` first; if you hit ZMQ or asyncio errors, fall back to mode B.

---

## Memory tuning

For a 24 GB GPU loading a 14B AWQ-4bit model:

| Setting | Recommended | Why |
|---|---|---|
| `gpu_memory_utilization` | 0.85 | Leaves headroom for activations and Triton workspace |
| `max_model_len` | 2048-4096 | Larger contexts eat KV cache linearly |
| `max_num_seqs` | 64-128 | Higher = more parallelism, more KV cache |
| `enforce_eager` | True | CUDAGraph capture is slow + uses extra memory |
| `kv_cache_dtype` | `auto` or `isoquant4` | iso4 doubles KV capacity at small quality cost |

For longer contexts, drop `max_num_seqs` to give each sequence more KV
budget. For high concurrency, drop `max_model_len`.

If you see `OutOfMemoryError`:
1. Lower `gpu_memory_utilization` by 0.1
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. Switch to a TQ KV cache dtype to halve cache pressure
4. Drop `max_num_seqs` and `max_model_len`

---

## Picking a KV cache dtype

| Method | When to use |
|---|---|
| `auto` (fp16) | Default. Best speed, most memory. |
| `isoquant4` | **Recommended TQ**. Half the memory, near-FP16 quality, no calibration needed. |
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
