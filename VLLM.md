# vLLM v0.21.0 Windows — Quick Reference

A condensed page for getting a model running fast. For full
documentation see [docs/](docs/).

## One-line install

```batch
install.bat
```

This downloads embedded Python 3.10.11, installs PyTorch 2.11.0+cu126,
triton-windows, and the v0.21.0 wheel. Self-contained — nothing touches
your system Python.

## One-line run

```batch
launch.bat --model E:\models\Qwen3-14B-AWQ-4bit --port 8000
```

Starts an OpenAI-compatible server at `http://127.0.0.1:8000`.

## Hello world (Python)

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
os.add_dll_directory(r"C:\path\to\venv\Lib\site-packages\torch\lib")
from vllm import LLM, SamplingParams

llm = LLM(
    model=r"E:\models\Qwen3-14B-AWQ-4bit",
    dtype="float16",
    kv_cache_dtype="isoquant4",   # 2x KV cache, near-FP16 quality
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
)

print(llm.generate(
    ["Explain CUDA streams in 3 sentences:"],
    SamplingParams(temperature=0.7, max_tokens=200),
)[0].outputs[0].text)
```

## KV cache compression options

Pass any of these as `kv_cache_dtype`:

| dtype | Bits | Memory | Quality | Backend | Notes |
|---|---|---|---|---|---|
| `auto` | 16 | 1× | best | Triton/FA2 | FP16 baseline |
| `fp8` | 8 | 0.5× | minimal loss | Triton/FA2 | requires `fp8_e4m3` GPU support |
| `turboquant_k8v4` | 8.25/4.25 | ~0.4× | minimal loss | TurboQuant (upstream) | mixed K/V, no calibration |
| `turboquant_4bit_nc` | 4.25 | 0.25× | minor loss | TurboQuant (upstream) | upstream default |
| `turboquant_k3v4_nc` | 3.25/4.25 | ~0.22× | visible loss | TurboQuant (upstream) | aggressive K |
| `turboquant_3bit_nc` | 3.25 | ~0.2× | visible loss | TurboQuant (upstream) | most aggressive upstream |
| `isoquant4` | 4.25 | **0.5×** | near-FP16 | TritonAttention (ours) | **recommended TQ default** |
| `planarquant4` | 4.25 | 0.5× | near-FP16 | TritonAttention (ours) | simpler transform |
| `isoquant3` | 3.25 | 0.5× | visible loss | TritonAttention (ours) | aggressive |
| `planarquant3` | 3.25 | 0.5× | visible loss | TritonAttention (ours) | aggressive |
| `turboquant35` | 3.25 | 0.5× | visible loss | TritonAttention (ours) | calibrated outliers |
| `turboquant25` | 2.25 | 0.5× | most loss | TritonAttention (ours) | most compression |

The 4 upstream `turboquant_*` variants use fused Triton kernels and
run at full speed. Our 6 methods use a PyTorch-fallback encode/decode
(slower, but memory savings are real and quality across the iso/planar
family is closer to FP16). See [docs/turboquant.md](docs/turboquant.md)
for details and [docs/benchmarks.md](docs/benchmarks.md) for measured numbers.

## Required env vars on Windows

```batch
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set VLLM_HOST_IP=127.0.0.1
```

The first one is **required** if your Windows pagefile is small or
disabled — without it the PyTorch allocator hits fragmentation and
crashes mid-run.

## Verifying it works

```batch
set VLLM_MODEL_PATH=E:\models\Qwen3-14B-AWQ-4bit
set VLLM_PYTHON=E:\vllm-windows-build\venv\Scripts\python.exe
%VLLM_PYTHON% tests\test_v19.py
```

Should finish in <60 seconds and print generated text. For the full TQ
validation sweep see [tests/README.md](tests/README.md).

## RTX 3090 performance profile (Qwen3-14B AWQ-4bit)

| | Value |
|---|---|
| Weights VRAM | 9.36 GiB |
| Loading time (custom mmap reader) | 6.5 s |
| KV cache @ FP16, gpu_util=0.5 | 16,336 tokens |
| KV cache @ TQ, gpu_util=0.5 | **32,672 tokens** |
| Max concurrency @ 512 ctx, FP16 | 31.91× |
| Max concurrency @ 512 ctx, TQ | **63.94×** |
| FP16 throughput | ~37 tok/s |

## Common issues

- **`OSError 1455`** → enable a Windows pagefile, see [docs/troubleshooting.md](docs/troubleshooting.md#oserror-1455)
- **`CUDA out of memory` with free GPU** → set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **`DLL load failed importing _C`** → add CUDA bin and torch lib to `os.add_dll_directory`
- **First inference is slow** → Triton JIT cold start, ~1-2 min for hybrid models like Qwen3.5

Full troubleshooting → [docs/troubleshooting.md](docs/troubleshooting.md)

## Environment

- vLLM 0.21.0+cu126
- PyTorch 2.11.0+cu126
- Triton 3.6.0 (triton-windows)
- Python 3.10.11
- CUDA 12.6
- Windows 10/11
