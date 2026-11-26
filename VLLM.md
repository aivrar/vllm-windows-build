# vLLM 0.34.0 Windows Quick Reference

Download the source ZIP for the desired [Python/CUDA release](docs/releases.md) and extract it into a new directory. Both wheels include the FP8 Marlin fallback. [Build record and validation](docs/v0.34.0-build.md)

## One-Line Install

```batch
install.bat
```

This downloads embedded Python 3.14.2, installs PyTorch 2.14.0+cu132,
triton-windows, structured-output backends, Multi-TurboQuant, and the
v0.34.0 wheel. It is self-contained in this directory.

## One-Line Run

```batch
launch.bat --model E:\models\Qwen3-14B-AWQ-4bit --port 8000
```

Starts an OpenAI-compatible server at `http://127.0.0.1:8000`.

KV offload is opt-in. For example, add `--kv-offload cpu-arc
--kv-offload-cpu-gb 8`, or add `--kv-offload fs-lru
--kv-offload-fs-root E:\vllm-kv-cache` for persistent filesystem tiering.
Filesystem caches have no automatic disk quota; see
[docs/usage.md](docs/usage.md#experimental-kv-offload).

## Hello World

```python
import os

os.environ["VLLM_HOST_IP"] = "127.0.0.1"

import torch  # Load the DLLs bundled with the installed CUDA-enabled PyTorch wheel.

from vllm import LLM, SamplingParams

llm = LLM(
    model=r"E:\models\Qwen2.5-0.5B-Instruct",
    dtype="float16",
    kv_cache_dtype="auto",
    max_model_len=512,
    gpu_memory_utilization=0.5,
)

outputs = llm.generate(
    ["Explain CUDA streams in 3 sentences:"],
    SamplingParams(temperature=0.0, max_tokens=32, seed=0),
)
print(outputs[0].outputs[0].text)
```

Use `auto` for the fast baseline and measure a second request after one-time
kernel/graph setup. `enforce_eager=True` is a compatibility/debugging option,
not a throughput default.

## KV Cache Compression Options

Pass any of these as `kv_cache_dtype`:

| dtype | Notes |
|---|---|
| `auto` | FP16 baseline |
| `fp8` | Half-size KV cache on supported GPUs |
| `turboquant_k8v4` | Upstream TurboQuant |
| `turboquant_4bit_nc` | Upstream TurboQuant |
| `turboquant_k3v4_nc` | Upstream TurboQuant |
| `turboquant_3bit_nc` | Upstream TurboQuant |
| `isoquant4` | Multi-TurboQuant, quality-first memory option; slow fallback path |
| `planarquant4` | Multi-TurboQuant |
| `isoquant3` | Multi-TurboQuant |
| `planarquant3` | Multi-TurboQuant |
| `turboquant35` | Multi-TurboQuant |
| `turboquant25` | Multi-TurboQuant |

The upstream `turboquant_*` variants use fused Triton kernels. The 6
Multi-TurboQuant methods use a slower PyTorch fallback path but preserve
the memory savings.

## Recommended Windows Environment

```batch
set VLLM_HOST_IP=127.0.0.1
```

For multi-GPU systems:

```batch
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
```

## Verify

```batch
python\python.exe -c "import importlib.metadata, vllm; print(vllm.__version__); print(importlib.metadata.version('vllm'))"
python\Scripts\vllm.exe --help
python\Scripts\vllm.exe serve --help
```

Expected runtime and distribution versions:

```text
0.34.0
0.34.0
```

## Environment

- vLLM 0.34.0 (SM 7.5/8.6/8.9/12.0)
- PyTorch 2.14.0+cu132
- Triton 3.8.0.post29 via triton-windows
- Python 3.14.2
- CUDA Toolkit 13.2 for builds
- Windows 10/11
