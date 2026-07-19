# vLLM v0.25.1 Windows Quick Reference

A condensed page for getting a model running fast. For full documentation
see [docs/](docs/).

## One-Line Install

```batch
install.bat
```

This downloads embedded Python 3.13.14, installs PyTorch 2.11.0+cu128,
triton-windows, structured-output backends, Multi-TurboQuant, and the
v0.25.1 wheel. It is self-contained in this directory.

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

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
os.add_dll_directory(r"C:\path\to\venv\Lib\site-packages\torch\lib")

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
python\python.exe -c "import vllm; print(vllm.__version__)"
python\Scripts\vllm.exe --help
python\Scripts\vllm.exe serve --help
```

Expected version:

```text
0.25.1+cu128
```

## Environment

- vLLM 0.25.1+cu128
- PyTorch 2.11.0+cu128
- Triton 3.6.0.post26 via triton-windows
- Python 3.13.14
- CUDA Toolkit 12.8 for builds
- Windows 10/11
