# vLLM v0.24.0 Windows Quick Reference

A condensed page for getting a model running fast. For full documentation
see [docs/](docs/).

## One-Line Install

```batch
install.bat
```

This downloads embedded Python 3.13.11, installs PyTorch 2.11.0+cu128,
triton-windows, structured-output backends, Multi-TurboQuant, and the
v0.24.0 wheel. It is self-contained in this directory.

## One-Line Run

```batch
launch.bat --model E:\models\Qwen3-14B-AWQ-4bit --port 8000
```

Starts an OpenAI-compatible server at `http://127.0.0.1:8000`.

## Hello World

```python
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
os.add_dll_directory(r"C:\path\to\venv\Lib\site-packages\torch\lib")

from vllm import LLM, SamplingParams

llm = LLM(
    model=r"E:\models\Qwen3-14B-AWQ-4bit",
    dtype="float16",
    kv_cache_dtype="isoquant4",
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
)

outputs = llm.generate(
    ["Explain CUDA streams in 3 sentences:"],
    SamplingParams(temperature=0.7, max_tokens=200),
)
print(outputs[0].outputs[0].text)
```

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
| `isoquant4` | Multi-TurboQuant, recommended local default |
| `planarquant4` | Multi-TurboQuant |
| `isoquant3` | Multi-TurboQuant |
| `planarquant3` | Multi-TurboQuant |
| `turboquant35` | Multi-TurboQuant |
| `turboquant25` | Multi-TurboQuant |

The upstream `turboquant_*` variants use fused Triton kernels. The 6
Multi-TurboQuant methods use a slower PyTorch fallback path but preserve
the memory savings.

## Required Windows Env Vars

```batch
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
0.24.0+cu128
```

## Environment

- vLLM 0.24.0+cu128
- PyTorch 2.11.0+cu128
- Triton 3.6.0.post26 via triton-windows
- Python 3.13.11
- CUDA Toolkit 12.8 for builds
- Windows 10/11
