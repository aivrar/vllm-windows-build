# vLLM 0.34.0 for Windows · Python 3.14 / CUDA 13.2

This release installs Python 3.14.2, PyTorch 2.14.0+cu132, Triton Windows 3.8.0.post29 and the native Windows vLLM wheel. Model inference uses CUDA. TorchAudio is the CPU build. For the default Python 3.13 / CUDA 13.0 installation, choose the [Latest release](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.34.0-win-cu130).

## Quick Start

1. Download the [Python 3.14 source ZIP](https://github.com/aivrar/vllm-windows-build/archive/refs/tags/v0.34.0-win-cu132-py314.zip) and extract it into a new folder.
2. Run **install.bat**. It downloads portable Python and verifies the matching wheels.
3. Run **launch.bat** to select a model, or use the command below.

    launch.bat --model "D:\models\YourModel"

Both editions include the Windows v14 port contributed by [Astemiir in #23](https://github.com/aivrar/vllm-windows-build/issues/23). Both passed real-model serving, all six Multi-TurboQuant formats, FP8 Marlin and CPU KV-cache restore on an RTX 3090. Native Blackwell FP4 acceleration is outside the released wheel.

[Wheel and release notes](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.34.0-win-cu132-py314) · [Install](docs/install.md) · [Build inputs, hashes and validation](docs/v0.34.0-build.md) · [Usage](docs/usage.md) · [Wiki](https://github.com/aivrar/vllm-windows-build/wiki)

## Hello world

```python
import os
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

# CUDA + torch DLL search paths
import torch  # Load the DLLs bundled with the installed CUDA-enabled PyTorch wheel.

# Both uvloop and flashinfer fallbacks are baked into the wheel.
# Multi-GPU host? Don't forget CUDA_DEVICE_ORDER + CUDA_VISIBLE_DEVICES
# so vLLM lands on the GPU you actually want.

from vllm import LLM, SamplingParams

llm = LLM(
    model=r"E:\models\Qwen2.5-0.5B-Instruct",
    dtype="float16",
    kv_cache_dtype="auto",        # Fast FP16 baseline
    max_model_len=512,
    gpu_memory_utilization=0.5,
)

outputs = llm.generate(
    ["Explain CUDA streams in three sentences:"],
    SamplingParams(temperature=0.0, max_tokens=32, seed=0),
)
print(outputs[0].outputs[0].text)
```

`auto` is deliberate here: it establishes normal baseline performance. The
first request can include one-time JIT or CUDA-graph setup, so benchmark a
second request in the same process. Add `enforce_eager=True` only when
diagnosing graph/compile compatibility; it disables those optimizations.

For OpenAI-compatible HTTP serving and more usage patterns:
**→ [docs/usage.md](docs/usage.md)**

---

## Credits

| | |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | The original engine |
| [PyTorch](https://github.com/pytorch/pytorch) | Tensor library + CUDA bindings |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | NVIDIA |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FA2 kernels |
| [triton-windows](https://github.com/triton-lang/triton-windows) | Triton compiler ported to Windows |
| [Multi-TurboQuant](https://github.com/aivrar/multi-turboquant) | KV cache compression methods (ours) |
| [Upstream TurboQuant](https://github.com/vllm-project/vllm/pull/38479) | TurboQuant attention backend (vLLM PR #38479) |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | GEMM kernels (CUTLASS 4.4.2 with MSVC patches) |
| [TurboQuant paper](https://arxiv.org/abs/2501.06725) | Walsh-Hadamard quantization |

### Community contributors

| | |
|---|---|
| [jfbobier](https://github.com/jfbobier) | Native SM120 (Blackwell) FP8/NVFP4 add-on patch, tested on an RTX 5090 ([#19](https://github.com/aivrar/vllm-windows-build/pull/19)) |
| [Astemiir](https://github.com/Astemiir) | [vLLM 0.34.0 Windows port](https://github.com/aivrar/vllm-windows-build/issues/23), launcher warm-up, FlashInfer attention port for RTX 30xx, measurements and write-up ([#20](https://github.com/aivrar/vllm-windows-build/issues/20)) |

Built with the help of [Claude](https://claude.ai).

---

## License

MIT. See [LICENSE](LICENSE).
