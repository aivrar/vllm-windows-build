# Install vLLM 0.34.0 on Windows with Python 3.14 / CUDA 13.2

Download the [source ZIP](https://github.com/aivrar/vllm-windows-build/archive/refs/tags/v0.34.0-win-cu132-py314.zip), extract it into a fresh folder and run install.bat. The installer supplies portable CPython 3.14.2. Run launch.bat to select a model. Keep installations for different Python or CUDA versions in separate folders.

The stack uses PyTorch 2.14.0+cu132, torchvision 0.29.0+cu132, CPU TorchAudio 2.11.0 and triton-windows 3.8.0.post29. The exact pins are in [constraints-cu132-py314.txt](../constraints-cu132-py314.txt). Video input through TorchCodec requires compatible shared FFmpeg DLLs.

If you already downloaded both wheel assets, place them in dist-v0.34.0 beside install.bat. The installer verifies their sizes and hashes. [Release assets](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.34.0-win-cu132-py314)

## Build from source

Start with the [common build steps](https://github.com/aivrar/vllm-windows-build/blob/v0.34.0-win-cu130/docs/install.md#build-from-source), substituting Python 3.14, CUDA 13.2 and torch 2.14.0+cu132 throughout. Apply the Python 3.14 allowlist patch to the pinned FlashAttention checkout and point VLLM_FLASH_ATTN_SRC_DIR there before compiling. Check the resulting cp314 wheel with tests/test_wheel_contents.py. [Revisions and validation](v0.34.0-build.md)
