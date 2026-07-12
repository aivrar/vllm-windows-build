"""Validate the portable vLLM runtime contract used by install.bat."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

EXPECTED_VLLM_VERSION = "0.24.0+cu128"

REQUIRED_MODULES = (
    "llguidance",
    "multi_turboquant",
    "xgrammar",
    "vllm.model_executor.models.qwen3_5",
    "vllm.model_executor.models.qwen3_vl",
    "vllm.vllm_flash_attn.layers.rotary",
    "vllm.vllm_flash_attn.ops.triton.rotary",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    site_packages = (root / "python" / "Lib" / "site-packages").resolve()

    print("Checking PyTorch, Triton, and vLLM imports...", flush=True)
    import torch
    import triton
    import vllm
    import multi_turboquant

    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"vLLM version is {vllm.__version__!r}, expected {EXPECTED_VLLM_VERSION!r}"
        )
    if not torch.__version__.startswith("2.11.0"):
        raise RuntimeError(f"PyTorch version is {torch.__version__!r}, expected 2.11.0")
    if not triton.__version__.startswith("3.6.0"):
        raise RuntimeError(f"Triton version is {triton.__version__!r}, expected 3.6.0")
    if multi_turboquant.__version__ != "0.1.0":
        raise RuntimeError(
            f"Multi-TurboQuant version is {multi_turboquant.__version__!r}, expected '0.1.0'"
        )

    vllm_file = Path(vllm.__file__).resolve()
    if site_packages not in vllm_file.parents:
        raise RuntimeError(
            f"vLLM loaded from {vllm_file}, expected it under {site_packages}"
        )

    for module_name in REQUIRED_MODULES:
        print(f"Checking {module_name}...", flush=True)
        importlib.import_module(module_name)

    from vllm.v1.attention.ops.multi_turboquant_kv import get_packed_dim

    for cache_dtype in (
        "isoquant3",
        "isoquant4",
        "planarquant3",
        "planarquant4",
        "turboquant25",
        "turboquant35",
    ):
        packed_dim = get_packed_dim(cache_dtype, 128)
        if not 0 < packed_dim <= 128:
            raise RuntimeError(f"invalid {cache_dtype} packed dimension: {packed_dim}")

    if args.cuda:
        print("Checking Triton and FlashAttention CUDA execution...", flush=True)
        import triton.runtime.driver as driver
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

        active = getattr(driver, "active", None)
        active = active if active is not None else driver.driver.active
        target = active.get_current_target()
        if target.backend != "cuda":
            raise RuntimeError(f"Triton backend is {target.backend!r}, expected 'cuda'")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA")

        x = torch.randn((1, 4, 2, 8), device="cuda", dtype=torch.float16)
        cos = torch.randn((4, 4), device="cuda", dtype=torch.float16)
        sin = torch.randn((4, 4), device="cuda", dtype=torch.float16)
        output = apply_rotary_emb(x, cos, sin)
        torch.cuda.synchronize()
        if output.shape != x.shape or output.device.type != "cuda":
            raise RuntimeError("FlashAttention rotary CUDA smoke test returned invalid output")
        print(f"Triton CUDA target: {target.backend} {target.arch}")

    print(f"vLLM {vllm.__version__} runtime contract passed from {vllm_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
