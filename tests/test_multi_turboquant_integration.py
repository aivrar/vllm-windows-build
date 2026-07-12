"""Exercise all six Multi-TurboQuant cache paths without loading a model."""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.multi_turboquant_kv import (
    tq_decode_active_blocks,
    tq_write_kv_cache,
)

METHODS = (
    "isoquant3",
    "isoquant4",
    "planarquant3",
    "planarquant4",
    "turboquant25",
    "turboquant35",
)


def main() -> int:
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")
    block_size = 4
    head_size = 128
    num_heads = 2

    key = torch.randn((2, num_heads, head_size), device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    slots = torch.tensor([0, 1], device=device, dtype=torch.int64)
    block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([2], device=device, dtype=torch.int32)

    for method in METHODS:
        key_cache = torch.zeros(
            (2, block_size, num_heads, head_size), device=device, dtype=torch.uint8
        )
        value_cache = torch.zeros_like(key_cache)
        tq_write_kv_cache(key, value, key_cache, value_cache, slots, method)
        decoded_key, decoded_value, remapped = tq_decode_active_blocks(
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            method,
            torch.float16,
            block_size,
        )
        assert decoded_key.shape == (1, block_size, num_heads, head_size)
        assert decoded_value.shape == decoded_key.shape
        assert torch.isfinite(decoded_key).all()
        assert torch.isfinite(decoded_value).all()
        assert remapped.tolist() == [[0]]
        print(f"{method}: write/decode passed")

    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
