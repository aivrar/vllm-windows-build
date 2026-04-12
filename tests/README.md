# Tests

End-to-end test scripts for the Windows vLLM 0.19.0 build with Multi-TurboQuant.

## Setup

All tests need two environment variables:

```bat
set VLLM_MODEL_PATH=E:\path\to\Qwen3-14B-AWQ-4bit
set VLLM_PYTHON=E:\vllm-windows-build\venv\Scripts\python.exe
```

And on Windows where the pagefile is small or disabled:

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## test_v19.py — smoke test

The fastest way to confirm the build works. Loads a model, generates a
couple of short prompts, prints them.

```bat
%VLLM_PYTHON% tests\test_v19.py
```

Should finish in well under 60 seconds for a 14B AWQ-4bit model.

## test_tq_real.py — Multi-TurboQuant correctness sweep

Runs the FP16 baseline plus all 6 Multi-TurboQuant compression methods
in fresh subprocesses (so each gets a clean GPU state). Verifies that
each method:

1. Loads, runs, and generates output without crashing
2. Produces output **different** from the FP16 baseline (proves the
   compression noise is actually entering the inference pipeline)
3. Produces output **unique** from the other TQ methods (proves each
   method has its own numerical signature)

```bat
%VLLM_PYTHON% tests\test_tq_real.py
```

Takes ~15-20 minutes total (FP16 ~30s; each TQ method ~3 min). Output
is a table:

```
Method          Differs from FP16    Differs from others
isoquant3       DIFFERS              UNIQUE
isoquant4       DIFFERS              UNIQUE
planarquant3    DIFFERS              UNIQUE
planarquant4    DIFFERS              UNIQUE
turboquant25    DIFFERS              UNIQUE
turboquant35    DIFFERS              UNIQUE
```

If any method shows `IDENTICAL (placebo!)` it means the dispatch isn't
hitting the real encode/decode path — likely a bug in the integration.

## test_tq_thorough.py — full benchmark

Like `test_tq_real.py` but with more rigour:
- 8 prompts batched (uses `max_num_seqs=8`)
- 80 tokens of generation each
- Captures KV cache token counts to prove memory is actually smaller
- Captures load time, generation time, throughput per method
- Coherence checks on each output (length, repetition, printability)

```bat
%VLLM_PYTHON% tests\test_tq_thorough.py
```

Takes ~60-80 minutes for all 7 methods. Produces a results table with
real numbers.

## Notes

- `test_tq_real.py` and `test_tq_thorough.py` spawn each method in a
  fresh subprocess to make sure GPU state from one run doesn't bleed
  into the next. This is slower but eliminates cross-test interference.
- All tests use `enforce_eager=True` and `temperature=0.0` so output is
  deterministic. Output diffs reflect KV math differences, not RNG.
- The default settings (`max_model_len=512`, `gpu_memory_utilization=0.5`)
  fit comfortably on a single 24 GB RTX 3090 with the 14B AWQ-4bit
  model. Larger models or smaller GPUs will need adjustment.
