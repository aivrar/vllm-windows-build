# Troubleshooting

Common errors when building or running vLLM v0.19.0 on Windows.

## Runtime errors

### `ImportError: DLL load failed while importing _C: The specified module could not be found.`

vLLM's compiled `_C.pyd` extension can't find its CUDA / torch DLLs.
The fix is to add the CUDA bin and the torch lib directories to the
Python DLL search path **before** importing vLLM:

```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
os.add_dll_directory(r"E:\path\to\venv\Lib\site-packages\torch\lib")
import vllm
```

The test scripts in `tests/` already do this. If you're embedding vLLM
in your own code, copy the pattern.

### `OSError: The paging file is too small for this operation to complete. (os error 1455)` <a id="oserror-1455"></a>

Windows uses *commit charge* (RAM + pagefile) to back any process
allocation. If your pagefile is small or set to zero, even a process
with plenty of free physical RAM can fail to allocate large buffers
because the system commit limit is exhausted.

This shows up most often when loading large model weights — the
embedding tensor of a 14B model is ~1.5 GB and needs a contiguous
allocation.

**Fix 1**: enable a system pagefile.
1. Win+R → `sysdm.cpl` → Advanced → Performance → Settings → Advanced → Virtual memory → Change
2. Uncheck "Automatically manage paging file size for all drives"
3. Pick a drive with ≥16 GB free, choose "System managed size"
4. **Reboot**

**Fix 2**: this build's custom safetensors reader (in
`vllm/model_executor/model_loader/weight_utils.py`) bypasses the
problem by using `numpy.memmap` (file-backed, no commit charge) plus
chunked GPU streaming. It's already enabled — if you're seeing this
error you may have an older patched build. Re-apply
`vllm-windows-v3.patch`.

### `torch.OutOfMemoryError: CUDA out of memory. ... X GiB is free` <a id="oom-with-free-gpu"></a>

PyTorch's caching allocator can't satisfy a contiguous allocation even
when the GPU has plenty of free memory. This is fragmentation.

**Fix**: enable expandable segments before importing vLLM:

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Or in Python:

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import vllm  # must be after the env var
```

If you still get OOM, lower `gpu_memory_utilization` (try 0.5 first,
then 0.4).

### `ValueError: not enough values to unpack (expected 2, got 1)` in `torch.unique`

You're running an old PyTorch + new vLLM combo. The custom code uses
`torch.unique(t, sorted=True)` which returns a single tensor on
PyTorch 2.10+. Make sure your venv has `torch==2.10.0`.

### `pyo3_runtime.PanicException: Python API call failed`

safetensors crashed inside vLLM's engine context. This is the same
root cause as OSError 1455 — Windows commit limit. Enable a pagefile
or use the patched safetensors reader. See above.

### `Segmentation fault` during model loading

Same root cause family as the two errors above. Apply the
[OSError 1455](#oserror-1455) fix.

## Build errors

### `CMake Error: Generator: Visual Studio 17 2022 does not match the generator used previously: Ninja`

There are stale CMake caches in `vllm-source/.deps/` from a previous
build that used a different generator. Clean them:

```bat
rmdir /s /q vllm-source\.deps
del /s /q vllm-source\build
```

Then re-run `build.bat`.

### `cl : error C2018: unknown character 'or' / 'and' / 'not'`

MSVC doesn't accept `or` / `and` / `not` as keywords by default. The
patch fixes every known instance — make sure `vllm-windows-v3.patch`
applied cleanly:

```bat
cd vllm-source
git apply --check ..\vllm-windows-v3.patch
```

If the check fails, the patch is partially applied or the source has
been modified. Reset:

```bat
cd vllm-source
git checkout v0.19.0
git reset --hard v0.19.0
git apply ..\vllm-windows-v3.patch
```

### `nvcc fatal: Unsupported gpu architecture 'compute_120'`

Your CUDA toolkit doesn't support Blackwell (SM 12.0). Either:
- Lower `TORCH_CUDA_ARCH_LIST` (drop `12.0` from the list)
- Or upgrade to CUDA 12.8+

### `error C1061: compiler limit: blocks nested too deeply`

You hit MSVC's nested-block limit on the auto-generated Marlin kernel
selector. The patch converts these from `else if` chains to flat `if`
chains, which avoids the limit. If you still see this, the patch isn't
applied — see the previous section.

### `catastrophic error: out of memory` from nvcc

nvcc ran out of memory during template instantiation. This is most
common on the Marlin kernel files. Lower `MAX_JOBS`:

```bat
set MAX_JOBS=2
```

Or `MAX_JOBS=1` on a 16 GB RAM machine.

### `is not able to compile a simple test program. ... rc /fo ... no such file or directory`

The Windows SDK Resource Compiler (`rc.exe`) isn't on PATH. The build
needs both MSVC and the Windows SDK 10.0.19041 or newer. Open a
"Developer Command Prompt for VS 2022" and run `where rc.exe` — if
nothing is found, install the Windows SDK component in the VS Installer.

### `ImportError: cannot import name 'X' from 'vllm.v1...'`

The patch is partially applied. Run:

```bat
cd vllm-source
git status
```

to see what's modified. If the modifications don't match the patch's
expected changes, reset and re-apply:

```bat
cd vllm-source
git checkout .
cd ..
build.bat
```

## Runtime warnings (safe to ignore)

### `WARNING: Distributed backend nccl is not available; falling back to fake.`

NCCL doesn't ship with PyTorch on Windows. The patch wires up
`FakeProcessGroup` so single-GPU operation still works. This warning
is expected.

### `triton kernels are Linux-only. Pure PyTorch fallbacks will be used on Windows.`

This is from `multi_turboquant`. The build uses `triton-windows`
(separate package, ships only Triton) so Triton itself does work. The
warning is misleading — your build does have Triton kernels.

### `Failed to compute shorthash for libnvrtc.so`

`libnvrtc.so` is a Linux library; Windows uses `nvrtc64_*.dll`. Harmless.
