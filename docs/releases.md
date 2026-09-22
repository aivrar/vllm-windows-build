# Choose a Windows release

| Channel | vLLM | Installer Python | Torch / CUDA | Release |
|---|---|---|---|---|
| **Stable / Latest** | 0.27.1 | 3.13.14 | 2.13.0+cu130 / 13.0 | [Stable](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.27.1-win-cu130) |
| Regular release (not Latest) | 0.29.0 | 3.14.2 | 2.13.0+cu132 / 13.2 | [Python 3.14 / cu132](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu132-py314) |
| Prerelease | 0.29.0 | 3.13.14 | 2.13.0+cu130 / 13.0 | [Python 3.13 / cu130](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu130-rc1) |

The two 0.29.0 builds are runtime variants of the same vLLM version. Neither
replaces the 0.27.1 Latest release. The Python 3.14 variant uses **CPU
TorchAudio**; GPU audio processing and audio-model serving were not validated.
That narrower audio scope is why 0.27.1 remains the default Latest release.
vLLM model inference still uses CUDA.

The final Python 3.14 release includes the corrected `launch.bat` in its source
ZIP. The earlier `rc1` source ZIP does not; existing `rc1` installs can replace
their launcher with the [corrected asset](https://github.com/aivrar/vllm-windows-build/releases/download/v0.29.0-win-cu132-py314-rc1/launch.bat).

## Install a prebuilt wheel

Download **Source code (zip)** from your chosen release, extract into a new
directory, and run `install.bat`, then `launch.bat`. The installer downloads
Python and the matching prebuilt wheels; **no compilation is required**.
Keep existing installations separate and use a fresh filesystem KV-cache
directory when switching release variants.

A default `git clone` checks out `master`, whose installer provides stable
0.27.1. Git itself does not download a Python runtime. For the Python 3.14 release:

```bat
git clone --branch v0.29.0-win-cu132-py314 https://github.com/aivrar/vllm-windows-build.git vllm-py314-cu132
```

For the Python 3.13 variant, substitute tag `v0.29.0-win-cu130-rc1` and a
separate destination directory. Release tags identify fixed installer/source
snapshots. The maintained branches are `prerelease/v0.29.0-win-cu130` and
`release/v0.29.0-win-cu132-py314`.

## Compile your own wheel

`install.bat` installs prebuilt wheels; it is not the source-build workflow.
Use the build record for the selected version, including its Python/CUDA/Torch
pins, upstream source revision, patches and vendor revisions:

- [Stable 0.27.1 source/build record](https://github.com/aivrar/vllm-windows-build/blob/v0.27.1-win-cu130/docs/v0.27.1-build-candidate.md).
- [0.29.0 Python 3.13 / cu130 source/build record](https://github.com/aivrar/vllm-windows-build/blob/v0.29.0-win-cu130-rc1/docs/v0.29.0-build-candidate.md).
- [0.29.0 Python 3.14 / cu132 source deltas and build pins](https://github.com/aivrar/vllm-windows-build/blob/v0.29.0-win-cu132-py314/docs/v0.29.0-cu132-py314.md#exact-source-and-build-changes).

The Python patch version in a historical build record can differ from the
portable installer's patch version; the table above describes the installers.
For the new cu132 build, both use Python 3.14.2. Historical root `build.bat`
targets 0.25.1; running it is not a way to build either 0.29.0 variant.
Maintainer workspace paths in build records describe the recorded environment;
substitute your own paths when reproducing it.

## Validation and limitations

Both 0.29.0 wheels passed targeted RTX 3090 checks and include the issue #16
Marlin fallback fixes. An RTX 5090 tester also confirmed the Python 3.14 build
with one specific Qwen3.8-27B INT4 model at 128K context, FP8 KV and MTP.
Exact scope is recorded on each release page. **Native Blackwell FP4 is not
added**. No speed advantage is claimed for Python 3.14/cu132 or for MTP.
The stable 0.27.1 workaround remains documented for users staying on stable.
