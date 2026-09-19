# Choose a Windows release

| Channel | vLLM | Installer Python | Torch / CUDA | Release |
|---|---|---|---|---|
| **Stable / Latest** | 0.27.1 | 3.13.14 | 2.13.0+cu130 / 13.0 | [Stable](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.27.1-win-cu130) |
| Prerelease | 0.29.0 | 3.13.14 | 2.13.0+cu130 / 13.0 | [Python 3.13 / cu130](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu130-rc1) |
| Prerelease | 0.29.0 | 3.14.2 | 2.13.0+cu132 / 13.2 | [Python 3.14 / cu132](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu132-py314-rc1) |

The two prereleases are runtime variants of the same vLLM version. Neither
replaces stable/Latest. **The cu132 variant uses CPU TorchAudio**; GPU audio
processing and audio-model serving were not validated. vLLM inference uses CUDA.

## Install a prebuilt wheel

Download **Source code (zip)** from your chosen release, extract into a new
directory, and run `install.bat`, then `launch.bat`. The installer downloads
Python and the matching prebuilt wheels; **no compilation is required**.
Keep existing installations separate and use a fresh filesystem KV-cache
directory when testing a prerelease.

A default `git clone` checks out `master`, whose installer provides stable
0.27.1. Git itself does not download a Python runtime. For a prerelease checkout:

```bat
git clone --branch v0.29.0-win-cu132-py314-rc1 https://github.com/aivrar/vllm-windows-build.git vllm-py314-cu132
```

For the Python 3.13 variant, substitute tag `v0.29.0-win-cu130-rc1` and a
separate destination directory. Release tags identify fixed installer/source
snapshots. The maintained branches are `prerelease/v0.29.0-win-cu130` and
`prerelease/v0.29.0-win-cu132-py314`.

## Compile your own wheel

`install.bat` installs prebuilt wheels; it is not the source-build workflow.
Use the build record for the selected version, including its Python/CUDA/Torch
pins, upstream source revision, patches and vendor revisions:

- [Stable 0.27.1 source/build record](https://github.com/aivrar/vllm-windows-build/blob/v0.27.1-win-cu130/docs/v0.27.1-build-candidate.md).
- [0.29.0 Python 3.13 / cu130 source/build record](https://github.com/aivrar/vllm-windows-build/blob/v0.29.0-win-cu130-rc1/docs/v0.29.0-build-candidate.md).
- [0.29.0 Python 3.14 / cu132 source deltas and build pins](https://github.com/aivrar/vllm-windows-build/blob/v0.29.0-win-cu132-py314-rc1/docs/v0.29.0-cu132-py314.md#exact-source-and-build-changes).

The Python patch version in a historical build record can differ from the
portable installer's patch version; the table above describes the installers.
For the new cu132 build, both use Python 3.14.2. Historical root `build.bat`
targets 0.25.1; running it is not a way to build either 0.29.0 variant.
Maintainer workspace paths in build records describe the recorded environment;
substitute your own paths when reproducing it.

## Validation and limitations

Both 0.29.0 wheels passed targeted RTX 3090 checks and include the issue #16
Marlin fallback fixes. Exact scope is recorded on each release page.
**Native Blackwell FP4 is not added**, and RTX 3090 validation does not establish
Blackwell execution. No speed advantage is claimed for Python 3.14/cu132.
The stable 0.27.1 workaround remains documented for users staying on stable.
