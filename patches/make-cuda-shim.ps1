<#
.SYNOPSIS
  Create a shim directory with a copy of the CUDA toolkit's cuda.h whose CUtensorMap is alignas(64).

.DESCRIPTION
  MSVC cannot pass 128-byte-aligned by-value kernel parameters (error C2719 in the generated
  .cudafe1.stub.c). CUDA 13.x declares CUtensorMap as alignas(128); only 12.6/12.8 use alignas(64).
  Building the native SM120 CUTLASS kernels on Windows therefore needs a cuda.h with alignas(64)
  placed first on the nvcc include path. Point VLLM_CUDA_SHIM_DIR at the output directory.
  See docs/blackwell-sm120.md. The toolkit itself is not modified.

.EXAMPLE
  .\patches\make-cuda-shim.ps1 -Out C:\vllm-build\shim
  $env:VLLM_CUDA_SHIM_DIR = 'C:/vllm-build/shim'
#>
param(
  [Parameter(Mandatory)][string]$Out,
  [string]$Cuda = $env:CUDA_PATH
)
$ErrorActionPreference = 'Stop'
if (-not $Cuda) { throw 'Pass -Cuda <toolkit root> or set CUDA_PATH.' }
$src = Join-Path $Cuda 'include\cuda.h'
if (-not (Test-Path $src)) { throw "cuda.h not found: $src" }
$text = [IO.File]::ReadAllText($src)
$n = ([regex]::Matches($text, 'alignas\(128\)')).Count + ([regex]::Matches($text, '_Alignas\(128\)')).Count
if ($n -eq 0) { Write-Warning "No 128-byte alignment found in $src - this toolkit probably does not need the shim."; return }
New-Item -ItemType Directory -Force $Out | Out-Null
$text = $text.Replace('alignas(128)', 'alignas(64)').Replace('_Alignas(128)', '_Alignas(64)')
[IO.File]::WriteAllText((Join-Path $Out 'cuda.h'), $text, (New-Object Text.UTF8Encoding $false))
"Replaced $n occurrence(s); shim written to $(Join-Path $Out 'cuda.h')"
"Set VLLM_CUDA_SHIM_DIR=$($Out -replace '\\','/') before building."
