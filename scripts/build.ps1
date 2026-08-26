param(
  [string]$Configuration = "Release",
  [string]$BuildDir = "build",
  [switch]$Cuda,
  [string]$CudaToolkit = "v13.1"
)
$ErrorActionPreference = "Stop"
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
  }
}
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$CudaToolkit\bin"
$env:PATH = "$cudaBin;" + $env:PATH
$genArgs = @("-S", ".", "-B", $BuildDir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=$Configuration")
$cudaOn = "$Cuda".ToLower()
if ($cudaOn -eq "true") {
  $genArgs += "-DIS_BUILD_CUDA=ON"
  $genArgs += "-DCMAKE_CUDA_COMPILER=$cudaBin\nvcc.exe"
  $genArgs += "-DCMAKE_CUDA_ARCHITECTURES=120"
} else {
  $genArgs += "-DIS_BUILD_CUDA=OFF"
}
Write-Host "Configuring: cmake $($genArgs -join ' ')"
& cmake @genArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
Write-Host "Building..."
& cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
Write-Host "Build OK ($Configuration) in $BuildDir"
