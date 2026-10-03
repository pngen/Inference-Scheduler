param(
  $ConsumerDir = (Join-Path ([System.IO.Path]::GetTempPath()) 'is_consumer'),
  $BuildDir = (Join-Path ([System.IO.Path]::GetTempPath()) 'is_consumer_build')
)
$ErrorActionPreference='Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
$installDir = Join-Path $repoRoot '_install'
& cmake -S $ConsumerDir -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DInferenceScheduler_DIR=(Join-Path $installDir 'lib\cmake\InferenceScheduler')
if ($LASTEXITCODE -ne 0) { throw 'consumer configure failed' }
& cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw 'consumer build failed' }
& (Join-Path $BuildDir 'consumer.exe')
Write-Host ('CONSUMER_EXIT=' + $LASTEXITCODE)
