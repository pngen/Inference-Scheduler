param($BuildDir = 'C:\Users\pauln\AppData\Local\Temp\is_consumer_build')
$ErrorActionPreference='Stop'
$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
& cmake -S 'C:\Users\pauln\AppData\Local\Temp\is_consumer' -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DInferenceScheduler_DIR='E:\The Journey\Coding\GitHub\production\Inference-Scheduler\_install\lib\cmake\InferenceScheduler'
if ($LASTEXITCODE -ne 0) { throw 'consumer configure failed' }
& cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw 'consumer build failed' }
& (Join-Path $BuildDir 'consumer.exe')
Write-Host ('CONSUMER_EXIT=' + $LASTEXITCODE)