param([int]$Port = 29833)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$build = Join-Path $root 'build'
$coord = Join-Path $build 'inference_scheduler_coordinator.exe'
$worker = Join-Path $build 'inference_scheduler_worker.exe'
$client = Join-Path $build 'inference_scheduler_client.exe'
$scr = Join-Path $root 'scratch'
New-Item -ItemType Directory -Force -Path $scr | Out-Null

Write-Host 'starting coordinator...'
$cp = Start-Process -FilePath $coord -ArgumentList @('127.0.0.1',$Port,'2') -PassThru -RedirectStandardOutput (Join-Path $scr 'coord.log') -RedirectStandardError (Join-Path $scr 'coord.err')
Start-Sleep -Milliseconds 600
$w1 = Start-Process -FilePath $worker -ArgumentList @('127.0.0.1',$Port,'11','1001','8','cpu','10,30') -PassThru -RedirectStandardOutput (Join-Path $scr 'w1.log') -RedirectStandardError (Join-Path $scr 'w1.err')
$w2 = Start-Process -FilePath $worker -ArgumentList @('127.0.0.1',$Port,'22','2002','8','cpu','20') -PassThru -RedirectStandardOutput (Join-Path $scr 'w2.log') -RedirectStandardError (Join-Path $scr 'w2.err')
Start-Sleep -Milliseconds 1000

Write-Host 'starting client...'
$cl = Start-Process -FilePath $client -ArgumentList @('127.0.0.1',$Port) -PassThru -Wait -NoNewWindow -RedirectStandardOutput (Join-Path $scr 'client.log') -RedirectStandardError (Join-Path $scr 'client.err')
Write-Host '--- client output ---'
if (Test-Path (Join-Path $scr 'client.log')) { Get-Content (Join-Path $scr 'client.log') -Raw }
Write-Host '--- worker1 log ---'
if (Test-Path (Join-Path $scr 'w1.log')) { Get-Content (Join-Path $scr 'w1.log') -Raw }
Write-Host '--- worker2 log ---'
if (Test-Path (Join-Path $scr 'w2.log')) { Get-Content (Join-Path $scr 'w2.log') -Raw }
Write-Host '--- coord log ---'
if (Test-Path (Join-Path $scr 'coord.err')) { Get-Content (Join-Path $scr 'coord.err') -Raw }
$exit = $cl.ExitCode
Stop-Process -Id $w1.Id,$w2.Id,$cp.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 300
Write-Host "CLIENT EXIT: $exit"
exit $exit
