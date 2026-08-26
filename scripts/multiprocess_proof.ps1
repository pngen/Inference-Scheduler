param([int]$Port = 29833, [string]$BuildDir = 'build')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$build = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $root $BuildDir }
$coord = Join-Path $build 'inference_scheduler_coordinator.exe'
$worker = Join-Path $build 'inference_scheduler_worker.exe'
$client = Join-Path $build 'inference_scheduler_client.exe'
$scr = Join-Path $root 'scratch'
New-Item -ItemType Directory -Force -Path $scr | Out-Null
Remove-Item (Join-Path $scr 'resume_id.txt'),(Join-Path $scr 'proceed.flag') -ErrorAction SilentlyContinue

Write-Host "extended multiprocess proof: port=$Port build=$build"
Write-Host 'starting coordinator...'
$cp = Start-Process -FilePath $coord -ArgumentList @('127.0.0.1',$Port,'2') -PassThru -RedirectStandardOutput (Join-Path $scr 'coord.log') -RedirectStandardError (Join-Path $scr 'coord.err')
Start-Sleep -Milliseconds 700
# worker1a: boot 1001, models 10,30 ; worker2: boot 2002, models 20
$w1a = Start-Process -FilePath $worker -ArgumentList @('127.0.0.1',$Port,'11','1001','8','cpu','10,30,40') -PassThru -RedirectStandardOutput (Join-Path $scr 'w1a.log') -RedirectStandardError (Join-Path $scr 'w1a.err')
$w2 = Start-Process -FilePath $worker -ArgumentList @('127.0.0.1',$Port,'22','2002','8','cpu','20') -PassThru -RedirectStandardOutput (Join-Path $scr 'w2.log') -RedirectStandardError (Join-Path $scr 'w2.err')
Start-Sleep -Milliseconds 1200

Write-Host 'starting client (phase A + wait + phase B)...'
$cl = Start-Process -FilePath $client -ArgumentList @('127.0.0.1',$Port,'extended',$scr) -PassThru -RedirectStandardOutput (Join-Path $scr 'client.log') -RedirectStandardError (Join-Path $scr 'client.err')

# wait for PHASE_A_DONE
$phaseDone = $false
for ($i = 0; $i -lt 300; $i++) {
  if (Test-Path (Join-Path $scr 'client.log')) { $t = Get-Content (Join-Path $scr 'client.log') -Raw; if ($t -match 'PHASE_A_DONE') { $phaseDone = $true; break } }
  if ($cl.HasExited) { break }
  Start-Sleep -Milliseconds 100
}
if (-not $phaseDone) {
  Write-Host 'FAIL: PHASE_A_DONE not observed';
  Stop-Process -Id $w1a.Id,$w2.Id,$cp.Id -Force -ErrorAction SilentlyContinue
  exit 1
}

Write-Host 'terminating worker1a (boot 1001) as a real process...'
Stop-Process -Id $w1a.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 600
Write-Host 'starting restarted worker1b (boot 9999, new identity)...'
$w1b = Start-Process -FilePath $worker -ArgumentList @('127.0.0.1',$Port,'11','9999','8','cpu','10,30,40') -PassThru -RedirectStandardOutput (Join-Path $scr 'w1b.log') -RedirectStandardError (Join-Path $scr 'w1b.err')
Start-Sleep -Milliseconds 1400
Set-Content -Path (Join-Path $scr 'proceed.flag') -Value 'go'

Write-Host 'waiting for client to finish...'
Wait-Process -Id $cl.Id
$out = ''
if (Test-Path (Join-Path $scr 'client.log')) { $out = Get-Content (Join-Path $scr 'client.log') -Raw }
Write-Host '--- client output ---'
Write-Host $out

Stop-Process -Id $w1b.Id,$w2.Id,$cp.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 300
if ($out -match 'CLIENT EXTENDED PASS') { Write-Host 'MULTIPROCESS_EXTENDED PASS'; exit 0 }
Write-Host 'MULTIPROCESS_EXTENDED FAIL'; exit 1