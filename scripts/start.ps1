param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'local-common.ps1')
Set-Location -LiteralPath $ProjectRoot
$pythonPath = Get-ProjectPython
$nodePath = Get-ProjectNode
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'node_modules\vinext'))) {
    throw 'Frontend dependencies missing. Run setup.cmd first.'
}
New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
$env:Path = (Split-Path $nodePath) + ';' + $env:Path

$backendOwner = Get-PortOwner 8000
if ($null -eq $backendOwner) {
    $backendProcess = Start-Process -FilePath $pythonPath -ArgumentList '-m backend.cli serve --host 127.0.0.1 --port 8000' -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RuntimeRoot 'backend.out.log') -RedirectStandardError (Join-Path $RuntimeRoot 'backend.err.log')
    Save-ServiceRecord 'backend' $backendProcess
    Write-Host 'Starting Python training service...'
}
$frontendOwner = Get-PortOwner 5173
if ($null -eq $frontendOwner) {
    $frameworkPath = Join-Path $ProjectRoot 'scripts\run-framework.mjs'
    $frontendProcess = Start-Process -FilePath $nodePath -ArgumentList ('"' + $frameworkPath + '" dev') -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RuntimeRoot 'frontend.out.log') -RedirectStandardError (Join-Path $RuntimeRoot 'frontend.err.log')
    Save-ServiceRecord 'frontend' $frontendProcess
    Write-Host 'Starting visualization website...'
}

$deadline = (Get-Date).AddSeconds(180)
$backendReady = $false
$frontendReady = $false
while ((Get-Date) -lt $deadline) {
    if (-not $backendReady) { $backendReady = Test-LocalHttp 'http://127.0.0.1:8000/health' }
    if (-not $frontendReady) { $frontendReady = Test-LocalHttp 'http://localhost:5173/training' }
    if ($backendReady -and $frontendReady) { break }
    Start-Sleep -Seconds 2
}
if (-not ($backendReady -and $frontendReady)) {
    foreach ($logName in @('backend.err.log','frontend.err.log')) {
        $logPath = Join-Path $RuntimeRoot $logName
        if (Test-Path -LiteralPath $logPath) { Get-Content -LiteralPath $logPath -Tail 15 }
    }
    throw "Startup failed or timed out. Logs: $RuntimeRoot"
}
$health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 5
Write-Host "Ready. GPU available: $($health.cuda_available)"
Write-Host 'Website: http://localhost:5173/training'
Write-Host 'API docs: http://127.0.0.1:8000/docs'
Write-Host "Logs: $RuntimeRoot"
if (-not $NoBrowser) { Start-Process 'http://localhost:5173/training' }
