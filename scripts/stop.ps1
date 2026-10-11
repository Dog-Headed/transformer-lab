$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'local-common.ps1')
foreach ($name in @('frontend', 'backend')) {
    $recordPath = Join-Path $RuntimeRoot "$name.json"
    if (-not (Test-Path -LiteralPath $recordPath)) { continue }
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    if ($record.root -ne $ProjectRoot) { throw 'Unexpected service record root.' }
    $serviceProcess = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if ($null -eq $serviceProcess) { continue }
    if ($serviceProcess.StartTime.ToUniversalTime().Ticks -ne ([DateTimeOffset]$record.started).UtcDateTime.Ticks) {
        throw "Service PID $($record.pid) has been reused; refusing to stop it."
    }
    # Stop descendants only after identifying the original project-owned parent.
    $snapshot = @(Get-CimInstance Win32_Process)
    function Stop-OwnedTree([int]$ProcessId) {
        foreach ($child in @($snapshot | Where-Object { $_.ParentProcessId -eq $ProcessId })) {
            Stop-OwnedTree $child.ProcessId
        }
        Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
    }
    Stop-OwnedTree $record.pid
    Write-Host "Stopped $name."
}
