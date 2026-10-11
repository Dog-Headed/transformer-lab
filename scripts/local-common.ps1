$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RuntimeRoot = Join-Path $ProjectRoot '.runtime'

function Get-ProjectPython {
    $pythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw 'Python environment missing. Run setup.cmd first.'
    }
    return $pythonPath
}

function Get-ProjectNode {
    $nodePath = Join-Path $ProjectRoot '.tools\node\node.exe'
    if (-not (Test-Path -LiteralPath $nodePath)) {
        $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
        if ($null -eq $nodeCommand) { throw 'Install Node.js 22.13+ or provide .tools\node\node.exe.' }
        $nodePath = $nodeCommand.Source
    }
    $nodeVersion = (& $nodePath --version).TrimStart('v')
    if ([version]$nodeVersion -lt [version]'22.13.0') { throw 'Node.js 22.13+ is required.' }
    return $nodePath
}

function Test-LocalHttp([string]$Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch { return $false }
}

function Get-PortOwner([int]$Port) {
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) { return $null }
    $ownerId = $listeners[0].OwningProcess
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerId"
    # A Windows venv launcher starts a child with the base Python executable.
    # Match the recorded parent and its start time, then verify the listener is
    # that process or a descendant. Do not infer ownership from the base path.
    $serviceName = if ($Port -eq 8000) { 'backend' } else { 'frontend' }
    $recordPath = Join-Path $RuntimeRoot "$serviceName.json"
    if (Test-Path -LiteralPath $recordPath) {
        $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
        $recordedProcess = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
        if ($record.root -eq $ProjectRoot -and $null -ne $recordedProcess -and
            $recordedProcess.StartTime.ToUniversalTime().Ticks -eq ([DateTimeOffset]$record.started).UtcDateTime.Ticks) {
            $candidate = $owner
            for ($depth = 0; $depth -lt 8 -and $null -ne $candidate; $depth++) {
                if ($candidate.ProcessId -eq $record.pid) { return $owner }
                $candidate = Get-CimInstance Win32_Process -Filter "ProcessId = $($candidate.ParentProcessId)"
            }
        }
    }
    throw "Port $Port belongs to an unrecorded service. Stop that service before starting this project."
}

function Save-ServiceRecord([string]$Name, $Process) {
    $record = @{pid=$Process.Id; started=$Process.StartTime.ToUniversalTime().ToString('o'); root=$ProjectRoot}
    $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RuntimeRoot "$Name.json") -Encoding UTF8
}
