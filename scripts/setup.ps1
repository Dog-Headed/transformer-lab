param([ValidateSet('cuda','cpu')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'local-common.ps1')
Set-Location -LiteralPath $ProjectRoot
$pythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $basePython = Get-Command python.exe -ErrorAction Stop
    & $basePython.Source -m venv (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment.' }
}
if ($Device -eq 'cuda') {
    & $pythonPath -m pip install -r backend/requirements-cuda.txt
} else {
    & $pythonPath -m pip install --force-reinstall torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $pythonPath -m pip install -e '.[test]'
if ($LASTEXITCODE -ne 0) { throw 'Python project installation failed.' }
$nodePath = Get-ProjectNode
$env:Path = (Split-Path $nodePath) + ';' + $env:Path
$localPnpm = Join-Path $ProjectRoot '.tools\pnpm\bin\pnpm.mjs'
if (Test-Path -LiteralPath (Join-Path $ProjectRoot 'node_modules\vinext')) {
    Write-Host 'Frontend dependencies already installed.'
} elseif (Test-Path -LiteralPath $localPnpm) {
    & $nodePath $localPnpm install --prefer-offline
} else {
    $npmCommand = Get-Command npm.cmd -ErrorAction Stop
    & $npmCommand.Source run install:ci
}
if ($LASTEXITCODE -ne 0) { throw 'Frontend installation failed.' }
& $pythonPath -m backend doctor
if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed.' }
Write-Host 'Setup complete. Run start.cmd.'
