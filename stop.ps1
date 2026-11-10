param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$listeners = @(Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -eq 0) {
    Write-Host 'Whisper is already stopped.'
    exit 0
}

foreach ($processId in ($listeners.OwningProcess | Select-Object -Unique)) {
    $serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
    $command = $serverProcess.CommandLine
    if (-not $command -or
        $command.IndexOf($projectPython, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
        $command -notmatch '(app\.py|import app;)') {
        Write-Error "Port 7860 belongs to another application (PID $processId). Nothing was stopped."
        exit 1
    }
    if ($CheckOnly) {
        Write-Host "Found this project's Whisper server (PID $processId). Check only; nothing was stopped."
    } else {
        # Inference runs in a child process; stop it along with the server.
        $processSnapshot = @(Get-CimInstance Win32_Process)
        function Stop-WhisperChildren([int]$parentProcessId) {
            foreach ($child in ($processSnapshot | Where-Object { $_.ParentProcessId -eq $parentProcessId })) {
                Stop-WhisperChildren $child.ProcessId
                Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
            }
        }
        Stop-WhisperChildren $processId
        Stop-Process -Id $processId -ErrorAction Stop
        Write-Host "Whisper stopped (PID $processId)."
    }
}
