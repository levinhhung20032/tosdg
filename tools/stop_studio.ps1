# Stops every running tosdg Studio before start_studio.bat starts a new one (a forced restart), and frees the port.
#   - every python process running "tosdg studio" (python / python3 / pythonw / py, any version) is stopped, with the
#     start_studio.bat window it ran in;
#   - the calling start_studio.bat window (this script's parent) is left alone;
#   - whatever still listens on the port after that is closed too (its name is printed); exit code 1 only when the port
#     cannot be freed (e.g. a system service or a program run as administrator), so the .bat can say so.
# The Studio tab of the old run closes itself when the new run opens its tab (see retire() in studio/static/index.html).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools\stop_studio.ps1 [-Port 8765]
param([int]$Port = 8765)

$self = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$callerCmd = $self.ParentProcessId                       # the cmd.exe running the start_studio.bat that called us

$studios = @(Get-CimInstance Win32_Process -Filter "Name LIKE 'python%.exe' OR Name='py.exe'" |
    Where-Object { $_.CommandLine -match '\btosdg\s+studio\b' })
foreach ($p in $studios) {
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.ParentProcessId)" -ErrorAction SilentlyContinue
    Write-Host "Closing the running Studio (PID $($p.ProcessId))"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    if ($parent -and $parent.ProcessId -ne $callerCmd -and $parent.Name -eq 'cmd.exe' -and $parent.CommandLine -match 'start_studio\.bat') {
        Stop-Process -Id $parent.ProcessId -Force -ErrorAction SilentlyContinue   # its old window would otherwise ask S/R
    }
}

# the programs listening on the port (none: $null); waits up to $Seconds for it to become free
function Wait-PortFree([double]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    do {
        $holders = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
        if (-not $holders) { return $null }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    return $holders
}

$holders = Wait-PortFree $(if ($studios) { 5 } else { 0 })   # a Studio just stopped may take a moment to release it
if (-not $holders) { exit 0 }

# still taken: close whatever holds it (PID 0/4 are the kernel - never touched)
foreach ($pid_ in @($holders | ForEach-Object { $_.OwningProcess } | Sort-Object -Unique)) {
    if ($pid_ -le 4) { continue }
    $proc = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
    Write-Host "Port $Port is used by $($proc.ProcessName) (PID $pid_): closing it"
    Stop-Process -Id $pid_ -Force -ErrorAction SilentlyContinue
}
$holders = Wait-PortFree 5
if (-not $holders) { exit 0 }

foreach ($h in $holders) {
    $proc = Get-Process -Id $h.OwningProcess -ErrorAction SilentlyContinue
    Write-Host "Port $Port is still used by $($proc.ProcessName) (PID $($h.OwningProcess)) and could not be closed. Close it and start again."
}
exit 1
