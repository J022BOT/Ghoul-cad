$ErrorActionPreference = 'Stop'
$Prefer = 'E'
function Say($m) { Write-Host ("  " + $m) }

Say "Scanning removable disks ..."

$cands = @()
foreach ($d in (Get-Disk | Where-Object { $_.BusType -in @('USB','SD') })) {
    if ($d.IsOffline) { try { Set-Disk -Number $d.Number -IsOffline $false } catch { } }
    if ($d.IsReadOnly) { try { Set-Disk -Number $d.Number -IsReadOnly $false } catch { } }
    foreach ($p in (Get-Partition -DiskNumber $d.Number -ErrorAction SilentlyContinue)) {
        if ($p.DriveLetter) { continue }
        if ($p.Type -eq 'Reserved') { continue }
        if ($p.Size -lt 100MB) { continue }
        $cands += [pscustomobject]@{ Disk = $d.Number; Part = $p.PartitionNumber; Size = $p.Size; Model = $d.FriendlyName }
    }
}

if (-not $cands) {
    Say "Nothing to repair: every removable partition already has a drive letter."
    Say "If the stick is still invisible, unplug it and try another USB port."
    exit 0
}

$used = @(Get-Volume | Where-Object { $_.DriveLetter } | ForEach-Object { $_.DriveLetter })

function Get-FreeLetter {
    param([string]$Want)
    if ($Want -and ($used -notcontains $Want)) { return $Want }
    foreach ($c in @('E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z')) {
        if ($used -notcontains $c) { return $c }
    }
    return $null
}

$fail = 0
foreach ($c in $cands) {
    $letter = Get-FreeLetter $Prefer
    if (-not $letter) { Say "No free drive letter left."; $fail = 1; break }
    try {
        Set-Partition -DiskNumber $c.Disk -PartitionNumber $c.Part -NewDriveLetter $letter -ErrorAction Stop
        $used += $letter
        Say ("[v] disk " + $c.Disk + " partition " + $c.Part + " (" + $c.Model + ", " + [math]::Round($c.Size/1GB,1) + " GB) -> " + $letter + ":")
    } catch {
        Say ("[!] failed: " + $_.Exception.Message)
        $fail = 1
    }
}

Start-Sleep -Seconds 2
Write-Host ""
Say "Current volumes:"
Get-Volume | Where-Object { $_.DriveType -eq 'Removable' } | ForEach-Object {
    $l = if ($_.DriveLetter) { $_.DriveLetter + ':' } else { '(no letter)' }
    Write-Host ("      " + $l + "  " + $_.FileSystemLabel + "  " + $_.FileSystem + "  " + [math]::Round($_.Size/1GB,1) + " GB  " + $_.HealthStatus)
}

if ($fail -eq 0) { exit 0 } else { exit 1 }
