<#
  eject_native.ps1  -  release a removable volume the same way Windows'
  "Safely Remove Hardware" does, BUT without deleting the drive letter.

  Why this file exists
  --------------------
  The old shutdown script used:  diskpart  select volume E / remove all dismount
  "remove all" deletes EVERY mount point of the volume, including the drive
  letter. Windows stores that letter nowhere else for removable media, so the
  next time the stick is plugged in it shows up with no letter at all: nothing
  in Explorer, no Eject entry in the tray, only Disk Management can fix it.

  This script instead talks to the volume handle directly:
      FSCTL_LOCK_VOLUME       - ask for exclusive use (best effort)
      FSCTL_DISMOUNT_VOLUME   - force the file system off the volume
      IOCTL_STORAGE_MEDIA_REMOVAL - clear "prevent removal"
      IOCTL_STORAGE_EJECT_MEDIA   - the real eject
  The mount point (drive letter) stays registered, so re-plugging gives the
  same letter back.

  Exit codes
  ----------
      0  ejected - safe to unplug
      2  could not open the volume at all
      3  volume is still locked / busy (a process outside this folder)
      4  eject request refused by the device
#>
param(
    [string]$Drive = 'E:',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$letter = $Drive.TrimEnd('\').TrimEnd(':')
if ($letter.Length -ne 1) {
    Write-Host "  [!] bad drive argument: $Drive"
    exit 2
}

$src = @'
using System;
using System.Runtime.InteropServices;

public static class J022Vol
{
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    public static extern IntPtr CreateFileW(string name, uint access, uint share,
        IntPtr sa, uint disposition, uint flags, IntPtr template);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool DeviceIoControl(IntPtr h, uint code, IntPtr inBuf,
        uint inSize, IntPtr outBuf, uint outSize, out uint returned, IntPtr overlapped);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr h);
}
'@

if (-not ('J022Vol' -as [type])) { Add-Type -TypeDefinition $src }

$FSCTL_LOCK_VOLUME     = 0x00090018
$FSCTL_DISMOUNT_VOLUME = 0x00090020
$IOCTL_MEDIA_REMOVAL   = 0x002D4804
$IOCTL_EJECT_MEDIA     = 0x002D4808

# 0xC0000000 does not fit a signed int in PowerShell - use the decimal form
$GENERIC_RW  = [uint32]3221225472
$SHARE_RW    = 0x00000003
$OPEN_EXIST  = 3
$INVALID     = [IntPtr](-1)
$path        = '\\.\' + $letter + ':'

function Open-Vol([uint32]$access, [uint32]$share) {
    [J022Vol]::CreateFileW($path, $access, $share, [IntPtr]::Zero, $OPEN_EXIST, 0, [IntPtr]::Zero)
}

$h = Open-Vol $GENERIC_RW $SHARE_RW
if ($h -eq $INVALID) {
    # 0 access rights can succeed even when another process holds the volume
    $h = Open-Vol 0 7
}
if ($h -eq $INVALID) {
    Write-Host "  [!] cannot open $letter`: (error $([System.Runtime.InteropServices.Marshal]::GetLastWin32Error()))"
    exit 2
}

$bytes = [uint32]0

if ($DryRun) {
    Write-Host "  [v] volume handle opened for $letter`: - interop OK (dry run, nothing changed)"
    [void][J022Vol]::CloseHandle($h)
    exit 0
}

# 1) try to get exclusive use; not fatal if it fails
$locked = [J022Vol]::DeviceIoControl($h, $FSCTL_LOCK_VOLUME, [IntPtr]::Zero, 0,
    [IntPtr]::Zero, 0, [ref]$bytes, [IntPtr]::Zero)
if ($locked) { Write-Host "  [v] volume locked" } else { Write-Host "  [i] lock refused - forcing dismount anyway" }

# 2) force the file system off the volume (letter is NOT touched)
$dismounted = [J022Vol]::DeviceIoControl($h, $FSCTL_DISMOUNT_VOLUME, [IntPtr]::Zero, 0,
    [IntPtr]::Zero, 0, [ref]$bytes, [IntPtr]::Zero)
if ($dismounted) { Write-Host "  [v] volume dismounted (drive letter kept)" } else { Write-Host "  [i] dismount refused" }

# 3) clear "prevent media removal", then eject
$buf = [System.Runtime.InteropServices.Marshal]::AllocHGlobal(1)
[System.Runtime.InteropServices.Marshal]::WriteByte($buf, 0, 0)
$prevCleared = [J022Vol]::DeviceIoControl($h, $IOCTL_MEDIA_REMOVAL, $buf, 1,
    [IntPtr]::Zero, 0, [ref]$bytes, [IntPtr]::Zero)
$ejected = [J022Vol]::DeviceIoControl($h, $IOCTL_EJECT_MEDIA, [IntPtr]::Zero, 0,
    [IntPtr]::Zero, 0, [ref]$bytes, [IntPtr]::Zero)
[void][J022Vol]::CloseHandle($h)
[System.Runtime.InteropServices.Marshal]::FreeHGlobal($buf)

if ($ejected) {
    Write-Host "  [v] ejection request accepted"
    exit 0
}
if (-not $dismounted) {
    Write-Host "  [!] volume is still in use by another program"
    exit 3
}
Write-Host "  [!] device refused the eject command"
exit 4
