<#
.SYNOPSIS
  Snapshot the irreplaceable, gitignored artifact roots before a rebuild.
  This is **isolation criterion 5** of docs/CODE_REVIEW_AUDIT_2026-08-06.md, the last gate on the
  batched v2 rebuild.

.DESCRIPTION
  The rebuild overwrites the only copy of these trees. They are gitignored, so git cannot restore
  them; pushing commits does not back them up; and hard links are not backups at all (writing
  through either name mutates the same bytes -- the trap that leaked through read_only_cache).
  A checksum is not a control either: it detects damage *after* the write and offers no rollback.

  READ-ONLY on every source. The only writes are under -Destination.

  What is excluded, and why. cache/ctx_tiles (41.2 GB) and cache/hirise_jp2 (19.8 GB) are
  re-downloadable archives, and cache_v2 reaches them through JUNCTIONS. robocopy /E follows
  junctions, so without /XJ this would silently duplicate ~61 GB of Murray zips and HiRISE JP2s.
  Both /XJ and an explicit /XD are used -- belt and braces, because that duplication is the
  documented trap.

  Everything else is copied, deliberately more than the audit's 110 GB table enumerated. That
  table listed four derived cache subdirs (4.3 GB) and missed cache_v2/hirise_color (8.9 GB),
  cache_v2/validation (2.2 GB), craters, minconf_sweep, stage7 and the pds_* trees -- the cached
  PDS .LBLs among them, which are load-bearing for the HiRISE SP1 fix and for src.size_floor's
  MAP_SCALE. With ~1 TB free the simpler rule is safer: take all of it except the two big
  re-downloadable archives.

.PARAMETER DryRun
  robocopy /L -- list what would be copied, write nothing. Run this first.

.PARAMETER SkipCopy
  Re-verify an existing backup without copying again.

.PARAMETER Hash
  Also SHA-256 every file on both sides. Strongest verification; roughly doubles the I/O.

.EXAMPLE
  powershell -File scripts/backup_artifacts.ps1 -DryRun
  powershell -File scripts/backup_artifacts.ps1
  powershell -File scripts/backup_artifacts.ps1 -SkipCopy -Hash
#>
[CmdletBinding()]
param(
    [string]$Destination = "D:\HiRISE2CTX Backup",
    [switch]$DryRun,
    [switch]$SkipCopy,
    [switch]$Hash
)

$ErrorActionPreference = 'Stop'
$ScriptPath = $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptPath)
$ParentDir = Split-Path -Parent $RepoRoot

# Re-downloadable archives. Excluded by NAME at any depth, matching robocopy /XD semantics.
$ExcludeDirs = @('ctx_tiles', 'hirise_jp2')

# The snapshot set. `Out` keeps the two trees separable in the backup, because the out-of-repo
# detections are the ~4.2 GB the audit found were outside every artifact manifest it took.
$Sources = @(
    @{ Out = 'repo\dataset_v2'; Path = (Join-Path $RepoRoot 'dataset_v2') }
    @{ Out = 'repo\reports';    Path = (Join-Path $RepoRoot 'reports') }
    @{ Out = 'repo\dataset';    Path = (Join-Path $RepoRoot 'dataset') }
    @{ Out = 'repo\models';     Path = (Join-Path $RepoRoot 'models') }
    @{ Out = 'repo\cache';      Path = (Join-Path $RepoRoot 'cache') }
    @{ Out = 'repo\cache_v2';   Path = (Join-Path $RepoRoot 'cache_v2') }
    @{ Out = 'external\hirise_40_vClaire';            Path = (Join-Path $ParentDir 'hirise_40_vClaire') }
    @{ Out = 'external\hirise_priority10_detections'; Path = (Join-Path $ParentDir 'hirise_priority10_detections') }
)

function Test-Excluded([string]$rel) {
    foreach ($seg in ($rel -split [regex]::Escape('\'))) {
        if ($ExcludeDirs -contains $seg) { return $true }
    }
    return $false
}

function Get-Manifest([string]$root) {
    # Get-ChildItem -Recurse does not traverse reparse points on this stack, which is exactly what
    # /XJ does on the copy side -- the two must agree or verification reports phantom misses.
    $full = (Resolve-Path -LiteralPath $root).Path
    $out = New-Object System.Collections.ArrayList
    Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            $rel = $_.FullName.Substring($full.Length).TrimStart([char]92)
            if (-not (Test-Excluded $rel)) {
                [void]$out.Add([pscustomobject]@{ rel = $rel; len = $_.Length })
            }
        }
    return $out
}

# ---------------------------------------------------------------- preflight
if (-not (Test-Path -LiteralPath $Destination)) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}
$destFull = (Resolve-Path -LiteralPath $Destination).Path
if ($destFull.StartsWith($ParentDir, [StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "destination $destFull is inside the source tree; that would recurse, and it would put the backup where a producer glob can reach it"
    exit 2
}

Write-Output "=== snapshot plan ==="
$plan = New-Object System.Collections.ArrayList
$totalBytes = [int64]0
foreach ($s in $Sources) {
    if (-not (Test-Path -LiteralPath $s.Path)) {
        Write-Warning "source missing, skipped: $($s.Path)"
        continue
    }
    $m = Get-Manifest $s.Path
    $b = [int64]0
    if ($m.Count -gt 0) { $b = [int64](($m | Measure-Object -Property len -Sum).Sum) }
    [void]$plan.Add([pscustomobject]@{
        Out = $s.Out; Path = $s.Path; Files = $m.Count; Bytes = $b; Manifest = $m })
    $totalBytes += $b
    Write-Output ("{0,-44} {1,9:N0} files {2,9:N2} GB" -f $s.Out, $m.Count, ($b / 1GB))
}
if ($plan.Count -eq 0) { Write-Error "nothing to back up"; exit 2 }
Write-Output ("{0,-44} {1,9:N0} files {2,9:N2} GB" -f 'TOTAL',
    (($plan | Measure-Object -Property Files -Sum).Sum), ($totalBytes / 1GB))

$destDrive = (Split-Path -Qualifier $destFull).TrimEnd(':')
$free = (Get-Volume -DriveLetter $destDrive).SizeRemaining
Write-Output ("destination {0}" -f $destFull)
Write-Output ("free {0:N1} GB   required {1:N1} GB incl. 5% headroom" -f ($free / 1GB), ($totalBytes * 1.05 / 1GB))
if ($free -lt $totalBytes * 1.05) { Write-Error "not enough free space on ${destDrive}:"; exit 2 }

$srcDrive = (Split-Path -Qualifier $RepoRoot).TrimEnd(':')
if ($srcDrive -eq $destDrive) {
    Write-Warning "source and destination are the same volume (${srcDrive}:). This still protects against the rebuild overwriting the originals -- the risk criterion 5 names -- but NOT against drive failure."
}

# ---------------------------------------------------------------- copy
$logDir = Join-Path $destFull '_backup_meta'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$log = Join-Path $logDir "robocopy_$stamp.log"

if (-not $SkipCopy) {
    foreach ($p in $plan) {
        $dst = Join-Path $destFull $p.Out
        # NOT $args -- that is a PowerShell automatic variable and assigning it here would be
        # silently overwritten inside the call.
        $rcArgs = @($p.Path, $dst, '/E', '/XJ', '/COPY:DAT', '/DCOPY:DAT',
                    '/R:2', '/W:5', '/MT:8', '/NP', '/NFL', '/NDL', "/LOG+:$log")
        foreach ($x in $ExcludeDirs) { $rcArgs += '/XD'; $rcArgs += $x }
        if ($DryRun) { $rcArgs += '/L' }
        Write-Output "--- robocopy $($p.Out) -> $dst"
        & robocopy @rcArgs | Out-Null
        # robocopy exit codes are a BITMASK: 0-7 are success (1 = copied, 2 = extras, 4 =
        # mismatched), 8+ are real failures. Testing -ne 0 would fail every successful run.
        if ($LASTEXITCODE -ge 8) {
            Write-Error "robocopy failed for $($p.Out), exit $LASTEXITCODE -- see $log"
            exit 1
        }
        Write-Output "    exit $LASTEXITCODE (<8 = ok)"
    }
}
if ($DryRun) {
    Write-Output ""
    Write-Output "--DryRun: nothing written except the log. Re-run without -DryRun to copy."
    exit 0
}

# ---------------------------------------------------------------- verify
Write-Output ""
Write-Output "=== verifying ==="
$bad = 0
$report = New-Object System.Collections.ArrayList
foreach ($p in $plan) {
    $dst = Join-Path $destFull $p.Out
    if (-not (Test-Path -LiteralPath $dst)) {
        Write-Warning "missing in backup: $($p.Out)"; $bad++; continue
    }
    $dstM = Get-Manifest $dst
    $srcIdx = @{}; foreach ($f in $p.Manifest) { $srcIdx[$f.rel] = $f.len }
    $dstIdx = @{}; foreach ($f in $dstM)       { $dstIdx[$f.rel] = $f.len }

    $missing = @($srcIdx.Keys | Where-Object { -not $dstIdx.ContainsKey($_) })
    $extra   = @($dstIdx.Keys | Where-Object { -not $srcIdx.ContainsKey($_) })
    $sizeBad = @($srcIdx.Keys | Where-Object { $dstIdx.ContainsKey($_) -and $dstIdx[$_] -ne $srcIdx[$_] })

    $hashBad = @()
    if ($Hash) {
        foreach ($rel in $srcIdx.Keys) {
            if (-not $dstIdx.ContainsKey($rel)) { continue }
            $a = (Get-FileHash -LiteralPath (Join-Path $p.Path $rel) -Algorithm SHA256).Hash
            $b = (Get-FileHash -LiteralPath (Join-Path $dst $rel) -Algorithm SHA256).Hash
            if ($a -ne $b) { $hashBad += $rel }
        }
    }

    $ok = ($missing.Count -eq 0 -and $sizeBad.Count -eq 0 -and $hashBad.Count -eq 0)
    if (-not $ok) { $bad++ }
    [void]$report.Add([pscustomobject]@{
        root = $p.Out; src = $p.Manifest.Count; dst = $dstM.Count
        missing = $missing.Count; extra = $extra.Count
        size_bad = $sizeBad.Count; hash_bad = $hashBad.Count
        status = $(if ($ok) { 'OK' } else { 'FAIL' }) })
    if ($missing.Count) { Write-Output "  $($p.Out) missing e.g. $($missing[0])" }
    if ($sizeBad.Count) { Write-Output "  $($p.Out) size mismatch e.g. $($sizeBad[0])" }
    if ($hashBad.Count) { Write-Output "  $($p.Out) HASH mismatch e.g. $($hashBad[0])" }
}

$report | Format-Table -AutoSize | Out-String | Write-Output
$summary = [pscustomobject]@{
    stamp = $stamp
    destination = $destFull
    source_repo = $RepoRoot
    total_files = (($plan | Measure-Object -Property Files -Sum).Sum)
    total_bytes = $totalBytes
    hashed = [bool]$Hash
    excluded_dirs = ($ExcludeDirs -join ',')
    roots = $report
    verdict = $(if ($bad -eq 0) { 'VERIFIED' } else { 'FAILED' })
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $logDir "backup_$stamp.json") -Encoding utf8

if ($bad -eq 0) {
    Write-Output ("VERIFIED: {0:N0} files / {1:N2} GB match by path and size{2}." -f `
        $summary.total_files, ($totalBytes / 1GB),
        $(if ($Hash) { ' and SHA-256' } else { ' -- add -Hash for content verification' }))
    Write-Output "Record it in DECISIONS and close isolation criterion 5."
    exit 0
}
Write-Error "FAILED: $bad root(s) did not verify. Do NOT treat this as a backup."
exit 1
