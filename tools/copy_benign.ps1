param(
  [int]$Count = 100,
  [string]$OutDir = "..\data\benign",
  [string[]]$SourceDirs = @("C:\Windows\System32"),
  [string[]]$Extensions = @(".exe", ".dll"),
  [switch]$Random
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$dest = Resolve-Path (Join-Path $PSScriptRoot $OutDir) -ErrorAction SilentlyContinue
if (-not $dest) {
  $dest = Join-Path $PSScriptRoot $OutDir
}
Ensure-Dir $dest

Write-Host "Copying benign samples..." -ForegroundColor Cyan
Write-Host "  SourceDirs: $($SourceDirs -join ', ')"
Write-Host "  Extensions: $($Extensions -join ', ')"
Write-Host "  OutDir:      $dest"
Write-Host "  Count:       $Count"

# Gather candidates
$candidates = @()
foreach ($dir in $SourceDirs) {
  if (-not (Test-Path $dir)) { continue }

  $files = Get-ChildItem -Path $dir -File -Recurse -ErrorAction SilentlyContinue
  foreach ($f in $files) {
    if ($Extensions -contains $f.Extension.ToLowerInvariant()) {
      $candidates += $f.FullName
    }
  }
}

$candidates = $candidates | Sort-Object -Unique
if ($candidates.Count -eq 0) {
  throw "No candidate binaries found in SourceDirs."
}

if ($Random) {
  $candidates = $candidates | Get-Random -Count ([Math]::Min($Count * 5, $candidates.Count))
}

$copied = 0
$skipped = 0
$errors = 0

foreach ($path in $candidates) {
  if ($copied -ge $Count) { break }

  try {
    $leaf = Split-Path $path -Leaf
    $target = Join-Path $dest $leaf

    # Avoid overwriting; add suffix if duplicate name
    if (Test-Path $target) {
      $base = [System.IO.Path]::GetFileNameWithoutExtension($leaf)
      $ext = [System.IO.Path]::GetExtension($leaf)
      $i = 1
      do {
        $target = Join-Path $dest ("{0}_{1}{2}" -f $base, $i, $ext)
        $i++
      } while (Test-Path $target)
    }

    Copy-Item -LiteralPath $path -Destination $target -Force
    $copied++
  } catch {
    $errors++
  }
}

Write-Host "Done." -ForegroundColor Green
Write-Host "  Copied:  $copied"
Write-Host "  Errors:  $errors"
Write-Host "  Output:  $dest"

Write-Host "" 
Write-Host "Next: train with:" -ForegroundColor Yellow
Write-Host "  python train.py --benign-dir data\benign --malicious-dir data\malicious --epochs 5 --batch-size 2 --out malconv_model.pth"
