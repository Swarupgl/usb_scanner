param(
    [string]$SourceDir = "C:\\Windows\\System32",
    [int]$Count = 50,
    [string]$OutBenign = "data\\benign",
    [string]$OutPacked = "data\\malicious",
    [string[]]$Extensions = @(".exe"),
    [switch]$Random
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Get-UpxCommand {
    $cmd = Get-Command upx -ErrorAction SilentlyContinue
    if ($null -ne $cmd) { return $cmd.Source }

    $local = Join-Path $PSScriptRoot "upx.exe"
    if (Test-Path $local) { return $local }

    throw "UPX not found. Install UPX and ensure 'upx' is in PATH, or place upx.exe next to this script (tools\\upx.exe)."
}

Ensure-Dir $OutBenign
Ensure-Dir $OutPacked

$upx = Get-UpxCommand
Write-Host "Using UPX: $upx"
Write-Host "Source: $SourceDir"
Write-Host "Benign out: $OutBenign"
Write-Host "Packed out: $OutPacked"
Write-Host "Target pairs: $Count"

$files = Get-ChildItem -Path $SourceDir -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {
    -not $_.PSIsContainer -and ($Extensions -contains $_.Extension.ToLower())
}

if (-not $files) {
    throw "No files found in $SourceDir with extensions: $($Extensions -join ', ')"
}

if ($Random) {
    # Shuffle
    $files = $files | Get-Random -Count (($files | Measure-Object).Count)
}

$made = 0
foreach ($f in $files) {
    if ($made -ge $Count) { break }

    $base = [IO.Path]::GetFileNameWithoutExtension($f.Name)
    $ext = $f.Extension

    $benignName = "$base$ext"
    $benignPath = Join-Path $OutBenign $benignName

    # Avoid collisions
    $i = 1
    while (Test-Path $benignPath) {
        $benignName = "$base`_$i$ext"
        $benignPath = Join-Path $OutBenign $benignName
        $i++
    }

    Copy-Item -LiteralPath $f.FullName -Destination $benignPath -Force

    $packedName = ([IO.Path]::GetFileNameWithoutExtension($benignName)) + "_packed" + $ext
    $packedPath = Join-Path $OutPacked $packedName

    try {
        # Create packed copy from the benign copy so we keep a clean original
        & $upx --best --force -o $packedPath $benignPath 1>$null 2>$null
        if (Test-Path $packedPath) {
            $made++
            Write-Host "Packed: $packedName"
        } else {
            # Ensure we keep balanced pairs only
            if (Test-Path $benignPath) { Remove-Item $benignPath -Force -ErrorAction SilentlyContinue }
        }
    } catch {
        # Some binaries can't be packed; skip those and keep dataset balanced
        if (Test-Path $packedPath) { Remove-Item $packedPath -Force -ErrorAction SilentlyContinue }
        if (Test-Path $benignPath) { Remove-Item $benignPath -Force -ErrorAction SilentlyContinue }
    }
}

Write-Host "\nDone. Created $made packed samples."
Write-Host "Next: train with --benign-dir $OutBenign --malicious-dir $OutPacked"