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

$files = Get-ChildItem -Path $SourceDir -ErrorAction Stop -Recurse | Where-Object {
    -not $_.PSIsContainer -and ($Extensions -contains $_.Extension.ToLower())
}

if (-not $files) {
    throw "No files found in $SourceDir with extensions: $($Extensions -join ', ')"
}

if ($Random) {
    $files = $files | Get-Random -Count ([Math]::Min($Count, ($files | Measure-Object).Count))
} else {
    $files = $files | Select-Object -First $Count
}

$made = 0
foreach ($f in $files) {
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
        & $upx --best -o $packedPath $benignPath | Out-Null
        if (Test-Path $packedPath) {
            $made++
            Write-Host "Packed: $packedName"
        }
    } catch {
        # Some binaries can't be packed; skip those
        if (Test-Path $packedPath) { Remove-Item $packedPath -Force -ErrorAction SilentlyContinue }
        Write-Host "Skip (UPX failed): $($f.Name)"
    }
}

Write-Host "\nDone. Created $made packed samples."
Write-Host "Next: train with --benign-dir $OutBenign --malicious-dir $OutPacked"