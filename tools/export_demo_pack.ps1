param(
    [string]$BenignDir = "data\\benign",
    [string]$PackedDir = "data\\malicious",
    [string]$Checkpoint = "malconv_model.pth",
    [string]$OutDir = "demo_pack",
    [int]$Pairs = 10,
    [switch]$Random
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

if (-not (Test-Path $BenignDir)) { throw "BenignDir not found: $BenignDir" }
if (-not (Test-Path $PackedDir)) { throw "PackedDir not found: $PackedDir" }
if (-not (Test-Path $Checkpoint)) { throw "Checkpoint not found: $Checkpoint" }

$cleanOut = Join-Path $OutDir "clean"
$packedOut = Join-Path $OutDir "packed"

# fresh output
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
Ensure-Dir $cleanOut
Ensure-Dir $packedOut

$cleanFiles = Get-ChildItem -Path $BenignDir -ErrorAction Stop | Where-Object { -not $_.PSIsContainer -and $_.Extension -eq '.exe' }
$packedFiles = Get-ChildItem -Path $PackedDir -ErrorAction Stop | Where-Object { -not $_.PSIsContainer -and $_.Name -like '*_packed.exe' }

if (-not $cleanFiles) { throw "No clean .exe files found in $BenignDir" }
if (-not $packedFiles) { throw "No packed files found in $PackedDir" }

# pick subset
if ($Random) {
    $cleanPick = $cleanFiles | Get-Random -Count ([Math]::Min($Pairs, ($cleanFiles | Measure-Object).Count))
    $packedPick = $packedFiles | Get-Random -Count ([Math]::Min($Pairs, ($packedFiles | Measure-Object).Count))
} else {
    $cleanPick = $cleanFiles | Select-Object -First $Pairs
    $packedPick = $packedFiles | Select-Object -First $Pairs
}

foreach ($f in $cleanPick) { Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $cleanOut $f.Name) -Force }
foreach ($f in $packedPick) { Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $packedOut $f.Name) -Force }

Copy-Item -LiteralPath $Checkpoint -Destination (Join-Path $OutDir (Split-Path $Checkpoint -Leaf)) -Force

$readme = @"
USB Scanner Demo Pack

Contents:
- clean\\  : clean Windows executables (benign)
- packed\\ : UPX-packed executables (still benign, but "packed")
- malconv_model.pth : trained checkpoint for packed-vs-clean demo

Friend laptop steps:
1) git clone https://github.com/Swarupgl/usb_scanner.git
2) cd usb_scanner
3) pip install -r requirements.txt
4) Copy malconv_model.pth from this folder into the repo folder
5) Run scan:
   python usb_monitor.py --scan <this_demo_pack_path> --checkpoint malconv_model.pth --extensions .exe --top 20 --metadata

Expected:
- clean\\ files score low
- packed\\ files score high
"@

$readmePath = Join-Path $OutDir "RUN_DEMO.txt"
$readme | Out-File -FilePath $readmePath -Encoding UTF8

# zip
$zipPath = "$OutDir.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path "$OutDir\\*" -DestinationPath $zipPath

Write-Host "Created: $OutDir"
Write-Host "Created zip: $zipPath"
Write-Host ("Clean files: {0} | Packed files: {1}" -f ($cleanPick.Count), ($packedPick.Count))
