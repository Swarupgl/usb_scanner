param(
    [string]$OutDir = "outputs\\demo\\av_demo",
    [string]$BenignExe = "$env:WINDIR\\System32\\notepad.exe",
    [switch]$IncludeEicar
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section([string]$Title) {
    Write-Host "" 
    Write-Host "==== $Title ===="
}

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Try-GetDefenderStatus() {
    if (Get-Command Get-MpComputerStatus -ErrorAction SilentlyContinue) {
        try {
            $s = Get-MpComputerStatus
            return [PSCustomObject]@{
                RealTimeProtectionEnabled = $s.RealTimeProtectionEnabled
                AntivirusEnabled          = $s.AntivirusEnabled
                AMServiceEnabled          = $s.AMServiceEnabled
                AntispywareEnabled        = $s.AntispywareEnabled
                NISEnabled                = $s.NISEnabled
                EngineVersion             = $s.AMEngineVersion
                SignatureVersion          = $s.AntivirusSignatureVersion
            }
        } catch {
            return $null
        }
    }
    return $null
}

function New-EicarBytes() {
    # EICAR standard test string (safe). AVs may quarantine it.
    $s = 'X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*'
    return [System.Text.Encoding]::ASCII.GetBytes($s)
}

function Write-Bytes([string]$Path, [byte[]]$Bytes) {
    [System.IO.File]::WriteAllBytes((Resolve-Path -LiteralPath (Split-Path -Parent $Path)).Path + "\\" + (Split-Path -Leaf $Path), $Bytes)
}

Write-Section "Output folder"
Ensure-Dir $OutDir
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
Write-Host "OutDir: $OutDir"

Write-Section "Windows Defender status (best effort)"
$def = Try-GetDefenderStatus
if ($null -ne $def) {
    $def | Format-List | Out-String | Write-Host
} else {
    Write-Host "Defender status cmdlets not available (ok)."
}

Write-Section "1) Create a safe 'extension spoof' file (benign PE renamed to .jpg)"
if (-not (Test-Path -LiteralPath $BenignExe)) {
    throw "BenignExe not found: $BenignExe"
}

$benignCopy = Join-Path $OutDir "benign_notepad.exe"
$disguisedJpg = Join-Path $OutDir "vacation_photo.jpg"
Copy-Item -LiteralPath $BenignExe -Destination $benignCopy -Force
Copy-Item -LiteralPath $BenignExe -Destination $disguisedJpg -Force

Write-Host "Created: $benignCopy"
Write-Host "Created: $disguisedJpg (this is NOT a real JPEG; it's a PE with .jpg extension)"

Write-Host "SHA256(original copy) / SHA256(disguised):"
$h1 = (Get-FileHash -Algorithm SHA256 -LiteralPath $benignCopy).Hash
$h2 = (Get-FileHash -Algorithm SHA256 -LiteralPath $disguisedJpg).Hash
Write-Host "  benign_notepad.exe : $h1"
Write-Host "  vacation_photo.jpg : $h2"
if ($h1 -eq $h2) {
    Write-Host "OK: Bytes are identical; only the extension changed."
} else {
    Write-Host "Warning: Hashes differ (unexpected)."
}

if ($IncludeEicar) {
    Write-Section "2) (Optional) Create EICAR test files (AV may quarantine immediately)"
    $eicar = New-EicarBytes
    $eicarCom = Join-Path $OutDir "eicar.com"
    $eicarExe = Join-Path $OutDir "eicar.exe"
    $eicarJpg = Join-Path $OutDir "eicar.jpg"
    $eicarPhotoJpg = Join-Path $OutDir "eicar_photo.jpg"
    $eicarTxt = Join-Path $OutDir "eicar.txt"

    try {
        [System.IO.File]::WriteAllBytes($eicarCom, $eicar)
        [System.IO.File]::WriteAllBytes($eicarExe, $eicar)
        [System.IO.File]::WriteAllBytes($eicarJpg, $eicar)
        [System.IO.File]::WriteAllBytes($eicarTxt, $eicar)
        Write-Host "Created: $eicarCom"
        Write-Host "Created: $eicarExe"
        Write-Host "Created: $eicarJpg"
        Write-Host "Created: $eicarTxt"

        # Disguise the EICAR bytes under a photo-looking name (still not a real JPEG).
        Copy-Item -LiteralPath $eicarExe -Destination $eicarPhotoJpg -Force
        Write-Host "Created: $eicarPhotoJpg (this is NOT a real JPEG; it's the EICAR bytes with .jpg extension)"

        $zipPath = Join-Path $OutDir "eicar.zip"
        if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
        Compress-Archive -Path $eicarCom -DestinationPath $zipPath
        Write-Host "Created: $zipPath"
    } catch {
        Write-Host "EICAR files may have been quarantined by AV (this is expected)."
        Write-Host $_
    }
}

Write-Section "Next steps"
Write-Host "- Copy ONLY the extension-spoof file (vacation_photo.jpg) to your USB for the demo."
Write-Host "- Do NOT double-click/run it during your demo; it's a real Windows executable." 
Write-Host "- Run your sanitizer on the folder and show it quarantines due to magic/extension mismatch."
