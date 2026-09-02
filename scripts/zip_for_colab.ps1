# zip_for_colab.ps1 — Creates two zips for upload to Google Drive
# Run from the film_factory project root:
#   cd C:\Users\tejap\OneDrive\Desktop\film_factory
#   .\scripts\zip_for_colab.ps1

$root    = "C:\Users\tejap\OneDrive\Desktop\film_factory"
$outDir  = "C:\Users\tejap\OneDrive\Desktop"

Set-Location $root

# ── Code zip ─────────────────────────────────────────────────────────────────
# Includes all Python source but excludes venv, __pycache__, temp files, secrets
$codeZip = "$outDir\film_factory_code.zip"
if (Test-Path $codeZip) { Remove-Item $codeZip -Force }

$codeItems = @(
    "agents",
    "utils",
    "colab",
    "context.py",
    "orchestrator.py"
)

Write-Host "Creating code zip: $codeZip"
Compress-Archive -Path ($codeItems | ForEach-Object { "$root\$_" }) `
                 -DestinationPath $codeZip
$codeSizeMB = [math]::Round((Get-Item $codeZip).Length / 1MB, 1)
Write-Host "  Done: $codeSizeMB MB"

# ── Outputs zip ──────────────────────────────────────────────────────────────
# Includes generated images, audio, context — preserves outputs/ folder structure
# so Colab extracts to /content/film_factory/outputs/* correctly.
$outZip = "$outDir\film_factory_outputs.zip"
if (Test-Path $outZip) { Remove-Item $outZip -Force }

# Zip from project root using relative paths so outputs/ prefix is preserved
Push-Location $root

$subdirs = @("audio","anchors","scene_images","reveal_images","insert_images","alignment")
$toAdd   = @()

# context.json first
if (Test-Path "outputs\context.json") { $toAdd += "outputs\context.json" }

# subdirectories
foreach ($d in $subdirs) {
    if (Test-Path "outputs\$d") { $toAdd += "outputs\$d" }
}

Write-Host "Creating outputs zip: $outZip"
Compress-Archive -Path $toAdd -DestinationPath $outZip
Pop-Location

$outSizeMB = [math]::Round((Get-Item $outZip).Length / 1MB, 1)
Write-Host "  Done: $outSizeMB MB"

Write-Host ""
Write-Host "Upload both files to Google Drive at:"
Write-Host "  My Drive/film_factory/film_factory_code.zip"
Write-Host "  My Drive/film_factory/film_factory_outputs.zip"
Write-Host ""
Write-Host "Then open colab_runner.ipynb in Colab and run all cells."
