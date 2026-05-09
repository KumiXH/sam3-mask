param(
    [string]$Python = "python",
    [string]$Sam3RepoDir = "..\\sam3"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Upgrade pip and install setuptools<70"
& $Python -m pip install --upgrade pip
& $Python -m pip install "setuptools<70.0.0"

Write-Host "[2/5] Install sam3-mask"
& $Python -m pip install -e .[dev]

if (-not (Test-Path $Sam3RepoDir)) {
    throw "SAM3 repo not found at '$Sam3RepoDir'. Clone https://github.com/facebookresearch/sam3 first."
}

Write-Host "[3/5] Install official SAM3 from $Sam3RepoDir"
Push-Location $Sam3RepoDir
try {
    & $Python -m pip install -e .
}
finally {
    Pop-Location
}

Write-Host "[4/5] Install extra runtime dependencies"
& $Python -m pip install einops pycocotools psutil

Write-Host "[5/5] Verify SAM3 imports"
& $Python -c "import pkg_resources; from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; print('SAM3 import OK')"
