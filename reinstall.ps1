$ErrorActionPreference = "Continue"
$root = "C:\AI_Team\Nehanth\face-swap-streamer"
$conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"

function Step($label, $sb) {
    Write-Host ""
    Write-Host "==> $label" -ForegroundColor Cyan
    & $sb
    Write-Host "==> exit: $LASTEXITCODE" -ForegroundColor Yellow
}

Step "FaceFusion install.py (onnxruntime cuda)" {
    Push-Location "$root\facefusion"
    & $conda run -n faceswap --no-capture-output python install.py --onnxruntime cuda
    Pop-Location
}

Step "pip install -r requirements-facefusion.txt (faceswap env)" {
    & $conda run -n faceswap --no-capture-output pip install -r "$root\requirements-facefusion.txt"
}

Step "pip install -r requirements.txt (dlc env, Deep-Live-Cam deps)" {
    Push-Location "$root\deep-live-cam"
    & $conda run -n dlc --no-capture-output pip install -r requirements.txt
    Pop-Location
}

Step "pip install -r requirements-webapp.txt (dlc env, webapp deps)" {
    & $conda run -n dlc --no-capture-output pip install -r "$root\requirements-webapp.txt"
}

Write-Host ""
Write-Host "==> ALL DONE" -ForegroundColor Green
