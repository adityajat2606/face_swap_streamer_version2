$ErrorActionPreference = "Continue"
$root = "C:\AI_Team\Nehanth\face-swap-streamer"
$conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
$wheel = "$env:TEMP\insightface-0.7.3-cp311-cp311-win_amd64.whl"

function Step($label, $sb) {
    Write-Host ""
    Write-Host "==> $label" -ForegroundColor Cyan
    & $sb
    Write-Host "==> exit: $LASTEXITCODE" -ForegroundColor Yellow
}

Step "pip install insightface wheel (dlc env)" {
    & $conda run -n dlc --no-capture-output pip install $wheel
}

Step "Re-run pip install -r requirements.txt (DLC deps)" {
    Push-Location "$root\deep-live-cam"
    & $conda run -n dlc --no-capture-output pip install -r requirements.txt
    Pop-Location
}

Step "Re-run pip install -r requirements-webapp.txt" {
    & $conda run -n dlc --no-capture-output pip install -r "$root\requirements-webapp.txt"
}

Step "Verify imports" {
    & $conda run -n dlc --no-capture-output python -c "import insightface, onnxruntime, cv2, flask, numpy; print('insightface', insightface.__version__); print('onnxruntime', onnxruntime.__version__); print('opencv', cv2.__version__); print('flask', flask.__version__); print('numpy', numpy.__version__); print('CUDAExecutionProvider available:', 'CUDAExecutionProvider' in onnxruntime.get_available_providers())"
}

Write-Host ""
Write-Host "==> ALL DONE" -ForegroundColor Green
