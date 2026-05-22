# ===========================================================================
# start-webapp.ps1  --  one-command launch of the VERSION2 (dev) face-swap app
#
# Run from anywhere (others can run it too):
#   powershell -ExecutionPolicy Bypass -File C:\AI_Team\Nehanth\face-swap-streamer-version2\start-webapp.ps1
#
# What it does:
#   1. self-elevates to Administrator (needed for portproxy + firewall)
#   2. starts the version2 dev server in WSL on the GPU (port 8090)
#   3. forwards Windows :8090 -> WSL and opens the firewall so OTHER devices
#      on your LAN / Tailscale can reach it
#   4. prints the links (local / LAN / Tailscale)
#
# version2 runs on port 8090 and is SEPARATE from production v1 (port 8080).
# ===========================================================================
param([int]$Port = 8090)
$ErrorActionPreference = 'Stop'

# --- 1. self-elevate -------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "elevating to Administrator (needed to open the port for others)..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList `
        "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Port $Port"
    return
}

$wslScript = "/mnt/c/AI_Team/Nehanth/face-swap-streamer-version2/run-dev.sh"

# --- 2. start the dev server inside WSL (GPU path) ------------------------
Write-Host "starting version2 dev server in WSL on the GPU (port $Port, warmup ~30s)..." -ForegroundColor Yellow
wsl -e bash -lc "FACESWAP_PORT=$Port bash '$wslScript'"

# --- 3. forward the port to WSL + open firewall ---------------------------
$wslIp = (wsl -e bash -lc "ip -4 addr show eth0 | grep -oP 'inet \K[0-9.]+'").Trim()
if (-not $wslIp) { $wslIp = ((wsl hostname -I).Trim() -split ' ')[0] }

Write-Host "forwarding Windows :$Port -> WSL ${wslIp}:$Port ..." -ForegroundColor DarkGray
netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0 2>$null | Out-Null
netsh interface portproxy add    v4tov4 listenport=$Port listenaddress=0.0.0.0 `
      connectport=$Port connectaddress=$wslIp | Out-Null

$ruleName = "Faceswap Streamer $Port"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -EA SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $Port | Out-Null
}

# --- 4. print links --------------------------------------------------------
$lan = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.InterfaceAlias -eq 'Wi-Fi' }).IPAddress | Select-Object -First 1
$ts  = try { ((tailscale ip -4) -split "`n" | Select-Object -First 1).Trim() } catch { '' }

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  face-swap-streamer  VERSION2 (dev)  is UP on the GPU"      -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  local:     http://localhost:$Port/"  -ForegroundColor Cyan
if ($lan) { Write-Host "  LAN:       http://${lan}:$Port/" -ForegroundColor Cyan }
if ($ts)  { Write-Host "  Tailscale: http://${ts}:$Port/"  -ForegroundColor Cyan }
Write-Host "  (share the LAN or Tailscale link with others)"           -ForegroundColor DarkGray
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close this window (the server keeps running)"
