# GoktugGPT Profesyonel Deploy Scripti - V2 (Full Kontrol)
$ErrorActionPreference = "Stop"

# 1. Klasör Kontrolü
if (-not (Test-Path "pom.xml")) {
    Write-Host "HATA: GoktugGPT projesinin kök dizininde değilsiniz!" -ForegroundColor Red
    return
}

Write-Host "=== [1] Gecmis Temizleniyor (Sifir Hata Yapisi) ===" -ForegroundColor Cyan
# Mevcut hatalı .git klasörünü sil
Remove-Item -Recurse -Force .git -ErrorAction SilentlyContinue

# Alt klasörlerdeki gizli .git klasörlerini temizle (Submodule çakışmalarını önler)
Get-ChildItem -Path . -Filter ".git" -Recurse -Hidden | ForEach-Object { 
    Write-Host "Alt repo temizleniyor: $($_.FullName)" -ForegroundColor Gray
    Remove-Item -Recurse -Force $_.FullName 
}

Write-Host "=== [2] Yeni Depo Hazirlaniyor (Branch: main) ===" -ForegroundColor Cyan
git init
git checkout -b main
git remote add origin git@github.com:Goktugberke/goktugGPT.git

Write-Host "=== [3] Servis Bazli Commitler Atiliyor ===" -ForegroundColor Yellow

# Commit 1: Infrastructure
Write-Host "> Commit 1: Infrastructure..."
git add ".github/" ".gitignore" "docker-compose.yml" "pom.xml" "docs/README.md"
git commit -m "chore: initial infrastructure, docker-compose and parent pom"

# Commit 2: API Gateway & Identity
Write-Host "> Commit 2: Gateway & Identity..."
git add "api-gateway/" "identity-service/"
git commit -m "feat: implement edge routing and identity management"

# Commit 3: Core Logic
Write-Host "> Commit 3: Core Logic (Conversation & Orchestrator)..."
git add "conversation-service/" "inference-orchestrator/"
git commit -m "feat: implement conversation engine and inference saga orchestration"

# Commit 4: Storage & Billing
Write-Host "> Commit 4: Storage & Billing..."
git add "asset-service/" "billing-service/"
git commit -m "feat: implement asset storage and token-based billing system"

# Commit 5: Frontend & AI components
Write-Host "> Commit 5: Frontend & Python Services..."
if (Test-Path "frontend-web") { git add "frontend-web/" }
if (Test-Path "guardrail-service") { git add "guardrail-service/" }
if (Test-Path "llm-server") { git add "llm-server/" }
if (Test-Path "router-service") { git add "router-service/" }
git commit -m "feat: add frontend-web, guardrail and llm-worker services"

# Commit 6: Support & Shared
Write-Host "> Commit 6: Supporting Components..."
git add "infra/" "scripts/" "shared-contracts/" "telemetry-consumer/" "notification-service/" "docs/"
git commit -m "feat: add observability infra, shared contracts and documentation"

# Commit 7: Final Catch-all
Write-Host "> Commit 7: Final Cleanup..."
git add .
git commit -m "chore: final project migration cleanup"

Write-Host "=== [4] GitHub'a Pushlaniyor (Branch: main) ===" -ForegroundColor Cyan
# Uzak depoyu temiz varsayıyoruz
git push -u origin main

Write-Host ""
Write-Host "ISLEM TAMAMLANDI!" -ForegroundColor Green
Write-Host "Repo: https://github.com/Goktugberke/goktugGPT" -ForegroundColor Cyan
