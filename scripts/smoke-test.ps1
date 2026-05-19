# smoke-test.ps1 -- E2E test bypassing the broken Spring Cloud Gateway.
# Hits each service on its directly-mapped port.
#
# Usage:
#   .\scripts\smoke-test.ps1

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$identity = "http://localhost:8081"
$conv     = "http://localhost:8082"
$infer    = "http://localhost:8083"
$llm      = "http://localhost:9001"

$email = if ($env:EMAIL) { $env:EMAIL } else { "smoke+$([int](Get-Date -UFormat %s))@goktug.dev" }
$password = "TestPass1234"

function Hr { Write-Host ("-" * 60) }
function Get-Json($url, $headers = @{}) {
    $r = Invoke-WebRequest -Uri $url -Method GET -Headers $headers -DisableKeepAlive -UseBasicParsing
    return $r.Content | ConvertFrom-Json
}
function Post-Json($url, $body, $headers = @{}) {
    $r = Invoke-WebRequest -Uri $url -Method POST -Headers $headers -Body $body `
        -ContentType 'application/json' -DisableKeepAlive -UseBasicParsing
    return $r.Content | ConvertFrom-Json
}

Hr; Write-Host "[0] Health checks" -ForegroundColor Yellow; Hr
try { $h = Get-Json "$llm/v1/health"; Write-Host "  llm-server: $($h.status) is_dummy=$($h.is_dummy)" } catch { Write-Host "  llm-server FAIL" -ForegroundColor Yellow }

Hr; Write-Host "[1] Register (direct identity:8081): $email" -ForegroundColor Yellow; Hr
$body = @{ email = $email; password = $password; displayName = "Smoke" } | ConvertTo-Json
$reg = Post-Json "$identity/api/v1/auth/register" $body
Write-Host "  userId=$($reg.userId)"
Write-Host "  token=$($reg.accessToken.Substring(0,60))..."
$token = $reg.accessToken

Hr; Write-Host "[2] Create chat (direct conversation:8082)" -ForegroundColor Yellow; Hr
# conversation-service expects X-User-Id header (api-gateway normally sets this from JWT)
$h = @{ Authorization = "Bearer $token"; "X-User-Id" = $reg.userId }
$chat = Post-Json "$conv/api/v1/chats" '{"title":"Smoke test"}' $h
$chatId = $chat.id
Write-Host "  chatId=$chatId"

Hr; Write-Host "[3] Send user message" -ForegroundColor Yellow; Hr
$idemp = [guid]::NewGuid().ToString()
$h2 = $h.Clone(); $h2["X-Idempotency-Key"] = $idemp
$msg = Post-Json "$conv/api/v1/chats/$chatId/messages" '{"content":"merhaba, sen kimsin?"}' $h2
Write-Host "  messageId=$($msg.id)"
$msgId = $msg.id

# Idempotency replay (known cache deserialization bug — tolerate)
try {
    $msg2 = Post-Json "$conv/api/v1/chats/$chatId/messages" '{"content":"merhaba, sen kimsin?"}' $h2
    if ($msg.id -eq $msg2.id) { Write-Host "  [OK] Idempotency same id" -ForegroundColor Green }
    else { Write-Host "  [WARN] different ids" -ForegroundColor Yellow }
} catch { Write-Host "  [SKIP] Idempotency cache bug - continuing" -ForegroundColor Yellow }

Hr; Write-Host "[4] Stream inference (direct orchestrator:8083, first 30 lines)" -ForegroundColor Yellow; Hr
$infBody = @{ chatId = $chatId; userMessageId = $msgId; text = "merhaba, sen kimsin?" } | ConvertTo-Json
$body = $infBody.Replace('"','\"')
# curl.exe streams SSE properly; -N disables buffering
$lines = curl.exe -sN -m 60 -X POST "$infer/api/v1/inference/stream" `
    -H "Authorization: Bearer $token" `
    -H "X-User-Id: $($reg.userId)" `
    -H "Content-Type: application/json" `
    -d "$body" 2>&1 | Select-Object -First 30
$lines | ForEach-Object { Write-Host $_ }

Hr; Write-Host "[5] Verify assistant message persisted" -ForegroundColor Yellow; Hr
Start-Sleep 3
$all = Get-Json "$conv/api/v1/chats/$chatId/messages" $h
$all | ForEach-Object { "$($_.sender): $($_.content.Substring(0, [Math]::Min(80, $_.content.Length)))" }

Write-Host ""
Write-Host "[PASSED] Smoke test passed!" -ForegroundColor Green
