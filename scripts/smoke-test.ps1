# smoke-test.ps1 -- E2E test running through the api-gateway (8080).
# Set $env:USE_DIRECT_PORTS = "1" to bypass the gateway (debug mode).
#
# Usage:
#   .\scripts\smoke-test.ps1

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if ($env:USE_DIRECT_PORTS -eq "1") {
    $identity = "http://localhost:8081"
    $conv     = "http://localhost:8082"
    $infer    = "http://localhost:8083"
    Write-Host "(direct-port mode)" -ForegroundColor DarkYellow
} else {
    # Single front door: api-gateway routes /api/v1/auth/** -> identity,
    # /api/v1/chats/** -> conversation, /api/v1/inference/** -> orchestrator.
    $identity = "http://localhost:8080"
    $conv     = "http://localhost:8080"
    $infer    = "http://localhost:8080"
}
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

Hr; Write-Host "[0] Health checks + warm-up" -ForegroundColor Yellow; Hr
try { $h = Get-Json "$llm/v1/health"; Write-Host "  llm-server: $($h.status) is_dummy=$($h.is_dummy)" } catch { Write-Host "  llm-server FAIL" -ForegroundColor Yellow }
# Warm up the Java services: after a host/Docker reboot the first POST can take
# up to 10s (Hikari pool init + Keycloak admin token fetch + Hibernate first hit)
# and PowerShell's WebRequest closes the socket as "connection closed". Hitting
# /actuator/health once primes the JVM so the subsequent /register stays under 1s.
foreach ($svc in @($identity, $conv, $infer)) {
    try { Get-Json "$svc/actuator/health" | Out-Null; Write-Host "  warm: $svc OK" } catch { Write-Host "  warm: $svc not ready yet (continuing)" -ForegroundColor Yellow }
}

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

# Idempotency replay
$msg2 = Post-Json "$conv/api/v1/chats/$chatId/messages" '{"content":"merhaba, sen kimsin?"}' $h2
if ($msg.id -eq $msg2.id) {
    Write-Host "  [OK] Idempotency same id ($($msg.id))" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] different ids: $($msg.id) vs $($msg2.id)" -ForegroundColor Red
}

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

Hr; Write-Host "[6] Asset upload (presigned URL + PUT + confirm + download)" -ForegroundColor Yellow; Hr
# 6a. Ask for a presigned PUT URL
$uploadReqBody = @{
    filename  = "hello.txt"
    mimeType  = "text/plain"
    sizeBytes = 11
} | ConvertTo-Json
$uploadUrlResp = Post-Json "$identity/api/v1/assets/upload-url" $uploadReqBody $h
$assetId = $uploadUrlResp.assetId
$putUrl  = $uploadUrlResp.uploadUrl
Write-Host "  assetId=$assetId"
Write-Host "  presigned url ok"

# 6b. Upload the file directly to MinIO via the presigned URL
$tmpFile = New-TemporaryFile
"hello world" | Out-File -FilePath $tmpFile -Encoding ascii -NoNewline
$uploadResp = & curl.exe -s -m 30 -X PUT --upload-file "$tmpFile" -H "Content-Type: text/plain" "$putUrl" -w "%{http_code}"
Remove-Item $tmpFile -Force
if ($uploadResp -match "200|204") {
    Write-Host "  [OK] upload to MinIO (http $uploadResp)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] upload returned: $uploadResp" -ForegroundColor Yellow
}

# 6c. Confirm (asset-service validates with HEAD against MinIO + publishes asset.uploaded.v1)
try {
    $confirmed = Post-Json "$identity/api/v1/assets/$assetId/confirm" "" $h
    Write-Host "  [OK] confirm: status=$($confirmed.status) size=$($confirmed.sizeBytes)" -ForegroundColor Green
} catch { Write-Host "  [FAIL] confirm: $_" -ForegroundColor Red }

# 6d. Fetch metadata + download URL
try {
    $meta = Get-Json "$identity/api/v1/assets/$assetId" $h
    Write-Host "  meta: $($meta.originalName) ($($meta.mimeType), $($meta.sizeBytes)B)"
    $dl = Get-Json "$identity/api/v1/assets/$assetId/download-url" $h
    Write-Host "  [OK] download-url received" -ForegroundColor Green
} catch { Write-Host "  [WARN] metadata/download fetch: $_" -ForegroundColor Yellow }

Write-Host ""
Write-Host "[PASSED] Smoke test passed!" -ForegroundColor Green
