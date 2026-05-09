# smoke-test.ps1 — PowerShell version of the E2E smoke test.
#
# Usage:
#   .\scripts\smoke-test.ps1
#   $env:GATEWAY_URL = "http://localhost:8080"; .\scripts\smoke-test.ps1

$ErrorActionPreference = "Stop"

$gw = if ($env:GATEWAY_URL) { $env:GATEWAY_URL } else { "http://localhost:8080" }
$email = if ($env:EMAIL) { $env:EMAIL } else { "test+$([int](Get-Date -UFormat %s))@goktug.dev" }
$password = if ($env:PASSWORD) { $env:PASSWORD } else { "TestPass1234" }

function Hr { Write-Host ("─" * 60) }

Hr; Write-Host "[0] Health checks" -ForegroundColor Yellow; Hr
Invoke-RestMethod "$gw/actuator/health" | Format-List
Invoke-RestMethod "http://localhost:9001/v1/health" | Format-List
Write-Host "  ✓ Health OK" -ForegroundColor Green

Hr; Write-Host "[1] Register: $email" -ForegroundColor Yellow; Hr
$body = @{ email = $email; password = $password; displayName = "Smoke" } | ConvertTo-Json
$reg = Invoke-RestMethod -Method Post "$gw/api/v1/auth/register" `
    -ContentType 'application/json' -Body $body
$reg | Format-List
$token = $reg.accessToken
Write-Host "  ✓ userId=$($reg.userId)" -ForegroundColor Green

Hr; Write-Host "[2] Create chat" -ForegroundColor Yellow; Hr
$h = @{ Authorization = "Bearer $token" }
$chat = Invoke-RestMethod -Method Post "$gw/api/v1/chats" -Headers $h `
    -ContentType 'application/json' -Body '{"title":"Smoke test"}'
$chat | Format-List
$chatId = $chat.id

Hr; Write-Host "[3] Send user message" -ForegroundColor Yellow; Hr
$idemp = [guid]::NewGuid().ToString()
$h2 = @{ Authorization = "Bearer $token"; "X-Idempotency-Key" = $idemp }
$msg = Invoke-RestMethod -Method Post "$gw/api/v1/chats/$chatId/messages" `
    -Headers $h2 -ContentType 'application/json' `
    -Body '{"content":"merhaba dünya, sen kimsin?"}'
$msg | Format-List
$msgId = $msg.id

# Idempotency replay
$msg2 = Invoke-RestMethod -Method Post "$gw/api/v1/chats/$chatId/messages" `
    -Headers $h2 -ContentType 'application/json' `
    -Body '{"content":"merhaba dünya, sen kimsin?"}'
if ($msg.id -eq $msg2.id) {
    Write-Host "  ✓ Idempotency OK (same id $($msg.id))" -ForegroundColor Green
} else {
    Write-Host "  ✗ Idempotency FAILED" -ForegroundColor Red
    exit 1
}

Hr; Write-Host "[4] Stream inference (first 30 lines)" -ForegroundColor Yellow; Hr
$infBody = @{ chatId = $chatId; userMessageId = $msgId; text = "merhaba dünya, sen kimsin?" } | ConvertTo-Json
# Invoke-WebRequest -OutFile streams to file; we use HttpClient style instead
$req = [System.Net.Http.HttpRequestMessage]::new('POST', "$gw/api/v1/inference/stream")
$req.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer', $token)
$req.Content = [System.Net.Http.StringContent]::new($infBody, [System.Text.Encoding]::UTF8, 'application/json')
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds(60)
$resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
$stream = $resp.Content.ReadAsStreamAsync().Result
$reader = [System.IO.StreamReader]::new($stream)
$count = 0
while (-not $reader.EndOfStream -and $count -lt 30) {
    $line = $reader.ReadLine()
    if ($line) { Write-Host $line }
    $count++
}
$reader.Dispose(); $client.Dispose()
Write-Host "  ✓ SSE stream OK" -ForegroundColor Green

Hr; Write-Host "[5] Verify assistant message persisted" -ForegroundColor Yellow; Hr
Start-Sleep -Seconds 3
$all = Invoke-RestMethod "$gw/api/v1/chats/$chatId/messages" -Headers $h
$all | ForEach-Object { "$($_.sender): $($_.content.Substring(0, [Math]::Min(80, $_.content.Length)))" }

Hr; Write-Host "[6] Search via Elasticsearch" -ForegroundColor Yellow; Hr
Start-Sleep -Seconds 2
$search = Invoke-RestMethod "$gw/api/v1/chats/search?q=Smoke" -Headers $h
$search.items | Format-Table

Write-Host ""
Write-Host "✅ Smoke test passed!" -ForegroundColor Green
Write-Host "Trace: http://localhost:16686 (Jaeger)"
