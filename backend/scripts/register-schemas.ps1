# Registers all event JSON Schemas (schemas/events/*.json) to Apicurio Registry
# with a BACKWARD compatibility rule per artifact + a global default rule.
#
# Usage:  pwsh ./scripts/register-schemas.ps1
# Env:    REGISTRY_URL (default http://localhost:8086)
#         GROUP        (default goktug-events)

$ErrorActionPreference = "Stop"
$RegistryUrl = if ($env:REGISTRY_URL) { $env:REGISTRY_URL } else { "http://localhost:8086" }
$Group       = if ($env:GROUP)        { $env:GROUP }        else { "goktug-events" }
$ApiBase     = "$RegistryUrl/apis/registry/v2"
$SchemaDir   = Join-Path $PSScriptRoot "..\schemas\events"

Write-Host "Registry : $RegistryUrl"
Write-Host "Group    : $Group"
Write-Host "Schemas  : $SchemaDir`n"

# 1) Wait for the registry to be reachable
Write-Host "Waiting for Apicurio..." -NoNewline
for ($i = 0; $i -lt 30; $i++) {
    try {
        Invoke-RestMethod -Uri "$ApiBase/system/info" -TimeoutSec 3 | Out-Null
        Write-Host " up."
        break
    } catch {
        Start-Sleep -Seconds 2
        Write-Host "." -NoNewline
        if ($i -eq 29) { throw "Apicurio did not become ready at $RegistryUrl" }
    }
}

# 2) Global default COMPATIBILITY = BACKWARD (idempotent; ignore 409 if exists)
try {
    Invoke-RestMethod -Method Post -Uri "$ApiBase/admin/rules" `
        -ContentType "application/json" `
        -Body '{"type":"COMPATIBILITY","config":"BACKWARD"}' | Out-Null
    Write-Host "[OK] global COMPATIBILITY=BACKWARD rule set"
} catch {
    if ($_.Exception.Response.StatusCode.value__ -eq 409) {
        Write-Host "[OK] global rule already present"
    } else { throw }
}

# 3) Register / update each schema as a JSON artifact
$files = Get-ChildItem -Path $SchemaDir -Filter *.json
foreach ($f in $files) {
    $artifactId = $f.BaseName            # e.g. "user.registered.v1"
    $content    = Get-Content $f.FullName -Raw

    # Create-or-update: ifExists=UPDATE bumps a new version
    $uri = "$ApiBase/groups/$Group/artifacts?ifExists=UPDATE"
    $headers = @{
        "X-Registry-ArtifactId"   = $artifactId
        "X-Registry-ArtifactType" = "JSON"
    }
    Invoke-RestMethod -Method Post -Uri $uri -Headers $headers `
        -ContentType "application/json" -Body $content | Out-Null
    Write-Host "[OK] registered $artifactId"

    # Per-artifact BACKWARD rule (idempotent)
    try {
        Invoke-RestMethod -Method Post `
            -Uri "$ApiBase/groups/$Group/artifacts/$artifactId/rules" `
            -ContentType "application/json" `
            -Body '{"type":"COMPATIBILITY","config":"BACKWARD"}' | Out-Null
    } catch {
        if ($_.Exception.Response.StatusCode.value__ -ne 409) { throw }
    }
}

Write-Host "`nDone. $($files.Count) schema(s) registered under group '$Group'."
Write-Host "Browse: $RegistryUrl/ui"
