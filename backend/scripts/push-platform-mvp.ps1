# push-platform-mvp.ps1
# Creates branch feat/platform-mvp-e2e, splits the current uncommitted work
# into ~9 logical commits, pushes the branch, merges back into main, pushes main.
#
# Usage: from repo root:  .\scripts\push-platform-mvp.ps1

$ErrorActionPreference = "Stop"
$branch = "feat/platform-mvp-e2e"

function Commit($msg, $paths) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
    $existing = @()
    foreach ($p in $paths) {
        if (Test-Path $p) { $existing += $p }
    }
    if ($existing.Count -eq 0) {
        Write-Host "  (no matching paths, skipping)" -ForegroundColor DarkGray
        return
    }
    git add -- $existing
    $staged = git diff --cached --name-only
    if (-not $staged) {
        Write-Host "  (nothing staged, skipping)" -ForegroundColor DarkGray
        return
    }
    git commit -m $msg | Out-Null
    Write-Host "  committed: $($staged.Count) files" -ForegroundColor Green
}

# --- 0. Sanity: clean working state baseline (main pulled, branch created) ---
$current = (git rev-parse --abbrev-ref HEAD).Trim()
if ($current -ne "main") {
    Write-Host "Switching to main from '$current'..." -ForegroundColor Yellow
    git checkout main
}
Write-Host "Pulling latest main..." -ForegroundColor Yellow
git pull --ff-only origin main

Write-Host "Creating branch $branch ..." -ForegroundColor Yellow
git checkout -b $branch

# --- 1. Build/infra plumbing: root-context Docker + multi-module + dockerignore ---
Commit "chore(build): root-context Docker builds + multi-module Maven layout" @(
    ".dockerignore",
    "docker-compose.yml",
    "api-gateway/Dockerfile",
    "identity-service/Dockerfile",
    "conversation-service/Dockerfile",
    "inference-orchestrator/Dockerfile",
    "asset-service/Dockerfile",
    "billing-service/Dockerfile",
    "telemetry-consumer/Dockerfile",
    "notification-service/Dockerfile"
)

# --- 2. Parent POM + Spring Boot 3.3 compatible versions ---
Commit "chore(deps): pin Spring Cloud 2023.0.4, springdoc 2.6.0, Lombok 1.18.34 for Spring Boot 3.3" @(
    "pom.xml"
)

# --- 3. llm-server: CPU-only torch + dummy fallback + Dockerfile slimming ---
Commit "feat(llm-server): CPU-only torch wheel + DummyGenerator fallback" @(
    "llm-server"
)

# --- 4. Faz 2 Python services ---
Commit "feat(guardrail): PII + prompt-injection + toxicity check with fail-safe" @(
    "guardrail-service/Dockerfile",
    "guardrail-service/app/main.py",
    "guardrail-service/requirements.txt"
)
Commit "feat(router): keyword + embedding-based prompt classifier" @(
    "router-service/Dockerfile",
    "router-service/app/main.py",
    "router-service/requirements.txt"
)

# --- 5. billing-service: full JPA model + endpoints + Kafka consumers ---
Commit "feat(billing): JPA Plan/Subscription/UsageRecord + /me/quota + Kafka consumers" @(
    "billing-service/Dockerfile",
    "billing-service/pom.xml",
    "billing-service/src",
    "billing-service/src/main/resources"
)

# --- 6. notification + telemetry consumer ---
Commit "feat(notification): WebSocket session registry + JavaMail welcome email" @(
    "notification-service/Dockerfile",
    "notification-service/pom.xml",
    "notification-service/src"
)
Commit "feat(telemetry): Elasticsearch index + MinIO NDJSON cold-storage fan-out" @(
    "telemetry-consumer/Dockerfile",
    "telemetry-consumer/pom.xml",
    "telemetry-consumer/src"
)

# --- 7. identity-service hardening ---
Commit "fix(identity): SecurityConfig permitAll + Keycloak client timeouts + logging" @(
    "identity-service/Dockerfile",
    "identity-service/src/main/java/com/goktug/identity/keycloak/KeycloakClient.java",
    "identity-service/src/main/java/com/goktug/identity/service/IdentityService.java",
    "identity-service/src/main/java/com/goktug/identity/config"
)

# --- 8. inference-orchestrator: rename beans, add Flyway plugin, CSRF disable ---
Commit "fix(orchestrator): rename WebClient beans, add flyway-postgresql, reactive SecurityConfig" @(
    "inference-orchestrator/Dockerfile",
    "inference-orchestrator/pom.xml",
    "inference-orchestrator/src/main/java/com/goktug/inference/client",
    "inference-orchestrator/src/main/java/com/goktug/inference/config",
    "inference-orchestrator/src/main/resources/application.yml"
)

# --- 9. gateway + smoke test ---
Commit "chore(smoke): direct-port bypass smoke test + gateway compatibility-verifier off" @(
    "api-gateway/src/main/resources/application.yml",
    "scripts/smoke-test.ps1"
)

# --- catch anything we missed ---
git add -A
$leftover = git diff --cached --name-only
if ($leftover) {
    Write-Host ""
    Write-Host "=== Catching leftover changes ===" -ForegroundColor Cyan
    git commit -m "chore: miscellaneous platform MVP follow-ups" | Out-Null
    Write-Host "  committed: $($leftover.Count) leftover files" -ForegroundColor Green
}

# --- push branch ---
Write-Host ""
Write-Host "Pushing branch $branch ..." -ForegroundColor Yellow
git push -u origin $branch

# --- merge into main ---
Write-Host ""
Write-Host "Merging $branch into main ..." -ForegroundColor Yellow
git checkout main
git pull --ff-only origin main
git merge --no-ff $branch -m "Merge branch '$branch' into main"
git push origin main

Write-Host ""
Write-Host "DONE." -ForegroundColor Green
Write-Host "  - Branch pushed: $branch"
Write-Host "  - main updated and pushed"
Write-Host ""
git log --oneline -12
