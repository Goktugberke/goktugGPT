# Initializes the runtime config-repo/ that Spring Cloud Config Server reads
# via its Git backend (file:///opt/config-repo, bind-mounted in docker-compose).
#
# Why: config-repo/ is a Git repo (JGit needs real git refs) and is gitignored,
# so it does not exist on a fresh clone. This script recreates it from the
# version-controlled seed at config-server/seed-config/ and runs git init.
#
# Idempotent: safe to re-run. start.bat calls this automatically.

$ErrorActionPreference = "Stop"
$repoRoot   = Split-Path $PSScriptRoot -Parent
$configRepo = Join-Path $repoRoot "config-repo"
$seedDir    = Join-Path $repoRoot "config-server\seed-config"

if (Test-Path (Join-Path $configRepo ".git")) {
    Write-Host "[init-config-repo] config-repo already initialized - syncing seed."
    Copy-Item (Join-Path $seedDir "*") $configRepo -Recurse -Force
    Push-Location $configRepo
    git add -A | Out-Null
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git -c user.email=config@goktug.local -c user.name=ConfigBot commit -m "chore: sync shared config from seed" | Out-Null
        Write-Host "[init-config-repo] committed seed changes."
    } else {
        Write-Host "[init-config-repo] no changes."
    }
    Pop-Location
    return
}

Write-Host "[init-config-repo] creating config-repo from seed..."
New-Item -ItemType Directory -Force -Path $configRepo | Out-Null
Copy-Item (Join-Path $seedDir "*") $configRepo -Recurse -Force

Push-Location $configRepo
git init -b main | Out-Null
git add -A | Out-Null
git -c user.email=config@goktug.local -c user.name=ConfigBot commit -m "init: shared platform config" | Out-Null
Pop-Location
Write-Host "[init-config-repo] done. config-repo is now a git repo for Spring Cloud Config."
