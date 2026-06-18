@echo off
REM ============================================================
REM  goktugGPT — One-click platform startup
REM ============================================================
REM  This script does:
REM   1. Checks Docker Desktop is running, starts it if not
REM   2. Waits for the Docker daemon to be ready
REM   3. Runs `docker compose --profile all up -d`
REM   4. Streams the wait for every healthcheck to pass
REM   5. Prints final status and useful URLs
REM
REM  Container healthchecks + depends_on (service_healthy) in
REM  docker-compose.yml handle ordering, so this script does NOT
REM  manually restart anything in a specific order.

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo === goktugGPT platform startup ===
echo.

REM --- 1) Ensure Docker Desktop is running ---
docker version >nul 2>&1
if errorlevel 1 (
    echo [1/5] Docker not responding, launching Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo      Waiting for Docker daemon to come up...
    set /a tries=0
    :wait_docker
    timeout /t 3 /nobreak >nul
    docker version >nul 2>&1
    if errorlevel 1 (
        set /a tries+=1
        if !tries! geq 40 (
            echo      ERROR: Docker did not start within 120 seconds.
            exit /b 1
        )
        goto wait_docker
    )
    echo      Docker is up.
) else (
    echo [1/5] Docker already running.
)

REM --- 1.5) Ensure config-repo exists (Spring Cloud Config git backend) ---
REM On a fresh clone config-repo/ is absent (.gitignore'd JGit repo); recreate
REM it from the version-controlled seed before the config-server starts.
if not exist "config-repo\.git" (
    echo [1.5] Initializing config-repo from seed...
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\init-config-repo.ps1"
)

REM --- 2) Compose up ---
echo [2/5] Starting all containers ^(infra + app services^)...
docker compose --profile all up -d
if errorlevel 1 (
    echo      ERROR: docker compose failed.
    exit /b 1
)

REM --- 3) Wait for all healthchecks ---
echo [3/5] Waiting for healthchecks ^(Elasticsearch, Kafka, Keycloak, Postgres, ...^)...
echo      Java services start automatically once their dependencies are healthy.
echo      First boot can take 60-90 seconds.
set /a waits=0
:wait_healthy
timeout /t 5 /nobreak >nul
REM Use HTTP probes for the four user-facing services. If any is unreachable, keep waiting.
curl -sf -m 3 http://localhost:8081/actuator/health >nul 2>&1 && set IDENTITY_OK=1 || set IDENTITY_OK=0
curl -sf -m 3 http://localhost:8082/actuator/health >nul 2>&1 && set CONV_OK=1 || set CONV_OK=0
curl -sf -m 3 http://localhost:8083/actuator/health >nul 2>&1 && set ORCH_OK=1 || set ORCH_OK=0
curl -sf -m 3 http://localhost:9001/v1/health      >nul 2>&1 && set LLM_OK=1 || set LLM_OK=0

if "!IDENTITY_OK!!CONV_OK!!ORCH_OK!!LLM_OK!"=="1111" goto all_ready

set /a waits+=1
if !waits! geq 36 (
    echo.
    echo      WARNING: Some services not healthy after 3 minutes.
    echo      identity=!IDENTITY_OK!  conversation=!CONV_OK!  orchestrator=!ORCH_OK!  llm-server=!LLM_OK!
    echo      Check:   docker compose ps
    echo      Logs:    docker compose logs --tail=80 ^<service^>
    goto done
)

REM Progress indicator every ~15s
set /a mod=!waits! %% 3
if !mod!==0 (
    echo      ... still waiting ^(identity=!IDENTITY_OK! conversation=!CONV_OK! orchestrator=!ORCH_OK! llm=!LLM_OK!^)
)
goto wait_healthy

:all_ready
echo [4/5] All services healthy.

REM --- 5) Summary ---
:done
echo.
echo [5/5] Status:
docker compose ps --format "table {{.Name}}\t{{.Status}}"
echo.
echo === Ready ===
echo   API Gateway:      http://localhost:8080
echo   Identity:         http://localhost:8081/actuator/health
echo   Conversation:     http://localhost:8082/actuator/health
echo   Orchestrator:     http://localhost:8083/actuator/health
echo   LLM server:       http://localhost:9001/v1/health
echo.
echo   Keycloak admin:   http://localhost:8180   ^(admin / admin^)
echo   Kafka UI:         http://localhost:8090
echo   MinIO console:    http://localhost:9090   ^(minio / minio12345^)
echo   Jaeger:           http://localhost:16686
echo   Grafana:          http://localhost:3000   ^(admin / admin^)
echo   Kibana:           http://localhost:5601
echo.
echo   Run smoke test:   .\scripts\smoke-test.ps1
echo.

endlocal
