@echo off
REM Cleanly stop all goktugGPT containers (keeps volumes intact).
cd /d "%~dp0"
echo Stopping all goktugGPT containers...
docker compose --profile all stop
echo.
echo Done. Data volumes preserved.
echo Run start.bat again to bring everything back up.
