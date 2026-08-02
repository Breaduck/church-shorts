@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo ================================================
echo   church-shorts editor   http://localhost:5000
echo ================================================
echo.

REM 이미 편집기가 떠 있으면(5000 응답) 브라우저만 연다.
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:5000/' -TimeoutSec 2 -UseBasicParsing; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
  echo [i] editor already running. opening browser...
  start "" http://localhost:5000/
  goto :eof
)

echo [i] starting editor...
start "church-shorts editor" venv\Scripts\python.exe -m src.web_app

REM 서버가 뜰 때까지 최대 20초 대기 후 브라우저 열기
for /l %%i in (1,1,20) do (
  powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://localhost:5000/' -TimeoutSec 1 -UseBasicParsing > $null; exit 0 } catch { exit 1 }"
  if not errorlevel 1 (
    start "" http://localhost:5000/
    echo [ok] editor ready.
    goto :eof
  )
  timeout /t 1 >nul
)
echo [!] editor did not respond in time. check the server window.
