@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
REM yt-dlp의 유튜브 추출에 JS 런타임(deno)이 필요하다(없으면 일부 포맷 누락/403 위험).
REM winget 설치본이 Links 별칭 없이 Packages 폴더에만 있어 PATH에 직접 추가한다.
set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%
REM Claude CLI: 네이티브 설치본(.local\bin)과 npm 전역 설치본(%APPDATA%\npm) 위치를 얹는다.
REM 서버(src/highlights.py)가 shutil.which("claude")로 찾으므로 이 프로세스의 PATH에 있어야 한다.
set PATH=%USERPROFILE%\.local\bin;%APPDATA%\npm;%PATH%

echo ================================================
echo   church-shorts editor   http://localhost:5000
echo ================================================
echo.

REM venv가 없으면(다른 PC에서 처음 클론한 경우) 설치부터 안내한다.
if not exist venv\Scripts\python.exe (
  echo [i] first run on this pc - starting setup...
  call "%~dp0설치.bat"
  goto :eof
)

REM Claude CLI가 없으면 AI 분석이 안 되므로 미리 알려준다(설치.bat이 설치해 준다).
where claude >nul 2>&1
if errorlevel 1 (
  echo [!] Claude Code CLI not found. run 설치.bat first, then run  claude  once to log in.
  pause
  goto :eof
)

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
