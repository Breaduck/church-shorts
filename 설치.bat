@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ================================================
echo   church-shorts   first-time setup
echo ================================================
echo.

REM ---------------- 1) python ----------------
where python >nul 2>&1
if errorlevel 1 (
  echo [!] python not found.
  echo     install first:  winget install Python.Python.3.12
  echo     then OPEN A NEW TERMINAL and run this file again.
  goto :fail
)

if exist venv\Scripts\python.exe (
  echo [1/6] venv ok.
) else (
  echo [1/6] creating venv...
  python -m venv venv
  if errorlevel 1 goto :fail
)

REM ---------------- 2) python packages ----------------
echo [2/6] installing python packages... this takes a few minutes.
venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail

REM ---------------- 3) ffmpeg ----------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo [3/6] ffmpeg not found - installing via winget...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  set NEEDNEWSHELL=1
) else (
  echo [3/6] ffmpeg ok.
)

REM ---------------- 4) deno (yt-dlp javascript runtime) ----------------
set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%
where deno >nul 2>&1
if errorlevel 1 (
  echo [4/6] deno not found - installing via winget...
  winget install --id DenoLand.Deno -e --accept-source-agreements --accept-package-agreements
  set NEEDNEWSHELL=1
) else (
  echo [4/6] deno ok.
)

REM ---------------- 5) claude code cli ----------------
where claude >nul 2>&1
if errorlevel 1 (
  echo [5/6] Claude Code CLI not found.
  echo     this project picks highlights with YOUR OWN claude subscription,
  echo     so it must be installed and logged in on this pc:
  echo         npm install -g @anthropic-ai/claude-code
  echo         claude          ^<- run once and log in
  set NEEDCLAUDE=1
) else (
  echo [5/6] claude CLI ok.
)

REM ---------------- 6) optional extra font ----------------
REM 자막 기본 글꼴은 저장소에 들어있는 Pretendard라 이 단계 없이도 잘 돈다.
REM 편집기 글꼴 목록에 선택지를 하나 늘려주는 보너스일 뿐이다.
if exist assets\fonts\malgunbd.ttf (
  echo [6/6] fonts ok. ^(Pretendard + malgun^)
) else (
  if exist "%WINDIR%\Fonts\malgunbd.ttf" (
    copy /y "%WINDIR%\Fonts\malgunbd.ttf" assets\fonts\malgunbd.ttf >nul
    echo [6/6] fonts ok. ^(Pretendard + malgun added as a bonus^)
  ) else (
    echo [6/6] fonts ok. ^(Pretendard^)
  )
)

echo.
echo ------------------------------------------------
if defined NEEDCLAUDE (
  echo [!] install + log in to Claude Code CLI before using the editor.
)
if defined NEEDNEWSHELL (
  echo [!] a tool was just installed. CLOSE this window, open a new one,
  echo     and run this file once more so PATH is refreshed.
  goto :end
)
echo [ok] setup done. run  편집기.bat  to open the editor.
echo ------------------------------------------------
goto :end

:fail
echo.
echo [x] setup failed - see the message above.

:end
echo.
pause
