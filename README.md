# 교회 설교 영상 → 자동 쇼츠 생성기

유튜브 설교 영상 링크 하나만 넣으면 하이라이트를 자동으로 찾아
세로형(9:16) 쇼츠 3~5개(20~60초, 한글 자막 번인 포함)를 만들어줍니다.

## 파이프라인

```
링크 → 다운로드(yt-dlp) → 전사(faster-whisper) → 오디오 피크 감지
     → 하이라이트 선정(로컬 Claude Code) → 컷+세로변환+자막(ffmpeg) → output/
```

하이라이트 선정은 유료 API가 아니라 **로컬에 설치된 Claude Code CLI**(`claude -p`)를
서브프로세스로 호출해서 처리합니다. 별도 API 키가 필요 없습니다.

## 설치

1. Python 3.11+ 설치 (venv 사용 권장: `python -m venv venv`)
2. `venv/Scripts/python.exe -m pip install -r requirements.txt`
3. ffmpeg 설치: `winget install Gyan.FFmpeg` 후 **새 터미널을 열어야** PATH가 반영됨
   (`ffmpeg -version`으로 확인)
4. Claude Code CLI 로그인 상태 확인: `claude --version`
5. (선택) 글꼴 추가 — 자막 기본 글꼴 Pretendard는 저장소에 들어있어 그대로 쓰면 됩니다.
   편집기 글꼴 목록을 늘리고 싶을 때만 ttf/otf를 `assets/fonts` 또는 `글씨체/`에 넣으면
   `src/fonts.py`가 자동으로 스캔합니다. 맑은 고딕은 재배포 라이선스가 불명확해 저장소엔
   커밋하지 않으므로, 쓰려면 로컬에서 직접 복사합니다.
   ```powershell
   copy C:\Windows\Fonts\malgunbd.ttf assets\fonts\malgunbd.ttf
   ```

## 다른 사람 PC에서 열기

이 프로젝트는 유료 API 키를 쓰지 않고 **그 PC에 로그인된 Claude Code CLI**를 호출하므로,
상대방이 Claude에 로그인만 되어 있으면 자기 컴퓨터에서 그대로 돌릴 수 있습니다
(내 계정/키를 넘길 필요 없음. AI 사용량도 각자 구독에서 나갑니다).

> **저장소가 비공개(private)면 상대방은 링크를 열어도 "404 Not Found"만 봅니다.**
> 배포하려면 GitHub 저장소를 public으로 바꾸거나(`gh repo edit Breaduck/church-shorts --visibility public --accept-visibility-change-consequences`),
> 상대방 GitHub 계정을 collaborator로 초대해야 합니다. `secrets/`·`.env`·`output/`은 .gitignore라 공개돼도 새지 않습니다.

1. 소스 받기 - 둘 중 하나
   - git 없이: <https://github.com/Breaduck/church-shorts/archive/refs/heads/master.zip> 내려받아 압축 해제
   - git으로: `git clone https://github.com/Breaduck/church-shorts.git`
2. 폴더 안의 **`install.bat`** 실행 - venv 생성, 파이썬 패키지, ffmpeg, deno,
   Claude Code CLI(없으면 공식 설치 스크립트로 자동 설치)까지 한 번에 처리합니다.
   (ffmpeg/deno가 새로 설치되면 창을 닫고 한 번 더 실행하라고 안내합니다.)
3. 새 터미널에서 `claude`를 한 번 실행해 **본인 Claude 계정으로 로그인**.
   (자동 설치가 실패했다면 `npm install -g @anthropic-ai/claude-code`로 수동 설치.)
4. **`editor.bat`** 실행 → 브라우저에서 `http://localhost:5000` 이 열립니다.
   (venv가 없으면 editor.bat이 알아서 install.bat을 먼저 돌립니다.)

Cloudflare Pages 소개 페이지에도 같은 zip 다운로드 버튼과 위 3단계가 적혀 있어
그 링크 하나만 전달하면 됩니다.

저장소에 없는 것: `글씨체/`(폰트 바이너리라 제외 - 없으면 글꼴 선택지만 줄고 동작엔 지장 없음),
`secrets/`(유튜브 업로드용 OAuth - 업로드 안 쓰면 불필요), `output/`(생성물).

### 같은 화면을 그냥 보여주기만 하려면

설치 없이 내 PC에서 돌아가는 편집기를 잠깐 공유하려면 터널을 쓰면 됩니다.

```powershell
cloudflared tunnel --url http://localhost:5000
```

출력된 `https://*.trycloudflare.com` 주소를 전달하면 상대는 브라우저만으로 접속합니다.
다만 이 경우 렌더링·AI 분석은 전부 **내 PC와 내 Claude 구독**을 씁니다.
(참고: Cloudflare Pages에 올라간 것은 정적 `index.html` 소개 페이지뿐이고,
실제 편집기 UI는 Flask 서버인 `src/web_app.py`입니다.)

## 사용법

```bash
venv/Scripts/python.exe -m src.main "https://www.youtube.com/watch?v=XXXXXXXX"
```

결과물은 `output/<video-id>/clips/short_1.mp4` ... 형태로 저장됩니다.
같은 영상으로 다시 실행하면 이미 받은 영상/전사/하이라이트 결과는 재사용합니다.

## 설정

`config.yaml`에서 쇼츠 개수, 길이, 자막 스타일, 세로 변환 방식(blur/crop),
하이라이트 카테고리 등을 조절할 수 있습니다.

## 진행 단계

- **Phase A** (현재): 링크 → 쇼츠 생성까지 완전 자동
- **Phase B**: 로컬 검토 웹 UI + YouTube 자동 업로드
- **Phase C**: Instagram Reels / TikTok API 연동 (계정·앱 심사 필요)
