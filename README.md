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
5. 한글 자막용 폰트: Windows 기본 맑은 고딕을 프로젝트로 복사 (라이선스상 저장소엔 커밋 안 함)
   ```powershell
   copy C:\Windows\Fonts\malgunbd.ttf assets\fonts\malgunbd.ttf
   ```

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
