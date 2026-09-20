"""editor.bat이 실행될 때마다 호출돼, GitHub master의 최신 코드로 조용히 갱신한다.

배경(사용자 요청 2026-09-21): "여기서 고친 게 상대방한테도 자동으로 가야지" — Cloudflare
Pages는 정적 페이지(index.html) 한 장만 배포하고, 실제 분석/편집 로직은 각자 PC에 내려받은
zip 사본으로 로컬에서 돈다(각자 자기 Claude 구독으로 AI를 쓰기 위해 일부러 이렇게 설계함,
[[project_deploy_topology]]). 그래서 지금까지는 코드를 고칠 때마다 모두가 zip을 다시 받아
폴더를 교체해야 했다. 이 스크립트가 그 간극을 없앤다 — editor.bat이 뜰 때마다 최신 커밋을
확인하고 다르면 코드 파일만 조용히 갱신한다(영상·설정·비밀키·가상환경은 절대 안 건드림).
AI 분석 자체는 여전히 각자 로컬에 설치된 Claude CLI로 돈다 — 이건 '코드'만 새로 받는 것이다.

중앙 서버로 전환하는 방안(Cloudflare가 실제 파이썬 앱을 직접 호스팅 + 각 PC의 로컬 Claude를
원격으로 호출)도 사용자에게 제시했으나, 상시 서버 비용과 로컬↔서버 중계 프로그램을 새로 만들어야
하는 큰 작업이라 사용자가 이 자동 업데이트 방식을 선택함(2026-09-21).

안전 규칙:
- output/, secrets/, venv/, .git/ 은 절대 덮어쓰지 않는다(사용자 영상·비밀키·가상환경).
- editor.bat/install.bat 자기 자신은 갱신 대상에서 뺀다 — cmd.exe가 지금 읽고 있는 배치
  파일을 실행 중에 덮어쓰면(자기 자신 수정) 이후 줄이 토큰 단위로 깨질 수 있다(실측 경험:
  [[project_bat_filename_ascii_2026-09-20]]). 이 두 파일이 바뀌면 다음 zip 재다운로드에서만
  반영되고, 그건 드문 일이다(런처는 거의 안 바뀜, 실제로 자주 바뀌는 건 src/ 쪽).
- 실패해도(오프라인·방화벽 등) 조용히 넘어간다 — 에디터 실행 자체를 막지 않는다.
- 최초 실행(마커 파일이 없음)은 다운로드하지 않고 현재 커밋을 그대로 최신으로 기록한다 —
  방금 받은 zip이 이미 최신이므로 즉시 재다운로드하는 낭비를 피한다.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "Breaduck/church-shorts"
BRANCH = "master"
ROOT = Path(__file__).resolve().parent.parent  # scripts/ 의 부모 = 프로젝트 루트
MARKER = ROOT / ".installed_commit"
SKIP_DIRS = {"output", "secrets", "venv", ".git"}
SKIP_FILES = {"editor.bat", "install.bat"}  # 자기 자신 실행 중 덮어쓰기 방지
TIMEOUT_SEC = 6


def _latest_sha() -> str | None:
    url = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "church-shorts-auto-update"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as r:
            data = json.loads(r.read().decode("utf-8"))
        sha = data.get("sha")
        return sha if isinstance(sha, str) and sha else None
    except Exception:
        return None


def _current_sha() -> str:
    try:
        return MARKER.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _apply_update(sha: str) -> bool:
    zip_url = f"https://github.com/{REPO}/archive/{sha}.zip"
    try:
        req = urllib.request.Request(zip_url, headers={"User-Agent": "church-shorts-auto-update"})
        with urllib.request.urlopen(req, timeout=90) as r:
            blob = r.read()
    except Exception as e:  # noqa: BLE001 - 오프라인/방화벽 등, 이번엔 건너뜀
        print(f"[update] 다운로드 실패({e}) - 이번엔 건너뜀")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(tmp_path)
        except Exception as e:  # noqa: BLE001
            print(f"[update] 압축 풀기 실패({e}) - 이번엔 건너뜀")
            return False

        roots = [p for p in tmp_path.iterdir() if p.is_dir()]
        if not roots:
            print("[update] 내려받은 zip이 비어있음 - 건너뜀")
            return False
        src_root = roots[0]

        copied = 0
        for item in src_root.rglob("*"):
            if item.is_dir():
                continue
            rel = item.relative_to(src_root)
            if rel.parts and rel.parts[0] in SKIP_DIRS:
                continue
            if len(rel.parts) == 1 and rel.parts[0] in SKIP_FILES:
                continue
            dest = ROOT / rel
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
                copied += 1
            except OSError as e:
                print(f"[update] {rel} 갱신 실패({e}) - 건너뜀")
        try:
            MARKER.write_text(sha, encoding="utf-8")
        except OSError:
            pass
        print(f"[update] 최신 코드로 갱신됨 ({copied}개 파일)")
        return True


def main() -> int:
    latest = _latest_sha()
    if not latest:
        return 0  # 오프라인 등 - 조용히 넘어감(에디터 실행은 계속됨)

    if not MARKER.exists():
        # 최초 실행: 방금 받은 zip이 이미 최신 코드이므로, 재다운로드 없이 현재를 최신으로
        # 기록만 해둔다. 다음 실행부터 진짜로 최신과 비교한다.
        try:
            MARKER.write_text(latest, encoding="utf-8")
        except OSError:
            pass
        return 0

    current = _current_sha()
    if latest == current:
        return 0

    print(f"[update] 새 버전 발견 - 코드 갱신 중... ({(current[:7] or '?')} -> {latest[:7]})")
    return 1 if _apply_update(latest) else 0


if __name__ == "__main__":
    sys.exit(main())
