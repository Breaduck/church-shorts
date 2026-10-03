"""파일 저장 공용 유틸: 원자적 쓰기.

write_text는 대상 파일을 먼저 0바이트로 자르고 쓴다. 쓰는 도중 프로세스가 죽으면(서버 창 닫기,
강제종료) 반쯤 쓴 JSON이 남고, 그 뒤로는 읽을 때마다 JSONDecodeError로 매번 실패한다(예: 깨진
feedback.json 하나가 모든 선정을 멈춤). 임시파일에 다 쓴 뒤 os.replace로 바꿔 끼우면 '옛 내용'
아니면 '새 내용' 둘 중 하나만 남는다.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def atomic_write_text(path: Path, text: str, *, fsync: bool = False) -> None:
    """text를 path에 원자적으로 쓴다.

    - 임시파일 이름에 pid·스레드 id를 넣는다: 고정 이름(.tmp)이면 동시에 두 곳(웹 서버 스레드끼리,
      또는 CLI와 웹 서버)이 저장할 때 서로의 임시파일을 덮어쓴다.
    - Windows에서 os.replace는 다른 스레드가 대상 파일을 읽느라 열고 있는 순간 PermissionError
      (WinError 5/32)를 낸다. 읽기는 수 ms면 끝나므로 잠깐 기다렸다 몇 번 재시도한다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            if fsync:
                f.flush()
                os.fsync(f.fileno())  # 전원이 나가도 내용이 디스크에 도달했음을 보장
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
