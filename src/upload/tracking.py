"""업로드된 쇼츠의 성과를 주 1회, 최대 한달(기본 4회) 자동으로 체크해 feedback.json에 반영한다.

저장 위치: 프로젝트 루트의 uploads.json (channel 전체, feedback.json과 같은 패턴).
실행 방법 두 가지 모두 지원:
  1. web_app.py가 켜져 있는 동안: 백그라운드 스레드가 주기적으로 run_due_checks() 호출.
  2. web_app.py가 꺼져 있어도: `python -m src.upload.tracking` 단독 실행(Windows 작업 스케줄러가 매일 호출).
     -> 어느 쪽으로 실행되든 uploads.json이 진실의 원천이라 중복/누락 없이 이어진다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.feedback import PerformanceRecord, load_feedback, upsert_feedback
from src.upload.youtube import fetch_video_stats

UPLOADS_PATH = Path("uploads.json")

CHECK_INTERVAL_DAYS = 7
MAX_CHECKS = 4  # 7일 x 4 = 최대 한달


@dataclass
class UploadRecord:
    video_id: str          # 원본 설교 영상(=video_dir 이름)
    clip_index: int
    youtube_video_id: str
    title: str = ""
    uploaded_at: str = ""
    next_check_at: str = ""
    checks_done: int = 0
    max_checks: int = MAX_CHECKS
    done: bool = False
    # 매 체크 스냅샷 이력: [{"checked_at": ..., "views": ..., "likes": ..., "comments": ...}, ...]
    history: list = field(default_factory=list)


def load_uploads(path: Path = UPLOADS_PATH) -> list[UploadRecord]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for d in data:
        known = {k: v for k, v in d.items() if k in UploadRecord.__dataclass_fields__}
        records.append(UploadRecord(**known))
    return records


def save_uploads(records: list[UploadRecord], path: Path = UPLOADS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_upload(video_id: str, clip_index: int, path: Path = UPLOADS_PATH) -> Optional[UploadRecord]:
    for r in load_uploads(path):
        if r.video_id == video_id and r.clip_index == clip_index:
            return r
    return None


def record_upload(
    video_id: str,
    clip_index: int,
    youtube_video_id: str,
    title: str = "",
    path: Path = UPLOADS_PATH,
) -> UploadRecord:
    """업로드 직후 호출: 추적 레코드를 새로 만들고 1주일 뒤 첫 체크를 예약한다."""
    now = datetime.now()
    record = UploadRecord(
        video_id=video_id,
        clip_index=clip_index,
        youtube_video_id=youtube_video_id,
        title=title,
        uploaded_at=now.isoformat(timespec="seconds"),
        next_check_at=(now + timedelta(days=CHECK_INTERVAL_DAYS)).isoformat(timespec="seconds"),
        checks_done=0,
        max_checks=MAX_CHECKS,
        done=False,
    )
    records = load_uploads(path)
    records = [
        r for r in records
        if not (r.video_id == video_id and r.clip_index == clip_index)
    ]
    records.append(record)
    save_uploads(records, path)
    return record


def _apply_stats_to_feedback(record: UploadRecord, stats: dict) -> None:
    """자동 수집한 조회수/좋아요/댓글을 feedback.json에 반영한다.
    사람이 이미 입력해둔 rating/notes/saves/shares/retention_pct는 덮어쓰지 않고 보존한다
    (Data API v3는 그 지표들을 애초에 제공하지 않는다)."""
    existing = next(
        (r for r in load_feedback() if r.video_id == record.video_id and r.clip_index == record.clip_index),
        None,
    )
    week_note = f"[자동 {record.checks_done}주차] 조회 {stats['views']:,}"
    if stats.get("likes") is not None:
        week_note += f" · 좋아요 {stats['likes']:,}"
    if stats.get("comments") is not None:
        week_note += f" · 댓글 {stats['comments']:,}"

    if existing:
        existing.views = stats["views"]
        if stats.get("likes") is not None:
            existing.likes = stats["likes"]
        if stats.get("comments") is not None:
            existing.comments = stats["comments"]
        existing.notes = (existing.notes + "\n" + week_note).strip()
        upsert_feedback(existing)
    else:
        upsert_feedback(
            PerformanceRecord(
                video_id=record.video_id,
                clip_index=record.clip_index,
                title=record.title,
                views=stats["views"],
                likes=stats.get("likes"),
                comments=stats.get("comments"),
                notes=week_note,
            )
        )


def run_due_checks(now: Optional[datetime] = None, path: Path = UPLOADS_PATH) -> list[UploadRecord]:
    """지금 시점 기준으로 체크 예정일이 지난 업로드들을 모두 체크하고 uploads.json/feedback.json을 갱신한다.
    반환값: 이번 호출에서 실제로 체크가 수행된 레코드 목록."""
    now = now or datetime.now()
    records = load_uploads(path)
    updated = []
    for r in records:
        if r.done or not r.next_check_at:
            continue
        due_at = datetime.fromisoformat(r.next_check_at)
        if now < due_at:
            continue
        try:
            stats = fetch_video_stats(r.youtube_video_id)
        except Exception as e:  # noqa: BLE001 - 다음 주기에 재시도, 여기서 죽으면 안 됨
            print(f"[tracking] {r.youtube_video_id} 통계 조회 실패, 다음 주기에 재시도: {e}")
            continue

        r.checks_done += 1
        r.history.append({
            "checked_at": now.isoformat(timespec="seconds"),
            **stats,
        })
        _apply_stats_to_feedback(r, stats)

        if r.checks_done >= r.max_checks:
            r.done = True
            r.next_check_at = ""
        else:
            r.next_check_at = (now + timedelta(days=CHECK_INTERVAL_DAYS)).isoformat(timespec="seconds")
        updated.append(r)

    if updated:
        save_uploads(records, path)
    return updated


if __name__ == "__main__":
    done = run_due_checks()
    print(f"[tracking] {len(done)}건 체크 완료" if done else "[tracking] 체크할 항목 없음")
