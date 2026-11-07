"""교회 유튜브 채널을 직접 훑어 '손으로 올린' 쇼츠의 성과를 feedback.json에 자동 기록한다.

왜 필요한가:
  upload/tracking.py는 앱의 업로드 버튼(OAuth)으로 올린 클립만 추적한다. 실제로는 사람이 스튜디오에서
  직접 올리기 때문에 uploads.json이 비어 있었고, 성과 기록은 0건이었다(2026-11-07).
  공개 채널의 쇼츠 목록은 로그인 없이 yt-dlp로 읽히므로, 채널 쇼츠 제목을 우리 clips.json의 제목
  (제목 후보 포함)과 맞춰 조회수·좋아요·댓글을 채운다.

규칙:
  - 사람이 입력한 rating/메모/저장/공유/조회율/지속률 분석은 덮어쓰지 않는다.
  - rating은 사람이 매긴 적 없을 때만(auto_rated) 채널 중앙값 대비 배수로 자동 판정한다.
    올린 지 28일 안 된 영상은 아직 판정하지 않는다(조회수가 덜 쌓여 flop으로 오판).
  - 우리 클립과 안 맞는 쇼츠(예전에 다른 방식으로 만든 것)도 channel_shorts.json에 스냅숏으로 남긴다
    — 채널 중앙값 계산과 사람이 훑어보는 용도.

실행: web_app.py 백그라운드 스케줄러가 하루 1회, 또는 `python -m src.channel_tracking` 단독 실행.
"""
from __future__ import annotations

import json
import re
import statistics
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import yaml
import yt_dlp

from src.feedback import (
    RATING_FLOP, RATING_HIT, RATING_OK, PerformanceRecord, load_feedback, upsert_feedback,
)
from src.fsutil import atomic_write_text

OUTPUT_ROOT = Path("output")
SNAPSHOT_PATH = Path("channel_shorts.json")
MATCH_THRESHOLD = 0.85   # 제목 유사도. 올릴 때 조사·문장부호를 살짝 고친 정도까지 같은 클립으로 본다.
MIN_AGE_DAYS_FOR_RATING = 28   # 비교 대상(채널 중앙값)이 대부분 몇 달 묵은 영상이라 갓 올린 건 불리하다
HIT_RATIO = 2.0          # 채널 중앙값의 2배 이상 → hit
FLOP_RATIO = 0.5         # 절반 이하 → flop


def _norm(title: str) -> str:
    t = re.sub(r"#\S+", "", title or "")          # 업로드 때 붙인 해시태그
    t = re.sub(r"[\s\W_]+", "", t, flags=re.UNICODE)
    return t.lower()


def channel_url_from_config(config_path: Path = Path("config.yaml")) -> str:
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return ""
    return str((cfg.get("church") or {}).get("youtube_channel") or "").strip()


def list_channel_shorts(channel_url: str) -> list[dict]:
    """채널 쇼츠 탭 목록(최신순). flat 목록이라 빠르고 로그인이 필요 없다(조회수까지만 나옴)."""
    url = channel_url.rstrip("/")
    if not url.endswith("/shorts"):
        url += "/shorts"
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True,
                           "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    out = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        out.append({"id": e["id"], "title": e.get("title") or "", "views": e.get("view_count")})
    return out


def fetch_details(youtube_id: str) -> dict:
    """영상 한 개의 좋아요·댓글·게시일. 봇 차단 등으로 실패하면 빈 dict(조회수만으로 진행)."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/shorts/{youtube_id}", download=False) or {}
    except Exception as e:  # noqa: BLE001 - 세부 지표는 덤, 실패해도 조회수 기록은 계속
        print(f"[channel] {youtube_id} 세부 지표 실패(조회수만 기록): {str(e)[:120]}", flush=True)
        return {}
    return {
        "views": info.get("view_count"),
        "likes": info.get("like_count"),
        "comments": info.get("comment_count"),
        "upload_date": info.get("upload_date") or "",   # YYYYMMDD
    }


def _clip_index(output_root: Path = OUTPUT_ROOT) -> list[tuple[str, str, int, object]]:
    """(정규화 제목, video_id, clip_index, Clip) — 확정 제목과 제목 후보 모두."""
    from src.highlights import load_clips_json
    rows = []
    for clips_path in sorted(output_root.glob("*/clips.json")):
        try:
            clips = load_clips_json(clips_path)
        except Exception:  # noqa: BLE001 - 깨진 clips.json 하나 때문에 전체 추적이 멈추면 안 됨
            continue
        vid = clips_path.parent.name
        for i, c in enumerate(clips):
            titles = [c.title] + list(getattr(c, "title_candidates", None) or [])
            for t in titles:
                n = _norm(t if isinstance(t, str) else (t or {}).get("title", ""))
                if n:
                    rows.append((n, vid, i, c))
    return rows


def match_clip(title: str, index: list) -> Optional[tuple[str, int, object, float]]:
    n = _norm(title)
    # 찬양은 업로드 제목에 "[찬양]" 같은 머리표가 붙어 있고, 우리 쪽 제목엔 없을 수 있다.
    n_bare = _norm(re.sub(r"^\s*[\[{(【][^\]})】]*[\]})】]\s*", "", title or ""))
    best = None
    for cand, vid, i, clip in index:
        r = max(SequenceMatcher(None, n, cand).ratio(),
                SequenceMatcher(None, n_bare, cand).ratio() if n_bare else 0.0)
        if best is None or r > best[3]:
            best = (vid, i, clip, r)
    if best and best[3] >= MATCH_THRESHOLD:
        return best
    return None


def _auto_rating(views: Optional[int], median: float, upload_date: str, now: datetime) -> Optional[str]:
    if views is None or median <= 0 or not upload_date:
        return None
    try:
        age = (now - datetime.strptime(upload_date, "%Y%m%d")).days
    except ValueError:
        return None
    if age < MIN_AGE_DAYS_FOR_RATING:
        return None
    ratio = views / median
    if ratio >= HIT_RATIO:
        return RATING_HIT
    if ratio <= FLOP_RATIO:
        return RATING_FLOP
    return RATING_OK


def _record_from_clip(vid: str, idx: int, clip) -> PerformanceRecord:
    hook = ""
    if getattr(clip, "caption_overrides", None):
        hook = (clip.caption_overrides[0] or {}).get("text", "") or ""
    hook = hook or (getattr(clip, "hook_line", "") or "").strip()
    return PerformanceRecord(
        video_id=vid, clip_index=idx, title=clip.title, hook_text=hook,
        start=clip.start, end=clip.end,
        predicted_score=clip.score, predicted_core=clip.core_score, predicted_viral=clip.viral_score,
        predicted_subscores={
            "hook": clip.hook_score, "retention": clip.retention_score,
            "emotion": clip.emotion_score, "relatability": clip.relatability_score,
            "payoff": clip.payoff_score, "quotability": clip.quotability_score,
        },
        auto_rated=True,
    )


def sync_channel(channel_url: str = "", now: Optional[datetime] = None, details: bool = True) -> dict:
    """채널 쇼츠를 훑어 feedback.json·channel_shorts.json을 갱신한다. 반환: 요약 dict."""
    channel_url = channel_url or channel_url_from_config()
    if not channel_url:
        return {"error": "config.yaml church.youtube_channel이 비어 있어요"}
    now = now or datetime.now()
    shorts = list_channel_shorts(channel_url)
    prev = {}
    if SNAPSHOT_PATH.exists():
        try:
            prev = {s["id"]: s for s in json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")).get("shorts", [])}
        except (OSError, ValueError, KeyError, TypeError):
            prev = {}
    index = _clip_index()

    for s in shorts:
        m = match_clip(s["title"], index)
        s["matched"] = {"video_id": m[0], "clip_index": m[1], "similarity": round(m[3], 3)} if m else None
        old = prev.get(s["id"]) or {}
        s["upload_date"] = old.get("upload_date", "")
        s["likes"], s["comments"] = old.get("likes"), old.get("comments")
        # 세부 지표(개별 요청)는 우리 클립과 맞은 것만 — 요청 수를 줄여 봇 차단을 피한다.
        if details and m:
            d = fetch_details(s["id"])
            for k in ("likes", "comments", "upload_date"):
                if d.get(k) not in (None, ""):
                    s[k] = d[k]
            if d.get("views") is not None:
                s["views"] = d["views"]
        s["history"] = (old.get("history") or [])[-60:]
        if s.get("views") is not None:
            s["history"].append({"checked_at": now.isoformat(timespec="seconds"), "views": s["views"],
                                 "likes": s.get("likes"), "comments": s.get("comments")})

    view_list = [s["views"] for s in shorts if isinstance(s.get("views"), int)]
    median = float(statistics.median(view_list)) if view_list else 0.0

    feedback = {(r.video_id, r.clip_index): r for r in load_feedback()}
    matched = 0
    for s in shorts:
        if not s["matched"]:
            continue
        matched += 1
        key = (s["matched"]["video_id"], s["matched"]["clip_index"])
        rec = feedback.get(key)
        if rec is None:
            clip = next(c for _, v, i, c in index if (v, i) == key)
            rec = _record_from_clip(key[0], key[1], clip)
        rec.youtube_video_id = s["id"]
        if s.get("views") is not None:
            rec.views = s["views"]
        if s.get("likes") is not None:
            rec.likes = s["likes"]
        if s.get("comments") is not None:
            rec.comments = s["comments"]
        if rec.auto_rated:
            rec.rating = _auto_rating(rec.views, median, s.get("upload_date", ""), now) or RATING_OK
        upsert_feedback(rec)

    atomic_write_text(SNAPSHOT_PATH, json.dumps(
        {"channel": channel_url, "synced_at": now.isoformat(timespec="seconds"),
         "median_views": median, "shorts": shorts}, ensure_ascii=False, indent=2))
    return {"total": len(shorts), "matched": matched, "median_views": median}


if __name__ == "__main__":
    print(f"[channel] {sync_channel()}")
