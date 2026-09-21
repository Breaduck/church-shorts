"""유튜브 자체 자동 자막(auto-caption)을 가져와 전사 대신 사용한다.

로컬 Whisper 전사는 26분 영상 기준 CPU에서 수십 분이 걸리는 반면,
유튜브가 이미 만들어둔 자동 자막은 몇 초 안에 받을 수 있고 품질도 충분히 좋다.
그래서 이 파이프라인은 자동 자막이 있으면 그걸 우선 쓰고,
자동 자막이 없는(비공개 자막 옵션이 꺼진) 영상에 한해서만 로컬 Whisper로 폴백한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import yt_dlp

from src.transcribe import Segment, Transcript, Word


def fetch_youtube_captions_json3(url: str, output_path: Path, lang: str = "ko") -> Path | None:
    """유튜브 자동 자막을 json3 형식으로 다운로드한다. 없으면 None."""
    outtmpl = str(output_path.with_suffix(""))
    ydl_opts = {
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "json3",
        "skip_download": True,
        "outtmpl": outtmpl,
        "quiet": True,
        "noprogress": True,
    }
    # 자막이 없거나 네트워크·403 등으로 실패해도 여기서 삼킨다. 계약이 "없으면 None"이고
    # 호출부는 None을 받으면 로컬 Whisper로 폴백하는데, 예외가 올라가면 폴백 대신 분석
    # 전체가 중단되기 때문이다.
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception:
        return None

    candidate = Path(f"{outtmpl}.{lang}.json3")
    return candidate if candidate.exists() else None


def parse_json3_to_transcript(json3_path: Path, video_duration_sec: float) -> Transcript:
    data = json.loads(json3_path.read_text(encoding="utf-8"))
    events = data.get("events", [])

    segments: list[Segment] = []
    for event in events:
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue

        event_start = event["tStartMs"] / 1000
        event_end = (event["tStartMs"] + event.get("dDurationMs", 0)) / 1000

        words: list[Word] = []
        raw_words = [(s.get("utf8", ""), s.get("tOffsetMs", 0)) for s in segs]
        for i, (word_text, offset_ms) in enumerate(raw_words):
            word_text = word_text.strip()
            if not word_text:
                continue
            w_start = event_start + offset_ms / 1000
            if i + 1 < len(raw_words):
                w_end = event_start + raw_words[i + 1][1] / 1000
            else:
                w_end = event_end
            w_end = max(w_end, w_start + 0.05)
            words.append(Word(start=w_start, end=w_end, text=word_text))

        if words:
            segments.append(
                Segment(start=words[0].start, end=words[-1].end, text=text, words=words)
            )

    return Transcript(language="ko", duration_sec=video_duration_sec, segments=segments)


def get_transcript_from_youtube(
    url: str, cache_dir: Path, video_duration_sec: float, lang: str = "ko"
) -> Transcript | None:
    """자동 자막이 있으면 Transcript로 변환해 반환, 없으면 None (로컬 Whisper로 폴백 필요)."""
    json3_path = cache_dir / "youtube_auto_caption"
    result = fetch_youtube_captions_json3(url, json3_path, lang=lang)
    if result is None:
        return None
    return parse_json3_to_transcript(result, video_duration_sec)
