"""사용자가 붙여넣은 자막 텍스트를 Transcript로 변환한다.

동기: 로컬 Whisper 전사는 느리고, 유튜브/노트북LM 등 외부 도구가 훨씬 빠르고 정확한
자막을 준다. 사용자가 그 자막을 붙여넣으면 전사 단계를 통째로 건너뛴다.

지원 형식(자동 감지):
  1) SRT           : "00:00:01,000 --> 00:00:04,000" 블록
  2) WebVTT        : "00:00:01.000 --> 00:00:04.000" 블록
  3) 유튜브 스크립트 : 타임스탬프(0:00 / 00:00 / 1:02:03)가 줄 단독 또는 줄머리에 오는 형식
  4) 순수 텍스트     : 타임스탬프 없음 → parse가 실패(None start)로 표시. 호출부에서
                      유튜브 자동자막 시간축에 정렬(align_plain_text_to_reference)해 시간 복원.

핵심 원칙: 여기서 만든 Transcript의 타임스탬프는 원본 영상 절대초 기준이어야 한다.
"""
from __future__ import annotations

import bisect
import re

from src.transcribe import Segment, Transcript, Word

# HH:MM:SS 또는 MM:SS, 밀리초는 , 또는 . 로.
_TS = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_TS_RE = re.compile(_TS)
_SRT_LINE_RE = re.compile(rf"({_TS})\s*-->\s*({_TS})")
# 줄이 타임스탬프로 시작(단독 또는 뒤에 텍스트)하는지.
_LEADING_TS_RE = re.compile(rf"^\s*\[?\s*({_TS})\s*\]?\s*(.*)$")


def _ts_to_sec(h: str | None, m: str, s: str, ms: str | None) -> float:
    total = int(m) * 60 + int(s)
    if h:
        total += int(h) * 3600
    if ms:
        total += int(ms.ljust(3, "0")) / 1000.0
    return float(total)


def _mk_segment(start: float, end: float, text: str) -> Segment:
    text = text.strip()
    # 단어 타임스탬프는 구간 안에 균등 분배(대략). 자막 표시엔 세그먼트 단위면 충분.
    tokens = text.split()
    words: list[Word] = []
    if tokens and end > start:
        step = (end - start) / len(tokens)
        for i, tok in enumerate(tokens):
            ws = start + i * step
            words.append(Word(start=ws, end=ws + step, text=tok))
    return Segment(start=start, end=end, text=text, words=words)


def _parse_srt_vtt(text: str) -> list[Segment]:
    segments: list[Segment] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        m = _SRT_LINE_RE.search(block)
        if not m:
            continue
        g = m.groups()
        start = _ts_to_sec(g[1], g[2], g[3], g[4])
        end = _ts_to_sec(g[6], g[7], g[8], g[9])
        # 시간줄 이후를 본문으로.
        lines = block.splitlines()
        body_lines = []
        for ln in lines:
            if "-->" in ln or ln.strip().isdigit() or ln.strip().upper() == "WEBVTT":
                continue
            body_lines.append(ln.strip())
        body = " ".join(x for x in body_lines if x)
        if body:
            segments.append(_mk_segment(start, end, body))
    return segments


def _parse_leading_ts(text: str) -> list[Segment]:
    """유튜브 '스크립트 표시' 복붙: 타임스탬프가 줄 단독 또는 줄머리에 오는 형식."""
    entries: list[tuple[float, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LEADING_TS_RE.match(line)
        if not m:
            # 타임스탬프 없는 줄 → 직전 항목 본문에 이어붙임.
            if entries:
                entries[-1] = (entries[-1][0], (entries[-1][1] + " " + line).strip())
            continue
        g = m.groups()
        start = _ts_to_sec(g[1], g[2], g[3], g[4])
        body = (g[5] or "").strip()
        entries.append((start, body))

    segments: list[Segment] = []
    for i, (start, body) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else start + 4.0
        if end <= start:
            end = start + 1.0
        if body:
            segments.append(_mk_segment(start, end, body))
    return segments


def parse_pasted_transcript(text: str, video_duration_sec: float) -> Transcript | None:
    """붙여넣은 자막을 Transcript로 파싱. 타임스탬프를 못 찾으면 None(순수 텍스트)."""
    text = (text or "").strip()
    if not text:
        return None

    if "-->" in text:
        segments = _parse_srt_vtt(text)
    elif _LEADING_TS_RE.match(text.strip().splitlines()[0]) or any(
        _LEADING_TS_RE.match(ln.strip()) for ln in text.splitlines()[:20]
    ):
        segments = _parse_leading_ts(text)
    else:
        return None  # 타임스탬프 없음 → 호출부에서 정렬 필요

    if not segments:
        return None
    dur = video_duration_sec or (segments[-1].end if segments else 0.0)
    return Transcript(language="ko", duration_sec=dur, segments=segments)


def align_plain_text_to_reference(
    plain_text: str, reference: Transcript, video_duration_sec: float
) -> Transcript:
    """타임스탬프 없는 순수 텍스트를, 시간축이 있는 참조 자막(유튜브 자동자막)에 정렬해
    시간을 복원한다. 정확한 단어정렬이 아니라 '순차 비례 매핑'의 실용적 근사다:

      참조 자막의 전체 단어 수 대비, 붙여넣은 텍스트를 문장 단위로 나눠 누적 글자수
      비율로 참조 타임라인 상의 위치를 추정한다. 붙여넣은(정확한) 텍스트를 화면/선정
      본문으로 쓰되, 시간만 참조에서 빌려온다.
    """
    ref_words = [w for seg in reference.segments for w in seg.words]
    if not ref_words:
        # 참조가 비면 균등 분배로라도.
        ref_start, ref_end = 0.0, video_duration_sec or 1.0
    else:
        ref_start, ref_end = ref_words[0].start, ref_words[-1].end

    # 문장 단위 분할(한국어 종결부호 기준, 없으면 줄 단위).
    sentences = re.split(r"(?<=[.!?。…])\s+|\n+", plain_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return reference

    total_chars = sum(len(s) for s in sentences) or 1
    span = max(0.1, ref_end - ref_start)

    segments: list[Segment] = []
    acc = 0
    for s in sentences:
        frac_start = acc / total_chars
        acc += len(s)
        frac_end = acc / total_chars
        start = ref_start + span * frac_start
        end = ref_start + span * frac_end
        segments.append(_mk_segment(start, max(end, start + 0.5), s))

    return Transcript(
        language="ko",
        duration_sec=video_duration_sec or ref_end,
        segments=segments,
    )


_NORM_RE = re.compile(r"[\s.,!?…·\"'`~()\[\]{}<>:;\-–—「」『』“”‘’]+")


def _normalize(s: str) -> str:
    """공백·문장부호를 모두 제거해 자막 간 띄어쓰기·부호 차이에 강인하게 만든다."""
    return _NORM_RE.sub("", (s or "")).lower()


def _build_ref_index(reference: Transcript):
    """참조 자막의 모든 단어를 이어붙인 정규화 문자열 + 문자→(시작,끝)초 매핑을 만든다."""
    norm_chars: list[str] = []
    char_time: list[tuple[float, float]] = []
    for seg in reference.segments:
        words = seg.words or [Word(start=seg.start, end=seg.end, text=seg.text)]
        for w in words:
            for ch in _normalize(w.text):
                norm_chars.append(ch)
                char_time.append((w.start, w.end))
    return "".join(norm_chars), char_time


def snap_clips_to_reference(clips, aligned: Transcript, reference: Transcript) -> list:
    """내용 기반으로 고른 클립의 start/end를, 타임스탬프가 있는 참조 자막(유튜브 자동자막)에서
    그 클립의 첫/끝 문장을 '역으로 찾아' 실제 시각으로 스냅한다.

    - aligned: 붙여넣은 텍스트를 비례배분한 Transcript(대략 시각). 여기서 각 클립 구간의
      실제 대사(첫·끝 문장 텍스트)를 얻는다.
    - reference: 유튜브 자동자막(정확한 시각). 여기서 그 문장을 찾아 정밀 시각을 얻는다.
    비례배분 오차를 없애 클립 경계를 실제 발화 지점에 맞춘다. 재전사 없이 문자열 검색이라 즉시.
    """
    norm, char_time = _build_ref_index(reference)
    if not norm:
        return clips
    char_starts = [t[0] for t in char_time]  # 시각→문자위치 변환용(단조 증가)

    def _time_to_pos(sec: float) -> int:
        return bisect.bisect_left(char_starts, sec)

    def _find(phrase: str, est_sec: float, want: str) -> float | None:
        q = _normalize(phrase)
        if len(q) < 5:
            return None
        # 비례추정 위치 ±120초 창으로 검색을 한정 → 중복 문구 오매칭 방지 + 정확도↑.
        lo = max(0, _time_to_pos(est_sec - 120))
        hi = min(len(norm), _time_to_pos(est_sec + 120) + 400)
        window = norm[lo:hi]
        for L in (28, 20, 14, 9):
            if want == "start":
                anchor = q[:L]
                p = window.find(anchor)
                if p != -1:
                    return char_time[lo + p][0]
            else:
                anchor = q[-L:]
                p = window.rfind(anchor)
                if p != -1:
                    idx = min(lo + p + len(anchor) - 1, len(char_time) - 1)
                    return char_time[idx][1]
        return None

    for clip in clips:
        segs = [s for s in aligned.segments if s.start < clip.end and s.end > clip.start]
        if not segs:
            continue
        new_start = _find(segs[0].text, clip.start, "start")
        new_end = _find(segs[-1].text, clip.end, "end")
        if new_start is not None:
            clip.start = round(new_start, 2)
        if new_end is not None and new_end > clip.start + 1.0:
            clip.end = round(new_end, 2)
    return clips
