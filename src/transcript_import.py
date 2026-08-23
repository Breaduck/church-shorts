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


def _split_sentences(text: str) -> list[str]:
    """붙여넣은 텍스트를 매칭 단위(문장)로 나눈다. 한국어 종결부호/줄바꿈이 1차 기준이고,
    노트북LM처럼 부호·줄바꿈 없는 통짜 문단은 단어 20개 단위로 잘라 매칭 앵커를 확보한다.
    (옛 방식은 통짜 텍스트를 세그먼트 1개로 만들어 시간정보가 무너졌다 — 그 원인 제거.)"""
    parts = re.split(r"(?<=[.!?。…?!])\s+|\n+", (text or "").strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        words = p.split()
        if len(words) <= 30:
            out.append(p)
        else:
            for i in range(0, len(words), 20):
                chunk = " ".join(words[i : i + 20]).strip()
                if chunk:
                    out.append(chunk)
    return out


def _fill_gap(spans: list, lo: int, hi: int, t0: float, t1: float) -> None:
    """매칭 실패한 문장들(spans[lo:hi])의 시각을, 앞뒤 성공 문장 사이 [t0,t1]에 글자수 비례로 채운다."""
    total = sum(len(_normalize(spans[i][0])) for i in range(lo, hi)) or 1
    span = max(0.1, t1 - t0)
    acc = 0
    for i in range(lo, hi):
        clen = len(_normalize(spans[i][0]))
        s = t0 + span * acc / total
        acc += clen
        e = t0 + span * acc / total
        spans[i][1] = s
        spans[i][2] = max(e, s + 0.3)


def _align_proportional(
    plain_text: str, reference: Transcript, video_duration_sec: float
) -> Transcript:
    """폴백: 문자열 매칭이 전부 실패했을 때만 쓰는 '순차 비례 매핑'(옛 방식)."""
    ref_words = [w for seg in reference.segments for w in seg.words]
    if not ref_words:
        ref_start, ref_end = 0.0, video_duration_sec or 1.0
    else:
        ref_start, ref_end = ref_words[0].start, ref_words[-1].end

    sentences = _split_sentences(plain_text)
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
        language="ko", duration_sec=video_duration_sec or ref_end, segments=segments
    )


def align_plain_text_to_reference(
    plain_text: str, reference: Transcript, video_duration_sec: float
) -> Transcript:
    """타임스탬프 없는 '정확한' 붙여넣기 텍스트(노트북LM 등)에, 시간축이 있는 참조 자막
    (유튜브 자동자막)의 실제 시각을 '문자열 매칭'으로 부여한다.

    두 자막은 같은 오디오의 전사라 표기가 거의 일치하므로, 정규화(공백·부호 제거) 후
    부분열 검색이 잘 맞는다. 참조 전체를 이어붙인 정규화 문자열에서 각 문장의 앞부분
    앵커를 '커서 이후'로만 찾아(단조 증가) 반복 어구 오매칭과 뒤로 점프를 막는다.
    비례배분(옛 방식)과 달리 시간이 추정이 아니라 실제 매칭 위치라 오차가 작다.
    화면 본문은 붙여넣은 정확본을 그대로 쓰고, 매칭 실패한 문장만 앞뒤 사이를 비례로 채운다.
    """
    norm, char_time = _build_ref_index(reference)
    sentences = _split_sentences(plain_text)
    if not norm or not sentences:
        return _align_proportional(plain_text, reference, video_duration_sec)

    n = len(norm)
    ref_start, ref_end = char_time[0][0], char_time[-1][1]

    cursor = 0
    matched: list[int] = []
    spans: list[list] = []  # [display, start_t|None, end_t|None]
    for disp in sentences:
        q = _normalize(disp)
        pos = -1
        for L in (24, 16, 10, 6):
            if len(q) >= L:
                p = norm.find(q[:L], cursor)
                if p != -1:
                    pos = p
                    break
        if pos == -1:
            spans.append([disp, None, None])
            continue
        end_pos = min(pos + max(1, len(q)), n)
        st = char_time[pos][0]
        en = char_time[min(end_pos, n) - 1][1]
        spans.append([disp, st, max(en, st + 0.3)])
        matched.append(len(spans) - 1)
        cursor = end_pos

    if not matched:
        return _align_proportional(plain_text, reference, video_duration_sec)

    # 실패한 문장 시각을 앞뒤 성공 문장 사이에 채운다(선두/중간/말미).
    first = matched[0]
    if first > 0:
        _fill_gap(spans, 0, first, ref_start, spans[first][1])
    for a, b in zip(matched, matched[1:]):
        if b - a > 1:
            _fill_gap(spans, a + 1, b, spans[a][2], spans[b][1])
    last = matched[-1]
    if last < len(spans) - 1:
        _fill_gap(spans, last + 1, len(spans), spans[last][2], max(ref_end, spans[last][2] + 0.5))

    segments = [_mk_segment(st, max(en, st + 0.3), disp) for disp, st, en in spans if st is not None]
    return Transcript(
        language="ko", duration_sec=video_duration_sec or ref_end, segments=segments
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

    # 역스냅은 비례배분 오차(수초~수십초)를 바로잡는 '미세 보정'일 뿐이다. 그런데 끝 문장의
    # 짧은 앵커("~습니다" 등 설교에 반복되는 어구)가 클립 밖 먼 지점에 오매칭되면 clip.end가
    # 몇 분 뒤로 튀어 4분·13분짜리 '쇼츠'가 만들어졌다(실측: 3k01rvZhW_M clip0=237초).
    # 그래서 추정 위치에서 이 허용범위(초)를 넘게 이동시키는 스냅 결과는 오매칭으로 보고 버린다.
    SNAP_TOLERANCE_SEC = 40.0
    for clip in clips:
        segs = [s for s in aligned.segments if s.start < clip.end and s.end > clip.start]
        if not segs:
            continue
        new_start = _find(segs[0].text, clip.start, "start")
        new_end = _find(segs[-1].text, clip.end, "end")
        if new_start is not None and abs(new_start - clip.start) <= SNAP_TOLERANCE_SEC:
            clip.start = round(new_start, 2)
        if (
            new_end is not None
            and new_end > clip.start + 1.0
            and abs(new_end - clip.end) <= SNAP_TOLERANCE_SEC
        ):
            clip.end = round(new_end, 2)
    return clips
