"""시청 지속률(Audience retention) CSV → '사람들이 정확히 어느 문장에서 나갔나'.

출처: github.com/Jakeschincariol/youtube-agent-skill 의 retention.py(MIT) 아이디어를 쇼츠·한국어 설교용으로
다시 짰다(2026-10-03). 원본은 30초 훅·롱폼 기준이라 그대로 쓰면 60초 쇼츠에선 '훅 구간'이 영상의 절반이 된다.

왜 필요한가:
  feedback.json에는 조회수/체감등급만 쌓여서 "이 클립이 망했다"는 알아도 **왜**(어느 문장에서) 망했는지는
  몰랐다. 유튜브 스튜디오의 지속률 CSV는 초 단위로 남은 시청자 비율을 주므로, 급락 지점의 출력 시각을
  원본 설교 시각으로 되돌려 그때 하던 말을 붙이면 선정 프롬프트에 "이런 시작/이음새에서 나간다"를
  구체적으로 되먹일 수 있다.

얻는 법: 스튜디오 → 쇼츠 → 분석 → 참여도 → 시청자 유지율 그래프 → 다운로드 아이콘.
열은 (동영상 위치, 시청자 유지율) 두 개면 된다. 위치는 %·초·0~1 비율, 유지율은 %·0~1 비율 모두 받는다.

보는 것 셋(원본과 같은 구분 — 원인과 처방이 서로 다르다):
  스와이프 이탈(HOOK)  첫 3초에 잃은 비율. 쇼츠는 여기서 대부분이 갈린다 → 첫 문장(훅) 문제.
  급락(CLIFF)          한 지점의 가파른 하락 → 그 순간의 문장(이음새·곁가지·긴 설명) 문제.
  흘러내림(SLIDE)      중간 구간의 평균 하락률 → 전체 호흡/길이 문제(자르는 게 처방).
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path
from typing import Optional

HOOK_SEC = 3.0          # 쇼츠 스와이프 판정 구간
CLIFF_MIN_LOST = 2.0    # 이보다 작은 낙폭(%p)은 급락으로 안 본다(쇼츠 곡선은 촘촘해 잡음이 크다)
MAX_CLIFFS = 3
SAID_BEFORE = 3.0       # 이탈 직전 이만큼(초) 동안 한 말을 '이탈 원인 후보'로 붙인다
SAID_AFTER = 0.5


def parse_retention_csv(text: str) -> list[tuple[float, float]]:
    """CSV 본문 → [(위치, 유지율%)]. 헤더/빈 줄/숫자 아닌 칸은 건너뛴다."""
    rows: list[tuple[float, float]] = []
    for r in csv.reader(io.StringIO(text.lstrip("﻿"))):
        vals = []
        for c in r:
            c = c.strip().replace("%", "").replace(",", "")
            try:
                vals.append(float(c))
            except ValueError:
                continue
        if len(vals) >= 2:
            rows.append((vals[0], vals[1]))
    rows.sort(key=lambda p: p[0])
    if rows and max(y for _, y in rows) <= 3.0:   # 0~1(반복 재생 시 1 초과) 비율 → %
        rows = [(x, y * 100.0) for x, y in rows]
    return rows


def probe_duration(path: Path) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _to_seconds(points: list[tuple[float, float]], duration: Optional[float]) -> list[tuple[float, float]]:
    """위치 축을 초로. 최대값이 1 이하면 비율, 100 근처면 %, 그 외는 이미 초로 본다."""
    mx = max(x for x, _ in points)
    if mx <= 1.01:
        if not duration:
            raise ValueError("위치가 비율(0~1)인데 영상 길이를 몰라 초로 바꿀 수 없습니다")
        return [(x * duration, y) for x, y in points]
    if 99.0 <= mx <= 100.5 and duration and abs(duration - mx) > 5:
        return [(x / 100.0 * duration, y) for x, y in points]
    return points


def output_to_source(t: float, ranges: list, content_dur: float) -> float:
    """쇼츠 출력 시각 → 원본 설교 시각. 무음 제거·배속까지 정확히 되돌릴 기록은 없어서, 남긴 구간 합을
    실제 본편 길이에 비례로 맞춘다(±1~2초 오차 — 이탈 직전 3초 문장을 고르는 용도엔 충분)."""
    total = sum(max(0.0, float(e) - float(s)) for s, e in ranges)
    if total <= 0:
        return float(ranges[0][0]) if ranges else t
    u = min(total, max(0.0, t * (total / content_dur if content_dur > 0 else 1.0)))
    for s, e in ranges:
        seg = float(e) - float(s)
        if u <= seg:
            return float(s) + u
        u -= seg
    return float(ranges[-1][1])


def _said_between(segments: list[dict], a: float, b: float) -> str:
    words = []
    for s in segments:
        if s["end"] < a or s["start"] > b:
            continue
        ws = s.get("words") or []
        if ws:
            words += [(w.get("text") or w.get("word") or "").strip() for w in ws if w["end"] >= a and w["start"] <= b]
        else:
            words.append(s["text"].strip())
    return " ".join(w for w in words if w)[:120]


def analyze(
    points: list[tuple[float, float]],
    duration: Optional[float],
    outro_sec: float = 0.0,
    ranges: Optional[list] = None,
    segments: Optional[list[dict]] = None,
) -> dict:
    """지속률 곡선 → {hook_leak, cliffs[{at, lost, said}], slide_per_sec, end_pct, avg_pct}."""
    if len(points) < 8:
        raise ValueError("CSV에서 데이터 점을 8개 이상 읽지 못했습니다")
    pts = _to_seconds(points, duration)
    dur = duration or pts[-1][0]
    content_end = max(HOOK_SEC, dur - outro_sec)
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    start = ys[0] or 100.0

    # 3초 경계를 걸친 하락도 훅 몫이다: 3초 이후 첫 점까지 본다(안 그러면 그 낙폭이 어디에도 안 잡힌다).
    first_after = next((i for i, x in enumerate(xs) if x >= HOOK_SEC), len(xs) - 1)
    hook_end = min(ys[: first_after + 1])
    hook_leak = max(0.0, start - hook_end)

    drops = []
    for i in range(1, len(pts)):
        if i <= first_after or xs[i - 1] >= content_end:   # 훅·로고 구간은 따로 본다
            continue
        lost = ys[i - 1] - ys[i]
        span = (xs[i] - xs[i - 1]) or 1.0
        drops.append((lost / span, xs[i - 1], lost))
    mid_rates = [r for r, _, _ in drops]
    slide = sum(mid_rates) / len(mid_rates) if mid_rates else 0.0

    cliffs = []
    for rate, at, lost in sorted(drops, reverse=True):
        if lost < CLIFF_MIN_LOST or len(cliffs) >= MAX_CLIFFS:
            continue
        if any(abs(at - c["at"]) < 2.0 for c in cliffs):   # 같은 하락의 이웃 점은 하나로
            continue
        c = {"at": round(at, 1), "lost": round(lost, 1)}
        if ranges and segments is not None:
            src = output_to_source(at, ranges, content_end)
            c["src"] = round(src, 1)
            c["said"] = _said_between(segments, src - SAID_BEFORE, src + SAID_AFTER)
        cliffs.append(c)
    cliffs.sort(key=lambda c: c["at"])

    hook_said = ""
    if ranges and segments is not None:
        s0 = output_to_source(0.0, ranges, content_end)
        hook_said = _said_between(segments, s0, output_to_source(HOOK_SEC, ranges, content_end))

    body = [y for x, y in pts if x < content_end]
    return {
        "hook_leak": round(hook_leak, 1),
        "hook_said": hook_said,
        "cliffs": cliffs,
        "slide_per_sec": round(slide, 2),
        "end_pct": round(body[-1] if body else ys[-1], 1),
        "avg_pct": round(sum(body) / len(body), 1) if body else None,
        "duration": round(dur, 1),
    }


def hook_verdict(leak: float) -> str:
    return "양호" if leak < 25 else "새는 중" if leak < 40 else "심각"


def summarize(a: dict) -> str:
    """한 줄 요약(UI 표시·프롬프트 공용)."""
    parts = [f"첫{HOOK_SEC:.0f}초 이탈 {a['hook_leak']:.0f}%({hook_verdict(a['hook_leak'])})"]
    for c in a.get("cliffs") or []:
        said = f' "{c["said"]}"' if c.get("said") else ""
        parts.append(f"{c['at']:.0f}초 -{c['lost']:.0f}%p{said}")
    if a.get("end_pct") is not None:
        parts.append(f"끝까지 {a['end_pct']:.0f}%")
    return " · ".join(parts)


def analyze_clip_csv(csv_text: str, video_dir: Path, clip, clip_number: int, outro_sec: float) -> dict:
    """웹 경로용: 렌더 파일 길이·클립 구간·전사를 찾아 analyze()를 돌린다."""
    points = parse_retention_csv(csv_text)
    mp4 = video_dir / "clips" / f"short_{clip_number}.mp4"
    duration = probe_duration(mp4) if mp4.exists() else None
    if duration is None:
        speed = float(getattr(clip, "playback_speed", 1.0) or 1.0)
        kr = getattr(clip, "keep_ranges", None) or [[clip.start, clip.end]]
        duration = sum(float(e) - float(s) for s, e in kr) / speed + outro_sec
    segments = None
    tp = video_dir / "transcript.json"
    if tp.exists():
        segments = json.loads(tp.read_text(encoding="utf-8")).get("segments")
    ranges = getattr(clip, "keep_ranges", None) or [[clip.start, clip.end]]
    return analyze(points, duration, outro_sec=outro_sec, ranges=ranges, segments=segments)
