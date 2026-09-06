"""오케스트레이터: 유튜브 링크 -> 세로형 쇼츠

두 단계로 나뉜다 (웹 검토 흐름에서 각각 따로 호출됨):
  1) analyze()        : 다운로드 + 전사 + 하이라이트 후보 선정 (아직 렌더링 안 함)
  2) render_selected() : 후보 중 유저가 고른 것만 정밀 재전사 + 렌더링

사용법 (CLI, 둘 다 한번에):
    python -m src.main "https://www.youtube.com/watch?v=XXXXXXXX"
"""
from __future__ import annotations

import copy
import hashlib
import os
import json
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml

from src.audio_peaks import detect_peak_hints
from src.download import DownloadResult, _probe_duration_sec, download_video, find_cached, probe_video
from src.render import _probe_display_resolution
from src.highlights import (
    CLIPS_LOCK,
    Clip,
    build_prompt,
    load_clips_json,
    save_clips_json,
    _distribute_lines_by_chars,
    correct_praise_lyrics,
    fetch_praise_lyrics_by_titles,
    guess_praise_title_from_snippet,
    _build_praise_clips_from_titles,
    save_prompt_for_manual_mode,
    select_highlights_auto,
    select_praise_songs,
)
from src.feedback import format_feedback_for_prompt, load_feedback
from src.render import render_clip
from src.transcribe import Segment, Word, transcribe_and_save, transcribe_clip_precise, Transcript
from src.transcript_import import (
    align_plain_text_to_reference,
    parse_pasted_transcript,
    snap_clips_to_reference,
)
from src.youtube_captions import get_transcript_from_youtube


def load_config(path: Path = Path("config.yaml")) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _default_progress(message: str, pct: float, eta_seconds: float | None = None) -> None:
    eta = "" if eta_seconds is None else f" (~{int(round(eta_seconds))}s)"
    print(f"[{pct:5.1f}%]{eta} {message}")


def _safe_progress(cb, message, pct, eta=None) -> None:
    """진행률 콜백은 순전히 장식(UI 표시)이다. 콜백이 던지는 예외가 전사/렌더 파이프라인을
    죽이거나, 더 나쁘게는 '정밀 인식 실패'로 오인돼 자막이 유튜브 폴백으로 떨어지게 해서는
    안 된다(실측 버그). 따라서 콜백 호출은 여기서 감싸 예외를 삼킨다."""
    try:
        cb(message, pct, eta)
    except Exception:  # noqa: BLE001 - 진행률 표시 실패는 파이프라인과 무관하게 무시
        pass


def render_signature(start: float, end: float) -> str:
    """렌더된 mp4가 '지금 이 클립'의 결과인지 식별하는 서명. 렌더 시 short_N.src에 기록하고
    검토 UI가 대조한다. clips.json이 바뀌어(재선정 등) 인덱스가 다른 클립을 가리켜도, 옛
    short_N.mp4가 새 클립 밑에 '이미 만들어진 것'처럼 붙어 보이던 문제를 막는다."""
    return f"{start:.2f}_{end:.2f}"


class _Stage:
    __slots__ = ("key", "message", "est", "span")

    def __init__(self, key: str, message: str, est: float, span: float | None = None):
        self.key = key
        self.message = message
        self.est = max(1.0, float(est))
        # span: 이 단계가 진행바에서 차지하는 '시각적 폭'(%). est(예상초)와 분리한다.
        # 예전엔 pct를 est 비율로 채웠는데, 준비 단계(다운로드/전사)는 est는 작지만 순식간에
        # 끝나고 하이라이트 선정(est 큼)만 몇 분 걸려서, 정작 선정 중인데 바가 14%에 있고
        # 스텝 라벨은 '다운로드'를 가리키는 심각한 불일치가 났다(사용자: "퍼센트 너무 부정확").
        # span을 스텝 라벨 구간과 일치시켜 "지금 몇% = 지금 무슨 단계"가 항상 맞게 한다.
        # span 미지정 시 est로 폴백(하위호환).
        self.span = float(span) if span is not None else self.est


class StageProgress:
    """단계 기반 진행률 엔진. 각 단계에 '예상 소요시간(est)'을 주고, 전체 바를 '경과시간 대비
    예상시간' 비율로 채운다. 그래서 진행 콜백이 없는 opaque 단계(하이라이트 선정 등)도 바가
    부드럽게 움직이고 절대 뒤로 튀지 않는다(단조 증가). ETA는 '남은 단계 예상시간의 합'이라
    링크 넣은 직후부터 끝까지 일관된 값이 나온다(예전의 구간별 점프/엉뚱한 ETA 해결).

    - set_fraction(frac): 실제 진행률이 있는 단계(다운로드 %, 로컬 전사 %)에서 호출.
    - set_current_est(sec): 런타임에 예상시간을 바꿀 때(예: 로컬 전사로 분기되면 크게).
    - advance(): 현재 단계 완료 처리 후 다음 단계로.
    - finish(): 100%로 마감하고 티커 종료."""

    def __init__(self, progress_cb, stages: list[_Stage], min_floor: float = 99.0):
        self._cb = progress_cb
        self._stages = stages
        self._i = 0
        self._frac = 0.0
        self._min_floor = min_floor  # finish() 전까지 넘지 않을 상한(거짓 100% 방지)
        self._stage_start = time.time()
        self._last_pct = 0.0
        self._overrun_sec = 0.0  # 예상시간을 넘겨 진행률이 0.95에 고정된 뒤 실제 경과시간(초)
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._ticker = threading.Thread(target=self._tick, daemon=True)
        self._ticker.start()

    def _total_span(self) -> float:
        return sum(s.span for s in self._stages) or 1.0

    def _emit_locked(self) -> None:
        cur = self._stages[self._i]
        # 진행바 %는 '시각적 폭(span)' 기준 — 스텝 라벨 구간과 일치해 "몇% = 무슨 단계"가 맞는다.
        span_done = sum(s.span for s in self._stages[: self._i]) + cur.span * self._frac
        gfrac = min(1.0, span_done / self._total_span())
        pct = min(self._min_floor, gfrac * 100.0)
        pct = max(pct, self._last_pct)  # 단조 증가 보장
        self._last_pct = pct
        # ETA는 '남은 예상초' 기준(span과 별개). 현재 단계의 남은 몫 + 이후 단계 est 합.
        remaining_est = cur.est * (1.0 - self._frac) + sum(
            s.est for s in self._stages[self._i + 1 :]
        )
        # 예상시간을 넘겨 frac이 0.95에 고정되면 eta도 같이 고정돼 "5초 남음"이 몇 분째
        # 안 바뀌는 것처럼 보인다(실제로는 멈춘 게 아님). 이 경우 거짓 ETA 대신 실제 경과시간을
        # 메시지에 보여줘 "아직 일하는 중"임을 알린다.
        if self._overrun_sec > 0:
            eta = None
            message = f"{cur.message} (예상보다 오래 걸리는 중... {int(self._overrun_sec)}초 경과)"
        else:
            eta = max(0.0, remaining_est)
            message = cur.message
        _safe_progress(self._cb, message, pct, eta)

    def _tick(self) -> None:
        while not self._done.wait(timeout=0.5):
            with self._lock:
                cur = self._stages[self._i]
                elapsed = time.time() - self._stage_start
                target = min(0.95, elapsed / cur.est)
                if target > self._frac:
                    self._frac = target
                self._overrun_sec = max(0.0, elapsed - cur.est)
                self._emit_locked()

    def set_fraction(self, frac: float, message: str | None = None) -> None:
        with self._lock:
            if message:
                self._stages[self._i].message = message
            self._frac = max(self._frac, min(1.0, frac))
            self._overrun_sec = 0.0  # 실제 진행률 콜백이 왔으니 더는 opaque 초과 상태가 아님
            self._emit_locked()

    def set_current_est(self, est_seconds: float) -> None:
        with self._lock:
            self._stages[self._i].est = max(1.0, float(est_seconds))
            self._emit_locked()

    def message(self, message: str) -> None:
        with self._lock:
            self._stages[self._i].message = message
            self._emit_locked()

    def advance(self, message: str | None = None) -> None:
        with self._lock:
            self._frac = 1.0
            self._emit_locked()
            if self._i < len(self._stages) - 1:
                self._i += 1
                self._frac = 0.0
                self._overrun_sec = 0.0
                self._stage_start = time.time()
                if message:
                    self._stages[self._i].message = message
                self._emit_locked()

    def stop(self) -> None:
        """티커만 정지(100% 안 찍음). 예외 경로의 finally에서 안전하게 부른다."""
        self._done.set()
        try:
            self._ticker.join(timeout=2.0)
        except RuntimeError:
            pass

    def finish(self, message: str = "완료") -> None:
        self.stop()
        with self._lock:
            self._last_pct = 100.0
            _safe_progress(self._cb, message, 100.0, 0.0)


def _run_with_progress_ticker(fn, start_pct: float, end_pct: float, progress, message: str, est_seconds: float):
    """분 단위로 걸릴 수 있는데 중간 진행률을 알 수 없는 단계(예: claude -p 서브프로세스 호출)를
    위한 흉내 진행률바. est_seconds에 걸쳐 start_pct -> end_pct*0.95 정도까지 서서히 채우고,
    실제로 더 오래 걸리면 end_pct 근처에서 멈춰 기다린다(거짓으로 100%를 찍지 않기 위함).
    progress(message, pct, eta_seconds)로 남은 예상시간(초)도 함께 넘긴다."""
    done = threading.Event()
    t0 = time.time()

    def _emit() -> None:
        elapsed = time.time() - t0
        frac = min(0.95, elapsed / est_seconds) if est_seconds > 0 else 0.95
        eta = max(0.0, est_seconds - elapsed)
        _safe_progress(progress, message, start_pct + (end_pct - start_pct) * frac, eta)

    def _tick() -> None:
        while not done.wait(timeout=0.5):
            _emit()

    _emit()
    ticker = threading.Thread(target=_tick, daemon=True)
    ticker.start()
    try:
        return fn()
    finally:
        done.set()
        ticker.join(timeout=2.0)


# 백그라운드 다운로드 스레드 레지스트리. 하이라이트 '후보 뽑기'는 자막만 있으면 되고 영상
# 파일은 렌더 때에야 필요하므로, 분석 크리티컬 패스에서 몇 분짜리 다운로드를 빼고 여기서
# 병렬로 받는다. 렌더는 wait_for_download()로 완료를 보장받은 뒤 시작한다.
_bg_downloads: dict[str, threading.Thread] = {}
_bg_downloads_lock = threading.Lock()


def _start_download_bg(url: str, output_root: Path) -> None:
    try:
        download_video(url, output_root)
    except Exception:  # noqa: BLE001 - 실패해도 렌더 단계의 자기치유(재다운로드)가 다시 시도한다
        traceback.print_exc()


def _rlog(video_dir: Path, msg: str) -> None:
    """렌더 진단 로그. 자막 폴백 같은 중요한 분기가 조용히 지나가 원인 추적이 불가능했던
    사고(오디오-자막 어긋남)를 겪은 뒤 추가 — 어떤 자막 소스를 왜 썼는지 파일로 남긴다."""
    try:
        with (video_dir / "render_log.txt").open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except OSError:
        pass


def wait_for_download(video_id: str, timeout: float = 1800) -> None:
    """백그라운드 다운로드가 돌고 있으면 완료(또는 timeout)까지 기다린다."""
    with _bg_downloads_lock:
        t = _bg_downloads.get(video_id)
    if t is not None and t.is_alive():
        t.join(timeout)


# 단계별 실측 소요시간 기록. 진행바 ETA가 "고정 예상값"이라 실제와 어긋나던 문제를,
# 직전 실행들의 실측 중앙값으로 보정한다(같은 채널 설교는 길이·모델이 비슷해 잘 맞는다).
_STAGE_TIMES_PATH = Path("output") / "_stage_times.json"
_STAGE_TIMES_LOCK = threading.Lock()


def _stage_time_est(key: str, default: float) -> float:
    """최근 실측(최대 5회)의 중앙값 × 1.15(여유)를 예상시간으로 쓴다. 기록 없으면 default."""
    try:
        data = json.loads(_STAGE_TIMES_PATH.read_text(encoding="utf-8"))
        vals = sorted(float(v) for v in data.get(key, [])[-5:])
        if vals:
            return max(20.0, vals[len(vals) // 2] * 1.15)
    except (OSError, ValueError):
        pass
    return default


def _record_stage_time(key: str, seconds: float) -> None:
    try:
        with _STAGE_TIMES_LOCK:
            data = {}
            if _STAGE_TIMES_PATH.exists():
                data = json.loads(_STAGE_TIMES_PATH.read_text(encoding="utf-8"))
            data.setdefault(key, []).append(round(float(seconds), 1))
            data[key] = data[key][-10:]
            _STAGE_TIMES_PATH.parent.mkdir(parents=True, exist_ok=True)
            _STAGE_TIMES_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except (OSError, ValueError):
        pass  # 통계 기록 실패가 파이프라인을 죽이면 안 됨


def json_load_transcript(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            words=[Word(**w) for w in s["words"]],
        )
        for s in data["segments"]
    ]
    return {"language": data["language"], "duration_sec": data["duration_sec"], "segments": segments}


def reanalyze_clip_region(
    video_dir: Path,
    orig: Clip,
    cfg: dict,
    model: str = "",
    on_progress=None,
) -> Clip:
    """한 클립의 '구간만' 다시 분석해 경계(시작/끝)를 새로 잡은 새 Clip을 만든다.

    쓰임새: 주제·제목은 좋은데 AI가 시작/끝 지점을 어설프게 잡았을 때, 그 장면 주변
    전사만 모델에 다시 줘서 같은 메시지의 정확한 경계를 재선정한다. 원본은 호출자가
    그대로 두고, 반환된 새 Clip을 후보 목록에 '추가'한다.

    원본의 title/title_candidates/insight(=좋은 주제)는 보존하고, start/end/hook_line/
    payoff_line/appeal/점수만 재선정 결과로 교체한다."""
    h = cfg["highlights"]
    tdata = json_load_transcript(video_dir / "transcript.json")
    full_segs: list[Segment] = tdata["segments"]
    total_dur = float(tdata["duration_sec"])

    # 이 장면 주변만 창(window)으로 잘라 모델의 탐색을 그 섹터로 좁힌다.
    pad = 75.0
    w_start = max(0.0, orig.start - pad)
    w_end = min(total_dur, orig.end + pad)
    window_segs = [s for s in full_segs if s.end > w_start and s.start < w_end]
    if not window_segs:
        window_segs = full_segs

    window = Transcript(language=tdata["language"], duration_sec=w_end - w_start, segments=window_segs)

    # 재선정 지시(주제 고정 + 경계 재탐색)를 feedback_block 통로로 프롬프트에 주입한다.
    focus = (
        "## ★★ 이번 작업은 '구간 재분석'이다 (일반 선정과 다름 — 반드시 읽어라)\n"
        "아래 전사는 전체 설교가 아니라, 이미 고른 한 장면의 **주변만 잘라낸 토막**이다.\n"
        f"이 장면의 주제/제목은 이미 좋다고 확정됐다: 제목 「{orig.title}」"
        + (f", 인사이트 「{orig.insight}」" if orig.insight else "")
        + ".\n"
        f"문제는 경계다 — 기존 시작/끝({orig.start:.0f}~{orig.end:.0f}초)이 어설프게 잘렸다.\n"
        "**같은 메시지·같은 장면**을 담되, 시작은 앞 맥락 없이도 이해되는 완결된 문장에서,\n"
        "끝은 '생각이 완결되는' 펀치라인 직후에서 끊기도록 **경계를 다시 정확히 잡아라.**\n"
        "새 주제를 찾지 말고, 이 장면의 가장 좋은 컷 하나(1개)만 내라. hook_line/payoff_line은\n"
        "이 토막 전사에서 글자 그대로 인용해 경계를 확정한다.\n"
    )

    def _try(thinking_tokens: int) -> list[Clip]:
        return select_highlights_auto(
            transcript=window,
            peak_hints=[],
            min_clips=1,
            max_clips=1,
            min_duration_sec=h["min_duration_sec"],
            max_duration_sec=h["max_duration_sec"],
            categories=h["categories"],
            feedback_block=focus,
            model=model or h.get("model", ""),
            thinking_tokens=thinking_tokens,
            on_progress=on_progress,
        )

    # 전체 선정용 thinking_tokens(8192, 2026-09-02 품질 상향)를 그대로 물려받으면 "구간 하나만
    # 다시 보는" 가벼운 작업도 몇 분씩 걸린다(thinking 1k ≈ 20~30초) — 그래서 가벼운 예산으로
    # 먼저 시도한다. 하지만 짧고 애매한 구간에서는 이 예산으로 인용문 경계를 못 찾아 빈 결과가
    # 나오는 경우가 실측됐다("후보를 얻지 못했습니다" 실패). 실패하면 조용히 전체 선정과 같은
    # 예산으로 한 번 더 시도한다(느려도 성공이 우선) — 사용자에게 "왜 실패했냐"는 재요청을
    # 시키지 않는다.
    fast_tokens = int(h.get("reanalyze_thinking_tokens", 2048))
    clips = _try(fast_tokens)
    if not clips:
        full_tokens = int(h.get("thinking_tokens", 8192))
        if full_tokens > fast_tokens:
            if on_progress:
                on_progress(0.05, "짧은 예산으로 실패 — 더 깊게 재시도 중...")
            clips = _try(full_tokens)
    if not clips:
        raise RuntimeError("재분석에서 후보를 얻지 못했습니다 (모델 응답 비어있음/한도 가능성)")

    hard_len = float(h.get("hard_max_duration_sec", float(h["max_duration_sec"]) + 5.0))
    new = clips[0]
    try:
        if anchor_clip_to_quotes(new, full_segs, hard_len, min_duration_sec=float(h["min_duration_sec"])):
            new.anchored = True
    except Exception:  # noqa: BLE001 - 앵커 실패 시 모델이 준 숫자 경계로 폴백
        traceback.print_exc()

    # 좋은 주제(제목·후보·인사이트)는 원본 것을 보존하고, 경계·훅/페이오프·점수만 새것으로.
    new.title = orig.title
    new.title_candidates = list(orig.title_candidates or [])
    new.insight = orig.insight or new.insight
    # 사용자 스타일/위치 편집도 새 후보에 물려준다(같은 장면이니 그대로 쓰는 게 자연스럽다).
    for attr in (
        "fill_mode", "title_font", "title_size", "title_align", "title_spacing",
        "caption_font", "caption_size", "caption_align", "caption_spacing",
    ):
        setattr(new, attr, getattr(orig, attr, getattr(new, attr)))
    # 구간이 새로 잡혔으니 이전 자막·분할은 물려주지 않는다(렌더 때 재전사).
    new.caption_overrides = []
    new.keep_ranges = []
    new.trimmed = False
    return new


def _title_based_praise_clips(
    dl: "DownloadResult", video_dir: Path, title_lines: list[str], cfg: dict, model: str, sp,
) -> list[Clip]:
    """곡 제목(들)으로 정식 가사를 가져와 클립을 만든다(글자 수 균등 분배 자막, 즉시 반환).

    title_lines가 사용자가 직접 입력한 것이든(analyze의 title_lines 분기), 짧은 스니펫으로
    자동 추정된 것이든(_quick_guess_praise_title) 동일하게 처리한다 — 두 경로가 결과를
    합치는 지점. 가사를 하나도 못 찾으면 빈 리스트를 반환해 호출자가 폴백하게 한다.
    노래 속도에 맞춘 정밀 싱크는 여기서 하지 않는다 — 호출자가 clips.json을 저장한 뒤
    _sync_praise_clips_bg로 백그라운드 정밀 매핑을 시작해야 한다."""
    p = cfg.get("praise", {}) or {}
    # 가사 검색/자막 텍스트 확정은 '자막 실행' 단계 — 분석용 모델(opus)과 무관하게 항상
    # Sonnet 고정(2026-09-06, 사용자 결정: 분석은 Opus, 자막 실행은 Sonnet).
    lyrics_by_idx = fetch_praise_lyrics_by_titles(
        title_lines, model="claude-sonnet-4-5",
        thinking_tokens=int(p.get("lyrics_thinking_tokens", 2048)),
        on_progress=lambda frac, msg: sp.set_fraction(frac, msg),
    )
    clips = _build_praise_clips_from_titles(title_lines, lyrics_by_idx, dl.duration_sec)
    if not clips:
        return []
    missing = [title_lines[i] for i in range(len(title_lines)) if not lyrics_by_idx.get(i)]
    if missing:
        print(f"[praise] 가사를 못 찾은 곡(자막 없음): {', '.join(missing)}")
    # '우리 영상의 노래 속도'에 맞춰 가사를 띄우는 정밀 매핑(전사는 타이밍 전용, 가사
    # 텍스트는 정식 가사 유지)은 large-v3를 CPU로 돌려 곡 길이만큼(몇 분) 걸린다 — 이걸
    # 분석 진행바 안에서 기다리게 했더니 "링크든 파일이든 너무 오래 걸린다"는 재신고를
    # 받았다(2026-09-06). 지금 반환하는 clips는 이미 글자 수 균등 분배 자막을 갖고 있어
    # (_build_praise_clips_from_titles) 그대로 써도 자막은 나온다 — 정밀 매핑은
    # 백그라운드로 미뤄 완료되는 대로 clips.json만 갱신한다(호출자가 저장한 뒤 시작).
    return clips


def _quick_guess_praise_title(video_path: Path, duration_sec: float, cfg: dict, model: str, sp) -> str:
    """전체 영상을 통째로 정밀 전사하지 않고, 짧은 구간만 빠르게 훑어 찬양 제목을 추정한다.

    사용자 요청(2026-09-05): "그냥 제목만 파악해서 검색해서 자막 확보되잖아. 전체 전사하지
    말고." — 직접 녹화해 올리는 전형적인 케이스(찬양 한 곡, 보통 몇 분)를 대상으로 한다.
    8분을 넘는 녹화는 여러 곡이 섞인 예배 실황일 가능성이 커, 한 제목으로 특정하는 게
    무의미하므로 아예 시도하지 않는다(호출자가 기존 전체 분석 경로로 안전하게 폴백).

    시작 구간(필요하면 중간 구간도) 몇 십 초만 small 모델로 빠르게 전사해(vad 끔 — 노래도
    받아적어야 함) Claude에게 곡을 추정하게 한다. 확신 없으면 빈 문자열(호출자가 폴백)."""
    if duration_sec > 480:  # 8분 초과 — 다곡 예배 실황 가능성 커 추정 생략
        return ""
    w = cfg["whisper"]
    sp.message("찬양 제목 추정 중 (일부만 빠르게 확인)...")

    def _snip(a: float, b: float) -> str:
        try:
            segs = transcribe_clip_precise(
                video_path, a, b, model_size="small",
                device=w["device"], compute_type=w["compute_type"], language=w["language"],
                vad_filter=False, cpu_threads=int(w.get("cpu_threads", 0)),
                batch_size=int(w.get("batch_size", 8)), batched=True,
            )
            return " ".join((s.text or "") for s in segs)
        except Exception:  # noqa: BLE001 - 스니펫 전사 실패는 치명적이지 않음(추정 실패로 처리)
            return ""

    text = _snip(0.0, min(45.0, duration_sec))
    if duration_sec > 90 and len(text.split()) < 15:
        # 도입부가 조용하면(간주 등) 중간 구간도 한 번 더 훑어 재료를 보탠다.
        text += " " + _snip(duration_sec * 0.4, min(duration_sec, duration_sec * 0.4 + 30.0))
    if len(text.split()) < 8:
        return ""  # 노래가 아직 안 잡혔거나 너무 조용함 — 추정 불가, 폴백
    p = cfg.get("praise", {}) or {}
    title, conf = guess_praise_title_from_snippet(
        text, model=model or p.get("model", ""),
        thinking_tokens=int(p.get("lyrics_thinking_tokens", 1024)),
    )
    return title if title and conf >= 6 else ""


def _prewarm_praise_sync_cache(video_path: Path, video_dir: Path, clips: list[Clip], cfg: dict) -> None:
    """백그라운드 예열: 찬양 클립들의 '타이밍 전사'(정밀 모델)를 미리 돌려 캐시한다.

    싱크 맞추기 버튼이 눌리는 시점에 캐시가 이미 있으면 몇 분 걸리던 게 즉시가 된다
    (사용자 신고: "싱크 맞추기 해도 시간이 너무 오래 걸리는데 더 빨리는 못하나" —
    CPU뿐인 환경이라 전사 자체를 빠르게 할 수는 없고, 미리 해두는 게 정답).
    데몬 스레드라 분석 응답을 막지 않고, 실패해도 조용히 넘어간다(버튼이 그때 전사)."""
    def _run():
        w = cfg["whisper"]
        model = w.get("precise_model_size", "large-v3")
        for c in clips:
            if getattr(c, "clip_type", "") != "praise":
                continue
            try:
                if _precise_cache_find(video_dir / "precise_cache", model, "praisesync2", c.start, c.end):
                    continue
                segs = transcribe_clip_precise(
                    video_path, c.start, c.end,
                    model_size=model, device=w["device"], compute_type=w["compute_type"],
                    language=w["language"], vad_filter=False,
                    cpu_threads=int(w.get("cpu_threads", 0)),
                    batch_size=int(w.get("batch_size", 8)), batched=True,
                )
                _precise_cache_save(video_dir / "precise_cache", model, "praisesync2", c.start, c.end, segs)
                print(f"[praise-sync] 예열 캐시 저장: {c.start:.0f}~{c.end:.0f}초")
            except Exception:  # noqa: BLE001 - 예열 실패는 치명적이지 않음
                traceback.print_exc()
    threading.Thread(target=_run, daemon=True).start()


def _sync_praise_clips_bg(video_path: Path, video_dir: Path, clips: list[Clip], cfg: dict) -> None:
    """제목 기반 찬양 클립의 '노래 속도 정밀 싱크'를 백그라운드에서 계산해 clips.json에 반영한다.

    _title_based_praise_clips가 반환하는 clips는 이미 글자 수 균등 분배 자막을 갖고 있어
    바로 써도 되지만(자막은 나온다), 노래 속도에 정확히 맞추려면 large-v3 정밀 전사가
    필요하고 CPU에서 곡 길이만큼(몇 분) 걸린다. 이걸 analyze()가 기다리게 했더니 "링크든
    파일이든 너무 오래 걸린다"는 재신고를 받았다(2026-09-06) — 그래서 저장은 먼저 하고
    정밀 매핑은 여기서 완료되는 대로 clips.json만 갱신한다.

    갱신 시 (start,end) 서명으로 대상 클립을 다시 찾는다 — 그 사이 사용자가 편집기에서
    클립 경계를 바꿨으면(서명 불일치) 그 클립은 건드리지 않아, 편집 중인 내용을 백그라운드
    결과가 덮어쓰는 경합을 피한다."""
    clips_path = video_dir / "clips.json"
    sigs = [(round(c.start, 2), round(c.end, 2)) for c in clips]
    texts_by_sig = {
        sig: [o["text"] for o in (c.caption_overrides or []) if o.get("text")]
        for sig, c in zip(sigs, clips)
    }

    def _run():
        # 싱크 전사 모델은 medium(praise_model_size)이 정답이다. 이 경로는 단어의 '시각'과
        # 자모 유사도만 쓰고 텍스트 품질은 안 쓰는데, large-v3는 같은 곡에서 3배 넘게
        # 느리면서 단어를 오히려 덜 잡았다(실측 2026-09-07, 124초 찬양 클립:
        # large-v3 순차 223초/63단어 vs medium 순차 68초/73단어). 배치+VAD는 15초로 빠르지만
        # VAD가 노래를 발화로 안 봐 단어가 1개까지 떨어져 못 쓴다(같은 실측).
        precise_model = (
            cfg["whisper"].get("praise_model_size")
            or cfg["whisper"].get("model_size", "medium")
        )
        # 이 정밀 전사는 사용자가 이미 편집기를 보고 있는 동안 도는 백그라운드 작업이다.
        # cpu_threads를 config 그대로(=코어 수) 쓰면 large-v3가 전 코어를 잡아, 편집기의
        # 미리보기/렌더/파형이 전부 굼떠진다("분석이 끝났는데도 계속 느리다", 2026-09-07).
        # 코어 절반만 쓰게 해 UI 응답성을 지킨다 — 싱크는 몇 분 늦어도 자막은 이미 나온다.
        w = dict(cfg["whisper"])
        w["cpu_threads"] = max(1, (os.cpu_count() or 4) // 2)
        for sig in sigs:
            texts = texts_by_sig.get(sig)
            if not texts:
                continue
            start, end = sig
            t_sync = time.time()
            try:
                mapped = map_lines_to_voice_times(
                    video_path, start, end, texts,
                    w, cache_dir=video_dir / "precise_cache",
                    model_override=precise_model,
                )
            except Exception:  # noqa: BLE001 - 백그라운드 싱크 실패해도 균등 분배 자막은 유지
                traceback.print_exc()
                continue
            if not mapped:
                print(f"[praise-sync] 곡 {start:.0f}~{end:.0f}초 노래 속도 매핑 실패(단어 부족) → 균등 분배 유지")
                continue
            with CLIPS_LOCK:
                if not clips_path.exists():
                    continue
                cur = load_clips_json(clips_path)
                for cc in cur:
                    if (round(cc.start, 2), round(cc.end, 2)) == sig:
                        cc.caption_overrides = mapped
                        save_clips_json(cur, clips_path)
                        print(f"[praise-sync] 곡 {start:.0f}~{end:.0f}초 정밀 싱크 반영 ({time.time() - t_sync:.0f}초 소요)")
                        break
    threading.Thread(target=_run, daemon=True).start()


def _vocal_band_envelope_db(video_path: Path, t0: float, t1: float) -> tuple[list[float], float]:
    """[t0,t1] 구간의 '목소리 대역(300~3400Hz)' 에너지 곡선(dB, 10ms 간격)을 돌려준다.
    반환: (dB 리스트, 프레임 간격 초). 실패하면 ([], 0.01)."""
    import subprocess

    import numpy as np
    from scipy import signal as _sig

    sr = 16000
    dur = max(0.0, t1 - t0)
    if dur <= 0.2:
        return [], 0.01
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, t0):.3f}", "-t", f"{dur:.3f}", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) < sr:
        return [], 0.01
    y = np.frombuffer(proc.stdout[: len(proc.stdout) - (len(proc.stdout) % 4)], dtype=np.float32)
    b, a = _sig.butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band")
    yv = _sig.lfilter(b, a, y).astype(np.float32)
    hop, win = 160, 400  # 10ms / 25ms
    n = max(0, (len(yv) - win) // hop + 1)
    if n < 10:
        return [], 0.01
    idx = np.arange(win)[None, :] + (np.arange(n) * hop)[:, None]
    rms = np.sqrt(np.mean(yv[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-6)
    db = np.convolve(db, np.ones(9) / 9.0, mode="same")  # 90ms 평활(음절 단위 요동 억제)
    return db.tolist(), hop / sr


def _refine_onsets_by_audio(
    video_path: Path, clip_start: float, clip_end: float,
    estimates: list[tuple[float, float, float, float]],
) -> list[float | None]:
    """각 소절의 추정 시작(estimates: (추정초, 창 앞 여유, 창 뒤 여유, 기대 오프셋))을
    실제 '발성 시작점'으로 정밀 보정한다.

    원리(실측 2026-09-06, 업로드 찬양 8소절): 회중 찬양은 소절 사이에 숨 쉬는 구간이
    있어 목소리 대역 에너지에 뚜렷한 '골'(주변 대비 5~11dB)이 파이고, 다음 소절은 그
    골 직후 급격한 상승으로 시작한다. whisper(large-v3)는 앞 소절 끝음(늘여 부르는
    음)을 다음 단어에 붙이는 버릇이 있어 단어 시작을 0.7~1.15초 이르게 찍는데(분산이
    커 고정 오프셋으론 ±0.5초가 남음), 골 직후 상승점을 잡으면 그 오차가 사라진다.

    각 추정마다 [추정-앞여유, 추정+뒤여유] 창 안의 골(국소 최소, 깊이≥4dB)들을 찾고,
    골 다음 에너지가 '골 깊이의 60%'를 회복하는 시점을 발성 시작 후보로 삼는다. 후보가
    여럿이면 '골 깊이 − 2×|후보−(추정+기대오프셋)|' 점수가 큰 쪽(더 깊은 골, 기대 위치에
    가까운 쪽). 골이 없으면 None(호출자가 예전 방식 폴백).
    반환: 추정별 보정된 절대 시각 또는 None."""
    import os as _os

    if not estimates:
        return []
    pad = 3.5
    a0 = max(0.0, clip_start - pad)
    db, dt = _vocal_band_envelope_db(video_path, a0, clip_end + pad)
    if not db:
        return [None] * len(estimates)
    n = len(db)

    def _fi(t: float) -> int:
        return int(round((t - a0) / dt))

    out: list[float | None] = []
    MIN_DEPTH = 4.0
    for est, back, fwd, exp_off in estimates:
        lo = max(0, _fi(est - back))
        hi = min(n - 1, _fi(est + fwd))
        ctx0, ctx1 = max(0, _fi(est - 3.0)), min(n, _fi(est + 3.0))
        if hi - lo < 5 or ctx1 - ctx0 < 10:
            out.append(None)
            continue
        ctx = sorted(db[ctx0:ctx1])
        level = ctx[len(ctx) // 2]  # 중앙값 = '노래하는 중' 수준
        best: tuple[float, float] | None = None  # (score, onset_abs)
        rad = max(1, int(round(0.15 / dt)))
        i = lo
        while i <= hi:
            v = db[i]
            depth = level - v
            if depth >= MIN_DEPTH and v <= min(db[max(0, i - rad): i + rad + 1]):
                # 골 → 회복점(골 깊이의 60% 회복) 탐색(골에서 0.8초 이내)
                # 회복은 '지속'되어야 한다: 숨 쉬는 구간 안의 잔물결(두 개의 작은 골)을
                # 발성으로 오인하지 않게, 회복점 뒤 0.15초 동안 평균이 임계 이상이어야 채택.
                thr = v + 0.6 * depth
                hold = max(1, int(round(0.15 / dt)))
                j = i + 1
                j_max = min(n - 1, i + int(round(0.9 / dt)))
                found = False
                while j <= j_max:
                    if db[j] >= thr and sum(db[j: j + hold]) / len(db[j: j + hold]) >= thr:
                        found = True
                        break
                    j += 1
                if found:
                    onset = a0 + j * dt
                    # 골의 '너비'(깊이의 절반보다 깊은 구간 길이): 소절 사이 숨 구간은
                    # 0.4~0.8초로 넓고, 소절 안 음절 사이 요동은 0.1~0.2초로 좁다 — 깊이가
                    # 비슷한 두 골이 경합할 때(실측: 소절 첫 단어가 오인식돼 창이 둘째
                    # 단어부터 시작한 경우) 넓은 쪽이 진짜 소절 경계다.
                    half = v + 0.5 * depth
                    wl = i
                    while wl > 0 and db[wl - 1] <= half:
                        wl -= 1
                    wr = i
                    while wr < n - 1 and db[wr + 1] <= half:
                        wr += 1
                    width = min(1.0, (wr - wl + 1) * dt)
                    score = depth + 4.0 * width - 2.0 * abs((onset - est) - exp_off)
                    if _os.environ.get("PRAISE_SYNC_DEBUG"):
                        print(f"      cand est={est:.2f} dip@{a0 + i * dt:.2f} depth={depth:.1f} width={width:.2f} onset={onset:.2f} score={score:.2f}")
                    if best is None or score > best[0]:
                        best = (score, onset)
                i = max(i + 1, j)  # 같은 골을 중복 탐색하지 않게 회복점 뒤로 건너뜀
                continue
            i += 1
        out.append(round(best[1], 2) if best else None)
    return out


def map_lines_to_voice_times(
    video_path: Path,
    clip_start: float,
    clip_end: float,
    texts: list[str],
    whisper_cfg: dict,
    cache_dir: Path | None = None,
    model_override: str = "",
) -> list[dict] | None:
    """가사 줄(texts)을 클립의 실제 '가창 단어 시각'에 비례로 매핑해 caption_overrides를 만든다.

    핵심: 전사는 '언제 노래하는지(타이밍)'에만 쓰고, 자막 '텍스트'는 정식 가사(texts) 그대로
    둔다 — 노래 오인식 문제를 피하면서도 '우리 영상의 노래 속도'에 맞춰 가사가 뜨게 한다
    (사용자 요청 2026-09-05). 단어 내용이 틀려도 발화 시각 진행은 맞으므로 인트로/간주를
    자연히 건너뛴다.

    모델은 small이 아니라 whisper.model_size(기본 medium)를 쓴다 — 실측(2026-09-05):
    small은 30초 분량을 통째로 놓치는 등 단어 밀도가 너무 낮아(0.47개/초) 퍼지 앵커링이
    앵커를 거의 못 잡고 "싱크가 헛돈다"는 재신고로 이어졌다. 타이밍 전용이라도 인식
    자체가 부실하면 앵커링의 기반이 무너지므로, 이 경로만은 정확도를 속도보다 우선한다.

    반환: [{start,end,text}, ...] (절대초). 전사 단어가 너무 적으면 None(호출자가 폴백)."""
    texts = [str(t).strip() for t in texts if str(t).strip()]
    if not texts:
        return None
    model = (
        model_override
        or whisper_cfg.get("praise_model_size")
        or whisper_cfg.get("model_size", "medium")
    )
    sig = "praisesync2"  # v1(small 모델) 캐시와 섞이지 않게 시그니처 분리(모델명은 파일명에 포함됨)
    segs = None
    if cache_dir is not None:
        # 정밀 모델(large-v3) 캐시가 이미 있으면 무조건 그걸 쓴다 — 분석 단계에서 미리
        # 정밀 전사를 돌려두므로(아래 _title_based_praise_clips), 싱크 맞추기 버튼은
        # 전사 없이 즉시+최고 정확도로 동작한다("정밀 재분석은 너무 오래 걸림" 해결).
        precise = whisper_cfg.get("precise_model_size", "large-v3")
        if precise != model:
            segs = _precise_cache_find(cache_dir, precise, sig, clip_start, clip_end)
        if segs is None:
            segs = _precise_cache_find(cache_dir, model, sig, clip_start, clip_end)
    if segs is None:
        segs = transcribe_clip_precise(
            video_path, clip_start, clip_end,
            model_size=model, device=whisper_cfg["device"], compute_type=whisper_cfg["compute_type"],
            language=whisper_cfg["language"], vad_filter=False,
            cpu_threads=int(whisper_cfg.get("cpu_threads", 0)),
            batch_size=int(whisper_cfg.get("batch_size", 8)), batched=True,
        )
        if cache_dir is not None:
            try:
                _precise_cache_save(cache_dir, model, sig, clip_start, clip_end, segs)
            except Exception:  # noqa: BLE001
                pass
    words = sorted(
        (
            (float(wd.start), float(wd.end), str(wd.text or ""))
            for s in segs for wd in s.words
            if clip_start <= float(wd.start) <= clip_end
        ),
        key=lambda w: w[0],
    )
    if len(words) < max(2, len(texts) // 3):
        return None  # 가창 단어가 거의 안 잡힘 → 폴백(글자수 균등 분배)

    # ── 소절별 퍼지 앵커링 ─────────────────────────────────────────────────────
    # 예전 글자수 비례 매핑은 "가사 진행 = 가창 단어 진행"을 가정하는데, 간주(단어 공백)와
    # 후렴 반복·전사 환각(같은 구절 반복 인식)에서 무너진다("실제 노래랑 다르다" 실신고).
    # 대신 각 소절 텍스트를 전사 단어열과 순차 퍼지 매칭(difflib)해 '실제로 그 소절을 부른
    # 위치'에 앵커한다 — whisper가 노래를 오인식해도 음절 일부는 비슷하게 받아적으므로
    # 낮은 임계값(0.42)이면 대부분 잡힌다. 앵커 안 된 소절은 이웃 앵커 사이를 글자수
    # 비례로 보간한다(전부 실패하면 예전 비례 방식으로 전체 폴백).
    import difflib
    import re as _re

    def _norm(s: str) -> str:
        return _re.sub(r"[^0-9가-힣a-zA-Z]", "", s or "")

    wnorm = [_norm(w[2]) for w in words]
    n = len(words)

    # ── 전역 최적 정렬(DP) 앵커링 ────────────────────────────────────────────
    # 순차 탐욕 매칭의 구조적 실패(실측): 후렴 반복곡에서 소절4("예수 늘 함께 하시네")가
    # 첫 후렴(오인식 심함, 0.6점) 대신 끝 후렴(깨끗, 0.93점)에 붙어, 사이 소절들("후회도
    # 염려도...")이 매칭 기회 자체를 잃고 꼬리에 뭉개졌다 — "이 2개가 왜 맨 뒤에 가 있냐"
    # 실신고. 전역 정렬은 '전체 소절의 매칭 총점'을 최대화하므로, 소절4를 첫 후렴에 앉혀
    # 5·6번까지 살리는 배치(0.6+0.45+0.93)가 하나만 잘 맞는 배치(0.93)를 이긴다.
    # 1) 소절별 후보 창 수집(임계 0.35 — 전역 제약이 오탐을 걸러주므로 낮춰도 안전).
    cands_per_line: list[list[tuple[float, int, int]]] = []
    for text in texts:
        target = _norm(text)
        cl: list[tuple[float, int, int]] = []
        if target:
            for i in range(n):
                acc = ""
                for j in range(i, min(i + 25, n)):
                    acc += wnorm[j]
                    if len(acc) > len(target) * 2 + 10:
                        break
                    sc = difflib.SequenceMatcher(None, acc, target).ratio()
                    # 커버리지 보정: 소절 일부(뒷부분만)에 붙은 짧은 창이 전체 창보다 근소하게
                    # 높은 점수를 받아 앵커 '시작'이 소절 중간(3초 뒤)에 찍히는 실측 사례
                    # ("그 무덤의 권세 다 깨뜨렸네"가 '방세 다 깨끄렸네'에 앵커). 창이 소절
                    # 길이의 75% 미만이면 그 비율만큼 깎아 온전한 창을 우선한다.
                    if len(acc) < 0.75 * len(target):
                        sc *= len(acc) / (0.75 * len(target))
                    if sc >= 0.35:
                        cl.append((sc, i, j))
        cl.sort(reverse=True)
        cands_per_line.append(cl[:40])  # 상위 40개면 충분(성능)
    # 2) DP: 소절 순서 = 단어 순서 제약 하에 (매칭 점수 + 소절당 보너스) 총합 최대 배치.
    #    보너스는 '한 소절이라도 더 실측에 앉히는' 배치를 선호하게 한다. 상태는 마지막
    #    사용 단어 인덱스별 최고점만 유지(가지치기).
    MATCH_BONUS = 0.25
    # 템포 일관성 항(2026-09-06 추가): 노래는 한 곡 안에서 '글자당 초'가 대체로 일정한데,
    # 문자열 점수만으로는 앞 소절 꼬리(whisper가 늘인 끝음을 엉뚱한 단어로 받아적은 것)에
    # 다음 소절이 근소한 차이로 앉아 그 소절 길이가 3.8초, 다음이 10.8초처럼 뒤틀렸다
    # (실측: "기뻐 찬송하세 밝은 빛이 왔네"가 4.5초 이르게). 1차 DP로 곡의 중앙값 글자당
    # 초를 재고, 2차 DP에서 이웃 앵커 간 속도가 중앙값에서 벗어나는 만큼(로그 비율×0.5)
    # 깎아 '박자에 맞는' 배치를 고른다. 앵커가 1개면 중앙값이 없어 1차 결과 그대로.
    RATE_W = 0.5
    _chars = [max(1, len(_norm(t))) for t in texts]

    def _anchor_t(li: int, i: int) -> float:
        # 첫 소절은 whisper가 여린 도입부를 놓쳐 늦게 찍는 버릇이 있어 아래 '첫 소절 스냅'
        # 규칙(클립 시작 6초 이내면 클립 시작)과 같은 가정으로 속도를 잰다 — 안 그러면
        # 첫 구간 속도가 실제보다 빨라 보여 2소절 앵커가 뒤쪽 창으로 밀린다(실측).
        t = words[i][0]
        if li == 0 and (t - clip_start) <= 6.0:
            return clip_start
        return t

    def _run_dp(median_rate: float | None) -> list:
        import math as _math

        states: list[tuple[int, float, list]] = [(-1, 0.0, [])]  # (last_j, 총점, [(li,i,j)])
        for li in range(len(texts)):
            new_states = list(states)  # 이 소절을 매칭 안 하는 선택지(상태 유지)
            for last_j, tot, path in states:
                for sc, i, j in cands_per_line[li]:
                    if i <= last_j:
                        continue
                    pen = 0.0
                    if path:
                        pl, pi, _pj = path[-1]
                        span = sum(_chars[pl:li])
                        rate = (words[i][0] - _anchor_t(pl, pi)) / span
                        if rate < 0.08:
                            continue  # 소절을 물리적으로 부를 수 없는 속도 → 불가
                        if median_rate:
                            pen = RATE_W * abs(_math.log(rate / median_rate))
                    new_states.append((j, tot + sc + MATCH_BONUS - pen, path + [(li, i, j)]))
            by_last: dict[int, tuple[int, float, list]] = {}
            for st in new_states:
                if st[0] not in by_last or st[1] > by_last[st[0]][1]:
                    by_last[st[0]] = st
            states = sorted(by_last.values(), key=lambda s: -s[1])[:60]
        return max(states, key=lambda s: s[1])[2]

    best_path = _run_dp(None)
    if len(best_path) >= 3:
        rates = []
        for (pl, pi, _), (cl, ci, _) in zip(best_path, best_path[1:]):
            rates.append((words[ci][0] - _anchor_t(pl, pi)) / sum(_chars[pl:cl]))
        rates.sort()
        best_path = _run_dp(rates[len(rates) // 2])

    # 창 가장자리 정리: 퍼지 창은 앞 소절의 마지막 단어나 다음 소절의 첫 단어를 곧잘
    # 물고 들어온다(실측: "주님|할렐루야 찬송하시" — '주님'은 앞 소절). 그러면 앵커
    # 시작이 1.2초 이르고 오디오 보정 창 밖으로 벗어난다. 실제 매칭된 글자가 하나도
    # 없는(또는 2자 미만 걸친 다음절) 가장자리 단어를 잘라낸다.
    def _jamo(txt: str) -> str:
        # 음절 → 초성·중성·종성 자모열. 노래 오인식은 음소 단위로 비슷한 경우가 많아
        # ("알레르기야"≈"할렐루야") 음절 비교로는 0자 일치인데 자모로는 절반이 맞는다.
        out = []
        for ch in txt:
            o = ord(ch)
            if 0xAC00 <= o <= 0xD7A3:
                o -= 0xAC00
                out.append(chr(0x1100 + o // 588))
                out.append(chr(0x1161 + (o % 588) // 28))
                if o % 28:
                    out.append(chr(0x11A7 + o % 28))
            else:
                out.append(ch)
        return "".join(out)

    def _trim_edges(li: int, i: int, j: int) -> tuple[int, int]:
        target = _jamo(_norm(texts[li]))
        parts = [_jamo(wnorm[k]) for k in range(i, j + 1)]
        acc = "".join(parts)
        if not acc or not target:
            return i, j
        matched = [False] * len(acc)
        for blk in difflib.SequenceMatcher(None, acc, target, autojunk=False).get_matching_blocks():
            for k in range(blk.a, blk.a + blk.size):
                matched[k] = True
        spans = []
        pos = 0
        i0 = i
        for part in parts:
            spans.append((pos, pos + len(part)))
            pos += len(part)

        def _keep(k: int) -> bool:
            # 자모의 절반 이상이 정식 가사와 정렬되면 그 소절의 단어로 본다.
            a, b = spans[k - i0]
            if b <= a:
                return False
            m = sum(1 for q in range(a, b) if matched[q])
            return m * 2 >= (b - a)

        while i < j and not _keep(i):
            i += 1
        while j > i and not _keep(j):
            j -= 1
        return i, j

    anchors: dict[int, tuple[int, int, float, float]] = {}
    import os as _os
    _dbg = bool(_os.environ.get("PRAISE_SYNC_DEBUG"))
    for (li, i, j) in best_path:
        i2, j2 = _trim_edges(li, i, j)
        if _dbg:
            print(f"    path li={li} ({i},{j}) '{' '.join(w[2] for w in words[i:j+1])}' -> ({i2},{j2}) @{words[i2][0]:.2f} | {texts[li]}")
        i, j = i2, j2
        anchors[li] = (i, j, words[i][0], max(words[j][1], words[i][0] + 0.3))

    # 앵커 밀집 정리: 두 앵커 '시작' 사이 간격이 그 사이 소절 수(앞 앵커 소절 포함)를
    # 부르기에 필요한 시간보다 좁으면, 그 구간은 인식 부실/후렴 반복 오탐일 가능성이 크다
    # (실측1: 4소절이 1.2초 안에 욱여넣어짐. 실측2: 후렴 반복곡에서 마지막 두 소절이
    # 1.4~1.6초로 뭉개짐 — "자막이 안 나온다" 체감 신고). 노래에서 한 소절이 1.5초
    # 미만일 수는 사실상 없으므로 기준을 1.5초로 둔다. 좁으면 뒤 앵커를 버리고 재보간.
    # (예전엔 앞 앵커의 '끝'을 기준으로 재서, 창이 늘인 끝음까지 물면 멀쩡한 이웃 앵커를
    # 버렸다 — "그 어둠의 권세" 0.82점 앵커가 그렇게 사라졌다. 시작끼리만 비교한다.)
    MIN_SEC_PER_LINE = 1.5
    idxs = sorted(anchors.keys())
    k = 1
    while k < len(idxs):
        prev_i, cur_i = idxs[k - 1], idxs[k]
        gap_lines = cur_i - prev_i  # 앞 앵커 소절 + 그 사이 미앵커 소절 수
        gap_sec = anchors[cur_i][2] - anchors[prev_i][2]
        if gap_sec < gap_lines * MIN_SEC_PER_LINE:
            del anchors[cur_i]
            idxs.pop(k)
            continue  # k는 그대로 두고 다음(당겨진) 항목과 다시 비교
        k += 1

    # (예전의 '2차 좁은 구간 재매칭'은 DP 전역 정렬이 대체한다 — 임계 0.35 후보를 순서
    # 제약과 총점 최대화로 배치하므로, 좁은 구간의 낮은 점수 매칭도 전역 최적이면 채택된다.)

    starts: list[float] = [0.0] * len(texts)
    if anchors:
        idxs = sorted(anchors.keys())
        # (1) 앵커 소절: whisper 단어 시작을 오디오 '발성 시작점'으로 정밀 보정.
        #     whisper는 앞 소절의 늘인 끝음을 다음 단어에 붙여 0.7~1.15초 이르게 찍으므로
        #     창은 [-0.5, +1.5], 기대 오프셋 +0.6. 골을 못 찾은 앵커만 예전 고정 +0.7초
        #     (2026-09-05 "자막이 1초 정도 빠르다" 피드백으로 정한 값) 폴백.
        est = [(anchors[li][2], 0.5, 1.5, 0.6) for li in idxs]
        try:
            refined = _refine_onsets_by_audio(video_path, clip_start, clip_end, est)
        except Exception as e:  # noqa: BLE001
            print(f"[praise-sync] 오디오 온셋 보정 실패({e}) → 고정 오프셋 폴백")
            refined = [None] * len(idxs)
        n_ref = 0
        for li, r in zip(idxs, refined):
            if r is not None:
                starts[li] = r
                n_ref += 1
            else:
                starts[li] = anchors[li][2] + 0.7
        # (2) 미앵커 소절: 이웃 앵커 '시작'끼리 사이를 글자수 비례로 보간한다. 예전엔
        #     앞 앵커의 '끝'에서 시작했는데, 퍼지 매칭이 소절 앞부분만 잡으면(오인식이
        #     심한 뒷부분 탈락) 그 끝이 실제보다 수 초 이르고 다음 소절도 그만큼 일찍
        #     떴다(실측: 4.7초 조기). 앵커 시작은 (1)로 정밀하므로 그것만 기준으로 삼는다.
        interp: list[int] = []
        interp_win: dict[int, float] = {}
        for li in range(len(texts)):
            if li in anchors:
                continue
            prev_a = max((k for k in idxs if k < li), default=None)
            next_a = min((k for k in idxs if k > li), default=None)
            t0 = starts[prev_a] if prev_a is not None else clip_start
            t1 = starts[next_a] if next_a is not None else clip_end
            lo = prev_a if prev_a is not None else 0
            hi = next_a if next_a is not None else len(texts)
            span_chars = sum(max(1, len(texts[k])) for k in range(lo, hi)) or 1
            cum = sum(max(1, len(texts[k])) for k in range(lo, li))
            starts[li] = t0 + (t1 - t0) * (cum / span_chars)
            interp.append(li)
            # 양쪽 앵커 사이에 홀로 낀 소절은 보간 오차가 작으므로 창을 ±0.6초로 좁힌다 —
            # 넓히면 앞 소절의 늘인 끝음 속 잔골(실측: 1.1초 앞)을 잡는다. 여럿이 끼거나
            # 한쪽이 클립 경계면 ±1.0초.
            both = prev_a is not None and next_a is not None
            interp_win[li] = 0.6 if (both and hi - lo <= 2) else 1.0
        # (3) 보간 소절도 오디오 골로 보정(보간은 편향이 없으니 기대 오프셋 0).
        if interp:
            try:
                refined2 = _refine_onsets_by_audio(
                    video_path, clip_start, clip_end,
                    [(starts[li], interp_win[li], interp_win[li], 0.0) for li in interp],
                )
            except Exception:  # noqa: BLE001
                refined2 = [None] * len(interp)
            for li, r in zip(interp, refined2):
                if r is not None:
                    starts[li] = r
                    n_ref += 1
        print(f"[praise-sync] 앵커 {len(anchors)}/{len(texts)}소절, 오디오 온셋 보정 {n_ref}/{len(texts)}")
    else:
        # 전부 매칭 실패(전사가 심하게 뭉개짐) → 예전 글자수 비례 방식 폴백(+0.7초 리드 보정).
        times = [w[0] for w in words]
        total_chars = sum(max(1, len(t)) for t in texts) or 1
        cum = 0
        for li, t in enumerate(texts):
            starts[li] = times[min(n - 1, int(cum / total_chars * n))] + 0.7
            cum += max(1, len(t))
        print("[praise-sync] 퍼지 앵커 0개 → 글자수 비례 폴백")
    starts = [max(clip_start, s) for s in starts]

    # ── 안 부른 소절 드랍 ────────────────────────────────────────────────────
    # 영상이 항상 곡 처음부터 시작하지는 않는다(후렴부터 찍었거나, 끝까지 안 부르고 끊김).
    # 그런데 정식 가사는 1절부터 끝까지 전부라, 안 부른 앞/뒤 소절들이 첫 앵커 앞·마지막
    # 앵커 뒤의 좁은 공백에 욱여넣어져 '부르지도 않은 가사'가 떴다(실신고 2026-09-06:
    # "영상이 항상 처음부터 시작하지는 않아 — 유도리 있게 날릴 건 날려야지").
    # 판정: 첫 앵커 앞 공백이 그 앞 미앵커 소절들을 실제로 부르기에 물리적으로 부족하면
    # (소절당 MIN_SEC_PER_LINE=1.5초 기준) 공백에 안 들어가는 만큼 앞에서부터 버린다.
    # 꼬리도 동일. 앵커가 하나도 없으면(전사 뭉개짐) 판단 근거가 없으니 그대로 둔다.
    dropped: set[int] = set()
    if anchors:
        first_a = min(anchors.keys())
        head_gap = anchors[first_a][2] - clip_start
        if first_a > 0 and head_gap < first_a * MIN_SEC_PER_LINE:
            n_fit = max(0, int(head_gap // MIN_SEC_PER_LINE))
            dropped.update(range(0, first_a - n_fit))
        last_a = max(anchors.keys())
        tail_lines = len(texts) - 1 - last_a
        tail_gap = clip_end - anchors[last_a][3]
        if tail_lines > 0 and tail_gap < tail_lines * MIN_SEC_PER_LINE:
            n_fit = max(0, int(tail_gap // MIN_SEC_PER_LINE))
            dropped.update(range(last_a + 1 + n_fit, len(texts)))
    keep = [i for i in range(len(texts)) if i not in dropped]
    if dropped:
        print(f"[praise-sync] 안 부른 소절 {len(dropped)}개 드랍 (영상이 곡 중간부터 시작/중간에 끝)")
    if not keep:
        return None

    # 첫 소절 스냅: 찬양 클립은 이미 '노래 시작' 기준으로 잘려 있다(제목 기반 업로드는
    # 0초=노래 시작, 자동 감지 곡도 전주 패딩 ~4초뿐). whisper는 노래의 여린 도입부를
    # 몇 초 놓치고 첫 단어를 늦게 찍는 버릇이 있어, 첫 소절이 4~5초 늦게 뜨는 신고가
    # 났다("0초부터 노래 나오는데 왜 4초부터로 나오지"). (드랍 후 남은) 첫 소절 시작이
    # 클립 시작에서 6초 이내면 인식 지연으로 보고 클립 시작으로 당긴다(진짜 긴 전주면 그대로 둠).
    if (starts[keep[0]] - clip_start) <= 6.0:
        starts[keep[0]] = clip_start

    out: list[dict] = [
        {"start": round(starts[i], 2), "end": 0.0, "text": texts[i]} for i in keep
    ]
    # 단조 증가 보정.
    for i in range(len(out)):
        if out[i]["start"] < clip_start:
            out[i]["start"] = round(clip_start, 2)
        if i > 0 and out[i]["start"] <= out[i - 1]["start"]:
            out[i]["start"] = round(out[i - 1]["start"] + 0.3, 2)
    # 끝 시각: 예전 '무조건 다음 소절 시작까지'는 간주(다음 소절까지 30~40초 공백)에서
    # 한 소절이 40초씩 떠 있는 사고를 냈다("한 소절이 8초부터 48초야" 실신고). 이제
    #   - 앵커된 소절: 실제로 부른 끝(anchor end) + 2초 여유까지만.
    #   - 미앵커 소절: 글자 수 기반 최대 노출(초당 1자 + 3초, 최소 5초)까지만.
    # 단, 다음 소절이 그보다 먼저 시작하면 거기서 끊는다(연속 가창은 기존처럼 이어짐).
    # 상한을 넘는 나머지 구간(간주)엔 자막이 꺼진다 — 노래 없는데 가사가 떠 있지 않게.
    #   - 앵커 소절의 '부른 끝'은 매칭 창의 끝이 아니라, 그 창에서 이어지는 whisper
    #     단어 사슬(단어 사이 공백 1초 미만)의 끝으로 본다: 퍼지 창이 소절 앞부분만
    #     잡으면(뒷부분 오인식) 창 끝이 실제보다 수 초 이르고, 노래가 계속되는데 자막이
    #     먼저 꺼졌다(실측: "할렐루야 찬송하세 다시 사셨도다"가 2.5초 일찍 사라짐).
    #     간주에선 whisper 단어가 끊기므로 사슬이 거기서 멈춘다.
    def _voiced_until(j: int) -> float:
        t_end = words[j][1]
        k = j + 1
        while k < n and words[k][0] - t_end < 1.0:
            t_end = max(t_end, words[k][1])
            k += 1
        return t_end

    for i in range(len(out)):
        oi = keep[i]  # 드랍 이후 out 위치 → 원래 소절 인덱스(anchors/texts 키)
        nxt = out[i + 1]["start"] if i + 1 < len(out) else clip_end
        if oi in anchors:
            cap = _voiced_until(anchors[oi][1]) + 2.0
        else:
            cap = out[i]["start"] + max(5.0, len(texts[oi]) * 1.0 + 3.0)
        out[i]["end"] = round(max(out[i]["start"] + 0.5, min(nxt, cap)), 2)

    # 최소 노출 보장(사용자 최우선 요구: "자막 누락 없는 게 제일 중요"): 후렴 반복 오탐 등으로
    # 소절들이 꼬리에 0.5~1.6초로 뭉개지면 사실상 안 보인다. 뒤에서부터 각 소절에 최소
    # 2.5초를 보장하고, 모자라면 앞 소절 시간/앞쪽 공백에서 연쇄적으로 당겨온다 — 어떤
    # 소절도 스쳐 지나가듯 사라지지 않는다(클립이 물리적으로 짧으면 가능한 만큼).
    # 단, 시작은 오디오로 정밀 보정한 '발성 시작점'이라 앞으로 당기면 자막이 노래보다
    # 먼저 뜬다("딱 타이트하게 발화 시점에" 요청 2026-09-06). 먼저 끝을 다음 소절 시작
    # 까지 늘려 확보하고, 그래도 모자랄 때만 시작을 최대 0.4초까지 당긴다.
    MIN_SHOW = 2.5
    MAX_PULL = 0.4
    for i in range(len(out) - 1, -1, -1):
        if out[i]["end"] - out[i]["start"] < MIN_SHOW:
            nxt = out[i + 1]["start"] if i + 1 < len(out) else clip_end
            out[i]["end"] = round(min(nxt, out[i]["start"] + MIN_SHOW), 2)
        if out[i]["end"] - out[i]["start"] < MIN_SHOW:
            out[i]["start"] = round(max(clip_start, out[i]["start"] - MAX_PULL, out[i]["end"] - MIN_SHOW), 2)
        if i > 0 and out[i - 1]["end"] > out[i]["start"]:
            out[i - 1]["end"] = out[i]["start"]  # 겹침 제거 → 다음 반복에서 i-1도 최소 노출 확보
    for i in range(len(out)):  # 안전망: 역전/0길이 정리
        if out[i]["end"] <= out[i]["start"]:
            out[i]["end"] = round(out[i]["start"] + 0.3, 2)
    return out


def analyze(
    url: str, config_path: Path = Path("config.yaml"), progress=_default_progress,
    transcript_text: str = "", force: bool = False, model: str = "",
    mode: str = "sermon", song_titles: str = "",
) -> tuple[Path, list[Clip]]:
    """다운로드 -> 전사 -> 하이라이트 후보 선정까지만 수행하고 (렌더링 없음),
    video_dir와 배열 순서=바이럴 예상 순위인 클립 후보 목록을 반환한다.

    progress(message, pct, eta_seconds)로 호출된다. pct는 0~100 전체 진행률, eta_seconds는
    남은 예상시간(초). 진행률은 단계별 '예상 소요시간' 비율로 채워지므로 바가 부드럽게 움직이고
    ETA도 처음부터 끝까지 일관된다(StageProgress 참고)."""
    cfg = load_config(config_path)
    output_root = Path("output")

    # 각 단계 예상시간(초). 실제 소요와 다르면 런타임에 보정한다(다운로드 %/전사 %/전사 분기).
    # 다운로드는 크리티컬 패스에서 뺐다(메타데이터만 몇 초 확인, 실제 파일은 백그라운드).
    # span = 진행바 시각적 폭(%), est = 예상초(ETA용). 준비 3단계는 순식간에 끝나므로
    # 시각 폭을 좁게(합 30%) 주고, 실제로 몇 분 걸리는 하이라이트 선정에 70%를 준다 →
    # "지금 30%면 선정 중"처럼 %와 단계 라벨이 항상 일치한다. 스텝 라벨 구간(웹): 다운로드
    # 0-10, 전사 10-30, 하이라이트 선정 30-100 과 정확히 맞춘다.
    if mode == "praise":
        # 찬양 모드: 오디오 힌트 단계가 없고, 선정 대신 '곡 구간 감지'가 마지막 단계다.
        stages = [
            _Stage("download", "영상 정보 확인 중...", est=8, span=10),
            _Stage("transcript", "자막 준비 중...", est=12, span=15),
            _Stage(
                "praise", "찬양 곡 구간 찾는 중...",
                est=_stage_time_est("praise_selection", 240.0), span=75,
            ),
        ]
    else:
        stages = [
            _Stage("download", "영상 정보 확인 중...", est=8, span=10),
            _Stage("transcript", "자막 준비 중...", est=12, span=15),
            _Stage("hints", "핵심 구간 분석 중...", est=5, span=5),
            # 하이라이트 선정(claude -p): 예상시간은 고정값 대신 직전 실행들의 실측 중앙값으로
            # 보정한다(_stage_time_est). 기록이 없을 때만 기본 200초(thinking 2048 + 출력
            # 다이어트 기준 실측 추정)를 쓴다.
            _Stage(
                "highlight", "하이라이트 후보 선정 중...",
                est=_stage_time_est("selection", 200.0), span=70,
            ),
        ]
    sp = StageProgress(progress, stages)
    try:
        # 업로드된 로컬 영상("local:<video_id>"): 사용자가 직접 찍어 파일로 올린 경우.
        # 다운로드/메타 조회가 필요 없고, 유튜브 자동자막도 없으므로 아래에서 whisper
        # 직접 전사 경로를 탄다(웹 라우트가 파일을 output/<vid>/source.mp4로 미리 저장).
        if url.startswith("local:"):
            _vid = url.split(":", 1)[1]
            _local_path = output_root / _vid / "source.mp4"
            if not _local_path.exists() or _local_path.stat().st_size == 0:
                raise RuntimeError("업로드된 영상 파일을 찾을 수 없습니다. 다시 업로드해 주세요.")
            dl = DownloadResult(
                video_id=_vid, title=_vid, video_path=_local_path,
                duration_sec=_probe_duration_sec(_local_path),
            )
            if dl.duration_sec <= 0:
                raise RuntimeError("업로드된 파일에서 영상 길이를 읽지 못했습니다(손상되었거나 영상이 아닐 수 있음).")
            sp.set_fraction(1.0, "업로드된 영상 사용")
            dl_local = True
        else:
            dl_local = False
        # 완성본이 이미 있으면 그대로, 없으면 메타데이터만 받고 다운로드는 백그라운드로.
        # (후보 뽑기는 자막 텍스트만 필요 — 영상 파일은 렌더 때 wait_for_download로 보장)
        dl = dl if dl_local else find_cached(url, output_root)
        if dl_local:
            pass
        elif dl is not None:
            sp.set_fraction(1.0, "영상 준비됨")
        else:
            dl = probe_video(url, output_root)
            with _bg_downloads_lock:
                t = _bg_downloads.get(dl.video_id)
                if t is None or not t.is_alive():
                    t = threading.Thread(
                        target=_start_download_bg, args=(url, output_root), daemon=True
                    )
                    _bg_downloads[dl.video_id] = t
                    t.start()
            sp.set_fraction(1.0, "영상은 백그라운드로 받는 중 (분석은 계속 진행돼요)")
        video_dir = output_root / dl.video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        # ── 업로드 찬양 + 곡 제목 직접 입력: 전사 없이 '정식 가사'로 자막 ─────────────
        # whisper가 회중 찬양(노래)을 심하게 오인식하는 문제를 근본 우회한다(사용자 요청
        # 2026-09-05: "전사 정확도가 너무 낮다. 곡 제목만 알면 정식 가사를 넣어달라").
        # 유명 찬송가·CCM은 모델이 정식 가사를 이미 알아 크롤링이 필요 없다. 전사를 통째로
        # 건너뛰므로 빠르고(노래 전사는 느리고 부정확), 자막 텍스트는 정확하다.
        title_lines = (
            [t.strip() for t in song_titles.splitlines() if t.strip()]
            if (mode == "praise" and dl_local and song_titles.strip()) else []
        )
        if title_lines:
            sp.advance("자막 준비 중...")
            sp.set_fraction(1.0, "전사 건너뜀 — 입력한 곡 제목의 정식 가사 사용")
            clips_path = video_dir / "clips.json"
            if clips_path.exists() and not force:
                sp.finish("완료: 기존 후보 재사용")
                return video_dir, load_clips_json(clips_path)
            sp.advance("입력한 곡 제목으로 정식 가사를 가져오는 중...")
            clips = _title_based_praise_clips(dl, video_dir, title_lines, cfg, model, sp)
            if not clips:
                raise RuntimeError(
                    "가사를 만들지 못했습니다. 곡 제목을 정확히 입력했는지 확인해 주세요."
                )
            with CLIPS_LOCK:
                save_clips_json(clips, clips_path)
            _sync_praise_clips_bg(dl.video_path, video_dir, clips, cfg)
            sp.finish(f"완료: 찬양 {len(clips)}곡 (가사 자동 싱크는 백그라운드에서 계속돼요)")
            return video_dir, clips

        # ── 업로드 찬양 + 곡 제목 미입력: 전체 전사 대신 짧은 구간만 훑어 제목 자동 추정 ──
        # 사용자 요청(2026-09-05): "그냥 제목만 파악해서 검색해서 자막 확보되잖아. 전체
        # 전사하지 말고." 제목을 안 넣어도, 영상 전체를 whisper로 정밀 전사하는 대신 짧은
        # 구간만 빠르게 훑어 추정하고, 확신이 있으면 위와 동일한 빠른 경로(정식 가사+속도
        # 싱크)로 간다. 추정 실패(8분 넘는 다곡 실황 등)면 아래 기존 전체 분석으로 폴백한다.
        if mode == "praise" and dl_local and not song_titles.strip():
            sp.advance("자막 준비 중...")
            guessed = _quick_guess_praise_title(dl.video_path, dl.duration_sec, cfg, model, sp)
            if guessed:
                clips_path = video_dir / "clips.json"
                if clips_path.exists() and not force:
                    sp.finish("완료: 기존 후보 재사용")
                    return video_dir, load_clips_json(clips_path)
                sp.message(f"'{guessed}' 정식 가사를 가져오는 중...")
                clips = _title_based_praise_clips(dl, video_dir, [guessed], cfg, model, sp)
                if clips:
                    with CLIPS_LOCK:
                        save_clips_json(clips, clips_path)
                    _sync_praise_clips_bg(dl.video_path, video_dir, clips, cfg)
                    sp.finish(f"완료: 찬양 '{guessed}' (제목 자동 추정 + 가사 자동 싱크는 백그라운드에서 계속돼요)")
                    return video_dir, clips
                sp.message(f"'{guessed}' 가사를 찾지 못해 정밀 분석으로 진행합니다...")
            else:
                sp.message("제목을 특정하지 못해 정밀 분석으로 진행합니다 (시간이 더 걸려요)...")

        # 2) 전사 --------------------------------------------------------------
        sp.advance("자막 준비 중...")
        transcript_path = video_dir / "transcript.json"
        # 순수 텍스트를 비례정렬한 경우, 선정 후 클립 경계를 실제 시각으로 스냅하기 위해 참조 자막을 보관.
        snap_reference: Transcript | None = None
        # 노트북LM 등 '다듬어진' 붙여넣기로 선정하면 매끈함에 속아 점수가 부풀려진다(요약 함정).
        # 이 경로에서만 채점을 냉정하게 하도록 프롬프트 경고를 켜는 플래그.
        transcript_is_cleaned = False
        # 붙여넣은 자막이 있으면 캐시된 transcript.json보다 그것을 우선(사용자 의도 존중).
        if transcript_path.exists() and not transcript_text.strip():
            sp.set_fraction(1.0, "기존 자막 재사용")
            transcript = Transcript(**json_load_transcript(transcript_path))
        elif transcript_text.strip():
            # 사용자가 붙여넣은 자막을 최우선으로 사용(전사 건너뜀).
            sp.message("붙여넣은 자막 사용 중...")
            transcript = parse_pasted_transcript(transcript_text, dl.duration_sec)
            if transcript is None:
                # 타임스탬프가 없는 '정확한' 텍스트(노트북LM 등). 텍스트 자체는 자동자막보다
                # 정확하므로 선정에 그대로 쓰되, 시간은 유튜브 자동자막(실제 시각)에 '문자열
                # 매칭'으로 정렬해 빌려온다(두 자막은 같은 오디오 전사라 표기가 거의 일치).
                # 예전 비례배분은 통짜 텍스트를 세그먼트 1개로 뭉개 [00:00:00] 하나만 남겨
                # 제목-내용 불일치를 냈다 → 매칭 정렬로 해결. 선정 후 클립 경계는 아래에서
                # snap_clips_to_reference로 자동자막의 실제 발화 시각에 다시 스냅한다.
                sp.message("붙여넣은 정확한 자막을 유튜브 자동자막 시각에 정렬하는 중...")
                reference = get_transcript_from_youtube(url, video_dir, dl.duration_sec)
                if reference and reference.segments:
                    transcript = align_plain_text_to_reference(
                        transcript_text, reference, dl.duration_sec
                    )
                    snap_reference = reference  # 선정 후 경계를 실제 시각으로 스냅하기 위해 보관
                    transcript_is_cleaned = True  # 다듬어진 텍스트 → 채점 냉정하게(요약 함정 경고 on)
                    (video_dir / "transcript_reference.json").write_text(
                        json.dumps(reference.to_json(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                else:
                    raise RuntimeError(
                        "붙여넣은 자막에 타임스탬프가 없고 유튜브 자동자막도 없어 시간을 매길 수 없습니다. "
                        "유튜브 '스크립트 표시'에서 타임스탬프 포함으로 복사하거나 SRT/VTT를 붙여넣으세요."
                    )
            sp.set_fraction(1.0, "붙여넣은 자막 사용")
            transcript_path.write_text(
                json.dumps(transcript.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            if dl_local:
                transcript = None  # 업로드 파일은 유튜브 자막이 없다 — 바로 whisper 직접 전사로
            else:
                sp.message("유튜브 자동 자막 확인 중...")
                transcript = get_transcript_from_youtube(url, video_dir, dl.duration_sec)
            if transcript is None or not transcript.segments:
                # 로컬 전사는 영상 길이에 비례해 오래 걸린다 → 예상시간을 크게 잡아 ETA를 맞춘다.
                sp.set_current_est(max(30.0, dl.duration_sec * 0.5))
                # 로컬 전사는 영상(오디오) 파일이 필요한 유일한 분석 단계 — 백그라운드
                # 다운로드가 아직이면 여기서만 기다린다(자동자막/붙여넣기 경로는 안 기다림).
                sp.message("전사를 위해 영상 다운로드를 기다리는 중...")
                wait_for_download(dl.video_id)
                sp.message("자동 자막이 없어 직접 전사 중 (시간이 걸릴 수 있어요)...")
                w = cfg["whisper"]
                # 찬양 모드: VAD가 노래(음악+합창)를 '음성 아님'으로 판단해 곡 구간 전체를
                # 통째로 건너뛴다(실측: 3.5분 업로드에서 찬송 2.5분이 전사 0단어). 가사를
                # 받아적어야 곡 식별·자막이 되므로 찬양 모드는 VAD를 끈다.
                # 찬양 전용 모델(config whisper.praise_model_size, 기본 medium=설교와 동일).
                # small로 낮추면 2~3배 빨라지지만 사용자 결정: 자막 정확도 우선 — medium 유지.
                _vad = False if mode == "praise" else w.get("vad_filter", True)
                _model_size = (
                    w.get("praise_model_size", w["model_size"]) if mode == "praise" else w["model_size"]
                )
                transcript = transcribe_and_save(
                    dl.video_path, transcript_path,
                    model_size=_model_size, device=w["device"], compute_type=w["compute_type"],
                    language=w["language"], vad_filter=_vad,
                    cpu_threads=int(w.get("cpu_threads", 0)),
                    on_segment=lambda seg_end, duration: sp.set_fraction(
                        min(1.0, seg_end / duration) if duration else 0.0,
                        f"직접 전사 중... {min(100, seg_end / duration * 100):.0f}%" if duration else "직접 전사 중...",
                    ),
                )
                # VAD(무음/비음성 감지)가 업로드 영상 오디오를 통째로 '음성 아님'으로 오판해
                # 세그먼트가 0개로 나오는 사고가 찬양이 아닌 설교/일반 업로드에서도 재현됨
                # (실측 2026-09-06: 실제로 사람이 말하는 138초 영상인데 vad_filter=True면
                # 0세그먼트, vad_filter=False면 32세그먼트 정상 전사됨). 이러면 하이라이트
                # 후보가 통째로 비어(clips.json=[]) 사용자에게는 "아무것도 안 뜬다"로 보인다.
                # VAD 끄고 한 번 더 시도해 되살린다 — 실패해도 기존 폴백(에러 메시지)은 그대로.
                if _vad and not transcript.segments:
                    sp.message("전사 결과가 비어 있어 무음 감지 없이 재시도하는 중...")
                    transcript = transcribe_and_save(
                        dl.video_path, transcript_path,
                        model_size=_model_size, device=w["device"], compute_type=w["compute_type"],
                        language=w["language"], vad_filter=False,
                        cpu_threads=int(w.get("cpu_threads", 0)),
                        on_segment=lambda seg_end, duration: sp.set_fraction(
                            min(1.0, seg_end / duration) if duration else 0.0,
                            f"직접 전사 재시도 중... {min(100, seg_end / duration * 100):.0f}%" if duration else "직접 전사 재시도 중...",
                        ),
                    )
            else:
                sp.set_fraction(1.0, "유튜브 자동 자막 사용")
            transcript_path.write_text(
                json.dumps(transcript.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
            )

        clips_path = video_dir / "clips.json"
        h = cfg["highlights"]
        # force(새로 분석) 또는 붙여넣은 자막이 있으면 캐시를 무시하고 재선정한다.
        regenerate = force or bool(transcript_text.strip())
        if clips_path.exists() and not regenerate:
            sp.finish(f"완료: 기존 후보 재사용")
            return video_dir, load_clips_json(clips_path)
        if clips_path.exists() and regenerate:
            # 기존 후보를 버전 백업하되 원본은 '복사'로 남긴다(예전엔 이동이었음).
            # 이동 방식은 재선정이 실패하면(세션 한도 등 — 2026-09-01 실제 발생) clips.json이
            # 사라진 채 남아 영상 페이지가 404가 되고 이전 후보까지 잃는다. 원본을 남기면
            # 실패 시 이전 후보가 그대로 살아 있고, 재선정 중 '옛 캐시를 완료로 오인'하는
            # 문제는 _clips_ready의 mtime(job.started 이후) 검사가 이미 막아준다.
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup = clips_path.with_name(f"clips.{ts}.bak.json")
            with CLIPS_LOCK:
                shutil.copy2(clips_path, backup)
            # 옛 렌더 산출물(short_N.mp4/.ass/.src)도 백업 폴더로 '이동'해 새 후보에 붙지
            # 않게 한다(삭제 아님 — 버전 보존 원칙). 재선정이 같은 장면을 다시 고르면 인용문
            # 앵커링 때문에 시작·끝(=서명)까지 동일해져, 예전 스타일로 구운 옛 영상이 새
            # 후보의 '완성됨'으로 그대로 붙어 보이는 잔해물 문제가 있었다(실신고).
            render_dir = video_dir / "clips"
            if render_dir.exists():
                stale = [
                    p for p in render_dir.iterdir()
                    if p.is_file() and p.name.startswith("short_")
                ]
                if stale:
                    arch = render_dir / f"backup_{ts}"
                    arch.mkdir(exist_ok=True)
                    for p in stale:
                        try:
                            shutil.move(str(p), str(arch / p.name))
                        except OSError:
                            pass  # 사용 중 파일 등은 남겨둔다(치명적이지 않음)

        # 찬양 모드: 곡별 구간 감지 → 시간순 후보 저장(채점/앵커링/오디오 힌트 없음) --------
        if mode == "praise":
            sp.advance("AI가 예배 실황에서 찬양 곡을 찾는 중...")
            p = cfg.get("praise", {}) or {}
            t_sel = time.time()
            clips = select_praise_songs(
                transcript=transcript,
                min_duration_sec=int(p.get("min_duration_sec", 60)),
                max_duration_sec=int(p.get("max_duration_sec", 420)),
                pad_start_sec=float(p.get("pad_start_sec", 4.0)),
                pad_end_sec=float(p.get("pad_end_sec", 6.0)),
                model=model or p.get("model", ""),
                thinking_tokens=int(p.get("thinking_tokens", 4096)),
                on_progress=lambda frac, msg: sp.set_fraction(frac, msg),
            )
            _record_stage_time("praise_selection", time.time() - t_sel)
            if not clips:
                raise RuntimeError(
                    "찬양 곡을 찾지 못했습니다. 영상에 찬양이 없거나 전사본에 가사가 거의 "
                    "안 잡혔을 수 있어요(자동자막 없는 실황은 직접 전사라 시간이 걸립니다)."
                )
            # 업로드 영상(화면에 가사 슬라이드 없음)은 가사 자막을 넣는다. whisper의 노래
            # 오인식("만유의"→"마녀의" 실측)을 그대로 구울 수 없으므로, 모델이 아는 정식
            # 가사로 교정해 caption_overrides(WYSIWYG 자막)로 확정한다.
            # 2026-09-05: 예전엔 "교정 줄 수가 whisper 줄 수와 정확히 같아야만" 적용했는데,
            # whisper가 숨쉬는 지점 기준으로 줄을 들쭉날쭉 나눠 이 조건이 실전에서 거의 항상
            # 깨져 교정이 사실상 안 먹혔다("자막 정확도 너무 안좋음" 실신고). 이제 whisper
            # 줄 구조와 무관하게 교정 텍스트를 문자 수 비례로 시간 배분한다.
            # 교정 실패(호출 자체 오류)는 치명적이지 않다 — 원문(whisper) 폴백으로 자막은 나온다.
            if dl_local:
                sp.message("가사 자막을 정식 가사로 교정하는 중...")
                try:
                    seg_lines: list[list[dict]] = []
                    for c in clips:
                        seg_lines.append([
                            {"start": s.start, "end": min(s.end, c.end), "text": (s.text or "").strip()}
                            for s in transcript.segments
                            if s.start >= c.start - 0.5 and s.start < c.end and (s.text or "").strip()
                        ])
                    req = [
                        {"index": ci, "title": c.title, "raw_text": " ".join(l["text"] for l in seg_lines[ci])}
                        for ci, c in enumerate(clips) if seg_lines[ci]
                    ]
                    if req:
                        # 자막 교정도 '자막 실행' 단계 — 분석용 모델(opus)과 무관하게 Sonnet 고정.
                        fixed = correct_praise_lyrics(
                            req, model="claude-sonnet-4-5",
                            thinking_tokens=int(p.get("lyrics_thinking_tokens", 2048)),
                        )
                        for ci, c in enumerate(clips):
                            lines = seg_lines[ci]
                            if not lines:
                                continue
                            corrected_text = fixed.get(ci)
                            corrected_lines = (
                                [ln.strip() for ln in corrected_text.split("\n") if ln.strip()]
                                if corrected_text else []
                            )
                            if corrected_lines:
                                c.caption_overrides = _distribute_lines_by_chars(
                                    corrected_lines, lines[0]["start"], lines[-1]["end"]
                                )
                            else:
                                print(f"[praise] 곡 {ci} 가사 교정 응답 없음 → whisper 원문 폴백")
                                c.caption_overrides = lines
                except Exception:  # noqa: BLE001 - 가사 교정 실패해도 후보 저장은 계속
                    traceback.print_exc()
            with CLIPS_LOCK:
                save_clips_json(clips, clips_path)
            # 업로드 찬양: 싱크 맞추기가 즉시 되도록 타이밍 전사를 백그라운드로 예열.
            if dl_local:
                _prewarm_praise_sync_cache(dl.video_path, video_dir, clips, cfg)
            sp.finish(f"완료: 찬양 {len(clips)}곡 감지")
            return video_dir, clips

        # 3) 오디오 에너지 힌트 -------------------------------------------------
        sp.advance("핵심 구간 분석 중...")
        peak_hints = []
        if cfg["audio_peaks"].get("enabled", True):
            wait_for_download(dl.video_id)  # 오디오 분석은 실제 파일 필요 (기본은 비활성)
            peak_hints = detect_peak_hints(
                dl.video_path,
                frame_length_sec=cfg["audio_peaks"]["frame_length_sec"],
                hop_length_sec=cfg["audio_peaks"]["hop_length_sec"],
            )

        # 피드백 루프: 과거 클립들의 실제 성과를 캘리브레이션 사례로 프롬프트에 주입한다.
        feedback_block = format_feedback_for_prompt(load_feedback())

        if h["mode"] == "manual":
            prompt = build_prompt(
                transcript=transcript, peak_hints=peak_hints,
                min_clips=h["min_clips"], max_clips=h["max_clips"],
                min_duration_sec=h["min_duration_sec"], max_duration_sec=h["max_duration_sec"],
                categories=h["categories"], video_duration_sec=transcript.duration_sec,
                feedback_block=feedback_block,
            )
            prompt_path = video_dir / "highlight_prompt.txt"
            save_prompt_for_manual_mode(prompt, prompt_path)
            raise RuntimeError(
                f"manual 모드: 프롬프트가 {prompt_path}에 저장됨. "
                f"Claude Code 세션에서 하이라이트를 골라 {clips_path}에 저장한 뒤 다시 시도하세요."
            )

        # 4) 하이라이트 선정 -----------------------------------------------------
        # 진행률은 이제 깜깜이 티커가 아니라 스트리밍 델타(생각 진행/몇 번째 클립 작성 중)로
        # 실제 진행을 반영한다. set_fraction은 단조 증가라 티커와 섞여도 뒤로 튀지 않는다.
        sp.advance("AI가 설교 전사본을 읽는 중...")
        t_selection = time.time()
        clips = select_highlights_auto(
            transcript=transcript, peak_hints=peak_hints,
            min_clips=h["min_clips"], max_clips=h["max_clips"],
            min_duration_sec=h["min_duration_sec"], max_duration_sec=h["max_duration_sec"],
            categories=h["categories"],
            feedback_block=feedback_block,
            # UI에서 고른 모델(model)이 있으면 그것을, 없으면 config 기본(sonnet)을 쓴다.
            model=model or h.get("model", ""),
            transcript_is_cleaned=transcript_is_cleaned,  # 다듬어진 붙여넣기면 채점 함정 경고 on
            thinking_tokens=int(h.get("thinking_tokens", 2048)),
            on_progress=lambda frac, msg: sp.set_fraction(frac, msg),
        )
        _record_stage_time("selection", time.time() - t_selection)
        # 경계 인용문 앵커링: 모델이 인용한 첫/끝 문장을 전사본에서 문자열로 찾아
        # start/end를 그 문장의 실제 발화 시각으로 확정한다. (모델이 읽은 것과 같은
        # transcript 기준이라 인용문이 반드시 이 텍스트 안에 있다.)
        hard_len_cfg = float(
            h.get("hard_max_duration_sec", float(h["max_duration_sec"]) + 5.0)
        )
        anchored_n = 0
        for c in clips:
            try:
                if anchor_clip_to_quotes(
                    c, transcript.segments, hard_len_cfg, min_duration_sec=float(h["min_duration_sec"])
                ):
                    c.anchored = True
                    anchored_n += 1
            except Exception:  # noqa: BLE001 - 앵커 실패 시 기존 숫자 경계로 조용히 폴백
                traceback.print_exc()
        print(f"[main] 경계 앵커링: {anchored_n}/{len(clips)}개 클립 인용문 매칭 성공", flush=True)
        # 순수 텍스트를 비례정렬해 선정한 경우, 클립 경계를 참조 자막의 실제 발화 시각으로 스냅한다.
        if snap_reference is not None:
            sp.message("클립 경계를 실제 자막 시각에 맞추는 중...")
            clips = snap_clips_to_reference(clips, transcript, snap_reference)
        # 안전망: 스냅(또는 다른 후처리)이 경계를 늘려 길이 상한을 넘긴 클립을 최종적으로 제외한다.
        # _validate_and_build_clips의 상한은 스냅 '이전'에만 적용되므로, 여기서 한 번 더 막는다.
        hard_max = h["max_duration_sec"] * 1.5
        kept = [c for c in clips if (c.end - c.start) <= hard_max]
        if len(kept) != len(clips):
            for c in clips:
                if (c.end - c.start) > hard_max:
                    print(f"[main] 스냅 후 과확장 클립 제외: {c.end - c.start:.0f}초 (상한 {hard_max:.0f}초) - {c.title!r}")
            clips = kept
        # 검토 UI는 clips.json 배열 순서 = 표시 순서다. 모델의 배열 순서("강한 순")와
        # 시스템이 따로 계산한 통합 score(scoring.py)가 어긋나면 목록이 뒤죽박죽으로
        # 보이므로(실측: 74,59,64,58), 저장 전에 score 내림차순으로 확정한다.
        # 렌더 전 시점이라 short_N 파일 매핑도 안 깨진다.
        clips.sort(key=lambda c: c.score or 0, reverse=True)
        with CLIPS_LOCK:
            save_clips_json(clips, clips_path)
        sp.finish(f"완료: {len(clips)}개 후보 선정")
        return video_dir, clips
    finally:
        sp.stop()


def _is_sentence_final(text: str) -> bool:
    """세그먼트 텍스트가 '문장이 여기서 끝났다'로 보이는지 판단한다.
    마침표류 문장부호나 한국어 종결어미로 끝나면 True. base 전사(유튜브 자동자막)는
    이런 신호가 있어, 펀치라인이 끝난 뒤 다음 새 주제로 끝을 끌고 가는 과확장을 막는다."""
    t = (text or "").strip().rstrip('"\'”’)』」')
    if not t:
        return False
    if t[-1] in ".?!…":
        return True
    # 한국어 종결어미(설교 말투): ~습니다/~합니다/~됩니다/~십시오/~세요/~예요/~에요/~죠/~다 등.
    endings = ("습니다", "합니다", "됩니다", "입니다", "니다", "십시오", "세요", "에요", "예요",
               "겁니다", "거예요", "거죠", "하죠", "이죠", "겠죠", "네요", "군요", "잖아요")
    return t.endswith(endings)


def _is_question_ending(text: str) -> bool:
    """물음표로 끝나는 문장인지. 설교체 수사의문문("~하면 어때요?")은 그 자체가 완결된
    펀치라인이 아니라 다음 문장이 답인 경우가 많아 끝 스냅에서 후순위로 다룬다."""
    t = (text or "").strip().rstrip('"\'”’)』」')
    return bool(t) and t[-1] == "?"


def _flatten_words(segs: list[Segment]) -> list[Word]:
    """세그먼트들의 단어를 시간순으로 평탄화하고, 유튜브 롤링 자막의 중복 단어를 제거한다.

    스냅의 기준을 세그먼트가 아니라 단어로 삼는 이유: 유튜브 자동자막(폴백 경로)은
    세그먼트가 서로 겹치는 '롤링' 구조라 세그먼트 경계가 문장 경계와 무관하다
    (실측: 세그먼트 끝 915.88에 스냅했지만 실제 문장은 918.08 '…것입니다.'에서 끝나
    "말이 안 끝났는데 뚝 끊기는" 사고). 단어 타임스탬프는 롤링과 무관하게 정확하다."""
    words = [w for s in segs for w in s.words if (w.text or "").strip()]
    words.sort(key=lambda w: (w.start, w.end))
    out: list[Word] = []
    for w in words:
        t = w.text.strip()
        if any(d.text == t and abs(d.start - w.start) < 0.25 for d in out[-3:]):
            continue
        out.append(Word(start=w.start, end=w.end, text=t))
    return out


def _word_true_end(words: list[Word], i: int) -> float:
    """단어 i의 '실질 발화 끝'. 롤링 자막은 세그먼트 마지막 단어의 end가 다음 발화
    시작까지 수 초씩 부풀어 있어(실측: '듣고서' 912.4→915.9), 다음 단어 start로 잘라야
    실제 발화 끝에 가깝다."""
    e = words[i].end
    if i + 1 < len(words) and words[i + 1].start > words[i].start:
        e = min(e, max(words[i + 1].start, words[i].start + 0.05))
    return e


_NORM_STRIP_RE = None  # 지연 컴파일 (re import를 함수 안에서)


def _normalize_for_match(text: str) -> str:
    """인용문↔전사본 매칭용 정규화: 공백·문장부호를 걷어내 '글자열'만 남긴다.
    (전사본과 모델 인용문은 띄어쓰기/문장부호가 어긋나기 쉽지만 글자 자체는 거의 같다.)"""
    global _NORM_STRIP_RE
    if _NORM_STRIP_RE is None:
        import re

        _NORM_STRIP_RE = re.compile(r"[\s\.,!\?…·\"'“”‘’()\[\]『』「」<>:;~\-—]+")
    return _NORM_STRIP_RE.sub("", text or "").lower()


def _find_quote_span(
    words: list[Word], quote: str, center_sec: float, window_sec: float = 75.0
) -> tuple[int, int] | None:
    """인용문(quote)을 발화 단어열에서 찾아 (첫 단어 idx, 끝 단어 idx)를 돌려준다.

    클립 경계를 '숫자 추측+종결어미 휴리스틱'이 아니라 모델이 인용한 실제 문장의 발화
    시각으로 확정하기 위한 핵심 부품. 같은 문구가 설교에서 반복될 수 있으므로 모델이
    말한 대략 시각(center_sec) 주변 window만 뒤진다. 정확 부분문자열 매칭을 먼저,
    실패하면 유사 매칭(연속 일치 75% 이상)으로 폴백한다."""
    nq = _normalize_for_match(quote)
    if len(nq) < 4:
        return None
    # 창 안의 단어들만 후보로. (idx 원본 보존)
    cand = [
        (i, w) for i, w in enumerate(words)
        if center_sec - window_sec <= w.start <= center_sec + window_sec
    ]
    if not cand:
        return None
    # 정규화 글자 스트림과 글자→단어 idx 매핑을 만든다.
    stream_parts: list[str] = []
    char_word_idx: list[int] = []
    for i, w in cand:
        nw = _normalize_for_match(w.text)
        stream_parts.append(nw)
        char_word_idx.extend([i] * len(nw))
    stream = "".join(stream_parts)
    if not stream:
        return None

    pos = stream.find(nq)
    if pos >= 0:
        return char_word_idx[pos], char_word_idx[pos + len(nq) - 1]

    # 유사 매칭 폴백: 전사 오탈자/조사 차이로 정확 일치가 깨진 경우.
    import difflib

    m = difflib.SequenceMatcher(None, stream, nq, autojunk=False).find_longest_match(
        0, len(stream), 0, len(nq)
    )
    if m.size >= max(6, int(len(nq) * 0.75)):
        # 인용문에서 매칭이 시작된 오프셋만큼 스트림 쪽 시작을 당겨 전체 인용 범위를 근사한다.
        a = max(0, m.a - m.b)
        b = min(len(stream) - 1, m.a + m.size - 1 + (len(nq) - (m.b + m.size)))
        return char_word_idx[a], char_word_idx[b]
    return None


def anchor_clip_to_quotes(
    clip: Clip, segs: list[Segment], hard_max_sec: float, min_duration_sec: float = 8.0
) -> bool:
    """클립 경계를 모델이 인용한 hook_line(첫 문장)/payoff_line(끝 문장)의 실제 발화
    시각으로 확정한다. 성공 시 True(→ clip.anchored).

    역할 분리가 핵심: "어디서 생각이 시작되고 완결되는가"는 전사본을 읽은 모델이 문장
    인용으로 답하고(의미 판단 — 모델의 강점), "그 문장이 몇 초인가"는 시스템이 문자열
    매칭으로 찾는다(정확 탐색 — 코드의 강점). 종결어미 휴리스틱으로 문장 끝을 '추측'하다
    변종 사고가 반복된 구조(문제 1·2)의 근본 대체물이다."""
    words = _flatten_words(segs)
    if not words:
        return False
    new_start, new_end = clip.start, clip.end
    end_anchored = False
    if clip.hook_line:
        span = _find_quote_span(words, clip.hook_line, clip.start)
        if span is not None:
            new_start = max(0.0, words[span[0]].start - 0.15)
    if clip.payoff_line:
        span = _find_quote_span(words, clip.payoff_line, clip.end)
        if span is not None:
            j = span[1]
            e = _word_true_end(words, j)
            pad = 0.35
            if j + 1 < len(words) and words[j + 1].start > e:
                pad = min(pad, max(0.1, words[j + 1].start - e))
            new_end = e + pad
            end_anchored = True
    # 앵커 결과가 말이 되는지 검증: config min_duration_sec 이상, 상한 이내. 아니면 원래 숫자 유지.
    # (예전엔 하드코딩 8초 바닥이라 인용문이 우연히 짧은 구간에서 매치되면 min_duration_sec
    # 20초 정책을 무시하고 16초짜리 클립이 그대로 통과했다 — "시간이 너무 짧다" 불만의 원인.)
    if not end_anchored or not (min_duration_sec <= new_end - new_start <= hard_max_sec):
        return False
    clip.start, clip.end = new_start, new_end
    return True


def _snap_clip_start_to_sentence(
    clip: Clip, segs: list[Segment], max_back: float = 4.0, tail_max: float = 3.0
) -> float:
    """clip.start가 문장 중간에 떨어지는 어색함을 잡는다 (실측: 클립이 앞 문장의 꼬리
    '겁니다.'로 시작). 단어 단위 문장 경계(이전 단어가 문장 종결로 끝난 다음 단어) 기준:
      - 시작이 문장 초입이면(문장 시작이 max_back초 이내 앞) → 문장 시작으로 살짝 당긴다.
      - 시작이 문장 꼬리면(다음 문장이 tail_max초 이내) → 다음 문장 시작으로 민다(꼬리 제거).
    반드시 clip.start 이전(START_BUFFER)까지 포함한 세그먼트를 넘겨야 당기기가 가능하다."""
    words = _flatten_words(segs)
    if not words:
        return clip.start
    # 문장 시작 시각 목록: 첫 단어, 그리고 문장 종결 단어 바로 다음 단어.
    sentence_starts = [words[0].start] + [
        words[i + 1].start
        for i in range(len(words) - 1)
        if _is_sentence_final(words[i].text)
    ]
    prev_start = None
    next_start = None
    for s in sentence_starts:
        if s <= clip.start + 0.05:
            prev_start = s
        else:
            next_start = s
            break
    if prev_start is not None and 0.35 <= clip.start - prev_start <= max_back:
        return max(0.0, prev_start - 0.05)
    if (
        next_start is not None
        and next_start - clip.start <= tail_max
        and next_start < clip.end - 5.0  # 클립이 사실상 사라질 정도로 밀지는 않는다
    ):
        return next_start - 0.1
    return clip.start


def _snap_clip_end_to_sentence(
    clip: Clip, segs: list[Segment], max_extend: float = 6.0
) -> float:
    """Claude가 대략 지정한 clip.end가 문장/발화 중간을 잘라 "말이 안 끝났는데 뚝 끊기는"
    문제를 막는다. clip.end 부근에서 처음으로 문장이 끝나는 '단어'까지만 살짝 늘린다.

    - 과확장 금지: clip.end 직전(0.8초 이내)에 이미 문장이 끝났으면 그대로 둔다.
      펀치라인이 끝났는데 다음 새 주제까지 끌고 가 마무리가 흐지부지되는 문제 방지(실측).
    - 세그먼트가 아니라 단어 기준인 이유는 _flatten_words 주석 참고(롤링 자막 사고).
    - 문장 종결 단어를 max_extend 안에서 못 찾으면 건드리지 않는다(기존 동작 유지).
    - 물음표 종결은 후순위: "죄가 들어오면 어때요?"처럼 수사의문문에서 바로 멈추면 답
      문장("아무리 좋은 관계도 깨져요")이 잘린다(실측). 예산 안에 평서형 종결이 더 있으면
      그쪽을 우선하고, 물음표 후보뿐이면 그걸로 폴백한다."""
    words = _flatten_words(segs)
    question_fallback: float | None = None
    for i, w in enumerate(words):
        e = _word_true_end(words, i)
        if e < clip.end - 0.8:
            continue
        if w.start > clip.end + max_extend:
            break
        if _is_sentence_final(w.text) and e <= clip.end + max_extend:
            if e < clip.end:
                return clip.end
            # 문장 끝 단어 뒤 살짝 여유를 줘 말끝이 딱 잘리지 않게 하되, 다음 단어
            # 시작을 넘지 않게 제한한다(넘으면 다음 문장 첫 단어가 끝에 깜빡 노출됨).
            pad = 0.25
            if i + 1 < len(words):
                pad = min(pad, max(0.0, words[i + 1].start - e))
            candidate = e + pad
            if _is_question_ending(w.text):
                if question_fallback is None:
                    question_fallback = candidate
                continue
            return candidate
    return question_fallback if question_fallback is not None else clip.end


def _advance_clip_start(min_start: float, segs: list[Segment]) -> float:
    """클립이 길이 상한을 넘을 때 시작점을 min_start 이후의 첫 발화(세그먼트) 시작으로 당긴다.

    쇼츠는 1분 내외여야 한다(절대 규칙). 상한을 넘으면 끝(가장 강한 펀치라인)은 지키고
    앞을 잘라내는 게 이 파이프라인의 원칙("펀치라인 하나만 남기고 앞을 잘라라")이므로
    end가 아니라 start를 움직인다. 문장 한가운데서 툭 시작하지 않도록 세그먼트 경계에
    스냅하되, 너무 멀면(8초 이상) 그냥 min_start에서 자른다(길이 보장이 우선)."""
    for s in sorted(segs, key=lambda s: s.start):
        if min_start - 0.01 <= s.start <= min_start + 8.0:
            return s.start
    return min_start


def _build_clip_hotwords(
    keywords: list[str], bible_dict: str, transcript_text: str, budget_chars: int = 260
) -> str | None:
    """클립별 hotwords를 '중요한 것부터, 예산 안에서' 구성한다.

    faster-whisper는 hotwords를 앞 223토큰까지만 쓰고 뒤는 조용히 버린다(실측 사고:
    500토큰짜리 성경 사전 전체 + 맨 뒤에 keywords를 이어 붙이던 기존 방식은 성경책
    이름들과 keywords가 통째로 잘려 무효 → 출애굽기가 '출애국기'로 나오는데도 사전이
    막지 못했다). 게다가 클립과 무관한 희귀 고유명사 수백 개는 디코딩을 이상한 단어
    쪽으로 편향시키는 부작용도 있다('아무리 존귀한'→'아우르 전기한' 류).
    그래서: (1) Claude가 클립 문맥을 읽고 뽑은 keywords를 맨 앞에, (2) 성경 사전 중
    이 설교 전사본에 실제로 등장하는 단어만 뒤에 더한다. 한글 1자≈1토큰꼴이라
    budget_chars(공백 포함 260자)면 223토큰 한도 안에 넉넉히 들어간다."""
    dict_words = list(dict.fromkeys(bible_dict.split()))
    if transcript_text.strip():
        # 1글자 항목(장/절/상/하 등)은 아무 데나 부분 매칭되므로 2글자 이상만 본다.
        matched = [d for d in dict_words if len(d) >= 2 and d in transcript_text]
    else:
        matched = dict_words  # 전사본이 없으면 사전 앞쪽(핵심 이름들)부터 예산만큼
    terms = list(dict.fromkeys([k.strip() for k in keywords if k and k.strip()] + matched))
    parts: list[str] = []
    used = 0
    for t in terms:
        if used + len(t) + 1 > budget_chars:
            break
        parts.append(t)
        used += len(t) + 1
    return " ".join(parts) or None


def _precise_worst_hole(
    base_segments: list[Segment], segs: list[Segment], a: float, b: float
) -> float:
    """base 전사에는 발화(단어)가 있는데 정밀 전사 결과에는 단어가 전혀 없는
    가장 긴 '구멍'(초)을 잰다.

    배치 정밀 전사가 조용한 발화 구간 20여 초를 통째로 빼먹는 사고가 있었는데(실측:
    Ywng7CLK3Ko — 초반 22초 자막 실종), 단어 수 비교(35% 기준)만으로는 '부분 실종'을
    못 잡는다(72/99단어로 통과). base 단어 시각마다 정밀 단어가 ±2.5초 안에 하나도
    없으면 그 지점은 구멍으로 보고, 연속 구멍의 최대 길이를 돌려준다."""
    if not base_segments or not segs:
        return 0.0
    base_ts = sorted(
        w.start for s in base_segments for w in s.words if a <= w.start <= b
    )
    precise_ts = sorted(w.start for s in segs for w in s.words if a - 3 <= w.start <= b + 3)
    if not base_ts:
        return 0.0
    if not precise_ts:
        return b - a
    import bisect

    worst = 0.0
    run_start: float | None = None
    for t in base_ts:
        i = bisect.bisect_left(precise_ts, t)
        near = min(
            (abs(precise_ts[j] - t) for j in (i - 1, i) if 0 <= j < len(precise_ts)),
            default=1e9,
        )
        if near > 2.5:  # 이 base 단어 주변에 정밀 단어가 없다 = 구멍
            if run_start is None:
                run_start = t
            worst = max(worst, t - run_start)
        else:
            run_start = None
    return worst


def _apply_corrections(segs: list[Segment], corrections: dict) -> None:
    """전사 오인식 확정 교정(config captions.corrections). 성경 용어처럼 절대 틀리면
    안 되는 단어의 최종 안전망 — hotwords 편향으로도 새는 반복 오탈자만 치환한다."""
    if not corrections or not segs:
        return
    for s in segs:
        for wrong, right in corrections.items():
            if wrong in (s.text or ""):
                s.text = s.text.replace(wrong, right)
        for wd in s.words:
            for wrong, right in corrections.items():
                if wrong in wd.text:
                    wd.text = wd.text.replace(wrong, right)


def _precise_cache_find(
    cache_dir: Path, model: str, sig: str, start: float, end: float
) -> list[Segment] | None:
    """이전 렌더에서 저장한 정밀 재전사 결과 중 [start, end]를 덮는 것을 찾는다.

    large-v3 CPU 재전사는 클립당 1~3분 걸리는데, 편집기에서 제목/폰트/위치만 바꿔
    재렌더할 때마다 같은 구간을 매번 다시 전사하는 게 렌더가 느린 주범이었다.
    세그먼트 타임스탬프는 원본 절대시간이라, 캐시 창이 현재 클립 구간을 포함하기만 하면
    구간이 다소 달라져도(스냅/살짝 트림) 그대로 재사용할 수 있다.

    sig(초기 프롬프트+hotwords 해시)가 파일명에 들어간다: hotwords 사전을 고치면 예전
    (틀린 어휘로 뽑힌) 캐시가 자동 무효화되어 다시 전사한다 — 이게 없으면 사전을 아무리
    고쳐도 캐시된 오탈자가 계속 나온다(실측)."""
    if not cache_dir.exists():
        return None
    for f in cache_dir.glob(f"*_{model}_{sig}.json"):
        try:
            a_str, b_str = f.stem.split("_")[:2]
            a, b = float(a_str), float(b_str)
        except ValueError:
            continue
        # 여유 0.3초: 시작 문장 스냅이 clip.start를 첫 단어보다 0.05초 앞으로 당겨 저장하므로,
        # 딱 맞는 비교(0.01)면 재렌더마다 캐시를 놓치고 매번 재전사한다(실측).
        if a <= start + 0.3 and b >= end - 0.3:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                return [
                    Segment(
                        start=s["start"], end=s["end"], text=s["text"],
                        words=[Word(**w) for w in s["words"]],
                    )
                    for s in data["segments"]
                ]
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
    return None


def _precise_cache_save(
    cache_dir: Path, model: str, sig: str, start: float, end: float, segs: list[Segment]
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{start:.2f}_{end:.2f}_{model}_{sig}.json"
    path.write_text(
        json.dumps(
            {"segments": [
                {"start": s.start, "end": s.end, "text": s.text,
                 "words": [{"start": w.start, "end": w.end, "text": w.text} for w in s.words]}
                for s in segs
            ]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def render_selected(
    video_dir: Path,
    clip_indices: list[int],
    config_path: Path = Path("config.yaml"),
    progress=_default_progress,
    outro_enabled: bool | None = None,
    sfx_enabled: bool | None = None,
    motion_enabled: bool | None = None,
    caption_preset: str = "",
    caption_lang: str = "",
    facetrack_enabled: bool | None = None,
    horizontal_indices: list[int] | None = None,
) -> list[Path]:
    """analyze()가 골라둔 후보 중 clip_indices(0-based, 배열 순서 기준)만 정밀
    재전사 + 렌더링한다. 결과 파일은 video_dir/clips/short_<원래순번>.mp4 로 저장된다.

    progress(message, pct)로 호출되며, 클립 개수만큼 균등 분할한 뒤 각 클립을
    재전사(전반 30%)/렌더링(후반 70%) 두 단계로 나눠 진행률을 채운다.

    outro_enabled: "끝에 로고 넣기" UI 체크박스 값(None이면 config 기본값 그대로 따름).
    horizontal_indices: 팝업의 '가로 원본' 버튼으로 고른 클립 인덱스들 — 찬양 클립을
    쇼츠 레이아웃 없이 원본 가로 비율 그대로, 자막·제목 없이 잘라만 낸다(곡별 개별
    업로드 용도, 사용자 요청 2026-09-06)."""
    # 진행률 콜백을 방어적으로 감싼다: UI 표시 오류가 전사/렌더를 죽이거나 폴백을 유발하지 않게.
    _raw_progress = progress
    progress = lambda message, pct, eta=None: _safe_progress(_raw_progress, message, pct, eta)  # noqa: E731
    cfg = load_config(config_path)
    if outro_enabled is not None:
        cfg["render"] = {
            **cfg["render"],
            "outro": {**cfg["render"].get("outro", {}), "enabled": outro_enabled},
        }
    if sfx_enabled is not None:
        cfg["render"] = {
            **cfg["render"],
            "sfx": {**cfg["render"].get("sfx", {}), "enabled": sfx_enabled},
        }
    if facetrack_enabled is not None:
        cfg["render"] = {**cfg["render"], "facetrack": facetrack_enabled}
    if caption_lang == "bilingual":
        # 이중언어(한글 아래 영어): 한국어는 그대로 유지하고, 렌더 시 영어 번역 트랙을 같은
        # 줄에 작게 이어붙인다(render.render_clip이 이 플래그를 읽어 bilingual_overrides
        # 전달). caption_lang=="en"(영어로 완전 대체)와는 다른 옵션 — 둘 다 켜지면 en이 우선.
        cfg["render"] = {**cfg["render"], "caption_bilingual": True}
    if motion_enabled is not None:
        # 모션그래픽(제목 팝 + 자막 페이드): captions.animate로 전달(렌더의 pr_captions도 상속).
        cfg["captions"] = {**cfg["captions"], "animate": motion_enabled}
    if caption_preset == "bold_yellow":
        # 인스타 레퍼런스 스타일: 정적(비카라오케) 큰 볼드 흰 글자 + 두꺼운 검정 외곽선,
        # 핵심어는 '형광펜(marker)' 강조. 현재 스타일은 그대로 두고 '이 스타일도' 선택 가능.
        _cap = cfg["captions"]
        cfg["captions"] = {
            **_cap,
            "template": "minimal",
            "highlight_style": "marker",
            "highlight_keywords_enabled": True,
            "primary_color": "&H00FFFFFF",           # 흰 글자
            "outline_color": "&H00000000",           # 검정 외곽선
            "outline_width": max(4, int(_cap.get("outline_width", 0) or 0)),
            "font_size": int(int(_cap.get("font_size", 72)) * 1.15),
        }
    with CLIPS_LOCK:
        clips = load_clips_json(video_dir / "clips.json")
    # 렌더는 몇 분씩 걸리고 그 사이 사용자가 편집기에서 clips.json을 고칠 수 있다. 렌더가
    # 끝날 때 이 낡은 메모리 사본으로 전체를 덮어쓰면 그 편집이 소실되므로, 종료 시점엔
    # 디스크를 다시 읽어 '렌더가 실제로 바꾼 필드(start/end)'만 병합한다. 어느 클립이
    # 렌더 중 편집됐는지 판별하기 위해 시작 시점 경계를 기억해 둔다.
    orig_bounds = {
        idx: (clips[idx].start, clips[idx].end)
        for idx in clip_indices
        if 0 <= idx < len(clips)
    }
    video_path = video_dir / "source.mp4"
    # 분석 단계에서 시작한 백그라운드 다운로드가 아직 진행 중이면 먼저 완료를 기다린다
    # (아래 자기치유가 같은 파일을 이중으로 받다 꼬이지 않게 하기 위함이기도 하다).
    if not video_path.exists() or video_path.stat().st_size == 0:
        progress("영상 다운로드 마무리 중...", 0)
        wait_for_download(video_dir.name)
    # 자기치유: 원본(source.mp4)이 없거나 깨졌으면(예: 이전 다운로드가 중단돼 조각만 남은
    # 경우) 렌더가 raw ffmpeg 오류로 죽지 않도록, video id로 유튜브 URL을 복원해 다시 받는다.
    # 단, 업로드된 로컬 영상(upload_*)은 유튜브에 없으므로 재다운로드가 불가능하다.
    if (not video_path.exists() or video_path.stat().st_size == 0) and video_dir.name.startswith("upload_"):
        raise RuntimeError("업로드된 원본 영상이 사라졌습니다. 파일을 다시 업로드해 주세요.")
    if not video_path.exists() or video_path.stat().st_size == 0:
        progress("영상 원본이 없어 다시 내려받는 중...", 0)
        download_video(
            f"https://www.youtube.com/watch?v={video_dir.name}",
            video_dir.parent,
            on_progress=lambda p: progress(f"영상 다시 받는 중... {p:.0f}%", p * 0.05),
        )
        if not video_path.exists():
            raise RuntimeError(
                f"원본 영상을 확보하지 못했습니다: {video_path}. 처음부터 다시 분석해 주세요."
            )
    w = cfg["whisper"]
    outputs = []
    END_BUFFER_SEC = 16.0   # clip.end 뒤로 이만큼 더 전사해서 문장이 끝나는 지점을 찾는다
                            # (8초였을 때 문장 끝이 창 밖이라 끝 스냅이 실패해 '이름을 그래서'처럼
                            #  말 중간에 뚝 끊긴 실측 사고 — 끝 스냅 허용폭 12초 + 여유)
    START_BUFFER_SEC = 4.0  # clip.start 앞도 전사해, 시작이 문장 중간이면 문장 시작으로 당긴다

    # 정밀 재전사가 이따금 한두 단어만 뱉고 사실상 실패할 때가 있다(자막이 통째로 비는
    # 치명적 결과 — 실제로 겪음). 그럴 때 폴백할 원본 전사(유튜브 자동자막/medium)를 미리 로드.
    base_segments: list[Segment] = []
    # 순수 텍스트 붙여넣기 경로에선 transcript.json이 '비례배분된 대략 시간'이라 문장 끝
    # 스냅/자막 폴백의 시간 기준으로 부적합하다(폴백 시 자막이 오디오와 통째로 어긋난 실측
    # 사고: tMJLm4Hrax8 — 모든 단어가 0.69초 균일 간격). 시간축 우선순위:
    #   1) transcript_reference.json (분석 때 따로 저장한 실제 발화 시각)
    #   2) youtube_auto_caption.ko.json3 (유튜브 자동자막 원본 — 과거 분석 폴더에도 있음)
    #   3) transcript.json (붙여넣기 경로면 비례배분이라 최후순위)
    reference_path = video_dir / "transcript_reference.json"
    json3_path = video_dir / "youtube_auto_caption.ko.json3"
    base_transcript_path = video_dir / "transcript.json"
    if reference_path.exists():
        base_segments = json_load_transcript(reference_path)["segments"]
        _rlog(video_dir, "base=transcript_reference.json")
    elif json3_path.exists():
        from src.youtube_captions import parse_json3_to_transcript

        base_segments = parse_json3_to_transcript(json3_path, 0.0).segments
        _rlog(video_dir, "base=youtube_auto_caption.json3")
    elif base_transcript_path.exists():
        base_segments = json_load_transcript(base_transcript_path)["segments"]
        _rlog(video_dir, "base=transcript.json (참조/json3 없음 - 붙여넣기 영상이면 시간 부정확 가능)")

    def _count_words(segments: list[Segment], a: float, b: float) -> int:
        return sum(1 for s in segments for wd in s.words if wd.start >= a and wd.end <= b)

    # 성경 사전 매칭용: 이 설교 전체에서 실제로 언급되는 고유명사만 hotwords에 넣는다.
    base_text_all = " ".join((s.text or "") for s in base_segments)
    corrections = cfg.get("captions", {}).get("corrections") or {}
    # base(폴백/스냅 기준) 전사에도 교정을 미리 적용해, 폴백 자막에서도 오탈자가 안 나가게 한다.
    _apply_corrections(base_segments, corrections)

    total = len(clip_indices)
    step = 100 / total if total else 100

    for i, idx in enumerate(clip_indices):
        clip = clips[idx]
        base = i * step

        # 영어 자막 옵션: 렌더 시점에만 한국어(caption_overrides) 대신 영어 트랙으로 바꿔
        # 굽는다(디스크의 한국어는 그대로 — 병합 저장은 start/end만 반영). 영어 번역이 없는
        # 클립은 그대로 한국어로 나간다.
        if caption_lang == "en" and getattr(clip, "caption_overrides_en", None):
            clip.caption_overrides = clip.caption_overrides_en

        # 찬양 곡 통편집: 정밀 재전사·문장 스냅·훅 배속을 모두 건너뛰고
        # 곡 구간 그대로 + 상단 제목(곡 제목)을 넣어 렌더한다. 노래에 문장 스냅은
        # 무의미하고, 배속은 곡 템포를 바꿔버린다.
        # 가사 자막: 유튜브 실황은 화면에 교회 자막(가사 슬라이드)이 이미 있어 안 넣지만,
        # 직접 찍어 업로드한 영상(upload_*)은 가사 표시가 없으므로 whisper 전사 기반
        # 카라오케 자막을 설교와 같은 스타일로 넣는다(사용자 요청, 2026-09-04).
        if getattr(clip, "clip_type", "") == "praise":
            out_path = video_dir / "clips" / f"short_{idx+1}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pr_render = {
                **cfg["render"],
                "remove_silence": False,           # 간주/조용한 피아노 구간을 무음으로 오인해 자르지 않게
                "hook_speedup": {"enabled": False},
            }
            # '가로 원본' 모드(팝업 버튼): 쇼츠 세로 카드 대신 원본 가로 비율 그대로,
            # 자막·제목 전부 없이 곡 구간만 잘라낸다(곡별 개별 업로드용 — 사용자 요청
            # 2026-09-06 "그냥 자르기 용도임. 자막도 필요없어").
            if horizontal_indices and idx in horizontal_indices:
                # 기본 source.mp4는 720p 상한(다운로드 시간 절약)이라 가로 풀프레임에선
                # 화질 저하가 그대로 보인다(실신고 2026-09-06) — 이 클립만 1080p 원본을
                # 따로 받아 쓴다(이미 받았으면 재사용, 실패하면 720p 폴백).
                hd_path = video_path
                if not video_dir.name.startswith("upload_"):
                    progress(f"[{i+1}/{total}] 고화질 원본 받는 중...", base)
                    from src.download import download_video_hd

                    _hd = download_video_hd(
                        video_dir,
                        on_progress=lambda p: progress(
                            f"[{i+1}/{total}] 고화질 원본 받는 중... {p:.0f}%",
                            base + step * 0.2 * (p / 100),
                        ),
                    )
                    if _hd is not None:
                        hd_path = _hd
                    else:
                        print("[render] 고화질 원본 확보 실패 — 기존 720p 원본으로 폴백")
                src_w, src_h = _probe_display_resolution(hd_path)
                if src_w > 1920:  # 인코딩 시간/용량 절약, 비율 유지
                    src_h = int(src_h * (1920 / src_w))
                    src_w = 1920
                out_res = (src_w - src_w % 2, src_h - src_h % 2)
                pr_render = {
                    **pr_render,
                    "resolution": out_res,
                    "background_mode": "crop",     # 스케일+크롭 — 패딩 여백이 원천적으로 없음
                    "hook": {"enabled": False},    # 상단 제목(곡명) 오버레이 제거
                }
                # 자막 이벤트 0개: 전사 세그먼트도, 확정 자막(caption_overrides)도 안 넘긴다.
                # 원본 clip은 건드리지 않는다(디스크의 가사 자막은 보존 — 얕은 복사로 충분,
                # 리스트 참조만 새로 비운 것이라 원본 리스트는 그대로다).
                clip = copy.copy(clip)
                clip.caption_overrides = []
                clip.caption_overrides_en = []
                _run_with_progress_ticker(
                    lambda: render_clip(hd_path, [], clip, out_path, pr_render, dict(cfg["captions"])),
                    start_pct=base, end_pct=base + step, progress=progress,
                    message=f"[{i+1}/{total}] 찬양 가로 원본 렌더링 중: {clip.title}",
                    est_seconds=max(30.0, (clip.end - clip.start) * 0.5),
                )
                out_path.with_suffix(".src").write_text(
                    render_signature(clip.start, clip.end), encoding="utf-8"
                )
                outputs.append(out_path)
                continue
            # 주의: captions enabled=False로 두면 ffmpeg subtitles 필터가 통째로 빠져
            # 같은 ASS에 든 '제목'까지 안 구워진다(실측: 제목 없는 찬양 렌더). 켠 채로 두고
            # 세그먼트로 자막 유무를 조절한다(빈 리스트 = 가사 자막 이벤트 0개).
            pr_captions = dict(cfg["captions"])
            is_upload = video_dir.name.startswith("upload_")
            pr_segments = base_segments if is_upload else []
            # 카라오케(파란/하늘색 강조, 목소리를 따라 색이 바뀜)는 업로드 여부와 무관하게
            # 찬양 전체 기본 꺼짐(정적 흰 자막) — 사용자가 팝업의 '🎨 파란 강조'를 눌러
            # 명시적으로 켠 클립만 예외. 예전엔 이 스위치가 아래 is_upload 블록 안에만
            # 있어서, 유튜브 링크 찬양 클립(is_upload=False)에 caption_overrides가 생기면
            # (가사 가져오기/AI 교정/번역 버튼은 업로드 여부를 안 가리고 동작함) 전역 설정
            # (config 기본 template="karaoke", 설교용)이 그대로 새어 들어가 파란 강조가
            # 켜지는 사고가 있었다(실신고 2026-09-05: "유튜브 링크 넣는 찬양도 파란색으로
            # 변해. 그 기능 모두 꺼줘 디폴트로").
            karaoke_on = bool(getattr(clip, "caption_karaoke", False))
            pr_captions["template"] = "karaoke" if karaoke_on else "minimal"
            if is_upload:
                # 직접 찍어 올린 영상은 원본 16:9를 그대로 유지하고(카드 세로 크롭 없음),
                # 가사 자막은 영상 화면 위에 흰 글씨로 오버레이한다(사용자 요청, 2026-09-04).
                # 싱크 정밀도보다 가사 정확도가 우선이라는 것도 같은 요청 — 이 부분은 이미
                # correct_praise_lyrics(정식 가사 교정)로 처리돼 있으므로 여기선 레이아웃만 바꾼다.
                src_w, src_h = _probe_display_resolution(video_path)
                if src_w > 1920:  # 너무 크면 다운스케일(인코딩 시간/용량 절약), 비율은 유지
                    src_h = int(src_h * (1920 / src_w))
                    src_w = 1920
                out_res = (src_w - src_w % 2, src_h - src_h % 2)  # 짝수 강제(인코더 요구사항)
                pr_render = {
                    **pr_render, "resolution": out_res,
                    # "pad"는 반올림 오차로 한두 픽셀 흰 여백이 생길 수 있다(사용자가 실제로
                    # 흰 배경/템플릿에 갇힌다고 신고, 2026-09-05) — "crop"은 채우기 색을 아예
                    # 쓰지 않는 스케일+크롭이라 여백이 원천적으로 생기지 않는다. out_res가 이미
                    # 소스 자체 비율이라 크롭도 사실상 0px(인코더 짝수 강제분 정도).
                    "background_mode": "crop",
                    # 제목(곡명) 오버레이도 없앤다(사용자 요청: "제목은 없애고 그냥 자막만").
                    "hook": {"enabled": False},
                    # 아웃트로는 이제 비율 유지+흰 패딩으로 만들어져(render._get_or_create_outro_segment)
                    # 16:9에 붙여도 안 찌그러진다 — 강제 off를 풀고 체크박스(outro_enabled)를 따른다
                    # (실신고 2026-09-05: "끝에 로고 넣기 2초도 작동을 안 하네 찬양에선").
                }
                # 카라오케 on/off는 위에서 이미 결정(업로드 여부 무관, 기본 꺼짐). 여기선
                # 업로드 전용 레이아웃(흰 글씨 오버레이·확대·강조색)만 덧붙인다.
                pr_captions = {
                    **pr_captions,
                    "position": "bottom",
                    # 1.35→1.28: "아주아주 조금만 줄여"(2026-09-05 미세조정, 97→92px 수준)
                    "font_size": int(pr_captions.get("font_size", 72) * 1.28),
                    "primary_color": "&H00FFFFFF",    # 흰색 자막(영상 위 오버레이라 대비 위해)
                    # 카라오케 켜졌을 때 강조색: 기존 진한 블루 대신 더 연한 하늘색(사용자 요청).
                    "karaoke_highlight_color": "&H00FACE87",  # 연한 하늘색(#87CEFA, ASS는 BGR)
                    "outline_color": "&H00000000",     # 검정 외곽선(어떤 배경에도 읽히도록)
                    "outline_width": max(3, int(pr_captions.get("outline_width", 0) or 0)),
                }
            _run_with_progress_ticker(
                lambda: render_clip(video_path, pr_segments, clip, out_path, pr_render, pr_captions),
                start_pct=base, end_pct=base + step, progress=progress,
                message=f"[{i+1}/{total}] 찬양 렌더링 중: {clip.title}",
                # 곡은 3~6분으로 길다 — 인코딩 시간도 대략 길이에 비례(QSV 기준 실측 보수치)
                est_seconds=max(30.0, (clip.end - clip.start) * 0.5),
            )
            out_path.with_suffix(".src").write_text(
                render_signature(clip.start, clip.end), encoding="utf-8"
            )
            outputs.append(out_path)
            continue

        # 최종 화면 자막은 유튜브 자동자막(부정확)이 아니라 이 정밀 재전사 결과를 쓴다.
        # precise_model_size로 정밀 재전사만 더 정확한 모델(예: large-v3)로 올릴 수 있다
        # (짧은 선택 클립에만 돌리므로 전체 영상을 큰 모델로 돌리는 부담 없이 정확도만 취함).
        # 성경 고유명사 상시 사전(정적) + 이 클립에서 뽑은 고유명사(동적)를 함께 hotwords로 넣어
        # large-v3가 이름 철자를 맞추게 한다(룻→'루시', 기드온→'기도원' 류 방지, 이중 방어).
        # 사용자가 자막 편집기에서 자막을 확정했으면 재전사 없이 그대로 렌더한다
        # (편집 결과가 최우선이고, 느린 large-v3 재전사도 건너뛰어 훨씬 빠르다).
        #
        # 2026-09-03에 이 fast-path를 없애고 편집 자막도 아래 정밀 재전사(최대 4단계 재시도
        # 캐스케이드: batched/순차/VAD끔 조합)를 거치게 했었는데, 되돌린다 — 그 재시도 단계마다
        # 결과가 근소하게 달라질 수 있어(폴백 임계값 근처일 때 특히) "고칠 때마다 결과가
        # 달라진다"·"한 부분만 고쳤는데 전체가 다시 처리된다"는 정확한 사용자 신고를 받았다.
        # 편집 자막은 텍스트가 이미 확정돼 있어 재전사로 얻을 게 없다(재전사는 '무슨 말인지'를
        # 알아내는 용도인데 이미 사용자가 알려줬다) — 그런데도 무겁고 비결정적인 파이프라인을
        # 태우는 건 손해뿐이었다. 카라오케 타이밍은 base_segments(참조 전사의 실제 발화 시각,
        # 롤링 중복 제거됨)로 충분히 정확하고, 이건 매번 똑같아 결과가 안정적이다.
        if getattr(clip, "caption_overrides", None):
            if not getattr(clip, "trimmed", False):
                last_ov_end = max((float(o["end"]) for o in clip.caption_overrides), default=clip.end)
                clip.end = max(clip.end, last_ov_end + 0.3)
                clip.end = _snap_clip_end_to_sentence(clip, base_segments)
            out_path = video_dir / "clips" / f"short_{idx+1}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # 편집 자막(caption_overrides)은 줄 단위라 단어별 시각이 없다 — 카라오케 강조가
            # 목소리 리듬을 따라가려면 단어별 '실제 발화 시각' 참조가 필요하다. 기준 우선순위:
            #   1) 정밀 재전사 캐시(있으면): whisper large-v3의 단어 시각 — 가장 정확.
            #      유튜브 자동자막은 단어 시작이 0.2~0.5초씩 '들쭉날쭉' 이르러 전역 오프셋
            #      (+0.25초)으로는 평균만 맞고 단어별 미세 오차가 남는다("파란색이 미세하게
            #      안 맞는다" 신고의 원인). 캐시는 '읽기만' 하므로(재전사 없음) 빠르고,
            #      항상 같은 파일이라 결과도 결정적이다.
            #   2) 캐시가 없으면 base_segments(참조 전사) — 예전 동작 그대로.
            ov_hotwords = _build_clip_hotwords(clip.keywords, w.get("bible_hotwords", ""), base_text_all)
            ov_sig = hashlib.md5(
                f"{w.get('initial_prompt', '')}|{ov_hotwords or ''}".encode("utf-8")
            ).hexdigest()[:8]
            ov_model = w.get("precise_model_size", w["model_size"])
            ref_segments = _precise_cache_find(
                video_dir / "precise_cache", ov_model, ov_sig, clip.start, clip.end + 4.0
            )
            if ref_segments is not None and _precise_worst_hole(
                base_segments, ref_segments, clip.start, clip.end
            ) >= 5.0:
                ref_segments = None  # 구멍 난 캐시는 신뢰하지 않는다(자막 씹힘 방지)
            if ref_segments is not None:
                _rlog(video_dir, f"clip{idx} 편집 자막 카라오케 기준: 정밀 캐시")
            else:
                ref_segments = base_segments
                _rlog(video_dir, f"clip{idx} 편집 자막 카라오케 기준: base(캐시 없음)")
            _run_with_progress_ticker(
                lambda: render_clip(video_path, ref_segments, clip, out_path, cfg["render"], cfg["captions"]),
                start_pct=base, end_pct=base + step, progress=progress,
                message=f"[{idx+1}/{total}] 편집 자막으로 렌더링 중: {clip.title}",
                est_seconds=max(15.0, (clip.end - clip.start) * 0.9),
            )
            out_path.with_suffix(".src").write_text(
                render_signature(clip.start, clip.end), encoding="utf-8"
            )
            outputs.append(out_path)
            continue

        hotwords = _build_clip_hotwords(clip.keywords, w.get("bible_hotwords", ""), base_text_all)
        clip_len = clip.end - clip.start
        precise_model = w.get("precise_model_size", w["model_size"])
        cache_dir = video_dir / "precise_cache"
        # 프롬프트/hotwords가 바뀌면 캐시도 무효가 되어야 한다(사전을 고쳐도 옛 오탈자
        # 캐시가 계속 나오는 문제 방지). 해시를 캐시 파일명에 넣는다.
        sig = hashlib.md5(
            f"{w.get('initial_prompt', '')}|{hotwords or ''}".encode("utf-8")
        ).hexdigest()[:8]

        # 같은 구간을 이미 정밀 재전사했다면 재사용한다. 편집기에서 제목/폰트/위치만 바꿔
        # 재렌더할 때도 클립당 1~3분짜리 large-v3 CPU 재전사를 매번 다시 돌리던 것이
        # 렌더가 느린 주범이었다 — 캐시 적중 시 그 시간이 통째로 사라진다.
        tr_a = max(0.0, clip.start - START_BUFFER_SEC)
        tr_b = clip.end + END_BUFFER_SEC
        # 캐시 요구 범위는 [시작, 끝+4초]면 충분하다: 시작 스냅으로 당겨져 저장된 start 때문에
        # tr_a(시작-4초)로 찾으면 자기 자신이 만든 캐시도 못 찾아 매번 재전사했다(실측).
        # 끝+4초는 "이미 문장 끝에 스냅돼 있는지" 확인에 필요한 최소 버퍼.
        cached_segs = _precise_cache_find(
            cache_dir, precise_model, sig, clip.start, clip.end + 4.0
        )
        if cached_segs is not None and _precise_worst_hole(
            base_segments, cached_segs, clip.start, clip.end
        ) >= 5.0:
            # 과거에 '부분 실종' 결과가 캐시된 경우(초반 20초 자막 실종 사고) 재사용하지 않는다.
            _rlog(video_dir, f"clip{idx} 캐시에 자막 구멍 발견 → 캐시 무시, 재전사")
            cached_segs = None
        if cached_segs is not None:
            progress(f"[{idx+1}/{total}] 이전 정밀 자막 재사용: {clip.title}", base + step * 0.5)
            segs = cached_segs
        else:
            # large-v3 CPU 재전사는 클립 하나에 1~3분씩 걸리는데 그동안 진행률이 한 지점에
            # 멈춰 있으면 사용자가 "안 만들어진다"고 오해한다(실제로 겪은 피드백). 재전사/렌더
            # 두 무진행 구간 모두 흉내 진행률 티커로 부드럽게 채워 "작동 중"임을 보여준다.
            # 정밀 재전사(faster-whisper)는 특정 클립에서 'maximum decoding length must be > 0'
            # 같은 예외로 통째로 죽는 경우가 있다(실제 발생). 그러면 렌더 전체가 실패하므로,
            # 예외는 삼키고 빈 결과로 둔 뒤 아래 폴백(원본 자막)이 자막을 채우게 한다.
            def _precise(vad: bool, batched: bool = True) -> list[Segment]:
                return transcribe_clip_precise(
                    video_path, tr_a, tr_b,
                    model_size=precise_model,
                    device=w["device"], compute_type=w["compute_type"],
                    language=w["language"], vad_filter=vad,
                    initial_prompt=w.get("initial_prompt"),
                    hotwords=hotwords,
                    cpu_threads=int(w.get("cpu_threads", 0)),
                    batch_size=int(w.get("batch_size", 8)),
                    batched=batched,
                )

            vad_default = w.get("vad_filter", True)
            try:
                segs = _run_with_progress_ticker(
                    lambda: _precise(vad_default),
                    start_pct=base, end_pct=base + step * 0.45, progress=progress,
                    message=f"[{idx+1}/{total}] 자막 정밀 인식 중: {clip.title}",
                    est_seconds=max(20.0, clip_len * 1.8),
                )
            except Exception as e:  # noqa: BLE001 - 정밀 재전사 실패해도 아래 재시도/폴백으로 계속
                progress(f"[{idx+1}/{total}] 정밀 인식 실패({e}) → 재시도", base + step * 0.45)
                _rlog(video_dir, f"clip{idx} 정밀 재전사 예외: {type(e).__name__}: {e}")
                segs = []

            # 배치 모드 '부분 실종' 검사: 단어 수(35% 기준)로는 못 잡는, base엔 발화가
            # 있는데 정밀 결과가 통째로 빈 구간(실측: 초반 22초 자막 실종)을 잡는다.
            # 구멍이 크면 느리지만 검증된 순차 모드로 다시 전사한다.
            hole = _precise_worst_hole(base_segments, segs, clip.start, clip.end)
            if segs and hole >= 5.0:
                _rlog(video_dir, f"clip{idx} 정밀(배치) 자막 구멍 {hole:.1f}초 → 순차 모드 재전사")
                try:
                    segs_seq = _run_with_progress_ticker(
                        lambda: _precise(vad_default, batched=False),
                        start_pct=base + step * 0.4, end_pct=base + step * 0.45, progress=progress,
                        message=f"[{idx+1}/{total}] 자막 재인식(빠짐 구간 복구): {clip.title}",
                        est_seconds=max(30.0, clip_len * 2.2),
                    )
                    if _precise_worst_hole(base_segments, segs_seq, clip.start, clip.end) < hole:
                        segs = segs_seq
                except Exception as e:  # noqa: BLE001 - 재시도 실패 시 기존 결과/폴백 유지
                    _rlog(video_dir, f"clip{idx} 순차 재전사 예외: {type(e).__name__}: {e}")

            base_n = _count_words(base_segments, clip.start, clip.end)
            precise_n = _count_words(segs, clip.start, clip.end)
            # 1차 정밀 재전사가 비었거나 원본보다 현저히 부실하면, 유튜브 자동자막(오인식 다수)으로
            # 폴백하기 전에 VAD를 끄고 한 번 더 정밀 재전사한다. VAD 필터가 짧은 클립에서 발화를
            # 통째로 무음 처리해 빈 결과나 'maximum decoding length must be > 0' 예외를 내는 사례가
            # 있어(실측: tMJLm4Hrax8), 이 재시도로 정확한 large-v3 자막을 되살린다. 순수 추가라
            # 재시도가 실패해도 결과는 기존과 동일(아래 원본 폴백).
            # 판정 여유: 정밀 전사는 실제 발화만 세고 base(유튜브 자막)는 롤링 중복·추임새가 섞여
            # 단어 수가 부풀기 쉽다. 50%로 잡으면 멀쩡한 정밀 자막이 51 vs 50처럼 아슬아슬하게
            # 버려지는 사고가 난다(실측) - 35%면 "진짜 부실"만 걸러진다.
            weak = (precise_n < max(3, int(base_n * 0.35))) if base_n >= 5 else (precise_n == 0)
            if weak and vad_default:
                try:
                    segs2 = _run_with_progress_ticker(
                        lambda: _precise(False),
                        start_pct=base + step * 0.45, end_pct=base + step * 0.5, progress=progress,
                        message=f"[{idx+1}/{total}] 자막 재인식(정밀·VAD 끔): {clip.title}",
                        est_seconds=max(20.0, clip_len * 1.8),
                    )
                    if _count_words(segs2, clip.start, clip.end) > precise_n:
                        segs, precise_n = segs2, _count_words(segs2, clip.start, clip.end)
                        # 이 재시도도 배치 모드(batched=True)라 같은 '구멍' 사고가 재발할 수 있다.
                        # 단어 수만 보고 그대로 채택하면 구멍이 다시 렌더로 새어나가므로 한 번 더 검사한다.
                        hole2 = _precise_worst_hole(base_segments, segs, clip.start, clip.end)
                        if hole2 >= 5.0:
                            _rlog(video_dir, f"clip{idx} VAD끔 재시도도 자막 구멍 {hole2:.1f}초 → 순차 모드 재전사")
                            try:
                                segs_seq2 = _run_with_progress_ticker(
                                    lambda: _precise(False, batched=False),
                                    start_pct=base + step * 0.45, end_pct=base + step * 0.5, progress=progress,
                                    message=f"[{idx+1}/{total}] 자막 재인식(빠짐 구간 복구 2차): {clip.title}",
                                    est_seconds=max(30.0, clip_len * 2.2),
                                )
                                if _precise_worst_hole(base_segments, segs_seq2, clip.start, clip.end) < hole2:
                                    segs, precise_n = segs_seq2, _count_words(segs_seq2, clip.start, clip.end)
                            except Exception as e:  # noqa: BLE001 - 재시도 실패 시 기존 결과/폴백 유지
                                _rlog(video_dir, f"clip{idx} 2차 순차 재전사 예외: {type(e).__name__}: {e}")
                except Exception:  # noqa: BLE001 - 재시도도 실패하면 아래 원본 폴백
                    pass
            # 그래도 부실하면 원본(유튜브 자동자막/medium) 전사로 폴백해 자막이 비는 것만은 막는다.
            # 단어 수(35%)만이 아니라 '구멍'도 본다: 재시도까지 다 해도 5초+ 구간이 통째로 빈
            # 정밀 결과는 단어 수 검사를 통과해도 그대로 구우면 그 구간 자막이 실종된다
            # (실측: base 54단어 vs precise 32단어=59%로 통과했지만 16초 구멍 → 자막 공백 렌더).
            final_hole = _precise_worst_hole(base_segments, segs, clip.start, clip.end) if segs else 999.0
            if base_n >= 5 and (precise_n < max(3, int(base_n * 0.35)) or final_hole >= 5.0):
                progress(
                    f"[{idx+1}/{total}] 정밀 자막 부실 → 원본 자막({base_n}단어)으로 대체",
                    base + step * 0.5,
                )
                _rlog(
                    video_dir,
                    f"clip{idx} 폴백: precise {precise_n}단어/구멍 {final_hole:.1f}초 (base {base_n}단어)",
                )
                segs = base_segments
            elif segs and precise_n > 0:
                _rlog(video_dir, f"clip{idx} 정밀 자막 사용: {precise_n}단어 (base {base_n}단어)")
                # 건강한 정밀 결과만 캐시한다(부실 결과를 캐시하면 다음 렌더가 재시도 기회를 잃는다).
                # 자막 구멍이 남아 있는 결과도 캐시하지 않는다(불량 캐시가 계속 재사용되는 사고 방지).
                if _precise_worst_hole(base_segments, segs, clip.start, clip.end) < 5.0:
                    _precise_cache_save(cache_dir, precise_model, sig, tr_a, tr_b, segs)
        # 확정 오탈자 교정(출애굽기/여호와 등). 캐시는 원본 그대로 저장하고 매번 여기서 교정한다
        # (교정 사전을 나중에 더 채워도 재전사 없이 다음 렌더부터 바로 반영되게).
        _apply_corrections(segs, corrections)
        # 경계 스냅 기준: 정밀 전사가 건강하면 그것을 쓴다 — large-v3는 구두점 있는 진짜
        # 문장 단위라 "문장 중간 끊김/앞 문장 꼬리 시작"을 정확히 잡는다. 유튜브 자막 조각
        # (base)은 문장 경계가 아니어서 스냅이 어색했다(실측 불만). 폴백 시에만 base 사용.
        # 사용자가 직접 구간을 자른 경우(trimmed)엔 건드리지 않는다.
        used_precise = bool(segs) and (segs is not base_segments)
        # 사용자가 직접 구간을 자른 경우(trimmed)엔 건드리지 않는다.
        if not getattr(clip, "trimmed", False):
            snap_src = segs if used_precise else (base_segments or segs)
            # 시작 스냅도 폴백(base) 경로에서 함께 돌린다: 이제 세그먼트가 아니라 단어 단위
            # 문장 경계 기준이라 롤링 자막에서도 안전하다(실측: 폴백 렌더가 앞 문장 꼬리
            # '겁니다.'로 시작하던 문제).
            new_start = _snap_clip_start_to_sentence(clip, snap_src)
            if new_start != clip.start:
                _rlog(video_dir, f"clip{idx} 시작 문장 스냅: {clip.start:.2f} -> {new_start:.2f}")
                clip.start = new_start
            # 끝 스냅 허용폭 12초: 6초였을 때 문장 끝이 조금 멀면 스냅이 포기해
            # '이름을 그래서'처럼 말 중간에 뚝 끊겼다(실측). 늘어난 길이가 상한을 넘으면
            # 아래 하드캡이 끝(펀치라인)을 지키고 시작을 당겨 해결한다.
            # 단, 인용문 앵커링이 성공한 클립(anchored)은 이미 '생각의 완결' 문장 끝에
            # 정렬돼 있으므로 정밀 전사 기준 미세 조정(4초)만 허용한다 — 휴리스틱이
            # 앵커를 다음 주제까지 끌고 가는 과확장을 막는다.
            snap_budget = 4.0 if getattr(clip, "anchored", False) else 12.0
            new_end = _snap_clip_end_to_sentence(clip, snap_src, max_extend=snap_budget)
            if new_end != clip.end:
                _rlog(video_dir, f"clip{idx} 끝 문장 스냅: -> {new_end:.2f}")
                clip.end = new_end
            # 길이 절대 상한(hard_max_duration_sec, 기본 80초): 2026 조사 결과 쇼츠/릴스 모두
            # 45~60초가 스위트스팟이지만 알고리즘의 실제 기준은 '완결 시청률'이라, 문장/맥락이
            # 60초 안에 안 끝나면 80초까지는 끊지 않고 완결시키는 게 낫다(말이 중간에 끊긴
            # 클립은 어떤 길이보다 성과가 나쁘다). 상한을 넘으면 끝(펀치라인)은 지키고
            # 시작을 당겨 상한 안으로 넣는다.
            hard_len = float(
                cfg["highlights"].get(
                    "hard_max_duration_sec", float(cfg["highlights"]["max_duration_sec"]) + 5.0
                )
            )
            if clip.end - clip.start > hard_len:
                new_start = _advance_clip_start(clip.end - hard_len, base_segments or segs)
                progress(
                    f"[{idx+1}/{total}] 길이 {clip.end - clip.start:.0f}초 → 상한 {hard_len:.0f}초로 앞부분 트림",
                    base + step * 0.5,
                )
                clip.start = new_start
        out_path = video_dir / "clips" / f"short_{idx+1}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _run_with_progress_ticker(
            lambda: render_clip(video_path, segs, clip, out_path, cfg["render"], cfg["captions"]),
            start_pct=base + step * 0.5, end_pct=base + step, progress=progress,
            message=f"[{idx+1}/{total}] 쇼츠 렌더링 중: {clip.title}",
            est_seconds=max(15.0, clip_len * 0.9),
        )
        out_path.with_suffix(".src").write_text(
            render_signature(clip.start, clip.end), encoding="utf-8"
        )
        outputs.append(out_path)

    # 렌더 과정에서 스냅/편집자막으로 clip.start/end가 조정됐을 수 있다. 이를 clips.json에
    # 반영해 검토 UI의 'N초' 라벨이 실제 렌더된 영상 길이와 일치하게 한다.
    _merge_render_bounds(video_dir / "clips.json", clips, clip_indices, orig_bounds)

    progress("모든 클립 렌더링 완료", 100)
    return outputs


def _merge_render_bounds(
    clips_path: Path,
    rendered_clips: list[Clip],
    clip_indices: list[int],
    orig_bounds: dict[int, tuple[float, float]],
) -> None:
    """렌더가 조정한 클립 경계(start/end)만 디스크 최신본에 병합 저장한다.

    렌더는 몇 분씩 걸리므로 시작 시점의 메모리 사본으로 전체를 덮어쓰면 그 사이 편집기에서
    저장한 수정(자막·제목·위치 등)이 통째로 소실된다(실제 시나리오). 그래서:
      - 락 안에서 디스크를 다시 읽어, 렌더가 실제로 바꾼 필드(start/end)만 써넣는다 →
        렌더 중 '다른' 클립에 한 편집은 그대로 보존된다.
      - 렌더 중인 '바로 그' 클립을 사용자가 편집했으면(경계가 시작 시점과 달라짐) 렌더의
        낡은 경계를 쓰지 않고 사용자 편집을 우선한다 — .src 서명이 어긋나 UI에 '미렌더'로
        표시되고, 다시 만들면 편집이 반영된다(올바른 동작).
      - 재선정으로 clips.json이 통째로 바뀐 경우도 경계 불일치로 걸러져 새 후보를 오염시키지 않는다."""
    with CLIPS_LOCK:
        fresh = load_clips_json(clips_path)
        for idx in clip_indices:
            if not (0 <= idx < len(fresh)) or idx not in orig_bounds:
                continue
            if (fresh[idx].start, fresh[idx].end) == orig_bounds[idx]:
                fresh[idx].start = rendered_clips[idx].start
                fresh[idx].end = rendered_clips[idx].end
        save_clips_json(fresh, clips_path)


def run(url: str, config_path: Path = Path("config.yaml")) -> None:
    """CLI 전용: 후보 선정부터 전체 렌더링까지 한 번에 (검토 없이)."""
    video_dir, clips = analyze(url, config_path)
    outputs = render_selected(video_dir, list(range(len(clips))), config_path)
    print("완료! 생성된 쇼츠:")
    for p in outputs:
        print(f"  - {p}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python -m src.main <youtube-url>")
        sys.exit(1)
    run(sys.argv[1])
