"""오케스트레이터: 유튜브 링크 -> 세로형 쇼츠

두 단계로 나뉜다 (웹 검토 흐름에서 각각 따로 호출됨):
  1) analyze()        : 다운로드 + 전사 + 하이라이트 후보 선정 (아직 렌더링 안 함)
  2) render_selected() : 후보 중 유저가 고른 것만 정밀 재전사 + 렌더링

사용법 (CLI, 둘 다 한번에):
    python -m src.main "https://www.youtube.com/watch?v=XXXXXXXX"
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml

from src.audio_peaks import detect_peak_hints
from src.download import download_video, find_cached, probe_video
from src.highlights import (
    CLIPS_LOCK,
    Clip,
    build_prompt,
    load_clips_json,
    save_clips_json,
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


def analyze(
    url: str, config_path: Path = Path("config.yaml"), progress=_default_progress,
    transcript_text: str = "", force: bool = False, model: str = "",
    mode: str = "sermon",
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
        # 완성본이 이미 있으면 그대로, 없으면 메타데이터만 받고 다운로드는 백그라운드로.
        # (후보 뽑기는 자막 텍스트만 필요 — 영상 파일은 렌더 때 wait_for_download로 보장)
        dl = find_cached(url, output_root)
        if dl is not None:
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
                transcript = transcribe_and_save(
                    dl.video_path, transcript_path,
                    model_size=w["model_size"], device=w["device"], compute_type=w["compute_type"],
                    language=w["language"], vad_filter=w.get("vad_filter", True),
                    cpu_threads=int(w.get("cpu_threads", 0)),
                    on_segment=lambda seg_end, duration: sp.set_fraction(
                        min(1.0, seg_end / duration) if duration else 0.0,
                        f"직접 전사 중... {min(100, seg_end / duration * 100):.0f}%" if duration else "직접 전사 중...",
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
            with CLIPS_LOCK:
                save_clips_json(clips, clips_path)
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
) -> list[Path]:
    """analyze()가 골라둔 후보 중 clip_indices(0-based, 배열 순서 기준)만 정밀
    재전사 + 렌더링한다. 결과 파일은 video_dir/clips/short_<원래순번>.mp4 로 저장된다.

    progress(message, pct)로 호출되며, 클립 개수만큼 균등 분할한 뒤 각 클립을
    재전사(전반 30%)/렌더링(후반 70%) 두 단계로 나눠 진행률을 채운다.

    outro_enabled: "끝에 로고 넣기" UI 체크박스 값(None이면 config 기본값 그대로 따름)."""
    # 진행률 콜백을 방어적으로 감싼다: UI 표시 오류가 전사/렌더를 죽이거나 폴백을 유발하지 않게.
    _raw_progress = progress
    progress = lambda message, pct, eta=None: _safe_progress(_raw_progress, message, pct, eta)  # noqa: E731
    cfg = load_config(config_path)
    if outro_enabled is not None:
        cfg["render"] = {
            **cfg["render"],
            "outro": {**cfg["render"].get("outro", {}), "enabled": outro_enabled},
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

        # 찬양 곡 통편집: 정밀 재전사·자막·문장 스냅·훅 배속을 모두 건너뛰고
        # 곡 구간 그대로 + 상단 제목(곡 제목)만 넣어 렌더한다. 노래에 문장 스냅은
        # 무의미하고, 배속은 곡 템포를 바꿔버리며, 자막은 사용자가 원치 않았다.
        if getattr(clip, "clip_type", "") == "praise":
            out_path = video_dir / "clips" / f"short_{idx+1}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pr_render = {
                **cfg["render"],
                "remove_silence": False,           # 간주/조용한 피아노 구간을 무음으로 오인해 자르지 않게
                "hook_speedup": {"enabled": False},
            }
            # 주의: captions enabled=False로 두면 ffmpeg subtitles 필터가 통째로 빠져
            # 같은 ASS에 든 '제목'까지 안 구워진다(실측: 제목 없는 찬양 렌더). 켠 채로 두고
            # segments=[]를 넘겨 가사 자막 이벤트만 0개가 되게 한다.
            pr_captions = dict(cfg["captions"])
            _run_with_progress_ticker(
                lambda: render_clip(video_path, [], clip, out_path, pr_render, pr_captions),
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
