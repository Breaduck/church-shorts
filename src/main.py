"""오케스트레이터: 유튜브 링크 -> 세로형 쇼츠

두 단계로 나뉜다 (웹 검토 흐름에서 각각 따로 호출됨):
  1) analyze()        : 다운로드 + 전사 + 하이라이트 후보 선정 (아직 렌더링 안 함)
  2) render_selected() : 후보 중 유저가 고른 것만 정밀 재전사 + 렌더링

사용법 (CLI, 둘 다 한번에):
    python -m src.main "https://www.youtube.com/watch?v=XXXXXXXX"
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml

from src.audio_peaks import detect_peak_hints
from src.download import download_video, find_cached, probe_video
from src.highlights import (
    Clip,
    build_prompt,
    load_clips_json,
    save_clips_json,
    save_prompt_for_manual_mode,
    select_highlights_auto,
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
    __slots__ = ("key", "message", "est")

    def __init__(self, key: str, message: str, est: float):
        self.key = key
        self.message = message
        self.est = max(1.0, float(est))


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

    def _total_est(self) -> float:
        return sum(s.est for s in self._stages) or 1.0

    def _emit_locked(self) -> None:
        cur = self._stages[self._i]
        completed = sum(s.est for s in self._stages[: self._i])
        done_est = completed + cur.est * self._frac
        total = self._total_est()
        gfrac = min(1.0, done_est / total)
        pct = min(self._min_floor, gfrac * 100.0)
        pct = max(pct, self._last_pct)  # 단조 증가 보장
        self._last_pct = pct
        # 예상시간을 넘겨 frac이 0.95에 고정되면 eta도 같이 고정돼 "5초 남음"이 몇 분째
        # 안 바뀌는 것처럼 보인다(실제로는 멈춘 게 아님). 이 경우 거짓 ETA 대신 실제 경과시간을
        # 메시지에 보여줘 "아직 일하는 중"임을 알린다.
        if self._overrun_sec > 0:
            eta = None
            message = f"{cur.message} (예상보다 오래 걸리는 중... {int(self._overrun_sec)}초 경과)"
        else:
            eta = max(0.0, total - done_est)
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


def analyze(
    url: str, config_path: Path = Path("config.yaml"), progress=_default_progress,
    transcript_text: str = "", force: bool = False,
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
    stages = [
        _Stage("download", "영상 정보 확인 중...", est=8),
        _Stage("transcript", "자막 준비 중...", est=12),
        _Stage("hints", "핵심 구간 분석 중...", est=5),
        # 하이라이트 선정(claude -p): sonnet-4.5 + thinking 상한 + 축소 출력 기준 실측 목표 ~2분.
        _Stage("highlight", "하이라이트 후보 선정 중...", est=150),
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
            # 기존 후보(사용자 편집이 담겼을 수 있음)를 지우지 않고 버전 백업 후 재생성.
            backup = clips_path.with_name(f"clips.{time.strftime('%Y%m%d_%H%M%S')}.bak.json")
            clips_path.replace(backup)

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

        # 4) 하이라이트 선정 (opaque: 티커가 부드럽게 채움) ---------------------
        sp.advance("AI가 설교 전체를 읽으며 하이라이트 선정 중 (보통 2~5분 걸려요)...")
        clips = select_highlights_auto(
            transcript=transcript, peak_hints=peak_hints,
            min_clips=h["min_clips"], max_clips=h["max_clips"],
            min_duration_sec=h["min_duration_sec"], max_duration_sec=h["max_duration_sec"],
            categories=h["categories"],
            feedback_block=feedback_block,
            model=h.get("model", ""),  # 기본 sonnet(config) — 하이라이트 선정 비용 절감
            transcript_is_cleaned=transcript_is_cleaned,  # 다듬어진 붙여넣기면 채점 함정 경고 on
        )
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


def _snap_clip_start_to_sentence(
    clip: Clip, segs: list[Segment], max_back: float = 4.0, tail_max: float = 3.0
) -> float:
    """clip.start가 문장 중간에 떨어지는 어색함을 잡는다 (실측: 클립이 앞 문장의 꼬리
    '…되기를 축원합니다 아멘'으로 시작). 정밀 전사(문장 단위 세그먼트)를 기준으로:
      - 시작이 문장 초입이면(문장 시작이 max_back초 이내 앞) → 문장 시작으로 살짝 당긴다.
      - 시작이 문장 꼬리면(남은 길이 tail_max초 이하) → 다음 문장 시작으로 민다(꼬리 제거).
    반드시 clip.start 이전(START_BUFFER)까지 포함해 전사한 세그먼트를 넘겨야 당기기가 가능하다."""
    ordered = sorted(segs, key=lambda s: s.start)
    for i, seg in enumerate(ordered):
        if seg.start <= clip.start <= seg.end:
            into = clip.start - seg.start
            remain = seg.end - clip.start
            if 0.35 <= into <= max_back and into <= remain:
                return seg.start
            if remain <= tail_max and i + 1 < len(ordered):
                nxt = ordered[i + 1].start
                if nxt < clip.end - 5.0:  # 클립이 사실상 사라질 정도로 밀지는 않는다
                    return nxt
            return clip.start
        if seg.start > clip.start:
            break
    return clip.start


def _snap_clip_end_to_sentence(
    clip: Clip, segs: list[Segment], max_extend: float = 6.0, pause_gap: float = 0.35
) -> float:
    """Claude가 대략 지정한 clip.end가 문장/발화 중간을 잘라 "말이 안 끝났는데 뚝 끊기는"
    문제를 막는다. clip.end가 발화 중간이면 그 발화가 끝나는 곳까지만 살짝 늘린다.

    단, 과확장 금지가 최우선: 현재 세그먼트가 이미 문장 종결(마침표/종결어미)로 끝나면
    거기서 멈춘다. 펀치라인이 끝났는데도 다음 새 주제(예: "그래서 어 6월 27일이죠…")까지
    끝을 끌고 가 마무리가 흐지부지되는 문제가 있어(실측), 문장 경계 신호를 우선한다.
    보조로 침묵 간격(pause_gap)과 확장 한도(max_extend)로 이중 제한한다.

    반드시 신뢰도 높은 원본 전사(base_segments)를 넘겨야 한다 — 정밀 재전사는 이따금 실패해
    세그먼트가 비어 스냅이 무력화된다(실제로 겪은 버그)."""
    if not segs:
        return clip.end
    segs = sorted(segs, key=lambda s: s.start)
    end = clip.end
    idx = -1
    for i, seg in enumerate(segs):
        if seg.start <= end <= seg.end:
            end = seg.end
            idx = i
            break
        if seg.start > end:
            idx = i - 1
            break
        idx = i
    if idx < 0:
        return clip.end
    # 이미 문장 종결로 끝나는 세그먼트에 걸쳐 있으면 확장하지 않는다(펀치라인에서 딱 끝).
    if _is_sentence_final(segs[idx].text):
        return end
    limit = clip.end + max_extend
    while idx + 1 < len(segs) and segs[idx + 1].start - end <= pause_gap and segs[idx + 1].end <= limit:
        idx += 1
        end = segs[idx].end
        if _is_sentence_final(segs[idx].text):
            break  # 문장이 끝나는 지점에 도달하면 더 늘리지 않는다
    return end


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


def _precise_cache_find(
    cache_dir: Path, model: str, start: float, end: float
) -> list[Segment] | None:
    """이전 렌더에서 저장한 정밀 재전사 결과 중 [start, end]를 덮는 것을 찾는다.

    large-v3 CPU 재전사는 클립당 1~3분 걸리는데, 편집기에서 제목/폰트/위치만 바꿔
    재렌더할 때마다 같은 구간을 매번 다시 전사하는 게 렌더가 느린 주범이었다.
    세그먼트 타임스탬프는 원본 절대시간이라, 캐시 창이 현재 클립 구간을 포함하기만 하면
    구간이 다소 달라져도(스냅/살짝 트림) 그대로 재사용할 수 있다."""
    if not cache_dir.exists():
        return None
    for f in cache_dir.glob(f"*_{model}.json"):
        try:
            a_str, b_str = f.stem.rsplit(f"_{model}", 1)[0].split("_")[:2]
            a, b = float(a_str), float(b_str)
        except ValueError:
            continue
        if a <= start + 0.01 and b >= end - 0.01:
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
    cache_dir: Path, model: str, start: float, end: float, segs: list[Segment]
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{start:.2f}_{end:.2f}_{model}.json"
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
) -> list[Path]:
    """analyze()가 골라둔 후보 중 clip_indices(0-based, 배열 순서 기준)만 정밀
    재전사 + 렌더링한다. 결과 파일은 video_dir/clips/short_<원래순번>.mp4 로 저장된다.

    progress(message, pct)로 호출되며, 클립 개수만큼 균등 분할한 뒤 각 클립을
    재전사(전반 30%)/렌더링(후반 70%) 두 단계로 나눠 진행률을 채운다."""
    # 진행률 콜백을 방어적으로 감싼다: UI 표시 오류가 전사/렌더를 죽이거나 폴백을 유발하지 않게.
    _raw_progress = progress
    progress = lambda message, pct, eta=None: _safe_progress(_raw_progress, message, pct, eta)  # noqa: E731
    cfg = load_config(config_path)
    clips = load_clips_json(video_dir / "clips.json")
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
    END_BUFFER_SEC = 8.0    # clip.end 뒤로 이만큼 더 전사해서 문장이 끝나는 지점을 찾는다
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

    total = len(clip_indices)
    step = 100 / total if total else 100

    for i, idx in enumerate(clip_indices):
        clip = clips[idx]
        base = i * step
        # 최종 화면 자막은 유튜브 자동자막(부정확)이 아니라 이 정밀 재전사 결과를 쓴다.
        # precise_model_size로 정밀 재전사만 더 정확한 모델(예: large-v3)로 올릴 수 있다
        # (짧은 선택 클립에만 돌리므로 전체 영상을 큰 모델로 돌리는 부담 없이 정확도만 취함).
        # 성경 고유명사 상시 사전(정적) + 이 클립에서 뽑은 고유명사(동적)를 함께 hotwords로 넣어
        # large-v3가 이름 철자를 맞추게 한다(룻→'루시', 기드온→'기도원' 류 방지, 이중 방어).
        # 사용자가 자막 편집기에서 자막을 확정했으면 재전사 없이 그대로 렌더한다
        # (편집 결과가 최우선이고, 느린 large-v3 재전사도 건너뛰어 훨씬 빠르다).
        if getattr(clip, "caption_overrides", None):
            # 사용자가 직접 구간을 자른 경우(trimmed)엔 그 길이를 그대로 존중한다.
            # 아니면, 편집 자막 마지막 줄이 잘리지 않게/문장 중간에 끊기지 않게 끝을 확장한다.
            if not getattr(clip, "trimmed", False):
                last_ov_end = max((float(o["end"]) for o in clip.caption_overrides), default=clip.end)
                clip.end = max(clip.end, last_ov_end + 0.3)
                clip.end = _snap_clip_end_to_sentence(clip, base_segments)
            out_path = video_dir / "clips" / f"short_{idx+1}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _run_with_progress_ticker(
                lambda: render_clip(video_path, [], clip, out_path, cfg["render"], cfg["captions"]),
                start_pct=base, end_pct=base + step, progress=progress,
                message=f"[{idx+1}/{total}] 편집 자막으로 렌더링 중: {clip.title}",
                est_seconds=max(15.0, (clip.end - clip.start) * 0.9),
            )
            out_path.with_suffix(".src").write_text(
                render_signature(clip.start, clip.end), encoding="utf-8"
            )
            outputs.append(out_path)
            continue

        hotwords = " ".join(filter(None, [w.get("bible_hotwords", ""), " ".join(clip.keywords)])).strip() or None
        clip_len = clip.end - clip.start
        precise_model = w.get("precise_model_size", w["model_size"])
        cache_dir = video_dir / "precise_cache"

        # 같은 구간을 이미 정밀 재전사했다면 재사용한다. 편집기에서 제목/폰트/위치만 바꿔
        # 재렌더할 때도 클립당 1~3분짜리 large-v3 CPU 재전사를 매번 다시 돌리던 것이
        # 렌더가 느린 주범이었다 — 캐시 적중 시 그 시간이 통째로 사라진다.
        tr_a = max(0.0, clip.start - START_BUFFER_SEC)
        tr_b = clip.end + END_BUFFER_SEC
        cached_segs = _precise_cache_find(cache_dir, precise_model, tr_a, clip.end)
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
            def _precise(vad: bool) -> list[Segment]:
                return transcribe_clip_precise(
                    video_path, tr_a, tr_b,
                    model_size=precise_model,
                    device=w["device"], compute_type=w["compute_type"],
                    language=w["language"], vad_filter=vad,
                    initial_prompt=w.get("initial_prompt"),
                    hotwords=hotwords,
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
                except Exception:  # noqa: BLE001 - 재시도도 실패하면 아래 원본 폴백
                    pass
            # 그래도 부실하면 원본(유튜브 자동자막/medium) 전사로 폴백해 자막이 비는 것만은 막는다.
            if base_n >= 5 and precise_n < max(3, int(base_n * 0.35)):
                progress(
                    f"[{idx+1}/{total}] 정밀 자막 부실({precise_n}단어) → 원본 자막({base_n}단어)으로 대체",
                    base + step * 0.5,
                )
                _rlog(video_dir, f"clip{idx} 폴백: precise {precise_n}단어 < base {base_n}단어의 35%")
                segs = base_segments
            elif segs and precise_n > 0:
                _rlog(video_dir, f"clip{idx} 정밀 자막 사용: {precise_n}단어 (base {base_n}단어)")
                # 건강한 정밀 결과만 캐시한다(부실 결과를 캐시하면 다음 렌더가 재시도 기회를 잃는다).
                _precise_cache_save(cache_dir, precise_model, tr_a, tr_b, segs)
        # 경계 스냅 기준: 정밀 전사가 건강하면 그것을 쓴다 — large-v3는 구두점 있는 진짜
        # 문장 단위라 "문장 중간 끊김/앞 문장 꼬리 시작"을 정확히 잡는다. 유튜브 자막 조각
        # (base)은 문장 경계가 아니어서 스냅이 어색했다(실측 불만). 폴백 시에만 base 사용.
        # 사용자가 직접 구간을 자른 경우(trimmed)엔 건드리지 않는다.
        used_precise = bool(segs) and (segs is not base_segments)
        if not getattr(clip, "trimmed", False):
            snap_src = segs if used_precise else (base_segments or segs)
            if used_precise:
                new_start = _snap_clip_start_to_sentence(clip, snap_src)
                if new_start != clip.start:
                    _rlog(video_dir, f"clip{idx} 시작 문장 스냅: {clip.start:.2f} -> {new_start:.2f}")
                    clip.start = new_start
            new_end = _snap_clip_end_to_sentence(clip, snap_src)
            if new_end != clip.end:
                _rlog(video_dir, f"clip{idx} 끝 문장 스냅: -> {new_end:.2f}")
                clip.end = new_end
            # 길이 절대 규칙: 쇼츠는 1분 내외. 스냅까지 끝난 최종 길이가 상한(max+5초)을
            # 넘으면 끝(펀치라인)은 지키고 시작을 당겨 상한 안으로 넣는다. 선정 단계의
            # 상한(1.5배=90초)만으로는 60~90초 후보가 그대로 렌더돼 "1분 내외"가 깨졌다.
            hard_len = float(cfg["highlights"]["max_duration_sec"]) + 5.0
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

    # 렌더 과정에서 스냅/편집자막으로 clip.end가 조정됐을 수 있다. 이를 clips.json에 반영해
    # 검토 UI의 'N초' 라벨(= end-start)이 실제 렌더된 영상 길이와 일치하게 한다.
    # (기존엔 원본 end를 그대로 표기해 "49초"인데 실제 58초처럼 어긋나던 문제.)
    save_clips_json(clips, video_dir / "clips.json")

    progress("모든 클립 렌더링 완료", 100)
    return outputs


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
