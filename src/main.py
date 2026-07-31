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
from pathlib import Path

import yaml

from src.audio_peaks import detect_peak_hints
from src.download import download_video
from src.highlights import (
    Clip,
    build_prompt,
    load_clips_json,
    save_clips_json,
    save_prompt_for_manual_mode,
    select_highlights_auto,
)
from src.render import render_clip
from src.transcribe import Segment, Word, transcribe_and_save, transcribe_clip_precise, Transcript
from src.youtube_captions import get_transcript_from_youtube


def load_config(path: Path = Path("config.yaml")) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _default_progress(message: str, pct: float, eta_seconds: float | None = None) -> None:
    eta = "" if eta_seconds is None else f" (~{int(round(eta_seconds))}s)"
    print(f"[{pct:5.1f}%]{eta} {message}")


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
        eta = max(0.0, total - done_est)
        self._cb(cur.message, pct, eta)

    def _tick(self) -> None:
        while not self._done.wait(timeout=0.5):
            with self._lock:
                cur = self._stages[self._i]
                elapsed = time.time() - self._stage_start
                target = min(0.95, elapsed / cur.est)
                if target > self._frac:
                    self._frac = target
                self._emit_locked()

    def set_fraction(self, frac: float, message: str | None = None) -> None:
        with self._lock:
            if message:
                self._stages[self._i].message = message
            self._frac = max(self._frac, min(1.0, frac))
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
            self._cb(message, 100.0, 0.0)


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
        progress(message, start_pct + (end_pct - start_pct) * frac, eta)

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
    url: str, config_path: Path = Path("config.yaml"), progress=_default_progress
) -> tuple[Path, list[Clip]]:
    """다운로드 -> 전사 -> 하이라이트 후보 선정까지만 수행하고 (렌더링 없음),
    video_dir와 배열 순서=바이럴 예상 순위인 클립 후보 목록을 반환한다.

    progress(message, pct, eta_seconds)로 호출된다. pct는 0~100 전체 진행률, eta_seconds는
    남은 예상시간(초). 진행률은 단계별 '예상 소요시간' 비율로 채워지므로 바가 부드럽게 움직이고
    ETA도 처음부터 끝까지 일관된다(StageProgress 참고)."""
    cfg = load_config(config_path)
    output_root = Path("output")

    # 각 단계 예상시간(초). 실제 소요와 다르면 런타임에 보정한다(다운로드 %/전사 %/전사 분기).
    stages = [
        _Stage("download", "영상 다운로드 중...", est=40),
        _Stage("transcript", "자막 준비 중...", est=12),
        _Stage("hints", "핵심 구간 분석 중...", est=15),
        _Stage("highlight", "하이라이트 후보 선정 중...", est=90),
    ]
    sp = StageProgress(progress, stages)
    try:
        dl = download_video(
            url, output_root,
            on_progress=lambda p: sp.set_fraction(p / 100.0, f"영상 다운로드 중... {p:.0f}%"),
        )
        video_dir = output_root / dl.video_id

        # 2) 전사 --------------------------------------------------------------
        sp.advance("자막 준비 중...")
        transcript_path = video_dir / "transcript.json"
        if transcript_path.exists():
            sp.set_fraction(1.0, "기존 자막 재사용")
            transcript = Transcript(**json_load_transcript(transcript_path))
        else:
            sp.message("유튜브 자동 자막 확인 중...")
            transcript = get_transcript_from_youtube(url, video_dir, dl.duration_sec)
            if transcript is None or not transcript.segments:
                # 로컬 전사는 영상 길이에 비례해 오래 걸린다 → 예상시간을 크게 잡아 ETA를 맞춘다.
                sp.set_current_est(max(30.0, dl.duration_sec * 0.5))
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
        if clips_path.exists():
            sp.finish(f"완료: 기존 후보 재사용")
            return video_dir, load_clips_json(clips_path)

        # 3) 오디오 에너지 힌트 -------------------------------------------------
        sp.advance("핵심 구간 분석 중...")
        peak_hints = []
        if cfg["audio_peaks"].get("enabled", True):
            peak_hints = detect_peak_hints(
                dl.video_path,
                frame_length_sec=cfg["audio_peaks"]["frame_length_sec"],
                hop_length_sec=cfg["audio_peaks"]["hop_length_sec"],
            )

        if h["mode"] == "manual":
            prompt = build_prompt(
                transcript=transcript, peak_hints=peak_hints,
                min_clips=h["min_clips"], max_clips=h["max_clips"],
                min_duration_sec=h["min_duration_sec"], max_duration_sec=h["max_duration_sec"],
                categories=h["categories"], video_duration_sec=transcript.duration_sec,
            )
            prompt_path = video_dir / "highlight_prompt.txt"
            save_prompt_for_manual_mode(prompt, prompt_path)
            raise RuntimeError(
                f"manual 모드: 프롬프트가 {prompt_path}에 저장됨. "
                f"Claude Code 세션에서 하이라이트를 골라 {clips_path}에 저장한 뒤 다시 시도하세요."
            )

        # 4) 하이라이트 선정 (opaque: 티커가 부드럽게 채움) ---------------------
        sp.advance("하이라이트 후보 선정 중 (AI 분석)...")
        clips = select_highlights_auto(
            transcript=transcript, peak_hints=peak_hints,
            min_clips=h["min_clips"], max_clips=h["max_clips"],
            min_duration_sec=h["min_duration_sec"], max_duration_sec=h["max_duration_sec"],
            categories=h["categories"],
        )
        save_clips_json(clips, clips_path)
        sp.finish(f"완료: {len(clips)}개 후보 선정")
        return video_dir, clips
    finally:
        sp.stop()


def _snap_clip_end_to_sentence(
    clip: Clip, segs: list[Segment], max_extend: float = 10.0, pause_gap: float = 0.7
) -> float:
    """Claude가 대략 지정한 clip.end가 문장/발화 중간을 잘라 "말이 안 끝났는데 뚝 끊기는"
    문제를 막는다. clip.end 지점부터 발화가 계속 이어지면(다음 세그먼트가 pause_gap 이내로
    시작) 자연스러운 멈춤(그보다 큰 침묵)이 나올 때까지 끝을 늘린다. max_extend까지만 확장.

    문장부호가 없어(정밀 재전사) 마침표로 판단 불가하므로 '침묵 간격'을 문장 경계 신호로 쓴다.
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
    limit = clip.end + max_extend
    while idx + 1 < len(segs) and segs[idx + 1].start - end <= pause_gap and segs[idx + 1].end <= limit:
        idx += 1
        end = segs[idx].end
    return end


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
    cfg = load_config(config_path)
    clips = load_clips_json(video_dir / "clips.json")
    video_path = video_dir / "source.mp4"
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
    END_BUFFER_SEC = 5.0  # clip.end 뒤로 이만큼 더 전사해서 문장이 끝나는 지점을 찾는다

    # 정밀 재전사가 이따금 한두 단어만 뱉고 사실상 실패할 때가 있다(자막이 통째로 비는
    # 치명적 결과 — 실제로 겪음). 그럴 때 폴백할 원본 전사(유튜브 자동자막/medium)를 미리 로드.
    base_segments: list[Segment] = []
    base_transcript_path = video_dir / "transcript.json"
    if base_transcript_path.exists():
        base_segments = json_load_transcript(base_transcript_path)["segments"]

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
            outputs.append(out_path)
            continue

        hotwords = " ".join(filter(None, [w.get("bible_hotwords", ""), " ".join(clip.keywords)])).strip() or None
        # large-v3 CPU 재전사는 클립 하나에 1~3분씩 걸리는데 그동안 진행률이 한 지점에
        # 멈춰 있으면 사용자가 "안 만들어진다"고 오해한다(실제로 겪은 피드백). 재전사/렌더
        # 두 무진행 구간 모두 흉내 진행률 티커로 부드럽게 채워 "작동 중"임을 보여준다.
        clip_len = clip.end - clip.start
        # 정밀 재전사(faster-whisper)는 특정 클립에서 'maximum decoding length must be > 0'
        # 같은 예외로 통째로 죽는 경우가 있다(실제 발생). 그러면 렌더 전체가 실패하므로,
        # 예외는 삼키고 빈 결과로 둔 뒤 아래 폴백(원본 자막)이 자막을 채우게 한다.
        try:
            segs = _run_with_progress_ticker(
                lambda: transcribe_clip_precise(
                    video_path, clip.start, clip.end + END_BUFFER_SEC,
                    model_size=w.get("precise_model_size", w["model_size"]),
                    device=w["device"], compute_type=w["compute_type"],
                    language=w["language"], vad_filter=w.get("vad_filter", True),
                    initial_prompt=w.get("initial_prompt"),
                    hotwords=hotwords,
                ),
                start_pct=base, end_pct=base + step * 0.5, progress=progress,
                message=f"[{idx+1}/{total}] 자막 정밀 인식 중: {clip.title}",
                est_seconds=max(20.0, clip_len * 1.8),
            )
        except Exception as e:  # noqa: BLE001 - 정밀 재전사 실패해도 원본 자막으로 계속 진행
            progress(f"[{idx+1}/{total}] 정밀 인식 실패({e}) → 원본 자막 사용", base + step * 0.5)
            segs = []
        # 방어: 정밀 재전사 단어 수가 원본 전사보다 현저히 적으면(재전사 실패로 판단) 원본
        # 전사 단어로 대체해 자막이 비는 걸 막는다. 원본에 충분한 단어가 있을 때만 발동.
        precise_n = _count_words(segs, clip.start, clip.end)
        base_n = _count_words(base_segments, clip.start, clip.end)
        if base_n >= 5 and precise_n < max(3, int(base_n * 0.5)):
            progress(
                f"[{idx+1}/{total}] 정밀 자막 부실({precise_n}단어) → 원본 자막({base_n}단어)으로 대체",
                base + step * 0.5,
            )
            segs = base_segments
        # 스냅은 반드시 신뢰도 높은 원본 전사로 판단한다(정밀 재전사는 실패 시 세그먼트가
        # 비어 문장 끝 감지가 무력화됨). 원본이 없을 때만 정밀 결과로 폴백.
        # 사용자가 직접 구간을 자른 경우(trimmed)엔 자동 확장하지 않는다.
        if not getattr(clip, "trimmed", False):
            new_end = _snap_clip_end_to_sentence(clip, base_segments or segs)
            if new_end != clip.end:
                clip.end = new_end
        out_path = video_dir / "clips" / f"short_{idx+1}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _run_with_progress_ticker(
            lambda: render_clip(video_path, segs, clip, out_path, cfg["render"], cfg["captions"]),
            start_pct=base + step * 0.5, end_pct=base + step, progress=progress,
            message=f"[{idx+1}/{total}] 쇼츠 렌더링 중: {clip.title}",
            est_seconds=max(15.0, clip_len * 0.9),
        )
        outputs.append(out_path)

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
