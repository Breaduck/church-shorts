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


def _default_progress(message: str, pct: float) -> None:
    print(f"[{pct:5.1f}%] {message}")


def _run_with_progress_ticker(fn, start_pct: float, end_pct: float, progress, message: str, est_seconds: float):
    """분 단위로 걸릴 수 있는데 중간 진행률을 알 수 없는 단계(예: claude -p 서브프로세스 호출)를
    위한 흉내 진행률바. est_seconds에 걸쳐 start_pct -> end_pct*0.95 정도까지 서서히 채우고,
    실제로 더 오래 걸리면 end_pct 근처에서 멈춰 기다린다 (거짓으로 100%를 찍지 않기 위함)."""
    done = threading.Event()

    def _tick() -> None:
        t0 = time.time()
        while not done.wait(timeout=1.0):
            frac = min(0.95, (time.time() - t0) / est_seconds)
            progress(message, start_pct + (end_pct - start_pct) * frac)

    progress(message, start_pct)
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

    progress(message, pct)로 호출되며 pct는 0~100 사이의 전체 진행률이다. 아래 단계별
    구간으로 나뉜다: 다운로드(0~30) / 전사(30~75) / 오디오 힌트(75~82) / 하이라이트 선정(82~100)."""
    cfg = load_config(config_path)
    output_root = Path("output")

    progress("다운로드 중...", 1)
    dl = download_video(
        url, output_root,
        on_progress=lambda p: progress(f"다운로드 중... {p:.0f}%", 1 + p * 0.29),
    )
    video_dir = output_root / dl.video_id

    transcript_path = video_dir / "transcript.json"
    if transcript_path.exists():
        progress("기존 전사 결과 재사용", 75)
        transcript = Transcript(**json_load_transcript(transcript_path))
    else:
        progress("유튜브 자동 자막 확인 중...", 32)
        transcript = get_transcript_from_youtube(url, video_dir, dl.duration_sec)
        if transcript is None or not transcript.segments:
            progress("자동 자막 없음. 로컬 전사로 대체 중 (시간이 걸릴 수 있음)...", 35)
            w = cfg["whisper"]
            transcript = transcribe_and_save(
                dl.video_path, transcript_path,
                model_size=w["model_size"], device=w["device"], compute_type=w["compute_type"],
                language=w["language"], vad_filter=w.get("vad_filter", True),
                on_segment=lambda seg_end, duration: progress(
                    f"전사 중... {min(100, seg_end / duration * 100):.0f}%",
                    35 + min(1.0, seg_end / duration) * 40,
                ),
            )
        else:
            progress("유튜브 자동 자막 사용", 75)
        transcript_path.write_text(
            json.dumps(transcript.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    clips_path = video_dir / "clips.json"
    h = cfg["highlights"]
    if clips_path.exists():
        progress("기존 하이라이트 선정 결과 재사용", 99)
        clips = load_clips_json(clips_path)
    else:
        progress("오디오 에너지 힌트 감지 중...", 78)
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

        clips = _run_with_progress_ticker(
            lambda: select_highlights_auto(
                transcript=transcript, peak_hints=peak_hints,
                min_clips=h["min_clips"], max_clips=h["max_clips"],
                min_duration_sec=h["min_duration_sec"], max_duration_sec=h["max_duration_sec"],
                categories=h["categories"],
            ),
            start_pct=82, end_pct=99, progress=progress,
            message="하이라이트 후보 선정 중 (로컬 Claude Code 호출)...",
            est_seconds=90,
        )
        save_clips_json(clips, clips_path)

    progress(f"완료: {len(clips)}개 후보 선정", 100)
    return video_dir, clips


def _snap_clip_end_to_sentence(clip: Clip, segs: list[Segment]) -> float:
    """Claude가 초 단위로 대략 지정한 clip.end가 실제 발화 중간을 자르는 경우가 있다
    (예: "기도는 마치 전부와 같습니다 바쁠 때는 그래요"가 575.0~578.2초인데
    end=576.0으로 잘라서 "그래요"가 통째로 잘림).

    주의: faster-whisper의 정밀 재전사 결과는 문장부호(마침표 등)를 전혀 붙이지 않으므로
    "문장부호로 끝나는지"로 판단할 수 없다 (실제로 시도했다가 전혀 매칭 안 됨을 확인함).
    대신 clip.end가 어떤 세그먼트의 '중간'에 걸쳐 있으면, 그 세그먼트를 통째로 포함하도록
    끝을 그 세그먼트의 끝까지 넓힌다 — 이미 말하기 시작한 발화 단위는 끝까지 들려준다."""
    for seg in segs:
        if seg.start < clip.end < seg.end:
            return seg.end
    return clip.end  # 정확히 세그먼트 경계에 걸리거나 못 찾으면 원래 값 유지


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
    w = cfg["whisper"]
    outputs = []
    END_BUFFER_SEC = 5.0  # clip.end 뒤로 이만큼 더 전사해서 문장이 끝나는 지점을 찾는다

    total = len(clip_indices)
    step = 100 / total if total else 100

    for i, idx in enumerate(clip_indices):
        clip = clips[idx]
        base = i * step
        progress(f"[{idx+1}] 정밀 재전사 중: {clip.title}", base)
        segs = transcribe_clip_precise(
            video_path, clip.start, clip.end + END_BUFFER_SEC,
            model_size=w["model_size"], device=w["device"], compute_type=w["compute_type"],
            language=w["language"], vad_filter=w.get("vad_filter", True),
        )
        new_end = _snap_clip_end_to_sentence(clip, segs)
        if new_end != clip.end:
            progress(f"[{idx+1}] 문장이 끊겨서 끝 지점 보정: {clip.end:.1f}s -> {new_end:.1f}s", base + step * 0.3)
            clip.end = new_end
        out_path = video_dir / "clips" / f"short_{idx+1}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        progress(f"[{idx+1}] 렌더링 중...", base + step * 0.35)
        render_clip(video_path, segs, clip, out_path, cfg["render"], cfg["captions"])
        outputs.append(out_path)
        progress(f"[{idx+1}] 완료", base + step)

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
