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


def analyze(url: str, config_path: Path = Path("config.yaml"), progress=print) -> tuple[Path, list[Clip]]:
    """다운로드 -> 전사 -> 하이라이트 후보 선정까지만 수행하고 (렌더링 없음),
    video_dir와 배열 순서=바이럴 예상 순위인 클립 후보 목록을 반환한다."""
    cfg = load_config(config_path)
    output_root = Path("output")

    progress("다운로드 중...")
    dl = download_video(url, output_root)
    video_dir = output_root / dl.video_id

    transcript_path = video_dir / "transcript.json"
    if transcript_path.exists():
        progress("기존 전사 결과 재사용")
        transcript = Transcript(**json_load_transcript(transcript_path))
    else:
        progress("유튜브 자동 자막 확인 중...")
        transcript = get_transcript_from_youtube(url, video_dir, dl.duration_sec)
        if transcript is None or not transcript.segments:
            progress("자동 자막 없음. 로컬 전사로 대체 중 (시간이 걸릴 수 있음)...")
            w = cfg["whisper"]
            transcript = transcribe_and_save(
                dl.video_path, transcript_path,
                model_size=w["model_size"], device=w["device"], compute_type=w["compute_type"],
                language=w["language"], vad_filter=w.get("vad_filter", True),
            )
        transcript_path.write_text(
            json.dumps(transcript.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    clips_path = video_dir / "clips.json"
    h = cfg["highlights"]
    if clips_path.exists():
        progress("기존 하이라이트 선정 결과 재사용")
        clips = load_clips_json(clips_path)
    else:
        progress("오디오 에너지 힌트 감지 중...")
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

        progress("하이라이트 후보 선정 중 (로컬 Claude Code 호출)...")
        clips = select_highlights_auto(
            transcript=transcript, peak_hints=peak_hints,
            min_clips=h["min_clips"], max_clips=h["max_clips"],
            min_duration_sec=h["min_duration_sec"], max_duration_sec=h["max_duration_sec"],
            categories=h["categories"],
        )
        save_clips_json(clips, clips_path)

    progress(f"완료: {len(clips)}개 후보 선정")
    return video_dir, clips


def render_selected(
    video_dir: Path,
    clip_indices: list[int],
    config_path: Path = Path("config.yaml"),
    progress=print,
) -> list[Path]:
    """analyze()가 골라둔 후보 중 clip_indices(0-based, 배열 순서 기준)만 정밀
    재전사 + 렌더링한다. 결과 파일은 video_dir/clips/short_<원래순번>.mp4 로 저장된다."""
    cfg = load_config(config_path)
    clips = load_clips_json(video_dir / "clips.json")
    video_path = video_dir / "source.mp4"
    w = cfg["whisper"]
    outputs = []

    for idx in clip_indices:
        clip = clips[idx]
        progress(f"[{idx+1}] 정밀 재전사 중: {clip.title}")
        segs = transcribe_clip_precise(
            video_path, clip.start, clip.end,
            model_size=w["model_size"], device=w["device"], compute_type=w["compute_type"],
            language=w["language"], vad_filter=w.get("vad_filter", True),
        )
        out_path = video_dir / "clips" / f"short_{idx+1}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        progress(f"[{idx+1}] 렌더링 중...")
        render_clip(video_path, segs, clip, out_path, cfg["render"], cfg["captions"])
        outputs.append(out_path)

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
