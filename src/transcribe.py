"""faster-whisper를 이용한 한국어 전사 (단어 단위 타임스탬프 포함)"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from faster_whisper import WhisperModel


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass
class Transcript:
    language: str
    duration_sec: float
    segments: list[Segment]

    def to_json(self) -> dict:
        return {
            "language": self.language,
            "duration_sec": self.duration_sec,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [asdict(w) for w in s.words],
                }
                for s in self.segments
            ],
        }

    def to_plain_text_with_timestamps(self) -> str:
        """Claude 프롬프트에 넣기 좋은, 타임스탬프가 붙은 평문 텍스트로 변환."""
        lines = []
        for seg in self.segments:
            ts = _format_ts(seg.start)
            lines.append(f"[{ts}] {seg.text.strip()}")
        return "\n".join(lines)


def _format_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


_model_cache: dict[str, WhisperModel] = {}


def _get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    key = f"{model_size}:{device}:{compute_type}"
    if key not in _model_cache:
        _model_cache[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model_cache[key]


def transcribe(
    audio_path: Path,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "ko",
    vad_filter: bool = True,
    on_segment: Optional[Callable[[float, float], None]] = None,
) -> Transcript:
    model = _get_model(model_size, device, compute_type)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=vad_filter,
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        words = [
            Word(start=w.start, end=w.end, text=w.word.strip())
            for w in (seg.words or [])
        ]
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text, words=words))
        # faster-whisper는 세그먼트를 순차 생성(streaming)하므로, 지금까지 처리한 지점(seg.end)을
        # 전체 길이(info.duration)와 비교하면 실시간 전사 진행률을 알 수 있다.
        if on_segment is not None and info.duration:
            on_segment(seg.end, info.duration)

    return Transcript(
        language=info.language,
        duration_sec=info.duration,
        segments=segments,
    )


def transcribe_and_save(
    audio_path: Path,
    output_json_path: Path,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "ko",
    vad_filter: bool = True,
    on_segment: Optional[Callable[[float, float], None]] = None,
) -> Transcript:
    transcript = transcribe(
        audio_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
        vad_filter=vad_filter,
        on_segment=on_segment,
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(transcript.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return transcript


def transcribe_clip_precise(
    video_path: Path,
    clip_start: float,
    clip_end: float,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "ko",
    vad_filter: bool = True,
) -> list[Segment]:
    """클립 구간(1~5분 이내의 짧은 분량)만 오려서 정밀 재전사한다.

    하이라이트 탐색 단계는 속도를 위해 유튜브 자동 자막을 쓰지만, 실제로 화면에
    번인되는 최종 자막은 이 함수로 얻은 결과를 쓴다 (짧은 구간만 다시 돌리므로
    전체 영상을 Whisper로 돌리는 것보다 훨씬 빠르면서도 정확도는 그대로 유지).
    반환되는 Segment/Word의 타임스탬프는 클립 상대시간이 아니라 원본 영상 기준
    절대시간이다 (clip_start만큼 이미 더해져 있음).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_audio_path = Path(tmpdir) / "clip.wav"
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(clip_start), "-to", str(clip_end), "-i", str(video_path),
            "-ac", "1", "-ar", "16000",
            str(clip_audio_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"클립 오디오 추출 실패:\n{proc.stderr[-2000:]}")

        transcript = transcribe(
            clip_audio_path,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_filter=vad_filter,
        )

    shifted_segments: list[Segment] = []
    for seg in transcript.segments:
        shifted_words = [
            Word(start=w.start + clip_start, end=w.end + clip_start, text=w.text)
            for w in seg.words
        ]
        shifted_segments.append(
            Segment(start=seg.start + clip_start, end=seg.end + clip_start, text=seg.text, words=shifted_words)
        )
    return shifted_segments


if __name__ == "__main__":
    import sys
    import time

    audio_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else audio_path.with_suffix(".transcript.json")

    t0 = time.time()
    result = transcribe_and_save(audio_path, out_path)
    elapsed = time.time() - t0
    print(f"전사 완료: {len(result.segments)}개 구간, 길이 {result.duration_sec:.1f}초, 소요 {elapsed:.1f}초")
    print(f"저장 위치: {out_path}")
