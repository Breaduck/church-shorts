"""오디오 에너지 피크 감지 (보조 힌트 전용)

주의: 이 모듈의 결과는 하이라이트 선정의 '참고 신호'일 뿐이다.
실제 하이라이트 판단은 highlights.py에서 Claude가 전사본 내용을 읽고 맥락을 이해해 내린다.
여기서 찾은 피크 구간은 프롬프트에 "이 근처에 반응이 있었다" 정도의 힌트로만 첨부된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class PeakHint:
    start: float
    end: float
    relative_energy: float  # 평균 대비 상대적 에너지 (1.0 = 평균, 2.0 = 평균의 2배)


def detect_peak_hints(
    audio_path: Path,
    frame_length_sec: float = 1.0,
    hop_length_sec: float = 0.5,
    top_k: int = 15,
) -> list[PeakHint]:
    """오디오 RMS 에너지가 평균보다 튀는 구간을 top_k개 찾아 반환 (참고용 힌트)."""
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    frame_length = int(frame_length_sec * sr)
    hop_length = int(hop_length_sec * sr)
    if frame_length <= 0 or hop_length <= 0:
        return []

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return []

    mean_energy = float(np.mean(rms)) or 1e-9
    relative = rms / mean_energy

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    order = np.argsort(relative)[::-1]
    hints: list[PeakHint] = []
    used_times: list[float] = []
    for idx in order:
        t = float(times[idx])
        if any(abs(t - u) < 5.0 for u in used_times):
            continue  # 너무 가까운 피크는 중복으로 취급
        hints.append(
            PeakHint(
                start=max(0.0, t - frame_length_sec / 2),
                end=t + frame_length_sec / 2,
                relative_energy=float(relative[idx]),
            )
        )
        used_times.append(t)
        if len(hints) >= top_k:
            break

    hints.sort(key=lambda h: h.start)
    return hints


def format_hints_for_prompt(hints: list[PeakHint]) -> str:
    """Claude 프롬프트에 곁들일 짧은 참고 힌트 텍스트."""
    if not hints:
        return "(오디오 에너지 힌트 없음)"
    lines = ["다음은 오디오 에너지가 평균보다 튀었던 구간들이다 (참고용 힌트일 뿐, 이것만으로 판단하지 말 것):"]
    for h in hints:
        m1, s1 = divmod(int(h.start), 60)
        m2, s2 = divmod(int(h.end), 60)
        lines.append(f"  - {m1:02d}:{s1:02d}~{m2:02d}:{s2:02d} (평균 대비 {h.relative_energy:.1f}배)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    hints = detect_peak_hints(Path(sys.argv[1]))
    print(format_hints_for_prompt(hints))
