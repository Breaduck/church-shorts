"""아웃트로 로고용 짧은 사운드 로고(징글)를 직접 합성한다 — 외부 음원·저작권 없음.

사용자 요청(2026-09-27): "정다운교회 로고 나올 때 어울리는 간단한 뮤직 — 현대 로고처럼".
기업 사운드 로고처럼 2초 안에 끝나는 짧고 따뜻한 상승 음형 3종을 만든다.

    venv\\Scripts\\python.exe scripts/make_outro_jingle.py

결과: assets/outro/jingle_chime.wav · jingle_warm.wav · jingle_amen.wav (48kHz 스테레오)
config.yaml의 render.outro.sound_path로 고른다(빈 값이면 예전처럼 무음).
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

SR = 48000
DUR = 2.0
OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "outro"


def hz(note: str) -> float:
    names = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    pitch = names[note[0]] + (1 if "#" in note else 0)
    octave = int(note[-1])
    return 440.0 * 2 ** ((pitch + 12 * (octave + 1) - 69) / 12)


def tone(freq: float, start: float, length: float, partials, decay: float, attack: float = 0.004) -> np.ndarray:
    """배음 합성 + 빠른 어택 + 지수 감쇠. partials = [(배수, 크기, 감쇠 가속)]."""
    out = np.zeros(int(SR * DUR))
    n0 = int(SR * start)
    n = min(int(SR * length), len(out) - n0)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for mult, amp, dmul in partials:
        sig += amp * np.sin(2 * np.pi * freq * mult * t) * np.exp(-t * decay * dmul)
    env = np.minimum(1.0, t / attack)
    out[n0:n0 + n] = sig * env
    return out


BELL = [(1, 1.0, 1.0), (2.0, 0.45, 1.6), (3.0, 0.18, 2.4), (4.2, 0.10, 3.5), (5.4, 0.05, 4.5)]
EPIANO = [(1, 1.0, 1.0), (2, 0.30, 1.8), (3, 0.08, 3.0), (7, 0.02, 6.0)]
ORGAN = [(1, 1.0, 0.0), (2, 0.5, 0.0), (3, 0.25, 0.0), (4, 0.12, 0.0)]


def reverb(x: np.ndarray, seconds: float = 1.1, wet: float = 0.28, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """노이즈 감쇠 임펄스로 만든 간단한 홀 잔향(좌우 다른 임펄스 → 넓게)."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    chans = []
    for _ in range(2):
        ir = rng.standard_normal(n) * np.exp(-t * 5.0)
        ir[0] = 0.0
        ir /= np.sqrt(np.sum(ir ** 2))
        chans.append((1 - wet) * x + wet * fftconvolve(x, ir)[: len(x)])
    return chans[0], chans[1]


def finish(left: np.ndarray, right: np.ndarray, name: str) -> Path:
    stereo = np.stack([left, right], axis=1)
    fade = int(SR * 0.35)
    stereo[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
    stereo /= max(1e-9, np.abs(stereo).max())
    stereo *= 10 ** (-3 / 20)  # -3 dBFS 피크
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"jingle_{name}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((stereo * 32767).astype("<i2").tobytes())
    return path


def chime() -> Path:
    """맑은 벨 상승 3음 + 화음 착지(G–C–E → C major add9). 가장 '사운드 로고'다운 느낌."""
    x = (
        tone(hz("G5"), 0.00, 1.6, BELL, 3.2) * 0.8
        + tone(hz("C6"), 0.16, 1.6, BELL, 3.0) * 0.8
        + tone(hz("E6"), 0.32, 1.6, BELL, 2.8) * 0.8
        + tone(hz("C5"), 0.50, 1.5, BELL, 2.0) * 0.6
        + tone(hz("G5"), 0.50, 1.5, BELL, 2.2) * 0.45
        + tone(hz("D6"), 0.50, 1.5, BELL, 2.4) * 0.35
        + tone(hz("C3"), 0.50, 1.5, EPIANO, 1.6) * 0.5
    )
    return finish(*reverb(x), "chime")


def warm() -> Path:
    """부드러운 일렉피아노로 5도→옥타브 상승 후 따뜻한 화음(C–G–C → Cmaj9). 차분·신뢰감."""
    x = (
        tone(hz("C4"), 0.00, 1.8, EPIANO, 2.2) * 0.7
        + tone(hz("G4"), 0.20, 1.8, EPIANO, 2.2) * 0.7
        + tone(hz("C5"), 0.40, 1.6, EPIANO, 1.8) * 0.7
        + tone(hz("E5"), 0.62, 1.4, EPIANO, 1.6) * 0.5
        + tone(hz("B4"), 0.62, 1.4, EPIANO, 1.8) * 0.3
        + tone(hz("D5"), 0.62, 1.4, EPIANO, 1.8) * 0.3
        + tone(hz("C3"), 0.62, 1.4, EPIANO, 1.4) * 0.6
    )
    return finish(*reverb(x, seconds=1.4, wet=0.33), "warm")


def amen() -> Path:
    """교회다운 '아-멘' 종지(F → C) 오르간 화음. 찬송가 끝 느낌."""
    def chord(notes, start, length):
        y = sum(tone(hz(n), start, length, ORGAN, 0.0, attack=0.06) for n in notes)
        n0, n1 = int(SR * start), int(SR * min(DUR, start + length))
        env = np.ones(int(SR * DUR))
        rel = int(SR * 0.12)
        env[n1 - rel:n1] = np.linspace(1, 0, rel)
        env[n1:] = 0
        env[:n0] = 0
        return y * env
    x = chord(["F3", "A3", "C4", "F4"], 0.00, 0.72) * 0.22 + chord(["C3", "G3", "C4", "E4"], 0.68, 1.3) * 0.22
    return finish(*reverb(x, seconds=1.6, wet=0.38), "amen")


if __name__ == "__main__":
    for fn in (chime, warm, amen):
        print(fn())
