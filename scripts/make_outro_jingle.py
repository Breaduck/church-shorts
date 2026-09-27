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


# ── 2차(2026-09-27): "다 별로, 현대처럼 경쾌하고 간편하게" → 짧게 끊어 치는 리듬 + 밝은 착지 ──
# 마림바(기음+4배·10배 배음, 빠른 감쇠)와 플럭(Karplus-Strong)으로 스타카토를 만들고, 마지막 음에만
# 작은 저음 '툭'과 반짝이는 윗배음을 얹어 로고가 '딱' 찍히는 느낌을 준다. 잔향은 얕게(깔끔하게).
MARIMBA = [(1, 1.0, 1.0), (3.9, 0.35, 3.0), (9.8, 0.12, 6.0)]


def pluck(freq: float, start: float, length: float, bright: float = 0.5, seed: int = 1) -> np.ndarray:
    out = np.zeros(int(SR * DUR))
    n0 = int(SR * start)
    n = min(int(SR * length), len(out) - n0)
    period = max(2, int(SR / freq))
    buf = np.random.default_rng(seed).uniform(-1, 1, period)
    sig = np.empty(n)
    for i in range(n):
        sig[i] = buf[i % period]
        buf[i % period] = 0.5 * (buf[i % period] + buf[(i + 1) % period]) * (0.994 + 0.005 * bright)
    out[n0:n0 + n] = sig
    return out


def thump(start: float, amp: float = 0.5) -> np.ndarray:
    """착지 순간의 작은 저음 '툭'(킥처럼 피치가 떨어지는 사인)."""
    out = np.zeros(int(SR * DUR))
    n0 = int(SR * start)
    n = int(SR * 0.25)
    t = np.arange(n) / SR
    f = 55 + 70 * np.exp(-t * 30)
    out[n0:n0 + n] = amp * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18)
    return out


def _mar(note: str, start: float, length: float = 0.5, decay: float = 9.0, amp: float = 0.8) -> np.ndarray:
    return tone(hz(note), start, length, MARIMBA, decay, attack=0.002) * amp


def pop() -> Path:
    """마림바 4연음 상승(도–미–솔–도) 후 높은 도 착지 + 툭. 통통 튀는 '따라라–띵'."""
    step = 0.085
    x = sum(_mar(n, i * step, 0.35, 12.0, 0.7) for i, n in enumerate(["C5", "E5", "G5"]))
    x = x + _mar("C6", 3 * step, 1.2, 3.5, 0.9) + _mar("G5", 3 * step, 1.0, 4.0, 0.35) + _mar("E6", 3 * step, 1.0, 5.0, 0.25)
    x = x + thump(3 * step, 0.45)
    return finish(*reverb(x, seconds=0.6, wet=0.12), "pop")


def bounce() -> Path:
    """플럭 '따–따–딴!'(솔 솔 도) — 짧고 리드미컬한 사운드 로고."""
    x = (
        pluck(hz("G5"), 0.00, 0.20, seed=1) * 0.6
        + pluck(hz("G5"), 0.14, 0.20, seed=2) * 0.6
        + pluck(hz("C6"), 0.30, 1.3, bright=1.0, seed=3) * 0.8
        + pluck(hz("E5"), 0.30, 1.3, seed=4) * 0.45
        + _mar("C7", 0.30, 0.8, 5.0, 0.18)
        + thump(0.30, 0.5)
    )
    return finish(*reverb(x, seconds=0.5, wet=0.10), "bounce")


def spark() -> Path:
    """빠른 5음계 상승 런(0.3초) 뒤 밝은 화음 '반짝' — 경쾌한 시작 느낌."""
    run = ["C5", "D5", "E5", "G5", "A5", "C6"]
    x = sum(_mar(n, i * 0.055, 0.3, 14.0, 0.55) for i, n in enumerate(run))
    land = len(run) * 0.055
    x = x + _mar("E6", land, 1.2, 3.0, 0.7) + _mar("C6", land, 1.2, 3.2, 0.45) + _mar("G6", land, 1.0, 4.0, 0.3)
    x = x + tone(hz("C8"), land + 0.02, 0.6, [(1, 1.0, 1.0)], 7.0) * 0.08 + thump(land, 0.4)
    return finish(*reverb(x, seconds=0.7, wet=0.14), "spark")


def hook() -> Path:
    """기억에 남는 4음 모티프(도–미–라–솔): 올라갔다 한 음 내려와 안정적으로 끝나는 '징글' 형태."""
    notes = [("C5", 0.00, 0.18), ("E5", 0.13, 0.18), ("A5", 0.26, 0.22), ("G5", 0.44, 1.2)]
    x = sum(
        pluck(hz(n), s, l, bright=0.8, seed=i + 5) * (0.8 if i == 3 else 0.6)
        + _mar(n, s, l, 6.0 if i == 3 else 12.0, 0.35)
        for i, (n, s, l) in enumerate(notes)
    )
    x = x + _mar("C5", 0.44, 1.2, 3.0, 0.3) + thump(0.44, 0.45)
    return finish(*reverb(x, seconds=0.6, wet=0.12), "hook")


if __name__ == "__main__":
    import sys

    fns = {f.__name__: f for f in (chime, warm, amen, pop, bounce, spark, hook)}
    for name in (sys.argv[1:] or fns):
        print(fns[name]())
