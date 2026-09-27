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


# ── 3차(2026-09-27): "기억되는 4음 괜찮은데 다른 악기 없나?" → 같은 선율·리듬을 악기만 바꿔 여러 벌 ──
HOOK_NOTES = [("C5", 0.00, 0.18), ("E5", 0.13, 0.18), ("A5", 0.26, 0.22), ("G5", 0.44, 1.2)]


def _env(n: int, attack: float, release: float) -> np.ndarray:
    t = np.arange(n) / SR
    env = np.minimum(1.0, t / max(attack, 1e-4))
    r = min(n, int(SR * release))
    if r > 0:
        env[-r:] *= np.linspace(1, 0, r)
    return env


def inst_note(kind: str, freq: float, start: float, length: float, last: bool) -> np.ndarray:
    out = np.zeros(int(SR * DUR))
    n0 = int(SR * start)
    n = min(int(SR * length), len(out) - n0)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    if kind == "piano":
        # 살짝 늘어난 배음(현의 강성) + 높은 배음일수록 빨리 사라짐 + 해머 잡음
        for k in range(1, 9):
            fk = freq * k * np.sqrt(1 + 0.0004 * k * k)
            sig += (1 / k ** 1.3) * np.sin(2 * np.pi * fk * t) * np.exp(-t * (2.5 + 1.2 * k))
        sig += np.random.default_rng(int(freq)).standard_normal(n) * np.exp(-t * 90) * 0.05
        sig *= _env(n, 0.002, 0.05)
    elif kind == "glock":
        # 글로켄슈필(뮤직박스 느낌): 한 옥타브 위, 비정수 배음, 길게 울림
        f = freq * 2
        for mult, amp, d in ((1, 1.0, 2.2), (2.76, 0.35, 5.0), (5.4, 0.15, 9.0), (8.93, 0.06, 14.0)):
            sig += amp * np.sin(2 * np.pi * f * mult * t) * np.exp(-t * d)
        sig *= _env(n, 0.001, 0.03)
    elif kind == "synth":
        # 모던 플럭 신스: 살짝 어긋난 톱니파 2개, 필터가 닫히듯 높은 배음부터 빠르게 사라짐
        for det in (-0.004, 0.004):
            for k in range(1, 24):
                sig += (1 / k) * np.sin(2 * np.pi * freq * (1 + det) * k * t) * np.exp(-t * (1.5 + 0.9 * k))
        sig *= _env(n, 0.003, 0.06) * 0.5
    elif kind == "flute":
        # 플루트/휘파람: 부드러운 어택, 비브라토, 숨소리
        vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.3)
        ph = 2 * np.pi * np.cumsum(freq * vib) / SR
        sig = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
        sig += np.random.default_rng(int(freq)).standard_normal(n) * 0.025
        sig *= _env(n, 0.035, 0.08) * (np.exp(-t * 1.2) if last else 1.0)
    elif kind == "brass":
        # 브라스: 어택 때 음이 살짝 아래서 올라오고, 세게 불수록 밝아지는 배음
        bend = 1 - 0.02 * np.exp(-t * 40)
        ph = 2 * np.pi * np.cumsum(freq * bend) / SR
        bright = 1 - np.exp(-t * 25)
        for k in range(1, 14):
            sig += (1 / k) * np.sin(k * ph) * (bright ** (k * 0.35)) * np.exp(-t * (0.8 if last else 0.3) * k * 0.15)
        sig *= _env(n, 0.02, 0.07) * 0.6
    elif kind == "uke":
        # 우쿨렐레: 밝은 플럭 + 작은 몸통 울림
        period = max(2, int(SR / freq))
        buf = np.random.default_rng(int(freq)).uniform(-1, 1, period)
        for i in range(n):
            sig[i] = buf[i % period]
            buf[i % period] = 0.5 * (buf[i % period] + buf[(i + 1) % period]) * 0.996
        body = np.sin(2 * np.pi * 220 * t[: int(SR * 0.03)]) * np.exp(-t[: int(SR * 0.03)] * 120)
        sig = np.convolve(sig, np.r_[1.0, 0.25 * body])[:n]
        sig *= _env(n, 0.001, 0.04)
    out[n0:n0 + n] = sig
    return out


def hook_as(kind: str, wet: float = 0.14) -> Path:
    x = sum(inst_note(kind, hz(nn), s, l, i == 3) * (1.0 if i == 3 else 0.8) for i, (nn, s, l) in enumerate(HOOK_NOTES))
    if kind in ("piano", "synth", "brass", "uke"):
        # 착지 음에 받쳐 주는 화음(도·미)을 작게 — 끝이 '완성된' 느낌
        x = x + inst_note(kind, hz("C4"), 0.44, 1.2, True) * 0.35 + inst_note(kind, hz("E4"), 0.44, 1.2, True) * 0.25
    x = x + thump(0.44, 0.3 if kind in ("glock", "flute") else 0.4)
    return finish(*reverb(x, seconds=0.8, wet=wet), f"hook_{kind}")


def hook_piano() -> Path: return hook_as("piano")
def hook_glock() -> Path: return hook_as("glock", 0.18)
def hook_synth() -> Path: return hook_as("synth")
def hook_flute() -> Path: return hook_as("flute", 0.2)
def hook_brass() -> Path: return hook_as("brass")
def hook_uke() -> Path: return hook_as("uke", 0.1)


# ── 4차(2026-09-27): "피아노가 전자피아노 같다 — 그랜드 피아노로, 현악기·하프·오르간도, 교회에 어울리게" ──
# 전자피아노처럼 들린 이유: 배음이 8개뿐이고 줄 1가닥·단일 감쇠라 소리가 얇고 '뿅' 했다. 그랜드는
# 한 음에 줄 3가닥(미세하게 어긋나 맥놀이), 배음 수십 개(현의 강성으로 조금씩 높아짐), 타현 위치에 따른
# 배음 빗살, '빠른 감쇠 + 긴 잔향음' 2단 감쇠, 해머 타격음, 페달을 밟아 울림이 이어지는 게 핵심이다.
from scipy.signal import lfilter  # noqa: E402


def hall(x: np.ndarray, seconds: float, wet: float, dark: float = 0.6, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """교회당처럼 길고 어두운 잔향(고음이 먼저 사라지게 저역 통과한 노이즈 임펄스, 좌우 따로)."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    chans = []
    for _ in range(2):
        ir = rng.standard_normal(n) * np.exp(-t * (6.9 / seconds))
        ir = lfilter([1 - dark], [1, -dark], ir)
        ir[: int(SR * 0.012)] = 0  # 첫 반사 전 짧은 틈 → 공간감
        ir /= np.sqrt(np.sum(ir ** 2))
        chans.append((1 - wet) * x + wet * fftconvolve(x, ir)[: len(x)])
    return chans[0], chans[1]


def grand(freq: float, start: float, vel: float = 0.8) -> np.ndarray:
    out = np.zeros(int(SR * DUR))
    n0 = int(SR * start)
    n = len(out) - n0
    t = np.arange(n) / SR
    B = 0.00025 * (freq / 261.6) ** 0.5          # 현의 강성(배음이 조금씩 높아짐)
    kmax = int(min(48, (SR * 0.45) / freq))
    k = np.arange(1, kmax + 1)
    comb = np.abs(np.sin(np.pi * k / 7.3))      # 해머가 현 길이 1/7 지점을 침 → 7번째 배음 부근이 약해짐
    tilt = (1 / k ** 0.85) * np.exp(-k * (0.09 / vel))
    amps = comb * tilt
    sig = np.zeros(n)
    for det in (-0.7, 0.0, 0.8):                # 줄 3가닥, 센트 단위로 어긋남 → 맥놀이
        fk = freq * k * np.sqrt(1 + B * k * k) * 2 ** (det / 1200)
        for kk in range(kmax):
            if fk[kk] > SR * 0.47:
                break
            fast = np.exp(-t * (2.2 + 0.55 * kk))
            slow = np.exp(-t * (0.35 + 0.09 * kk))
            sig += amps[kk] * np.sin(2 * np.pi * fk[kk] * t + det) * (0.6 * fast + 0.4 * slow)
    rng = np.random.default_rng(int(freq * 10))
    knock = lfilter([0.15], [1, -0.85], rng.standard_normal(n)) * np.exp(-t * 60) * 0.35
    sig = (sig / 3 + knock) * np.minimum(1, t / 0.0015) * vel
    out[n0:] = sig
    return out


def hook_grand() -> Path:
    """그랜드 피아노(페달 밟은 채로) + 착지에 왼손 화음."""
    # 착지에서 페달을 갈아 밟는다: 앞 세 음은 착지 직후 댐퍼로 잦아들게(라와 솔이 겹쳐 탁해지지 않게)
    repedal = np.ones(int(SR * DUR))
    r0 = int(SR * 0.47)
    repedal[r0:] = np.exp(-np.arange(len(repedal) - r0) / SR / 0.07)
    x = sum(grand(hz(n), s, 0.75) for n, s, _l in HOOK_NOTES[:3]) * repedal + grand(hz("G5"), 0.44, 0.9)
    x = x + grand(hz("C3"), 0.44, 0.55) + grand(hz("G3"), 0.44, 0.45) + grand(hz("E4"), 0.44, 0.4)
    return finish(*hall(x, 1.8, 0.22, dark=0.5), "hook_grand")


def bowed(freq: float, start: float, length: float, voices: int = 6, attack: float = 0.06) -> np.ndarray:
    """현악 합주: 톱니파 여러 대를 미세하게 어긋나게(합주의 두께) + 비브라토 + 부드러운 어택."""
    out = np.zeros(int(SR * DUR))
    n0 = int(SR * start)
    n = min(int(SR * length), len(out) - n0)
    t = np.arange(n) / SR
    rng = np.random.default_rng(int(freq))
    sig = np.zeros(n)
    for v in range(voices):
        det = rng.uniform(-9, 9)
        vib = 1 + 0.004 * np.sin(2 * np.pi * rng.uniform(4.8, 6.0) * t + rng.uniform(0, 6)) * np.minimum(1, t / 0.25)
        ph = 2 * np.pi * np.cumsum(freq * 2 ** (det / 1200) * vib) / SR
        for k in range(1, 16):
            sig += np.sin(k * ph) / k * np.exp(-k * 0.12)
    sig *= _env(n, attack, 0.25) / voices
    out[n0:n0 + n] = sig
    return out


def pizz(freq: float, start: float) -> np.ndarray:
    """피치카토(현을 손가락으로 튕김): 둥근 플럭."""
    out = np.zeros(int(SR * DUR))
    n0 = int(SR * start)
    n = min(int(SR * 0.6), len(out) - n0)
    t = np.arange(n) / SR
    sig = sum(np.sin(2 * np.pi * freq * k * t) / k ** 1.4 * np.exp(-t * (7 + 3 * k)) for k in range(1, 8))
    out[n0:n0 + n] = sig * np.minimum(1, t / 0.003)
    return out


def hook_strings() -> Path:
    """앞 세 음은 피치카토로 톡톡, 마지막 솔은 현악 합주가 화음으로 길게 받친다."""
    x = sum(pizz(hz(n), s) * 0.8 for n, s, _l in HOOK_NOTES[:3])
    for nn, amp in (("G5", 0.9), ("E5", 0.5), ("C5", 0.5), ("C4", 0.55), ("G3", 0.35)):
        x = x + bowed(hz(nn), 0.40, 1.5, attack=0.07) * amp
    return finish(*hall(x, 1.8, 0.28), "hook_strings")


def harp_note(freq: float, start: float, amp: float = 1.0) -> np.ndarray:
    """하프: 부드럽게 튕긴(저역 통과한) 들뜸 + 손실 적은 카플러스-스트롱 → 길고 둥근 울림."""
    out = np.zeros(int(SR * DUR))
    n0 = int(SR * start)
    n = len(out) - n0
    period = max(2, int(round(SR / freq)))
    rng = np.random.default_rng(int(freq))
    buf = lfilter([0.5], [1, -0.5], rng.uniform(-1, 1, period))
    sig = np.empty(n)
    for i in range(n):
        j = i % period
        sig[i] = buf[j]
        buf[j] = 0.5 * (buf[j] + buf[(j + 1) % period]) * 0.9985
    out[n0:] = sig * amp
    return out


def hook_harp() -> Path:
    """하프로 4음 + 착지에서 아래로부터 화음을 굴려(아르페지오) 펼친다."""
    x = sum(harp_note(hz(n), s, 0.8 if i < 3 else 1.0) for i, (n, s, _l) in enumerate(HOOK_NOTES))
    for j, nn in enumerate(["C3", "G3", "C4", "E4"]):
        x = x + harp_note(hz(nn), 0.44 + j * 0.035, 0.45)
    return finish(*hall(x, 1.6, 0.25), "hook_harp")


def pipe(freq: float, start: float, length: float) -> np.ndarray:
    """파이프오르간: 8'·4'·2⅔'·2' 스톱 + 공기 '칙'(chiff) 어택, 음이 줄지 않음."""
    out = np.zeros(int(SR * DUR))
    n0 = int(SR * start)
    n = min(int(SR * length), len(out) - n0)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for mult, amp in ((1, 1.0), (2, 0.55), (3, 0.28), (4, 0.3), (6, 0.1), (8, 0.08)):
        sig += amp * np.sin(2 * np.pi * freq * mult * t * (1 + 0.0003 * mult))
    chiff = lfilter([0.3], [1, -0.6], np.random.default_rng(int(freq)).standard_normal(n)) * np.exp(-t * 45) * 0.25
    sig = (sig + chiff) * _env(n, 0.03, 0.09)
    out[n0:n0 + n] = sig
    return out


def hook_organ() -> Path:
    """교회 파이프오르간 — 넓은 예배당 잔향 속에서 4음, 착지 화음은 페달 저음까지."""
    x = sum(pipe(hz(n), s, (l + 0.02) if i < 3 else 1.4) * 0.5 for i, (n, s, l) in enumerate(HOOK_NOTES))
    for nn, amp in (("C5", 0.3), ("E4", 0.28), ("C4", 0.3), ("C3", 0.35), ("C2", 0.3)):
        x = x + pipe(hz(nn), 0.44, 1.4) * amp
    return finish(*hall(x, 2.6, 0.38, dark=0.7), "hook_organ")


if __name__ == "__main__":
    import sys

    fns = {f.__name__: f for f in (chime, warm, amen, pop, bounce, spark, hook,
                                   hook_piano, hook_glock, hook_synth, hook_flute, hook_brass, hook_uke,
                                   hook_grand, hook_strings, hook_harp, hook_organ)}
    for name in (sys.argv[1:] or fns):
        if name in fns:
            print(fns[name]())


# ── 5차(2026-09-27): "오르간 어색, 현대처럼 가볍게" → 합성 대신 **실제 악기 녹음(VSCO-2 CE, CC0)** ──
# 합성 악기는 아무리 다듬어도 '가짜' 티가 나서 어색했다. VSCO-2 Community Edition(퍼블릭 도메인 CC0)의
# 실제 녹음을 음마다 가까운 샘플에서 조금만 음높이를 옮겨 쓴다. 가볍게: 저음 '툭' 없음, 착지엔 반짝임만.
# 샘플 받기: assets/cache/vsco/ (gitignore) — 없으면 fetch_vsco()가 GitHub에서 받는다.
from fractions import Fraction  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

VSCO_DIR = OUT_DIR.parent / "cache" / "vsco"
VSCO_BASE = "https://raw.githubusercontent.com/sgossner/VSCO-2-CE/master/"
VSCO_FILES = {
    "glock": ["Percussion/Glock/glock_medium_C6.wav", "Percussion/Glock/glock_medium_G5.wav",
              "Percussion/Glock/glock_medium_G6.wav", "Percussion/Glock/glock_medium_C7.wav"],
    "marimba": ["Percussion/Marimba/Marimba_hit_Outrigger_C4_loud_01.wav", "Percussion/Marimba/Marimba_hit_Outrigger_G4_loud_01.wav",
                "Percussion/Marimba/Marimba_hit_Outrigger_B4_loud_01.wav", "Percussion/Marimba/Marimba_hit_Outrigger_F5_loud_01.wav",
                "Percussion/Marimba/Marimba_hit_Outrigger_C6_loud_01.wav"],
    "xylo": ["Percussion/Xylo/Xylo_Medium_C5_ff_01_far.wav", "Percussion/Xylo/Xylo_Medium_G5_ff_01_far.wav",
             "Percussion/Xylo/Xylo_Medium_C6_ff_01_far.wav"],
    "harp": ["Strings/Harp/KSHarp_C5_mf.wav", "Strings/Harp/KSHarp_E5_mf.wav", "Strings/Harp/KSHarp_G5_mf.wav",
             "Strings/Harp/KSHarp_A4_mf.wav", "Strings/Harp/KSHarp_C3_mf.wav", "Strings/Harp/KSHarp_E3_mf.wav",
             "Strings/Harp/KSHarp_G3_mf.wav"],
    "pizz": ["Strings/Violin Section/Pizz/VlnEns_Pizz_C4_v2_rr1.wav", "Strings/Violin Section/Pizz/VlnEns_Pizz_E4_v2_rr1.wav",
             "Strings/Violin Section/Pizz/VlnEns_Pizz_B4_v2_rr1.wav", "Strings/Violin Section/Pizz/VlnEns_Pizz_D5_v2_rr1.wav"],
    "triangle": ["VSCO 1 Percussion/varMetal/triangle/1/triangle1_hit_pp.wav"],
}
_NOTE_RE = __import__("re").compile(r"_([A-G]#?\d)[_.]")


def fetch_vsco() -> None:
    import urllib.parse
    import urllib.request

    VSCO_DIR.mkdir(parents=True, exist_ok=True)
    for paths in VSCO_FILES.values():
        for p in paths:
            dst = VSCO_DIR / p.split("/")[-1]
            if not dst.exists():
                urllib.request.urlretrieve(VSCO_BASE + urllib.parse.quote(p), dst)


_SAMPLE_CACHE: dict[str, np.ndarray] = {}


def load_sample(name: str) -> np.ndarray:
    """ffmpeg로 48kHz 모노 float로 읽고, 앞의 무음을 떼어 타격 순간이 0초가 되게 한다."""
    if name in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[name]
    import subprocess

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(VSCO_DIR / name), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    x = np.frombuffer(raw, dtype="<f4").astype(float)
    x /= max(1e-9, np.abs(x).max())
    onset = int(np.argmax(np.abs(x) > 0.03))
    x = x[max(0, onset - int(SR * 0.002)):]
    _SAMPLE_CACHE[name] = x
    return x


def _midi(note: str) -> float:
    return 69 + 12 * np.log2(hz(note) / 440.0)


def play(inst: str, note: str, start: float, amp: float = 1.0, length: float | None = None) -> np.ndarray:
    """악기 샘플 중 목표 음에 가장 가까운 것을 골라 음높이를 옮겨(재샘플) start에 놓는다."""
    out = np.zeros(int(SR * DUR))
    names = [p.split("/")[-1] for p in VSCO_FILES[inst]]
    if inst == "triangle":
        src, ratio = names[0], 1.0
    else:
        src = min(names, key=lambda nm: abs(_midi(_NOTE_RE.search(nm).group(1)) - _midi(note)))
        ratio = hz(note) / hz(_NOTE_RE.search(src).group(1))
    x = load_sample(src)
    if abs(ratio - 1) > 1e-4:
        fr = Fraction(1 / ratio).limit_denominator(400)  # 음을 올리려면 길이를 줄인다
        x = resample_poly(x, fr.numerator, fr.denominator)
    n0 = int(SR * start)
    n = min(len(x), len(out) - n0, int(SR * length) if length else len(x))
    seg = x[:n].copy()
    if length:  # 손으로 음을 막듯 짧게 끊기(끝 40ms 페이드)
        r = min(n, int(SR * 0.04))
        seg[-r:] *= np.linspace(1, 0, r)
    out[n0:n0 + n] = seg * amp
    return out


def _motif(inst: str, amps=(0.7, 0.7, 0.75, 1.0), octave: int = 0) -> np.ndarray:
    def sh(nn: str) -> str:
        return nn[:-1] + str(int(nn[-1]) + octave)
    return sum(play(inst, sh(nn), s, a) for (nn, s, _l), a in zip(HOOK_NOTES, amps))


def real_glock_marimba() -> Path:
    """실제 마림바가 선율, 글로켄슈필이 한 옥타브 위에서 살짝 겹쳐 반짝 — 가볍고 산뜻."""
    x = _motif("marimba") + _motif("glock", (0.25, 0.25, 0.3, 0.45), octave=1)
    x = x + play("triangle", "C6", 0.44, 0.18)
    return finish(*hall(x, 1.2, 0.16, dark=0.4), "real_glock_marimba")


def real_harp() -> Path:
    """실제 하프 선율 + 착지에서 아래부터 굴리는 화음, 글로켄 한 점."""
    x = _motif("harp", (0.75, 0.75, 0.8, 1.0))
    for j, nn in enumerate(["C3", "G3", "E4"]):
        x = x + play("harp", nn, 0.44 + j * 0.04, 0.35)
    x = x + play("glock", "G6", 0.46, 0.2)
    return finish(*hall(x, 1.4, 0.2, dark=0.45), "real_harp")


def real_xylo() -> Path:
    """실제 실로폰 — 가장 가볍고 톡톡 튀는 소리, 착지에 트라이앵글 '팅'."""
    x = _motif("xylo", (0.65, 0.65, 0.7, 1.0)) + play("triangle", "C6", 0.44, 0.22)
    return finish(*hall(x, 1.0, 0.14, dark=0.4), "real_xylo")


def real_pizz_glock() -> Path:
    """실제 바이올린 합주 피치카토(한 옥타브 아래)로 톡톡, 착지에 글로켄 반짝."""
    x = _motif("pizz", (0.8, 0.8, 0.85, 1.0), octave=-1)
    x = x + play("glock", "G6", 0.44, 0.35) + play("glock", "C7", 0.5, 0.15)
    return finish(*hall(x, 1.3, 0.2, dark=0.45), "real_pizz_glock")


def real_glock() -> Path:
    """실제 글로켄슈필만 — 오르골처럼 맑고 가벼운 '띠리링'."""
    x = _motif("glock", (0.55, 0.55, 0.6, 0.85), octave=1) + play("triangle", "C6", 0.44, 0.12)
    return finish(*hall(x, 1.3, 0.2, dark=0.4), "real_glock")


REAL = (real_glock_marimba, real_harp, real_xylo, real_pizz_glock, real_glock)

if __name__ == "__main__" and (set(__import__("sys").argv[1:]) & {f.__name__ for f in REAL}):
    fetch_vsco()
    for f in REAL:
        if f.__name__ in __import__("sys").argv[1:]:
            print(f())
