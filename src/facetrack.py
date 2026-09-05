"""얼굴 추적 리프레이밍: 화자의 얼굴을 따라 세로(cover) 크롭 위치를 움직인다.

게시물 레퍼런스의 '화면 추적'(사람을 따라 화면을 다시 잡음)에 해당한다. 세로 쇼츠의
cover 크롭은 원래 화면 '중앙'을 고정으로 잘라내는데, 설교자가 무대 좌/우로 움직이면
화면 밖으로 나간다. 여기서 OpenCV(Haar)로 프레임마다 얼굴 위치를 찾아 부드럽게
스무딩한 뒤, ffmpeg crop 필터의 x를 시간에 따라 바꾸는 표현식을 만든다.

외부 모델 다운로드가 필요 없도록 Haar 캐스케이드 XML을 리포(assets/facetrack)에 넣어
버전 독립적으로 로드한다. 얼굴이 거의 안 잡히면 [] 를 반환 → 호출자가 중앙 크롭으로 폴백.
"""
from __future__ import annotations

from pathlib import Path

_CASCADE_PATH = Path("assets/facetrack/haarcascade_frontalface_default.xml")


def compute_face_centers(
    video_path: Path,
    clip_start: float,
    clip_end: float,
    sample_fps: float = 2.0,
    min_detect_ratio: float = 0.25,
) -> list[tuple[float, float]]:
    """[clip_start, clip_end] 구간에서 얼굴 중심 x를 표본추출한다.

    반환: [(t_rel, fx), ...]  t_rel=클립 시작 기준 상대초, fx=얼굴 중심의 가로 위치(0~1).
    스무딩(이동평균)과 빈 구간 보간을 적용한다. 검출률이 낮으면([]) 폴백을 유도한다.
    """
    try:
        import cv2  # 지연 import: 얼굴 추적을 안 쓰면 의존성 없이도 나머지가 동작
    except Exception:  # noqa: BLE001
        return []
    cascade_file = _CASCADE_PATH
    if not cascade_file.exists():
        # 리포에 없으면 cv2 번들 위치를 시도(구버전 호환)
        alt = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not alt.exists():
            return []
        cascade_file = alt
    cascade = cv2.CascadeClassifier(str(cascade_file))
    if cascade.empty():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    try:
        dur = max(0.1, clip_end - clip_start)
        step = 1.0 / max(0.5, sample_fps)
        n_samples = max(2, int(dur / step) + 1)
        raw: list[tuple[float, float | None]] = []
        t = 0.0
        for _ in range(n_samples):
            abs_ms = (clip_start + t) * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, abs_ms)
            ok, frame = cap.read()
            if not ok or frame is None:
                raw.append((t, None))
                t += step
                continue
            h, w = frame.shape[:2]
            # 속도: 가로 640으로 다운스케일 후 그레이스케일로 검출.
            scale = 640.0 / w if w > 640 else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) == 0:
                raw.append((t, None))
            else:
                # 가장 큰 얼굴(주 화자)을 택한다.
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                cx = (fx + fw / 2.0) / small.shape[1]  # 0~1 (다운스케일 무관, 비율)
                raw.append((t, float(cx)))
            t += step
    finally:
        cap.release()

    detected = [c for _, c in raw if c is not None]
    if len(detected) < max(2, int(len(raw) * min_detect_ratio)):
        return []  # 검출률 너무 낮음 → 중앙 크롭 폴백

    # 빈 구간 보간: 앞뒤 검출값으로 선형 채움(양끝은 최근값 유지).
    filled: list[float] = []
    last = detected[0]
    for _, c in raw:
        filled.append(c if c is not None else last)
        if c is not None:
            last = c
    # 뒤에서 앞으로 한 번 더(초반 None을 첫 검출값으로).
    nxt = detected[-1]
    for i in range(len(raw) - 1, -1, -1):
        if raw[i][1] is not None:
            nxt = raw[i][1]
        elif filled[i] == detected[0] and raw[i][1] is None:
            filled[i] = nxt

    # 이동평균 스무딩(윈도 ~1초).
    win = max(1, int(round(sample_fps)))
    smoothed: list[float] = []
    for i in range(len(filled)):
        a = max(0, i - win)
        b = min(len(filled), i + win + 1)
        smoothed.append(sum(filled[a:b]) / (b - a))

    return [(raw[i][0], smoothed[i]) for i in range(len(raw))]


def build_crop_x_expr(
    centers: list[tuple[float, float]],
    scaled_w: float,
    crop_w: float,
    max_x: float,
) -> str:
    """얼굴 중심(fx, 원본 가로 비율) 키프레임을 ffmpeg crop x 표현식으로 만든다.

    scaled_w: cover 스케일 후 프레임 가로 픽셀. crop_w: 잘라낼 가로(=박스 폭).
    max_x: 크롭 x 상한(=scaled_w-crop_w). 각 키프레임의 목표 x=clamp(fx*scaled_w-crop_w/2,0,max_x).
    시간 t(초)에 대해 구간별 선형보간하는 nested if 식을 만든다(부드러운 팬)."""
    if not centers or max_x <= 0:
        return ""
    pts: list[tuple[float, float]] = []
    for t, fx in centers:
        x = fx * scaled_w - crop_w / 2.0
        x = max(0.0, min(max_x, x))
        pts.append((t, x))
    # 중복 시각 제거(단조 증가 보장)
    dedup: list[tuple[float, float]] = []
    for t, x in pts:
        if not dedup or t > dedup[-1][0] + 1e-3:
            dedup.append((t, x))
    if len(dedup) == 1:
        return f"{dedup[0][1]:.1f}"
    # 뒤에서부터 nested if(lt(t,t_i), lerp, ...) 구성.
    expr = f"{dedup[-1][1]:.1f}"
    for i in range(len(dedup) - 1, 0, -1):
        t0, x0 = dedup[i - 1]
        t1, x1 = dedup[i]
        dt = max(1e-3, t1 - t0)
        slope = (x1 - x0) / dt
        seg = f"({x0:.1f}+({slope:.3f})*(t-{t0:.3f}))"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    # 첫 구간 이전(t<t0_first)은 첫 값으로.
    t_first, x_first = dedup[0]
    expr = f"if(lt(t,{t_first:.3f}),{x_first:.1f},{expr})"
    return expr
