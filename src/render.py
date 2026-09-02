"""ffmpeg로 클립 컷 -> 9:16 변환(+Ken Burns) -> 무음 제거 -> 자막 번인까지 한 번에 처리"""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.captions import build_ass_for_clip
from src.highlights import Clip
from src.transcribe import Segment


def _probe_fps(video_path: Path) -> float:
    """소스 영상의 실제 프레임레이트를 가져온다. zoompan에 하드코딩된 fps를 쓰면
    (예: 25) 실제 소스가 60fps일 때 영상 길이가 fps 비율만큼(60/25=2.4배) 늘어나버리는
    버그가 생기므로, 반드시 소스에 맞는 fps를 zoompan에 넘겨야 한다."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
         str(video_path)],
        capture_output=True, text=True,
    )
    raw = proc.stdout.strip()
    try:
        if "/" in raw:
            num, den = raw.split("/")
            return float(num) / float(den)
        return float(raw)
    except (ValueError, ZeroDivisionError):
        return 30.0  # 안전한 기본값


def _probe_resolution(video_path: Path) -> tuple[int, int]:
    """소스 영상의 실제 가로x세로 픽셀 크기를 가져온다 (원본 비율을 유지한 채 둥근 박스를
    만들려면 이 비율을 알아야 크롭/확대 없이 정확히 맞출 수 있다)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        w_str, h_str = proc.stdout.strip().split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError):
        return 1920, 1080  # 안전한 기본값 (16:9)


def _detect_silences(
    video_path: Path,
    start: float,
    end: float,
    threshold_db: float,
    min_duration_sec: float,
) -> list[tuple[float, float]]:
    """clip 구간(start~end, 원본 기준) 안에서 무음 구간을 찾아 (클립 내부 상대시작, 상대끝) 리스트로 반환."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner",
        "-ss", str(start), "-to", str(end), "-i", str(video_path),
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration_sec}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stderr = proc.stderr

    silences: list[tuple[float, float]] = []
    silence_start = None
    for line in stderr.splitlines():
        if "silence_start" in line:
            try:
                silence_start = float(line.split("silence_start:")[1].strip().split()[0])
            except (IndexError, ValueError):
                silence_start = None
        elif "silence_end" in line and silence_start is not None:
            try:
                part = line.split("silence_end:")[1].strip().split("|")[0].strip()
                silences.append((silence_start, float(part)))
            except (IndexError, ValueError):
                pass
            silence_start = None
    return silences


def _build_keep_segments(
    clip_duration: float, silences: list[tuple[float, float]], min_gap_sec: float = 0.15
) -> list[tuple[float, float]]:
    """무음 구간을 제외한, 실제로 남길 구간들(클립 내부 상대시간)을 계산.
    무음을 완전히 0으로 자르지 않고 min_gap_sec만큼만 남겨 자연스러운 호흡을 유지."""
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in silences:
        keep_until = s + min_gap_sec / 2 if e - s > min_gap_sec else e
        if keep_until > cursor:
            keep.append((cursor, keep_until))
        cursor = max(cursor, e - min_gap_sec / 2)
    if cursor < clip_duration:
        keep.append((cursor, clip_duration))
    return keep if keep else [(0.0, clip_duration)]


def _intersect_intervals(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """두 '남길 구간' 목록의 교집합(둘 다 남기라고 한 구간만 남긴다). 짧은 파편은 버린다."""
    out: list[tuple[float, float]] = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e - s > 0.05:
                out.append((s, e))
    out.sort()
    return out


def _combine_keep(
    duration: float,
    clip_start: float,
    user_keep_abs: list | None,
    silence_keep_rel: list[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    """사용자 분할·삭제(keep_ranges, 절대초)와 무음 제거(상대초)를 합쳐 최종 '남길 구간'(상대초)을 만든다.
    전체(0~duration) 한 조각이면 None을 반환해 select 필터 없는 빠른 경로를 탄다."""
    if user_keep_abs:
        user_rel = [
            (max(0.0, float(s) - clip_start), min(duration, float(e) - clip_start))
            for s, e in user_keep_abs
        ]
        user_rel = sorted((s, e) for s, e in user_rel if e - s > 0.05)
        if not user_rel:
            user_rel = [(0.0, duration)]
    else:
        user_rel = [(0.0, duration)]

    combined = user_rel if silence_keep_rel is None else _intersect_intervals(user_rel, silence_keep_rel)
    if not combined:
        return None
    if len(combined) == 1 and combined[0][0] <= 0.01 and combined[0][1] >= duration - 0.01:
        return None  # 통째로 남김 = 자를 것 없음(빠른 경로)
    return combined


def _vertical_transform(background_mode: str, resolution: tuple[int, int], pad_color: str = "white") -> str:
    w, h = resolution
    if background_mode == "crop":
        return f"scale={w}:-2,crop={w}:{h}"
    if background_mode == "pad":
        # 단색 여백 + 중앙 원본 (설교자 화면이 안 잘리고, 원본 자체 자막이 블러로 비치지 않아 산만하지 않음)
        return (
            f"scale={w}:-2:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={pad_color}"
        )
    # blur 배경 + 중앙 원본 (설교자 화면이 안 잘리도록)
    return (
        f"split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},gblur=sigma=20[bgblur];"
        f"[fg]scale={w}:-2:force_original_aspect_ratio=decrease[fgscaled];"
        f"[bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2"
    )


def _rounded_mask_expr(w: int, h: int, r: int) -> str:
    """geq 필터용 마스크 표현식: 네 모서리 RxR 영역에서 라운딩 원 밖이면 검정(0), 나머지는 흰색(255)."""
    corners = [
        f"lte(X,{r})*lte(Y,{r})*gt(pow({r}-X,2)+pow({r}-Y,2),pow({r},2))",
        f"lte({w}-X,{r})*lte(Y,{r})*gt(pow({r}-({w}-X),2)+pow({r}-Y,2),pow({r},2))",
        f"lte(X,{r})*lte({h}-Y,{r})*gt(pow({r}-X,2)+pow({r}-({h}-Y),2),pow({r},2))",
        f"lte({w}-X,{r})*lte({h}-Y,{r})*gt(pow({r}-({w}-X),2)+pow({r}-({h}-Y),2),pow({r},2))",
    ]
    expr = "255"
    for c in reversed(corners):
        expr = f"if({c},0,{expr})"
    return expr


_MASK_CACHE_DIR = Path("assets/cache")


def _get_or_create_rounded_mask(vbw: int, vbh: int, r: int) -> Path:
    """둥근 모서리 마스크 이미지를 캐시해서 재사용한다.

    geq는 픽셀마다 수식을 계산해야 해서 영상 전체 프레임에 매번 적용하면 극도로 느리다
    (30초 클립 하나에 몇 분씩 걸릴 정도). 그래서 이 마스크를 '정지 이미지 한 장'에만 한 번
    geq로 구워두고, 실제 영상에는 훨씬 가벼운 alphamerge로 그 마스크를 입힌다.
    """
    _MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = _MASK_CACHE_DIR / f"rounded_mask_{vbw}x{vbh}_r{r}.png"
    if mask_path.exists():
        return mask_path

    mask_expr = _rounded_mask_expr(vbw, vbh, r)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=white:s={vbw}x{vbh}",
        "-frames:v", "1",
        "-vf", f"geq=lum='{mask_expr}'",
        # -pix_fmt gray 없으면 PNG가 RGB로 저장되며 YUV<->RGB 변환 과정에서 흰색(255)이
        # 178로 떨어진다. 그러면 alphamerge 시 영상이 70%만 불투명해져 흰 배경이 비쳐
        # 화면 전체가 뿌옇게 보인다("불투명 박스 올린 것 같은" 현상). gray로 저장해야 255 유지.
        "-pix_fmt", "gray",
        str(mask_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"둥근 모서리 마스크 생성 실패:\n{proc.stderr[-2000:]}")
    return mask_path


# 소스 하단 크롭 기본값. 렌더(_build_card_filter_complex)와 편집 미리보기(web_app)가
# 서로 다른 기본값을 쓰면 config에 이 키가 없을 때 미리보기와 결과물이 어긋난다 — 반드시
# 이 상수 하나만 참조할 것.
SOURCE_CROP_BOTTOM_PCT_DEFAULT = 0.08


def card_source_video_filter(card: dict, vbw: int, vbh: int) -> str:
    """소스 프레임을 카드 영상 박스 크기로 만드는 crop+scale 필터 체인.

    실제 렌더(_build_card_filter_complex)와 편집 미리보기(web_app.clip_preview_frame)가
    이 함수 하나를 공유한다 — 각자 문자열을 복제하면 fill_mode/크롭 기본값이 조금만
    어긋나도 '편집 화면에서 본 프레임 ≠ 결과물'이 된다."""
    parts = []
    pct = card.get("source_crop_bottom_pct", SOURCE_CROP_BOTTOM_PCT_DEFAULT)
    if pct > 0:
        parts.append(f"crop=iw:ih*{1 - pct}:0:0")
    if card.get("fill_mode", "cover") == "cover":
        # 박스를 꽉 채우도록 확대 후 중앙을 박스 크기로 잘라낸다 (좌우 빈 배경만 트리밍,
        # 설교자는 중앙이라 안 잘림). 이렇게 해야 영상이 커져 화면이 유튜브 쇼츠처럼 꽉 찬다.
        parts.append(
            f"scale={vbw}:{vbh}:force_original_aspect_ratio=increase:flags=lanczos,crop={vbw}:{vbh}"
        )
    else:
        # fit: 원본 가로세로 비율 그대로(옆을 안 자름). vbh가 이미 그 비율로 계산됨.
        parts.append(f"scale={vbw}:{vbh}:flags=lanczos")
    return ",".join(parts)


def _compute_card_video_box_height(card: dict, source_resolution: tuple[int, int]) -> int:
    """영상 박스 높이를 결정한다.

    - fill_mode == "cover"(기본): 세로가 확 큰 소스(16:9)를 카드에 작게 letterbox하면
      화면 절반이 빈 흰 여백이 되어 "유튜브 쇼츠 같지 않고 하단이 텅 빈" 문제가 생긴다.
      그래서 config의 video_box_aspect(가로:세로 비, 예 [4,3])로 박스를 크게 잡고, 소스는
      좌우의 빈 배경만 살짝 잘라(cover) 박스를 꽉 채운다. 설교자는 화면 중앙에 있으므로
      좌우를 조금 잘라도 잘리지 않는다.
    - fill_mode == "fit": 원본 가로세로 비율을 그대로 유지(옛 동작, 옆을 안 자름).
    """
    vbw = card["video_box_width"]
    fill_mode = card.get("fill_mode", "cover")
    if fill_mode == "cover":
        aspect = card.get("video_box_aspect", [4, 3])
        return round(vbw * aspect[1] / aspect[0])
    source_crop_bottom_pct = card.get("source_crop_bottom_pct", SOURCE_CROP_BOTTOM_PCT_DEFAULT)
    src_w, src_h = source_resolution
    effective_src_h = src_h * (1 - source_crop_bottom_pct)
    return round(vbw * effective_src_h / src_w)


def _compute_card_video_box_y(card: dict, resolution: tuple[int, int], vbh: int) -> int:
    """영상 박스를 제목 영역과 캡션 영역을 제외한 나머지 중간 공간에 수직 중앙 정렬한다.
    (원본 비율을 유지하면 박스 높이가 짧아지는데, 고정된 y좌표에 그대로 두면 제목 바로
    아래 붙어서 아래쪽에 여백만 커 보인다 — 사용자 요청으로 중앙 정렬로 변경.)"""
    _, h = resolution
    title_h = card.get("title_area_height", 200)
    caption_h = card.get("caption_area_height", 380)
    available = h - title_h - caption_h
    return title_h + max(0, (available - vbh) // 2)


def _build_card_filter_complex(
    render_cfg: dict,
    captions_cfg: dict,
    resolution: tuple[int, int],
    clip_duration: float,
    select_expr: str | None,
    ass_path_ff: str,
    font_dir: str,
    source_fps: float,
    vbh: int,
    vby: int,
) -> tuple[str, str]:
    """카드형 레이아웃(흰 배경 + 둥근 모서리 영상 박스): filter_complex 문자열과
    최종 비디오 출력 라벨을 반환한다.
    blur/crop/pad와 달리 흰 배경을 합성해야 해서 단순 -vf 체인이 아니라 filter_complex가 필요하다.

    color 소스의 fps를 소스 영상과 반드시 맞춰야 한다 (fps=25 기본값과 실제 소스 fps가
    다르면 overlay가 두 입력의 프레임레이트를 맞추는 과정에서 길이가 틀어질 수 있음 —
    이미 zoompan에서 같은 이유로 길이가 늘어나는 버그를 겪었으므로 여기서도 명시적으로 맞춘다).

    둥근 모서리는 매 프레임 geq를 돌리지 않고, 미리 구워둔 마스크 이미지를 alphamerge로
    입힌다 (마스크는 입력 1번, `-loop 1 -i mask.png`로 별도 추가되어야 함 — render_clip 참고).

    vbh/vby(영상 박스 높이/y좌표)는 호출자가 미리 계산해서 넘긴다 (build_ass_for_clip의
    캡션 위치 계산과 같은 값을 써야 하므로 한 곳에서만 계산한다).
    """
    w, h = resolution
    card = render_cfg["card_layout"]
    vbw = card["video_box_width"]
    vbx = (w - vbw) // 2
    video_parts = []
    if select_expr:
        video_parts.append(f"select='{select_expr}'")
        video_parts.append("setpts=N/FRAME_RATE/TB")
    video_parts.append(card_source_video_filter(card, vbw, vbh))
    video_chain = ",".join(video_parts)

    filter_complex = (
        f"[0:v]{video_chain}[rgbsrc];"
        f"[1:v]format=gray[maskgray];"
        f"[rgbsrc][maskgray]alphamerge[rounded];"
        f"color=white:s={w}x{h}:d={clip_duration}:r={source_fps}[cardbg];"
        f"[cardbg][rounded]overlay={vbx}:{vby}[composited]"
    )
    if captions_cfg.get("enabled", True):
        filter_complex += f";[composited]subtitles='{ass_path_ff}':fontsdir='{font_dir}'[vout]"
        return filter_complex, "[vout]"
    return filter_complex, "[composited]"


def _ken_burns_filter(resolution: tuple[int, int], zoom_per_sec: float, fps: float) -> str:
    """이미 9:16으로 합성된 최종 프레임에 미세한 줌인 효과를 사후 적용한다.

    주의: crop 필터는 w/h를 프레임마다(eval=frame) 다시 계산하는 옵션 자체가 없어
    시간 가변 줌에 쓸 수 없다 (x/y만 프레임마다 재계산 가능). 그래서 zoompan을 쓰되,
    d=1(입력 프레임 하나당 출력 프레임도 하나)로 못박고 zoompan 내부에 유지되는
    자기참조 변수 'zoom'을 프레임마다 조금씩 늘리는 방식으로, 프레임 수/길이를
    전혀 바꾸지 않으면서 부드러운 줌인을 만든다. (d에 총 프레임 수를 넣으면
    입력 프레임 하나당 그만큼 프레임을 복제해버려 영상이 크게 길어지는 버그가 남.)"""
    w, h = resolution
    increment = zoom_per_sec / fps
    zoom_expr = f"zoom+{increment}"
    return (
        f"zoompan=z='{zoom_expr}':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}"
    )


def _build_video_filter(
    render_cfg: dict,
    captions_cfg: dict,
    resolution: tuple[int, int],
    select_expr: str | None,
    ass_path_ff: str,
    font_dir: str,
    source_fps: float,
) -> str:
    parts = []
    if select_expr:
        parts.append(f"select='{select_expr}'")
        parts.append("setpts=N/FRAME_RATE/TB")
    parts.append(_vertical_transform(
        render_cfg.get("background_mode", "blur"), resolution, render_cfg.get("pad_color", "white")
    ))
    ken_burns = render_cfg.get("ken_burns", {})
    if ken_burns.get("enabled", False):
        # 9:16으로 합성된 최종 프레임에 대해 줌인 (blur/crop 어느 모드든 동일하게 적용 가능)
        parts.append(_ken_burns_filter(resolution, ken_burns.get("zoom_per_sec", 0.008), source_fps))
    if captions_cfg.get("enabled", True):
        parts.append(f"subtitles='{ass_path_ff}':fontsdir='{font_dir}'")
    return ",".join(parts)


def render_clip(
    video_path: Path,
    segments: list[Segment],
    clip: Clip,
    output_path: Path,
    render_cfg: dict,
    captions_cfg: dict,
) -> None:
    resolution = tuple(render_cfg.get("resolution", [1080, 1920]))
    duration = clip.end - clip.start

    # 편집기에서 클립별로 화면모드(풀 화면=fit / 화면 확대=cover)를 고른 경우 config 기본값을 덮어쓴다.
    clip_fill = getattr(clip, "fill_mode", "") or ""
    if clip_fill:
        _card = {**render_cfg.get("card_layout", {}), "fill_mode": clip_fill}
        render_cfg = {**render_cfg, "card_layout": _card}

    silences: list[tuple[float, float]] = []
    if render_cfg.get("remove_silence", True):
        silences = _detect_silences(
            video_path,
            clip.start,
            clip.end,
            render_cfg.get("silence_threshold_db", -35),
            render_cfg.get("silence_min_duration_sec", 0.6),
        )
    # 자막 생성보다 먼저 계산해야 한다: 무음 제거로 영상 타임라인이 압축되는데,
    # 자막 타임스탬프도 똑같이 압축해서 리매핑하지 않으면 뒤로 갈수록 자막이 밀린다.
    silence_keep = _build_keep_segments(duration, silences) if silences else None
    # 확인 팝업에서 분할·삭제한 구간(keep_ranges)이 있으면 무음 제거와 합쳐 최종 남길 구간을 만든다.
    keep_segments = _combine_keep(
        duration, clip.start, getattr(clip, "keep_ranges", None) or None, silence_keep
    )

    is_card = render_cfg.get("background_mode", "blur") == "card"
    card_layout = render_cfg.get("card_layout") if is_card else None
    if is_card:
        # 영상 박스 높이는 원본 비율을 그대로 유지해서 계산 (옆을 잘라 확대하지 않음).
        # 자막 위치 계산(build_ass_for_clip)과 실제 필터(_build_card_filter_complex)가
        # 반드시 같은 값을 써야 캡션이 영상 박스 바로 아래에 정확히 붙는다.
        source_resolution = _probe_resolution(video_path)
        vbh = _compute_card_video_box_height(card_layout, source_resolution)
        vby = _compute_card_video_box_y(card_layout, resolution, vbh)
        card_layout = {**card_layout, "video_box_height": vbh, "video_box_y": vby}

    # 자막 싱크 교정용 무음 지도: Whisper가 쉼(pause)을 다음 단어 발화 시간에 흡수해
    # 자막이 실제 말보다 1~2초 먼저 뜨는 문제(실측)를, 실제 오디오의 무음 구간으로
    # 단어 start를 교정해 잡는다. remove_silence(-35dB/1.2s)와 별개로, 짧은 쉼까지
    # 잡도록 더 민감한 값(-32dB/0.3s)을 쓴다. ffmpeg 한 번이라 클립당 2~3초면 끝난다.
    voice_silences: list[tuple[float, float]] | None = None
    if captions_cfg.get("enabled", True):
        # 편집 자막(caption_overrides) 경로에도 적용한다: 카라오케 단어 시각은 어느 경로든
        # 전사 단어를 쓰므로, 여기서 빼면 편집 클립만 "자막이 말보다 빠른" 문제가 남는다
        # (실제 신고된 잔존 싱크 문제의 원인 중 하나).
        voice_silences = _detect_silences(video_path, clip.start, clip.end, -32.0, 0.3)

    ass_path = output_path.with_suffix(".ass")
    ass_content = build_ass_for_clip(
        segments=segments,
        clip_start=clip.start,
        clip_end=clip.end,
        config_captions=captions_cfg,
        config_hook=render_cfg.get("hook", {"enabled": False}),
        resolution=resolution,
        hook_text=clip.title,
        card_layout=card_layout,
        keep_segments=keep_segments,
        title_offset_x=clip.title_offset_x,
        title_offset_y=clip.title_offset_y,
        caption_offset_x=clip.caption_offset_x,
        caption_offset_y=clip.caption_offset_y,
        caption_overrides=getattr(clip, "caption_overrides", None) or None,
        font_style={
            "title_font": getattr(clip, "title_font", "") or "",
            "title_size": getattr(clip, "title_size", 0) or 0,
            "title_align": getattr(clip, "title_align", "") or "",
            "title_spacing": getattr(clip, "title_spacing", 0.0) or 0.0,
            "caption_font": getattr(clip, "caption_font", "") or "",
            "caption_size": getattr(clip, "caption_size", 0) or 0,
            "caption_align": getattr(clip, "caption_align", "") or "",
            "caption_spacing": getattr(clip, "caption_spacing", 0.0) or 0.0,
        },
        voice_silences=voice_silences,
    )
    ass_path.write_text(ass_content, encoding="utf-8")

    # 편집기에서 고른 글꼴이 어느 폴더에 있든 libass가 찾도록 모든 폰트를 모은 통합 폴더를 fontsdir로 넘긴다.
    from src.fonts import default_font_dir_for_ass

    font_dir = default_font_dir_for_ass()
    ass_path_ff = str(ass_path).replace("\\", "/").replace(":", "\\:")

    select_expr = None
    audio_filter = None
    if keep_segments:
        select_expr = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in keep_segments)
        audio_filter = f"aselect='{select_expr}',asetpts=N/SR/TB"

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", str(clip.start), "-to", str(clip.end), "-i", str(video_path),
    ]

    # 클립 끝 여운: 마지막 프레임을 정지(tpad clone)시키고 오디오엔 무음(apad)을 덧붙여
    # 문장이 끝나자마자 뚝 끊기는 느낌을 없앤다. 영상·오디오 둘 다 같은 길이만큼 늘려야
    # (-shortest / 기본 종료 조건에서) 여운이 잘리지 않는다.
    end_pad = float(render_cfg.get("end_pad_sec", 0.0) or 0.0)
    vpad_vf = f"tpad=stop_mode=clone:stop_duration={end_pad}" if end_pad > 0 else None
    apad_af = f"apad=pad_dur={end_pad}" if end_pad > 0 else None

    def _join_af(base: str | None) -> list[str]:
        chain = ",".join(p for p in (base, apad_af) if p)
        return ["-af", chain] if chain else []

    source_fps = _probe_fps(video_path)
    if is_card:
        mask_path = _get_or_create_rounded_mask(
            card_layout["video_box_width"], card_layout["video_box_height"], card_layout["corner_radius"]
        )
        cmd += ["-loop", "1", "-i", str(mask_path)]  # 입력 1번: 둥근 모서리 마스크 이미지
        filter_complex, vout_label = _build_card_filter_complex(
            render_cfg, captions_cfg, resolution, duration, select_expr, ass_path_ff, font_dir,
            source_fps, card_layout["video_box_height"], card_layout["video_box_y"],
        )
        if vpad_vf:
            filter_complex += f";{vout_label}{vpad_vf}[vpad]"
            vout_label = "[vpad]"
        cmd += ["-filter_complex", filter_complex, "-map", vout_label]
        cmd += _join_af(audio_filter)
        # 마스크 입력(-loop 1)이 무한 스트림이라 -shortest 없이는 ffmpeg가 멈출 시점을 몰라
        # 인코딩이 끝나지 않는다. 반드시 필요. (tpad/apad로 영상·오디오 둘 다 +end_pad가 되어
        # -shortest가 여운을 자르지 않는다.)
        cmd += ["-map", "0:a", "-shortest"]
    else:
        video_filter = _build_video_filter(
            render_cfg, captions_cfg, resolution, select_expr, ass_path_ff, font_dir, source_fps
        )
        if vpad_vf:
            video_filter += f",{vpad_vf}"
        cmd += ["-vf", video_filter]
        cmd += _join_af(audio_filter)

    # 인코더: 이 PC엔 Intel Arc iGPU가 있어 h264_qsv 하드웨어 인코딩이 가능하다(실측 인코딩
    # 17~28초 → 수 초). 쇼츠는 플랫폼이 재인코딩하므로 화질 차이는 체감 없음. QSV가 드라이버
    # 문제 등으로 실패하면 자동으로 libx264(소프트웨어)로 다시 인코딩한다.
    encoder = render_cfg.get("video_encoder", "h264_qsv")
    if encoder == "h264_qsv":
        video_args = ["-c:v", "h264_qsv", "-global_quality", "23", "-preset", "veryfast"]
    else:
        # preset slow/crf16은 화질은 최고지만 클립 하나에 1~2분씩 걸려 렌더가 느렸다.
        # crf 20/veryfast면 체감 화질 차이 없이 인코딩이 5~8배 빠르다(속도 우선).
        video_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    tail = ["-c:a", "aac", "-b:a", "192k", str(output_path)]

    proc = subprocess.run(cmd + video_args + tail, capture_output=True, text=True)
    if proc.returncode != 0 and encoder == "h264_qsv":
        fallback_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        proc = subprocess.run(cmd + fallback_args + tail, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 렌더링 실패:\n{proc.stderr[-3000:]}")


def render_all_clips(
    video_path: Path,
    segments: list[Segment],
    clips: list[Clip],
    output_dir: Path,
    render_cfg: dict,
    captions_cfg: dict,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for i, clip in enumerate(clips, start=1):
        out_path = output_dir / f"short_{i}.mp4"
        render_clip(video_path, segments, clip, out_path, render_cfg, captions_cfg)
        outputs.append(out_path)
    return outputs
