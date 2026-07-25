"""클립 구간에 대한 ASS 자막 생성 (카라오케 단어 강조 지원)

짧은 구 단위(max_words_per_line)로 줄바꿈하고, 플랫폼 UI(좋아요/팔로우 버튼 등)에
가리지 않도록 안전 영역(safe_area) 안에 배치한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.transcribe import Segment, Word


@dataclass
class CaptionLine:
    start: float   # 클립 시작 기준 상대 시간(초)
    end: float
    words: list[Word]  # 원본(절대 시간) 단어 타임스탬프 그대로 보관


def _collect_words_in_range(segments: list[Segment], clip_start: float, clip_end: float) -> list[Word]:
    words: list[Word] = []
    for seg in segments:
        if seg.end < clip_start or seg.start > clip_end:
            continue
        for w in seg.words:
            if w.start >= clip_start and w.end <= clip_end:
                words.append(w)
    return words


def chunk_words_into_lines(words: list[Word], max_words_per_line: int) -> list[CaptionLine]:
    lines: list[CaptionLine] = []
    for i in range(0, len(words), max_words_per_line):
        chunk = words[i : i + max_words_per_line]
        if not chunk:
            continue
        lines.append(CaptionLine(start=chunk[0].start, end=chunk[-1].end, words=chunk))
    return lines


def _ass_time(sec: float) -> str:
    if sec < 0:
        sec = 0
    cs = int(round(sec * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _fit_title_font_size(
    text: str, max_size: int, min_size: int, available_width_px: int, char_width_ratio: float = 0.62
) -> int:
    """제목이 항상 한 줄에 들어가도록, 글자 수에 맞춰 폰트 크기를 줄인다.
    (짧은 제목은 max_size 그대로, 긴 제목은 min_size까지 줄여서라도 한 줄 유지.)"""
    if not text:
        return max_size
    needed = available_width_px / (len(text) * char_width_ratio)
    return max(min_size, min(max_size, int(needed)))


def _remap_after_silence_removal(t: float, keep_segments: list[tuple[float, float]]) -> float:
    """무음 제거로 압축된 새 타임라인 기준으로 시간을 다시 계산한다.

    render.py가 무음 구간을 select 필터로 걷어내면 영상/오디오는 짧아지는데,
    자막(ASS) 타임스탬프가 원본(무음 포함) 기준 그대로면 뒤로 갈수록 자막이 밀린다
    (예: 10초 지점에 3초 무음이 있었다면, 그 뒤 모든 대사가 실제 화면보다 자막이
    3초 늦게 뜬다). keep_segments는 실제로 남긴 (원본시작, 원본끝) 구간 목록이며,
    이걸 이어붙인 새 타임라인 상의 위치로 t를 변환한다.
    """
    cursor_new = 0.0
    for seg_start, seg_end in keep_segments:
        seg_len = seg_end - seg_start
        if t < seg_start:
            return cursor_new  # 제거된(무음) 구간 안 -> 다음 유지 구간 시작점으로 스냅
        if t <= seg_end:
            return cursor_new + (t - seg_start)
        cursor_new += seg_len
    return cursor_new  # 마지막 유지 구간 이후


def _y_position(resolution: tuple[int, int], position: str, safe_bottom_pct: float, safe_top_pct: float) -> int:
    width, height = resolution
    if position == "center":
        return height // 2
    # bottom: 안전 영역(하단 UI) 바로 위에 배치
    return int(height * (1 - safe_bottom_pct) - 40)


def build_ass(
    clip_words: list[Word],
    clip_start: float,
    clip_end: float,
    resolution: tuple[int, int],
    font_family: str,
    font_size: int,
    primary_color: str,
    outline_color: str,
    outline_width: int,
    karaoke_highlight_color: str,
    max_words_per_line: int,
    position: str,
    safe_area_bottom_pct: float,
    safe_area_top_pct: float,
    template: str = "karaoke",
    hook_text: str | None = None,
    hook_duration_sec: float = 3.0,
    clip_duration: float = 0.0,
    hook_always_on: bool = False,
    title_font_size: int | None = None,
    title_font_family: str | None = None,
    card_layout: dict | None = None,
    keep_segments: list[tuple[float, float]] | None = None,
) -> str:
    """클립 하나에 대한 ASS 자막 문자열을 생성한다.

    clip_words의 타임스탬프는 원본(절대) 기준이며, 여기서 clip_start를 빼서
    클립 내부 상대 시간으로 변환한다.

    card_layout이 주어지면(카드형 레이아웃): 제목은 화면 최상단에 고정하고,
    캡션은 영상 박스 바로 아래 흰 여백 영역에 배치한다.
    card_layout이 None이면(레거시 blur/crop/pad): 기존처럼 영상 위에 오버레이한다.
    """
    width, height = resolution
    font_name = font_family
    title_font_name = title_font_family or font_family
    max_title_size = title_font_size or int(font_size * 1.3)
    # 제목은 무조건 한 줄에 들어가야 하므로, 글자 수에 맞춰 폰트 크기를 동적으로 줄인다
    # (긴 제목이 2줄로 자동 줄바꿈되면서 영상 박스와 겹치는 문제가 있었음).
    title_size = _fit_title_font_size(
        hook_text or "", max_title_size, min_size=font_size, available_width_px=width - 80
    )

    if card_layout:
        # 제목을 상단 고정이 아니라 영상 박스 바로 위, 가깝게 붙여서 배치한다
        # (요청: "제목을 영상 쪽으로 훨씬 아래로 내려라").
        # 이제 한 줄로 고정되므로 줄 높이는 1줄 기준으로만 여백을 잡으면 된다.
        video_box_y = card_layout["video_box_y"]
        line_height_estimate = int(title_size * 1.25)
        title_margin_v = max(20, video_box_y - line_height_estimate - 90)
        video_box_bottom = video_box_y + card_layout["video_box_height"]
        caption_margin_v = video_box_bottom + 60
        caption_alignment = 8  # 상단 기준 (캡션 영역 안에서 위쪽부터 채움)
    else:
        y = _y_position(resolution, position, safe_area_bottom_pct, safe_area_top_pct)
        title_margin_v = int(height * safe_area_top_pct)
        caption_margin_v = height - y
        caption_alignment = 2  # 하단 기준 (기존 방식)

    # Fontname은 폰트 파일명이 아니라 폰트 내부에 등록된 family name과 일치해야 libass가 찾는다.
    # (render.py가 font_path의 디렉터리를 fontsdir로 넘겨 해당 폴더의 폰트 파일들을 스캔하게 한다)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font_name},{font_size},{primary_color},{karaoke_highlight_color},{outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,{outline_width},0,{caption_alignment},40,40,{caption_margin_v},1
Style: Hook,{title_font_name},{title_size},{primary_color},{karaoke_highlight_color},{outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,{outline_width + 1},0,8,40,40,{title_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    # 무음 제거 후 실제 최종 영상 길이 (keep_segments가 있으면 원본 clip_duration보다 짧다)
    final_duration = (
        sum(e - s for s, e in keep_segments) if keep_segments else clip_duration
    )

    if hook_text:
        hook_end = final_duration if (hook_always_on and final_duration > 0) else hook_duration_sec
        events.append(
            f"Dialogue: 0,{_ass_time(0)},{_ass_time(hook_end)},Hook,,0,0,0,,{hook_text}"
        )

    rel_words = [Word(start=w.start - clip_start, end=w.end - clip_start, text=w.text) for w in clip_words]
    if keep_segments:
        rel_words = [
            Word(
                start=_remap_after_silence_removal(w.start, keep_segments),
                end=_remap_after_silence_removal(w.end, keep_segments),
                text=w.text,
            )
            for w in rel_words
        ]
        # 리매핑 후 순간적으로 start==end가 되는(무음 구간에 걸쳐있던) 단어는 아주 살짝 늘려서
        # 카라오케 \k 지속시간이 0이 되어 깨지는 걸 방지
        rel_words = [
            Word(start=w.start, end=max(w.end, w.start + 0.05), text=w.text) for w in rel_words
        ]
    lines = chunk_words_into_lines(rel_words, max_words_per_line)

    for line in lines:
        start_t = _ass_time(line.start)
        end_t = _ass_time(line.end)
        if template == "karaoke":
            text = ""
            for w in line.words:
                dur_cs = max(1, int(round((w.end - w.start) * 100)))
                text += f"{{\\k{dur_cs}}}{w.text} "
            text = text.strip()
        else:
            text = " ".join(w.text for w in line.words)
        events.append(f"Dialogue: 0,{start_t},{end_t},Caption,,0,0,0,,{text}")

    return header + "\n".join(events) + "\n"


def build_ass_for_clip(
    segments: list[Segment],
    clip_start: float,
    clip_end: float,
    config_captions: dict,
    config_hook: dict,
    resolution: tuple[int, int],
    hook_text: str | None,
    card_layout: dict | None = None,
    keep_segments: list[tuple[float, float]] | None = None,
) -> str:
    words = _collect_words_in_range(segments, clip_start, clip_end)
    return build_ass(
        clip_words=words,
        clip_start=clip_start,
        clip_end=clip_end,
        resolution=resolution,
        font_family=config_captions["font_family"],
        font_size=config_captions["font_size"],
        primary_color=config_captions["primary_color"],
        outline_color=config_captions["outline_color"],
        outline_width=config_captions.get("outline_width", 3),
        karaoke_highlight_color=config_captions["karaoke_highlight_color"],
        max_words_per_line=config_captions.get("max_words_per_line", 4),
        position=config_captions.get("position", "bottom"),
        safe_area_bottom_pct=config_captions.get("safe_area_bottom_pct", 0.2),
        safe_area_top_pct=config_captions.get("safe_area_top_pct", 0.1),
        template=config_captions.get("template", "karaoke"),
        hook_text=hook_text if config_hook.get("enabled") else None,
        hook_duration_sec=config_hook.get("duration_sec", 3),
        clip_duration=clip_end - clip_start,
        hook_always_on=config_hook.get("always_on", False),
        title_font_size=config_captions.get("title_font_size"),
        title_font_family=config_captions.get("title_font_family"),
        card_layout=card_layout,
        keep_segments=keep_segments,
    )
