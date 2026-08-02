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


def _clean_word_text(text: str) -> str:
    """유튜브 자동자막의 화자 표시(>>)나 잡토큰을 자막에서 걷어낸다.
    (정밀 재전사 실패로 원본 자동자막으로 폴백할 때 '>> 우리의…'처럼 노출되던 문제 방지.)"""
    t = text.strip()
    while t.startswith(">"):
        t = t[1:].strip()
    return t


# 뜻 없이 끼어드는 순수 간투사/추임새. 이것들만 자막에서 걷어낸다. 뜻을 가진 지시어
# ('그 나라'의 '그', '이제'='now' 등)와 혼동될 수 있는 단어는 부작용이 커서 넣지 않는다
# (그런 공격적 제거는 aggressive_filler 옵션에서만).
_FILLER_WORDS = {"음", "어", "에", "으", "엄", "아", "응", "어어", "으음", "음음", "아아", "어으"}
# 공격적 모드에서 추가로 제거할, 자주 군더더기지만 가끔 뜻을 갖는 단어들.
_FILLER_WORDS_AGGRESSIVE = _FILLER_WORDS | {"그", "그그", "인제", "막", "뭐", "저", "저기"}


def _is_filler(text: str, aggressive: bool = False) -> bool:
    """단어가 순수 추임새인지 — 앞뒤 문장부호는 떼고 판단한다."""
    t = text.strip().strip(".,!?…·").strip()
    return t in (_FILLER_WORDS_AGGRESSIVE if aggressive else _FILLER_WORDS)


def _collect_words_in_range(
    segments: list[Segment], clip_start: float, clip_end: float,
    strip_filler: bool = True, aggressive_filler: bool = False,
) -> list[Word]:
    words: list[Word] = []
    for seg in segments:
        if seg.end < clip_start or seg.start > clip_end:
            continue
        for w in seg.words:
            if w.start >= clip_start and w.end <= clip_end:
                cleaned = _clean_word_text(w.text)
                if cleaned:
                    words.append(Word(start=w.start, end=w.end, text=cleaned))
    # 유튜브 자동자막은 '롤링' 방식이라 연속 자막 이벤트가 같은 단어를 겹쳐서 반복한다.
    # 그대로 두면 자막이 겹쳐 2줄로 뜨고 싱크가 어긋난다. 시간 순 정렬 후 같은 단어가
    # 겹치거나 거의 같은 시각에 중복되면 하나만 남긴다.
    words.sort(key=lambda w: (w.start, w.end))
    deduped: list[Word] = []
    for w in words:
        # 롤링 중복만 제거: 같은 단어가 거의 같은 시각에 다시 나온 경우에만 버린다.
        # (단어 end 시간 비교로 거르면 롤링 자막의 긴 end 때문에 뒤 단어가 줄줄이 삭제되어
        #  자막이 반토막 나는 문제가 있었음 — start 근접만 본다.)
        if any(d.text == w.text and abs(d.start - w.start) < 0.25 for d in deduped[-3:]):
            continue
        deduped.append(w)
    # 뜻 없는 추임새(음·어·에…) 제거. 자막은 전사 기록이 아니라 읽기 보조물이라, 군더더기를
    # 걷어내면 훨씬 정갈하다. 오디오는 그대로라 남은 단어들의 카라오케 타이밍은 영향 없다.
    if strip_filler:
        deduped = [w for w in deduped if not _is_filler(w.text, aggressive_filler)]
    return deduped


def _lines_from_overrides(
    caption_overrides: list, clip_start: float
) -> list[CaptionLine]:
    """사용자가 편집기에서 확정한 자막 라인({start,end,text} 절대초)을 화면용 라인으로 변환한다.
    라인 텍스트를 단어로 쪼개 라인 구간에 균등 분배해, 편집된 자막도 카라오케 강조가 유지되게 한다."""
    lines: list[CaptionLine] = []
    for ov in caption_overrides:
        text = _clean_word_text(str(ov.get("text", "")))
        if not text:
            continue
        rel_start = float(ov["start"]) - clip_start
        rel_end = max(rel_start + 0.05, float(ov["end"]) - clip_start)
        toks = text.split()
        n = len(toks) or 1
        span = (rel_end - rel_start) / n
        ws = [
            Word(start=rel_start + i * span, end=rel_start + (i + 1) * span, text=tok)
            for i, tok in enumerate(toks)
        ]
        lines.append(CaptionLine(start=rel_start, end=rel_end, words=ws))
    return lines


def _clamp_lines_non_overlap(lines: list[CaptionLine]) -> list[CaptionLine]:
    """자막 라인들의 '표시 구간([start,end])'이 서로 겹치지 않게 정리한다.

    유튜브 자동자막으로 폴백하면 단어 end 타임스탬프가 다음 단어 위로 길게 겹치는
    '롤링' 특성 때문에, 4단어씩 끊은 인접 라인의 표시 구간이 시간상 겹쳐 화면에 자막이
    2줄로 동시에 뜬다. 시작 시간 순으로 정렬한 뒤 각 라인의 end를 다음 라인 start까지만
    보이도록 잘라, 어떤 순간에도 한 줄만 표시되게 한다. (라인 내부 \\k 카라오케 타이밍은
    상대값이라 영향 없음.) 길이가 0 이하가 되는 라인은 버린다."""
    ordered = sorted(lines, key=lambda l: l.start)
    result: list[CaptionLine] = []
    for i, line in enumerate(ordered):
        end = line.end
        if i + 1 < len(ordered):
            end = min(end, ordered[i + 1].start)
        if end - line.start < 0.05:
            continue
        result.append(CaptionLine(start=line.start, end=end, words=line.words))
    return result


def _ends_phrase(text: str) -> bool:
    """단어가 문장/절 끝(문장부호)으로 끝나는지 — 여기서 줄을 끊으면 자연스럽다."""
    t = (text or "").strip()
    return bool(t) and t[-1] in ".?!…"


def chunk_words_into_lines(
    words: list[Word], max_words_per_line: int, max_gap: float = 0.45
) -> list[CaptionLine]:
    """자막을 한 줄씩 자를 때 기계적으로 N단어에서 끊지 않고, 말의 자연스러운 경계에서
    우선 끊는다: (1) 문장부호로 끝나는 단어 뒤, (2) 다음 단어와 뚜렷한 쉼(gap≥max_gap)이
    있는 곳. 그래야 '…나타나지 않는 / 겁니다'처럼 한 구가 두 줄로 쪼개지는 어색함이 준다.
    자연 경계가 없으면 최대 max_words_per_line 단어에서 안전하게 끊는다.
    (쉼으로 끊을 땐 최소 2단어를 모아 한 단어짜리 줄이 깜빡이는 것을 막는다.)"""
    lines: list[CaptionLine] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        gap = (words[i + 1].start - w.end) if i + 1 < len(words) else 1e9
        at_cap = len(cur) >= max_words_per_line
        natural = _ends_phrase(w.text) or (gap >= max_gap and len(cur) >= 2)
        if natural or at_cap:
            lines.append(CaptionLine(start=cur[0].start, end=cur[-1].end, words=cur))
            cur = []
    if cur:
        lines.append(CaptionLine(start=cur[0].start, end=cur[-1].end, words=cur))
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


def compute_card_margins(card_layout: dict, resolution: tuple[int, int], title_size: int) -> tuple[int, int]:
    """카드 레이아웃에서 제목/캡션의 기본(오프셋 적용 전) MarginV를 계산한다.
    build_ass()와 위치 편집 웹 UI(web_app.py)가 반드시 같은 값을 써야 미리보기가
    실제 렌더링과 일치하므로, 계산 로직을 이 함수 하나로 모은다."""
    video_box_y = card_layout["video_box_y"]
    line_height_estimate = int(title_size * 1.25)
    title_margin_v = max(20, video_box_y - line_height_estimate - 90)
    video_box_bottom = video_box_y + card_layout["video_box_height"]
    caption_margin_v = video_box_bottom + 60
    return title_margin_v, caption_margin_v


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
    title_offset_x: float = 0.0,
    title_offset_y: float = 0.0,
    caption_offset_x: float = 0.0,
    caption_offset_y: float = 0.0,
    caption_overrides: list | None = None,
    title_font_override: str = "",
    title_size_override: int = 0,
    title_align: str = "",
    title_spacing: float = 0.0,
    caption_font_override: str = "",
    caption_size_override: int = 0,
    caption_align: str = "",
    caption_spacing: float = 0.0,
) -> str:
    """클립 하나에 대한 ASS 자막 문자열을 생성한다.

    clip_words의 타임스탬프는 원본(절대) 기준이며, 여기서 clip_start를 빼서
    클립 내부 상대 시간으로 변환한다.

    card_layout이 주어지면(카드형 레이아웃): 제목은 화면 최상단에 고정하고,
    캡션은 영상 박스 바로 아래 흰 여백 영역에 배치한다.
    card_layout이 None이면(레거시 blur/crop/pad): 기존처럼 영상 위에 오버레이한다.
    """
    width, height = resolution
    # 편집기에서 고른 글꼴/크기가 있으면 그것을 우선 사용(빈 값/0이면 config 기본값).
    font_name = caption_font_override or font_family
    caption_size = caption_size_override or font_size
    title_font_name = title_font_override or title_font_family or font_family
    max_title_size = title_size_override or title_font_size or int(font_size * 1.3)
    # 제목은 무조건 한 줄에 들어가야 하므로, 글자 수에 맞춰 폰트 크기를 동적으로 줄인다
    # (긴 제목이 2줄로 자동 줄바꿈되면서 영상 박스와 겹치는 문제가 있었음).
    # 사용자가 지정한 크기(max_title_size)를 상한으로 삼되, 폭을 넘으면 줄여 한 줄 유지.
    title_size = _fit_title_font_size(
        hook_text or "", max_title_size, min_size=min(font_size, max_title_size),
        available_width_px=width - 80,
    )

    def _align_num(a: str, default: int) -> int:
        # 카드 레이아웃은 상단 기준(7/8/9)으로 배치해야 MarginV 위치 계산과 맞는다.
        return {"left": 7, "center": 8, "right": 9}.get(a, default)

    # 위치 편집 웹 UI에서 사용자가 드래그로 조정한 픽셀 오프셋. MarginV는 커질수록 텍스트가
    # 아래로 내려가므로(상단 기준 정렬), offset_y를 그대로 더하면 된다. 좌우는 중앙 정렬
    # 기준 MarginL/MarginR을 반대 방향으로 움직여서 중심을 offset_x만큼 이동시킨다.
    base_margin_lr = 40
    title_margin_l = max(0, base_margin_lr + title_offset_x)
    title_margin_r = max(0, base_margin_lr - title_offset_x)
    caption_margin_l = max(0, base_margin_lr + caption_offset_x)
    caption_margin_r = max(0, base_margin_lr - caption_offset_x)

    if card_layout:
        # 제목을 상단 고정이 아니라 영상 박스 바로 위, 가깝게 붙여서 배치한다
        # (요청: "제목을 영상 쪽으로 훨씬 아래로 내려라").
        # 이제 한 줄로 고정되므로 줄 높이는 1줄 기준으로만 여백을 잡으면 된다.
        title_margin_v, caption_margin_v = compute_card_margins(card_layout, resolution, title_size)
        title_margin_v = max(0, title_margin_v + title_offset_y)
        caption_margin_v = max(0, caption_margin_v + caption_offset_y)
        caption_alignment = _align_num(caption_align, 8)  # 상단 기준(7/8/9), 기본 중앙
        title_alignment = _align_num(title_align, 8)
    else:
        y = _y_position(resolution, position, safe_area_bottom_pct, safe_area_top_pct)
        title_margin_v = max(0, int(height * safe_area_top_pct) + title_offset_y)
        caption_margin_v = max(0, height - y + caption_offset_y)
        caption_alignment = 2  # 하단 기준 (기존 방식)
        title_alignment = 8

    # Fontname은 폰트 파일명이 아니라 폰트 내부에 등록된 family name과 일치해야 libass가 찾는다.
    # (render.py가 font_path의 디렉터리를 fontsdir로 넘겨 해당 폴더의 폰트 파일들을 스캔하게 한다)

    # 카라오케 색 규칙: ASS \k는 "말하기 전=SecondaryColour, 말한 후=PrimaryColour"로 칠한다.
    # 따라서 '말소리를 따라 강조색으로 켜지게' 하려면 카라오케일 때 Primary=강조색(연두),
    # Secondary=기본색(검정)이어야 한다. (기존엔 반대로 되어 있어 단어가 연두로 떴다가 검정으로
    # 꺼지는 반전 버그가 있었음.) 비카라오케 템플릿은 Primary=기본색 그대로 둔다.
    if template == "karaoke":
        cap_primary, cap_secondary = karaoke_highlight_color, primary_color
    else:
        cap_primary, cap_secondary = primary_color, karaoke_highlight_color

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font_name},{caption_size},{cap_primary},{cap_secondary},{outline_color},&H00000000,-1,0,0,0,100,100,{caption_spacing},0,1,{outline_width},0,{caption_alignment},{caption_margin_l},{caption_margin_r},{caption_margin_v},1
Style: Hook,{title_font_name},{title_size},{primary_color},{karaoke_highlight_color},{outline_color},&H00000000,-1,0,0,0,100,100,{title_spacing},0,1,{outline_width + 1},0,{title_alignment},{title_margin_l},{title_margin_r},{title_margin_v},1

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

    # 사용자가 편집기에서 확정한 자막이 있으면 그것을 최우선으로 쓴다(재전사 결과 무시).
    if caption_overrides:
        lines = _clamp_lines_non_overlap(_lines_from_overrides(caption_overrides, clip_start))
        for line in lines:
            start_t = _ass_time(line.start)
            end_t = _ass_time(line.end)
            if template == "karaoke":
                text = "".join(
                    f"{{\\k{max(1, int(round((w.end - w.start) * 100)))}}}{w.text} " for w in line.words
                ).strip()
            else:
                text = " ".join(w.text for w in line.words)
            events.append(f"Dialogue: 0,{start_t},{end_t},Caption,,0,0,0,,{text}")
        return header + "\n".join(events) + "\n"

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
    lines = _clamp_lines_non_overlap(chunk_words_into_lines(rel_words, max_words_per_line))

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
    title_offset_x: float = 0.0,
    title_offset_y: float = 0.0,
    caption_offset_x: float = 0.0,
    caption_offset_y: float = 0.0,
    caption_overrides: list | None = None,
    font_style: dict | None = None,
) -> str:
    words = _collect_words_in_range(
        segments, clip_start, clip_end,
        strip_filler=config_captions.get("strip_filler", True),
        aggressive_filler=config_captions.get("aggressive_filler", False),
    )
    fs = font_style or {}
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
        title_offset_x=title_offset_x,
        title_offset_y=title_offset_y,
        caption_offset_x=caption_offset_x,
        caption_offset_y=caption_offset_y,
        caption_overrides=caption_overrides,
        title_font_override=fs.get("title_font", ""),
        title_size_override=int(fs.get("title_size", 0) or 0),
        title_align=fs.get("title_align", ""),
        title_spacing=float(fs.get("title_spacing", 0) or 0),
        caption_font_override=fs.get("caption_font", ""),
        caption_size_override=int(fs.get("caption_size", 0) or 0),
        caption_align=fs.get("caption_align", ""),
        caption_spacing=float(fs.get("caption_spacing", 0) or 0),
    )
