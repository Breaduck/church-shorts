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
_FILLER_WORDS = {"음", "어", "에", "으", "엄", "아", "응", "어어", "으음", "음음", "아아", "어으",
                  "예"}  # 설교 말버릇으로 자주 끼는 추임새 "예~"(대답의 "예"와 겹치지만 사용자 요청)
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
            # 완전 포함(start>=clip_start and end<=clip_end)으로 거르면 클립 경계에 걸친
            # 첫/끝 단어가 자막에서 통째로 빠져 "첫 마디부터 자막이 안 맞는" 체감 싱크 문제가
            # 생긴다. 단어 중간점이 클립 안이면 포함시키고, 시간만 클립 경계로 클램프한다.
            mid = (w.start + w.end) / 2.0
            if clip_start <= mid <= clip_end:
                cleaned = _clean_word_text(w.text)
                if cleaned:
                    ws = max(w.start, clip_start)
                    we = min(w.end, clip_end)
                    words.append(Word(start=ws, end=max(we, ws + 0.05), text=cleaned))
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


def _distribute_by_chars(toks: list[str], a: float, b: float) -> list["Word"]:
    """[a, b] 구간을 토큰들의 글자 수에 비례해 나눠 Word 리스트로 만든다(카라오케 강조용).
    균등 분배보다 자연스럽다 — 긴 단어가 더 오래 강조된다."""
    weights = [max(1, len(t)) for t in toks]
    total_w = sum(weights) or 1
    span = max(0.1, b - a)
    out: list[Word] = []
    cur = a
    for t, wt in zip(toks, weights):
        dur = span * wt / total_w
        out.append(Word(start=cur, end=cur + dur, text=t))
        cur += dur
    return out


def _lines_from_overrides(
    caption_overrides: list, clip_start: float, clip_words: list["Word"] | None = None
) -> list[CaptionLine]:
    """사용자가 편집기에서 확정한 자막 라인({start,end,text} 절대초)을 화면용 라인으로 변환한다.

    원칙: **편집기에 보이는 시간이 곧 영상에 구워지는 시간**(WYSIWYG). 각 줄은 사용자가
    선언한 [start, end]를 그대로 쓰고, 줄 안의 카라오케(\\k)는 글자 수 비례로 나눈다.

    예전엔 참조 전사(clip_words)에서 '실제 발화 리듬'을 빌려오는 3단계 로직이 있었지만
    전면 제거했다(2026-09-03) — 참조 시각은 편집기에 보이는 시간과 미묘하게 달라서
    (무음 보정 유무·롤링 겹침·압축 보정), 저장할 때마다 싱크가 흔들리고 단어가 유실되는
    사고가 연쇄로 났다("수정할수록 싱크가 깨진다" 신고의 구조적 원인). 이제 편집기 초안
    자체가 정밀 인식 시각에서 나오므로(web_app._caption_lines_for_clip), 선언된 줄 시간이
    이미 정확하고, 시간을 다시 맞추고 싶으면 '싱크 맞추기' 버튼(명시적 동작)을 쓴다.
    clip_words 인자는 하위 호환용으로만 남겨두고 사용하지 않는다."""
    lines: list[CaptionLine] = []
    for ov in caption_overrides:
        text = _clean_word_text(str(ov.get("text", "")))
        if not text:
            continue
        rel_start = float(ov["start"]) - clip_start
        rel_end = max(rel_start + 0.05, float(ov["end"]) - clip_start)
        lines.append(CaptionLine(
            start=rel_start, end=rel_end,
            words=_distribute_by_chars(text.split(), rel_start, rel_end),
        ))
    return lines


def _clamp_lines_non_overlap(lines: list[CaptionLine]) -> list[CaptionLine]:
    """자막 라인들의 '표시 구간([start,end])'이 서로 겹치지 않게 정리한다.

    유튜브 자동자막으로 폴백하면 단어 end 타임스탬프가 다음 단어 위로 길게 겹치는
    '롤링' 특성 때문에, 4단어씩 끊은 인접 라인의 표시 구간이 시간상 겹쳐 화면에 자막이
    2줄로 동시에 뜬다. 시작 시간 순으로 정렬한 뒤 각 라인의 end를 다음 라인 start까지만
    보이도록 잘라, 어떤 순간에도 한 줄만 표시되게 한다. 길이가 0 이하가 되는 라인은 버린다.

    라인 end를 자르면 그 안의 단어(\\k 카라오케 타이밍)도 같이 잘라야 한다 — "상대값이라
    영향 없다"는 예전 가정이 틀렸다: caption_overrides가 실제 단어(base_segments)의 원래
    타임스탬프를 그대로 물려받는 경우(실측: 롤링 자막에서 매칭된 단어들의 원래 구간이
    4초인데, 다음 줄과 겹쳐 표시 구간이 0.1초로 잘린 사례), \\k 합계(4초)가 실제 표시
    시간(0.1초)보다 훨씬 길게 남아 강조색이 글자가 사라진 뒤에도 계속 진행 중인 것처럼
    보였다("싱크가 안 맞는다"의 실제 원인). 단어를 라인의 최종 end 안으로 잘라 넣어
    \\k 합계가 항상 실제 표시 시간과 일치하게 만든다."""
    ordered = sorted(lines, key=lambda l: l.start)
    result: list[CaptionLine] = []
    for i, line in enumerate(ordered):
        end = line.end
        if i + 1 < len(ordered):
            end = min(end, ordered[i + 1].start)
        if end - line.start < 0.05:
            continue
        # 단어끼리도 겹치지 않게 순서대로 눌러 담고, 라인의 최종 표시 구간을 넘치면 단어를
        # '버리는' 게 아니라 구간 안으로 선형 압축한다. (예전엔 넘친 단어를 드롭했는데,
        # 인접 라인이 각자 실제 발화 시각을 빌려 쓰면 서로 침범하는 경우가 생겨 마지막
        # 단어가 통째로 사라졌다 — 실측: '안 되잖아요'에서 '되잖아요' 유실. 압축은 리듬이
        # 아주 살짝 빨라질 뿐 내용은 절대 잃지 않는다.)
        words = []
        prev_end = line.start
        for w in line.words:
            ws = max(w.start, prev_end)
            we = max(w.end, ws + 0.03)
            words.append(Word(start=ws, end=we, text=w.text))
            prev_end = we
        if words and words[-1].end > end:
            a = words[0].start
            src_span = max(1e-6, words[-1].end - a)
            scale = max(0.05, end - a) / src_span
            words = [
                Word(start=a + (w.start - a) * scale, end=a + (w.end - a) * scale, text=w.text)
                for w in words
            ]
        result.append(CaptionLine(start=line.start, end=end, words=words))
    return result


def _ends_phrase(text: str) -> bool:
    """단어가 문장/절 끝(문장부호)으로 끝나는지 — 여기서 줄을 끊으면 자연스럽다."""
    t = (text or "").strip()
    return bool(t) and t[-1] in ".?!…"


def _display_text(text: str) -> str:
    """화면에 실제로 보여줄 단어 텍스트. 정밀 전사가 붙이는 끝 마침표는 자막에서 어색해
    떼어낸다(사용자 요청). 물음표/느낌표는 뜻을 가지므로 남긴다. 분절(줄바꿈) 판단은
    마침표를 떼기 전 원본 텍스트로 하므로 여기서 떼도 경계 품질에는 영향이 없다."""
    t = (text or "").strip()
    stripped = t.rstrip(".")
    return stripped if stripped else t


# ---- 한국어 의미 단위 줄바꿈 규칙 ----------------------------------------
# 이 단어 '뒤'에서 끊으면 어색한 것들: 부정 부사(안/못)는 뒤 용언과 한 몸이고,
# 관형사(그/이/저/한/두…)는 뒤 명사를 꾸민다. 의존명사(수/것/줄…)는 앞뒤 모두와 묶인다.
_NO_BREAK_AFTER = {"안", "못", "잘", "더", "덜", "꼭", "왜", "그", "이", "저", "한", "두", "세", "네",
                   "수", "것", "거", "줄", "때", "뿐", "채", "지"}
# 이 단어 '앞'에서 끊으면 어색한 것들: 의존명사('감당할 수', '가시는 거')와
# 보조용언('되지 않고', '하지 못하고')은 앞 말에 붙어야 뜻이 산다.
_NO_BREAK_BEFORE_EXACT = {"수", "것", "거", "줄", "때", "뿐", "채", "만큼", "중", "등",
                          "수가", "수는", "수도", "수를", "수밖에", "줄로", "줄은", "줄을",
                          "것입니다", "것이다", "것이죠", "것이에요", "겁니다", "거예요", "거죠", "거야"}
_NO_BREAK_BEFORE_PREFIX = ("않", "못하", "없")
# 이 어미로 끝나면 절(節)이 일단락된 것 — 여기서 끊으면 자연스럽다.
_CLAUSE_ENDINGS = ("고", "서", "며", "면", "는데", "지만", "니까", "다가", "라서", "려고",
                   "다면", "거든요", "는데요", "어요", "아요")
# 문장 종결 어미 — 최상급 경계.
_SENTENCE_ENDINGS = ("습니다", "합니다", "됩니다", "입니다", "니다", "십시오", "세요", "에요",
                     "예요", "겠죠", "네요", "군요", "잖아요", "이죠", "하죠", "거죠")


def _break_score(words: list[Word], j: int, max_gap: float) -> float:
    """words[j-1]과 words[j] 사이에서 줄을 끊는 것의 자연스러움 점수.
    문장 끝 > 절 끝 > 뚜렷한 쉼 > 무표정 경계 > 조사 뒤 순이며, 부정어/관형사 뒤와
    의존명사·보조용언 앞은 금지(-100)한다."""
    prev = words[j - 1].text.strip()
    prev_bare = prev.strip(".,!?…·")
    nxt = words[j].text.strip()
    nxt_bare = nxt.strip(".,!?…·")
    # 문장이 끝난 지점은 금지 규칙보다 우선한다: '…가시는 거.'처럼 의존명사로 끝나는
    # 문장 뒤에서 못 끊으면 다음 문장('우리는 무지합니다')이 같은 줄에 섞인다(실측).
    if _ends_phrase(prev):
        return 12.0
    if prev_bare.endswith(_SENTENCE_ENDINGS):
        return 10.0
    if prev_bare in _NO_BREAK_AFTER:
        return -100.0
    if nxt_bare in _NO_BREAK_BEFORE_EXACT or nxt_bare.startswith(_NO_BREAK_BEFORE_PREFIX):
        return -100.0
    score = 0.0
    if prev_bare.endswith(_CLAUSE_ENDINGS):
        score += 6.0
    elif prev_bare.endswith(("을", "를")):
        score -= 2.0  # 목적어와 서술어 사이 — 되도록 붙여둔다
    elif prev_bare.endswith("의"):
        score -= 4.0  # 관형격 조사 뒤는 거의 항상 어색
    gap = words[j].start - words[j - 1].end
    if gap >= max_gap:
        score += 5.0
    elif gap >= 0.25:
        score += 2.0
    return score


def _char_units(ch: str) -> float:
    """글자 하나의 대략적 폭(폰트 크기 1.0 = 전각 1자 기준). 한글/한자는 전각(1.0),
    공백·라틴·숫자·문장부호는 반각 이하. libass 실제 렌더 폭의 근사치로, 자막이
    화면 폭을 넘어 2줄로 자동 랩핑되는지 판단하는 데 쓴다."""
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3 or 0x4E00 <= o <= 0x9FFF or ch in "…—":
        return 1.0
    if ch == " ":
        return 0.34
    if ch.isdigit() or "a" <= ch <= "z" or "A" <= ch <= "Z":
        return 0.56
    return 0.45


def _line_units(words: list[Word]) -> float:
    text = " ".join(_display_text(w.text) for w in words)
    return sum(_char_units(c) for c in text)


def chunk_words_into_lines(
    words: list[Word], max_words_per_line: int, max_gap: float = 0.45,
    max_units: float | None = None,
) -> list[CaptionLine]:
    """자막 줄바꿈을 기계적 N단어 컷이 아니라 말의 의미 경계에서 한다.

    기존 방식(4단어 강제 컷)은 '안 되지 / 않고', '가시는 / 거'처럼 한 뜻 단위를
    두 줄로 찢어 읽기 흐름을 깨뜨렸다(실측 불만). 대신 각 후보 지점의 자연스러움을
    _break_score로 채점해, 창(최대 max_words_per_line, 금지 경계 회피 시 +1단어까지)
    안에서 가장 좋은 지점을 골라 끊는다. 문장부호로 끝나는 단어 뒤 > 연결어미 뒤 >
    뚜렷한 쉼 순으로 선호하고, 부정어·관형사 뒤/의존명사·보조용언 앞은 절대 안 끊는다.

    max_units가 주어지면(화면 폭 ÷ 폰트 크기) 그 폭을 넘는 줄은 금지한다: 자막은
    무조건 화면에 1줄이어야 하고(사용자 요구), 넘치면 libass가 멋대로 2줄로 랩핑하므로
    아예 후보에서 제외해 각 조각이 따로따로 순차 표시되게 한다."""
    lines: list[CaptionLine] = []
    i = 0
    n = len(words)
    while i < n:
        # 후보: 현재 줄을 words[i:j]로 확정하는 j들. 기본 창은 max_words_per_line,
        # 금지 경계를 피해야 할 때를 위해 1단어 초과(오버플로 페널티)까지 본다.
        best_j, best_score = i + 1, -1e9
        for j in range(i + 1, min(i + max_words_per_line + 1, n) + 1):
            # 화면 폭 초과 줄은 후보 자체가 아니다 (단, 한 단어는 쪼갤 수 없어 허용).
            if max_units is not None and j - i > 1 and _line_units(words[i:j]) > max_units:
                break  # 단어를 더 붙일수록 더 넘치므로 이후 j도 전부 불가
            if j >= n:
                score = 100.0  # 마지막 단어까지 담으면 그대로 끝
            else:
                score = _break_score(words, j, max_gap)
                wlen = j - i
                score += wlen * 0.4          # 같은 값이면 줄을 조금 더 채우는 쪽 선호
                if wlen == 1:
                    score -= 6.0             # 한 단어짜리 깜빡이 줄 억제
                if wlen > max_words_per_line:
                    score -= 1.0             # 초과는 금지 경계 회피용으로만
            if score > best_score:
                best_j, best_score = j, score
        cur = words[i:best_j]
        lines.append(CaptionLine(start=cur[0].start, end=cur[-1].end, words=cur))
        i = best_j
    return lines


def _hi_bare(s: str) -> str:
    """하이라이트 매칭용 정규화: 앞뒤 문장부호·공백 제거 + 라틴 소문자화."""
    return (s or "").strip().strip(" .,!?…·\"'()[]{}~-").lower()


def _make_highlight_matcher(keywords: list[str] | None):
    """자막 단어가 강조 대상인지 판정하는 함수를 만든다.

    한국어 자막 단어는 조사가 붙어("사랑을","은혜가") 정식 키워드와 정확히 안 맞을 때가
    많다. 그래서 '키워드가 단어에 포함' 또는 '단어가 키워드에 포함'이면 강조로 본다.
    2글자 미만 키워드는 오탐이 커서 제외한다."""
    bares = [b for b in (_hi_bare(k) for k in (keywords or [])) if len(b) >= 2]
    if not bares:
        return None

    def _match(disp: str) -> bool:
        low = _hi_bare(disp)
        if len(low) < 2:
            return False
        return any(kw in low or low in kw for kw in bares)

    return _match


def _highlight_wrap(disp: str, hi_color: str) -> str:
    r"""단어를 형광색+볼드로 감싼다. {\r}로 스타일 기본값(흰색/현재 볼드)으로 복귀."""
    return f"{{\\c{hi_color}\\b1}}{disp}{{\\r}}"


def _ass_time(sec: float) -> str:
    if sec < 0:
        sec = 0
    cs = int(round(sec * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _karaoke_text(words: list[Word], hi_match=None, hi_color: str = "", base_color: str = "") -> str:
    r"""단어별 카라오케(\k) 강조 텍스트. 단어 사이의 '쉼(gap)'만큼 \k를 먼저 넣어,
    강조가 실제 목소리보다 앞서 달려나가지 않게 맞춘다(줄 시작 = 첫 단어 start 기준).
    이걸 안 하면 단어 사이 침묵이 무시돼 강조가 목소리를 앞질러 자막이 어긋난다
    (특히 필러 제거로 단어가 빠지면 그 자리 침묵이 커져 더 심해짐).

    hi_match/hi_color가 주어지면 핵심 키워드는 리빌(sung) 색을 형광색으로 바꿔서(\1c) 그
    단어만 형광으로 켜지게 한다 — \r을 쓰지 않아 카라오케 상태를 깨지 않고, 단어 뒤에서
    base_color(원래 리빌색)로 되돌린다. 볼드(\b1/\b0)도 그 단어에만 적용."""
    if not words:
        return ""
    parts: list[str] = []
    prev_end = words[0].start  # 줄은 첫 단어에서 시작 → 그 앞엔 쉼이 없다
    for w in words:
        gap_cs = int(round((w.start - prev_end) * 100))
        if gap_cs > 0:
            parts.append(f"{{\\k{gap_cs}}}")  # 쉼: 아무것도 강조 안 하고 시간만 소비
        dur_cs = max(1, int(round((w.end - w.start) * 100)))
        disp = _display_text(w.text)
        if hi_match and hi_color and hi_match(disp):
            parts.append(f"{{\\1c{hi_color}\\b1\\k{dur_cs}}}{disp}{{\\1c{base_color}\\b0}} ")
        else:
            parts.append(f"{{\\k{dur_cs}}}{disp} ")
        prev_end = w.end
    return "".join(parts).strip()


def _fit_title_font_size(
    text: str, max_size: int, min_size: int, available_width_px: int, char_width_ratio: float = 0.62,
    font_family: str = "", available_height_px: int = 0,
) -> int:
    """제목 폰트 크기를 정한다: (여러 줄이면 가장 긴 줄이) 폭에 들어가도록 상한을 잡는다.

    사용자가 팝업에서 제목을 직접 수정하며 Enter로 줄바꿈(\\n)할 수 있게 되면서, 이 함수도
    여러 줄을 지원한다: 폭은 '가장 긴 줄' 기준으로 맞추고, 높이는 줄 수 × 줄높이가
    available_height_px(영상 박스 위 여백)를 넘지 않게 상한을 건다(제목이 영상과 겹침 방지).

    font_family가 주어지고 그 폰트 파일을 찾을 수 있으면 실제 글리프 폭(fonts.py
    measure_text_width_px)으로 정확히 계산한다 — 미리보기(브라우저, 같은 폰트 파일)와
    렌더(libass) 크기가 어긋나던 문제의 근본 원인 수정. 측정 불가하면 char_width_ratio 근사."""
    if not text:
        return max_size
    lines = text.split("\n")
    non_empty = [ln for ln in lines if ln.strip()] or [text]
    size = max_size
    measure = None
    if font_family:
        from src.fonts import measure_text_width_px as measure
    for ln in non_empty:
        w = measure(ln, font_family, max_size) if measure else None
        if w is not None and w > 0:
            if w > available_width_px:
                size = min(size, int(available_width_px / w * max_size))
        else:
            approx = int(available_width_px / (max(1, len(ln)) * char_width_ratio))
            size = min(size, approx)
    if available_height_px and available_height_px > 0:
        # 줄높이 ≈ 1.25×글자크기. 줄 수(빈 줄 포함)만큼 쌓여도 여백을 안 넘게.
        height_cap = int(available_height_px / (max(1, len(lines)) * 1.25))
        size = min(size, height_cap)
    return max(min_size, min(max_size, size))


def warp_time(t: float, first_sec: float, factor: float) -> float:
    """훅 배속(초반 first_sec초를 factor배로 빠르게)이 적용된 '출력 타임라인'에서의 시각.

    렌더(ffmpeg setpts/atempo)와 자막(ASS 이벤트 시각)이 이 함수 하나를 공유해야 자막이
    영상과 정확히 맞는다. 초반 구간은 압축되고(t/factor), 그 이후는 압축량만큼 통째로 당겨진다.
    factor<=1 또는 first_sec<=0이면 항등(원래 시각)."""
    if factor <= 1.0 or first_sec <= 0:
        return t
    if t <= first_sec:
        return t / factor
    return first_sec / factor + (t - first_sec)


def _snap_word_starts_to_voice(
    words: list[Word], silences: list[tuple[float, float]]
) -> list[Word]:
    """Whisper가 단어 앞의 침묵을 단어 발화 시간에 흡수하는 문제를 실제 오디오 기준으로 교정한다.

    실측: 쉼(pause) 다음 첫 단어의 start가 실제 발화보다 1~1.8초 이르게 찍혀
    ('아우르' \\k182 = 1.82초처럼 단어 하나가 침묵 전체를 차지), 자막 줄이 목사님이
    말을 시작하기 한참 전에 미리 떠서 "자막이 말보다 빠르다"고 체감된다.
    ffmpeg silencedetect로 잰 무음 구간(silences, 클립 상대시간)을 받아, 단어 구간을
    무음이 덮고 있으면 start를 무음이 끝나는 지점(= 실제 발화 시작) 직전으로 민다.

    조건: 무음이 단어 시작 부근(start+0.35 이내)에서 시작하고, 단어가 끝나기 전에
    무음이 끝나는 경우만. (단어 중간·끝의 무음은 말끝 여운이므로 건드리지 않는다.)"""
    if not silences:
        return words
    out: list[Word] = []
    for w in words:
        new_start = w.start
        for s, e in silences:
            if s <= w.start + 0.35 and w.start + 0.15 < e <= w.end + 0.1:
                new_start = max(new_start, e - 0.08)
        out.append(Word(start=new_start, end=max(w.end, new_start + 0.05), text=w.text))
    return out


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
    # 제목 위치 변천: 중앙정렬 → "더 위쪽으로"(2026-08-24) 요청에 최상단 24px 고정 →
    # "너무 위쪽"(2026-09-02) 피드백. 24px는 폰 상단 상태바·쇼츠 UI가 덮는 위험 지역이었다
    # (과교정). 이제 상단 안전영역(높이의 6% ≈ 115px)에 붙인다 — 위쪽이되 UI에 안 가리고,
    # 제목 블록(최대 2줄)이 영상 박스 위 여백 안에 들어온다. 세부 취향은 편집기 드래그로.
    title_margin_v = max(24, int(resolution[1] * 0.06))
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
    voice_silences: list[tuple[float, float]] | None = None,
    sync_offset_sec: float = 0.0,
    hook_speedup: tuple[float, float] | None = None,
    highlight_keywords: list[str] | None = None,
    highlight_color: str = "",
    animate: bool = False,
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
    # 사용자가 지정한 크기(max_title_size)를 상한으로 삼되, (가장 긴 줄이) 폭을 넘으면 줄인다.
    # 제목이 여러 줄(사용자가 Enter로 줄바꿈)이면 영상 박스 위 여백 안에 다 들어가도록 높이도 제한.
    title_avail_h = 0
    if card_layout and card_layout.get("video_box_y"):
        _title_top = max(24, int(height * 0.06))
        title_avail_h = max(80, int(card_layout["video_box_y"]) - _title_top - 20)
    title_size = _fit_title_font_size(
        hook_text or "", max_title_size, min_size=min(font_size, max_title_size),
        available_width_px=width - 80, font_family=title_font_name,
        available_height_px=title_avail_h,
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

    # 훅 배속(초반 first_sec초를 factor배로): 자막 이벤트 시각도 렌더(setpts/atempo)와
    # 같은 워프를 태워야 영상과 맞는다. 항등(None/factor<=1)이면 원래 시각 그대로.
    _wf, _wk = hook_speedup if hook_speedup else (0.0, 1.0)

    # 자막 키워드 형광 강조(선택): highlight_color가 있고 강조어가 매칭될 때만 적용.
    # 카라오케 템플릿은 이미 단어별로 색이 움직이므로 정적(비카라오케) 자막에만 칠한다.
    _hi_match = _make_highlight_matcher(highlight_keywords) if highlight_color else None

    def _w(t: float) -> float:
        return warp_time(t, _wf, _wk)

    def _warp_line(line: CaptionLine) -> CaptionLine:
        if _wk <= 1.0 or _wf <= 0:
            return line
        return CaptionLine(
            start=_w(line.start), end=_w(line.end),
            words=[Word(start=_w(w.start), end=_w(w.end), text=w.text) for w in line.words],
        )

    if hook_text:
        hook_end = final_duration if (hook_always_on and final_duration > 0) else hook_duration_sec
        hook_end = _w(hook_end)  # 배속된 최종 길이에 맞춰 제목 표시 끝도 당긴다
        # 캡션은 _display_text/_karaoke_text가 끝 마침표를 떼지만, 제목(Hook)은 그 경로를
        # 안 타서 모델이 붙인 마침표가 그대로 나갈 수 있었다("자막 끝마다 점" 원인 중 하나).
        # 사용자가 팝업에서 Enter로 넣은 줄바꿈(\n)은 ASS의 강제 줄바꿈(\N)으로 변환한다.
        title_disp = "\\N".join(_display_text(ln) for ln in (hook_text or "").split("\n"))
        if animate:
            # 제목 등장 '팝': 살짝 작게+투명하게 시작해 살짝 오버슈트했다가 제자리로(스케일)
            # + 페이드인. 모션그래픽 옵션(사용자 요청, 인스타 레퍼런스의 타이틀 애니메이션).
            title_disp = (
                "{\\fad(180,0)\\fscx78\\fscy78"
                "\\t(0,220,\\fscx106\\fscy106)\\t(220,320,\\fscx100\\fscy100)}"
                + title_disp
            )
        events.append(
            f"Dialogue: 0,{_ass_time(0)},{_ass_time(hook_end)},Hook,,0,0,0,,{title_disp}"
        )

    def _emit_line(line: CaptionLine) -> None:
        # sync_offset_sec: 자막 표시를 오디오 대비 일괄 지연(+)/선행(-)하는 전역 노브.
        # 유튜브 자동자막 단어 시각이 전반적으로 0.2~0.5초 이른 '상시 리드'는 무음 교정
        # (긴 쉼만 잡음)으로는 안 잡혀서, 최종 이벤트 시각에서 통째로 민다. 카라오케 \k는
        # 라인 시작 기준 상대값이라 라인과 함께 자연히 밀린다.
        line = _warp_line(line)  # 훅 배속 반영(항등이면 그대로)
        start_t = _ass_time(max(0.0, line.start + sync_offset_sec))
        end_t = _ass_time(max(0.0, line.end + sync_offset_sec))
        if template == "karaoke":
            text = _karaoke_text(line.words, _hi_match, highlight_color, cap_primary)
        else:
            parts = []
            for w in line.words:
                disp = _display_text(w.text)
                if _hi_match and _hi_match(disp):
                    disp = _highlight_wrap(disp, highlight_color)
                parts.append(disp)
            text = " ".join(parts)
        if animate:
            # 자막 줄 등장/퇴장 페이드(부드러운 전환) — 모션그래픽 옵션.
            text = "{\\fad(120,80)}" + text
        events.append(f"Dialogue: 0,{start_t},{end_t},Caption,,0,0,0,,{text}")

    # 사용자가 편집기에서 확정한 자막이 있으면 그것을 최우선으로 쓴다(재전사 결과 무시).
    # 시간도 선언된 값 그대로(WYSIWYG) — 참조 전사로 리듬을 빌리거나 무음 보정을 다시
    # 적용하지 않는다(그게 저장할 때마다 싱크가 흔들리던 구조적 원인, _lines_from_overrides 주석).
    if caption_overrides:
        # 전역 sync_offset(+0.25초, 유튜브 자동자막의 '상시 리드' 보정용)도 여기선 끈다 —
        # 편집 자막 시각은 정밀 인식/사용자 지정이라 이미 정확한데 그 위에 0.25초를 또
        # 더하면 전체가 미세하게 늦어진다("전체적으로 미세하게 안 맞는다" 실신고 원인).
        # 전체를 밀고 싶으면 편집기의 '전체 밀기' 버튼으로 명시적으로 조절한다.
        sync_offset_sec = 0.0
        lines = _clamp_lines_non_overlap(_lines_from_overrides(caption_overrides, clip_start))
        for line in lines:
            _emit_line(line)
        return header + "\n".join(events) + "\n"

    rel_words = [Word(start=w.start - clip_start, end=w.end - clip_start, text=w.text) for w in clip_words]
    # 실제 오디오의 무음 구간 기준으로 단어 start를 교정해 "자막이 말보다 앞서 뜨는" 문제를
    # 잡는다. (voice_silences는 원본 타임라인 기준이라 무음 제거 리매핑 전에 적용해야 한다.)
    if voice_silences and not keep_segments:
        rel_words = _snap_word_starts_to_voice(rel_words, voice_silences)
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
    # 자막은 화면에 무조건 1줄: 사용 가능한 폭(픽셀)을 폰트 크기로 나눈 전각 단위 폭을
    # 상한으로 넘겨, 넘치는 줄은 아예 만들어지지 않게 한다(각 조각은 따로 순차 표시).
    usable_px = width - caption_margin_l - caption_margin_r - 24
    max_units = max(4.0, usable_px / max(1, caption_size))
    lines = _clamp_lines_non_overlap(
        chunk_words_into_lines(rel_words, max_words_per_line, max_units=max_units)
    )

    for line in lines:
        _emit_line(line)

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
    voice_silences: list[tuple[float, float]] | None = None,
    hook_speedup: tuple[float, float] | None = None,
    highlight_keywords: list[str] | None = None,
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
        voice_silences=voice_silences,
        sync_offset_sec=float(config_captions.get("sync_offset_sec", 0.0) or 0.0),
        hook_speedup=hook_speedup,
        highlight_keywords=(
            highlight_keywords if config_captions.get("highlight_keywords_enabled", True) else None
        ),
        highlight_color=(
            config_captions.get("highlight_color", "&H0000E5FF")
            if config_captions.get("highlight_keywords_enabled", True) else ""
        ),
        animate=bool(config_captions.get("animate", False)),
    )
