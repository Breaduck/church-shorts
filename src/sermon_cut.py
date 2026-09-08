"""설교 하이라이트 선정 v2 — '문장 단위 + 명제 우선 2패스 + 결정론적 검증'.

2026-09-08 구조 개편 배경(사용자: "핵심을 걍 못 잡아내잖아"):
  1) 예전 단일 프롬프트는 25분 전사본 읽기·주제 찾기·초 단위 경계·7축 채점·제목·캡션·
     해시태그·키워드를 JSON 하나로 한 번에 시켰다 → 핵심 판단이 형식 채우기에 밀렸다.
  2) 모델이 보는 전사본이 유튜브 자동자막 '조각'이라 문장이 없었다 → hook/payoff를 조각
     중간에서 잡고, 펀치라인 직전에서 끊기는 사고가 구조적으로 반복됐다.
  3) 뽑힌 클립이 벤치마크 뼈대(설정→명제→착지)에 맞는지 아무도 검증하지 않았다.

이 모듈의 흐름:
  build_sentences()      단어 시각을 이용해 전사본을 '문장' 단위로 재조립(문장별 start/end).
  pass 1 (Opus)          명제 지도: 설교자가 한 문장으로 못 박은 핵심 문장(core)과 그 문장을
                         감싸는 컷(start/end 문장 번호)만 뽑는다. 채점·제목 없음, 사고는 전부 여기.
  verify_and_fix()       결정론적 검증·보정: 핵심 문장 포함, 끝이 미완(접속형/질문)이면 착지까지
                         확장, 첫 문장이 구조 표지("둘째,")면 제거, 길이 상한/하한, 중복 제거.
  pass 2 (Sonnet)        확정된 컷의 '실제 대사'만 보고 세부 축 채점·insight·캡션·키워드·제목 초안.
  최종 제목은 기존 refine_titles 별도 패스가 맡는다(main.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.highlights import Clip, _invoke_claude_json
from src.scoring import compute_scores
from src.transcribe import Transcript


# ---------------------------------------------------------------------------
# 1. 문장 재조립
# ---------------------------------------------------------------------------
@dataclass
class Sentence:
    idx: int
    start: float
    end: float
    text: str


_NOISE_TOKEN = re.compile(r"^\[[^\]]*\]$|^>>$|^>>\S*$")  # [노래] [음악] [한숨] >> 등
_TERMINAL_PUNCT = re.compile(r"[.?!。]$")
# 마침표 없이도 문장이 끝났다고 볼 수 있는 종결어미(단어 끝). 자동자막은 마침표가 대체로 있지만
# 빠지는 곳이 있어, 긴 문장(12초/25단어 초과)에서만 이 규칙으로 보조 분할한다.
_FINAL_ENDING = re.compile(r"(습니다|ㅂ니다|입니다|니다|세요|어요|아요|해요|예요|이에요|거예요|네요|죠|지요|까|나요|잖아요|랍니다|습니까|십시오)$")
_MAX_SENT_SEC = 12.0
_MAX_SENT_WORDS = 25
_GAP_SPLIT_SEC = 1.2


def build_sentences(transcript: Transcript) -> list[Sentence]:
    """단어 시각으로 전사본을 문장 단위로 재조립한다.

    분할 규칙(우선순위): 문장 부호(. ? !) → 단어 사이 1.2초 이상 침묵 → 너무 길면 종결어미.
    단어 시각이 없는 세그먼트는 세그먼트 텍스트를 한 덩어리로 쓴다."""
    words: list[tuple[float, float, str]] = []
    for seg in transcript.segments:
        if seg.words:
            for w in seg.words:
                t = (w.text or "").replace(">>", "").strip()
                if not t or _NOISE_TOKEN.match(t):
                    continue
                words.append((float(w.start), float(w.end), t))
        else:
            t = (seg.text or "").strip()
            if t:
                words.append((float(seg.start), float(seg.end), t))

    sentences: list[Sentence] = []
    cur: list[tuple[float, float, str]] = []

    def _flush() -> None:
        nonlocal cur
        if not cur:
            return
        text = " ".join(w[2] for w in cur).strip()
        if text:
            sentences.append(Sentence(len(sentences), cur[0][0], max(w[1] for w in cur), text))
        cur = []

    for i, (ws, we, wt) in enumerate(words):
        if cur:
            gap = ws - cur[-1][1]
            if gap >= _GAP_SPLIT_SEC:
                _flush()
        cur.append((ws, we, wt))
        if _TERMINAL_PUNCT.search(wt):
            _flush()
            continue
        # 마침표 없는 긴 문장 보조 분할
        if (we - cur[0][0] > _MAX_SENT_SEC or len(cur) > _MAX_SENT_WORDS) and _FINAL_ENDING.search(
            wt.rstrip(",")
        ):
            _flush()
    _flush()
    return sentences


def _fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def format_sentences(sentences: list[Sentence]) -> str:
    return "\n".join(f"S{s.idx} [{_fmt_ts(s.start)}] {s.text}" for s in sentences)


# ---------------------------------------------------------------------------
# 2. 1차 패스 — 명제 지도 + 컷 (문장 번호로만 경계 지정)
# ---------------------------------------------------------------------------
BENCHMARK_BLOCK = """## 벤치마크 — 이렇게 잘라야 한다 (잘 되는 교회 쇼츠 채널의 조회수 상위 클립 실측)
상위 클립은 전부 같은 뼈대다: **설교자가 한 문장으로 못 박은 명제(핵심 문장)** 하나를 중심에 두고,
그 앞의 **설정**(공감 상황·질문·명제 자체)에서 시작해, 그 명제가 **착지**하는 축복·"~줄로 믿습니다"·아멘·
명제 재선언에서 딱 끊는다. 제목도 그 핵심 문장 그대로다. 실제 예:
- 「기도하는 사람은 오염되지 않습니다」(59초): 첫 문장 "기도하는 사람은 세상에 오염되지 않습니다." → 예수님이
  인기에 오염되지 않은 예 → 끝 "우리가 먼저 기도하는 것이 하나님 앞에 놀라운 은혜의 통로가 됩니다."
- 「뭐 하러 그렇게까지 하냐?」(60초): "우리가 믿음 생활 하다 보면 종종 듣게 되는 말이 있습니다. 뭐 하러 그렇게까지
  하냐?" → 더운 날 예배 안 나온 78명 유머 → "뭐 하러 그렇게까지 ~하냐" 5번 반복 크레센도 → 끝 "여러분 이게 믿음입니다."
- 「아무것도 보이지 않을 때」(69초): "내가 지금 어려운 상황인데 내 옆에 도와줄 사람이 하나도 없고…" → 요나 이야기를
  압축(성경 인물은 근거로 중간에) → 끝 "지금 보이지 않는다고 해서 없다는 말이 아닙니다 … 반드시 있을 줄로 믿습니다."
- 「나와 상관없는 것들과 이별할 수 있어야 합니다」(42초): "이별이 쉽지 않습니다. 이별 노래가 얼마나 많아요." → 명제 →
  끝 "내려놓는 은혜가 있기를 축복합니다 … 끝까지 사명을 감당하는 저와 여러분 될 줄로 믿습니다."
- 「규칙보다 구원이 먼저입니다」(59초): "규칙이 신앙을 지배하면 안 됩니다. 은혜가 우리를 붙잡아야 합니다." → 십계명은
  구원 뒤에 주신 것 → 끝 "좀 부족해도 연약해도 완벽하지 않아도 … 계명을 주신 줄로 믿습니다."
- 「딱 맞아떨어지면 무조건 하나님의 뜻일까?」(153초, 그 채널 1위): 목사 간증 단골 레퍼토리(쌀독 긁는 소리) 유머 재연 →
  반전 "딱 맞아떨어진다고 다 하나님의 뜻은 아닙니다" → 끝 "성도의 기준은 상황이 아닙니다." 이야기가 온전하면 길어도 터진다.
공통점: (1) 시청자는 **교회 다니는(또는 관심 있는) 한국 사람**이다 — 성경 인물·용어는 얼마든지 나오되, **첫 문장은
인물 소개·본문 설명이 아니라 명제/공감 상황/질문**이다. (2) 한 클립 = 명제 하나. (3) 끝은 가장 힘 있는 문장에서
딱 끊는다. (4) 40~60초가 주류, 온전한 이야기면 90초까지 괜찮다."""


def build_thesis_cut_prompt(
    sentences: list[Sentence],
    video_duration_sec: float,
    min_clips: int,
    max_clips: int,
    extra_block: str = "",
) -> str:
    extra = f"\n{extra_block}\n" if extra_block else ""
    return f"""너는 조회수가 잘 나오는 교회 쇼츠 채널의 수석 편집자다. 아래는 {video_duration_sec/60:.0f}분 설교를
**문장 단위**로 정리한 전사본이다(각 줄 = 문장 하나, 앞의 S번호가 문장 번호, [분:초]는 시작 시각).
이번 작업은 딱 하나다: **이 설교의 핵심 문장(명제)을 찾고, 명제마다 쇼츠 컷의 시작·끝 문장 번호를 정하는 것.**
채점·제목·캡션은 다른 단계가 한다. 사고는 전부 "어느 문장이 진짜 알맹이인가"에 써라.

{BENCHMARK_BLOCK}

## 1단계 — 설교의 뼈대와 명제 지도
먼저 이 설교의 주제와 대지(설교자가 나눈 큰 포인트들)를 파악하라. 그 다음 설교자가 **한 문장으로 못 박아 말한
명제**를 전부 찾아라. 찾는 결: "~은 ~입니다 / ~가 아닙니다" 선언, "~해야 됩니다" 단호한 권면, "왜 ~일까요?"
질문과 답, 통념을 뒤집는 말, 찔리는 직격("그거 점쟁이지 뭐예요?", "이단이 다른 사람이 아니에요"), 위로("~해도
괜찮아요", "하나님이 버리셨느냐? 그럴 수 없느니라"), 같은 말이 반복되며 고조되는 크레센도, 예화가 꺾이는 반전의
한마디, 설교자 자신의 고백, 청중이 웃었을 일상 흉내·자학. 25~40분 설교면 보통 8~15개다. 6개 미만이면 못 찾은
것이니 본문 해설 사이에 툭 튀어나온 일상 언어 문장, 후반부 고조 구간, 예화의 마지막 문장을 다시 훑어라.

## 2단계 — 명제마다 컷 잡기 (문장 번호로)
- core: 명제 문장 번호. 클립의 심장. 반드시 start~end 안에 있어야 한다.
- start: 그 명제를 이해하는 데 필요한 **최소 설정**이 시작되는 문장. 공감 상황·질문·명제 자체 중 하나.
  "그래서/이것도 마찬가지로/둘째," 같이 앞 문맥을 전제하는 문장, "오늘 본문은", "○○가 ~했는데" 식 인물·배경 설명으로
  시작하지 마라 — 그 뒤의 첫 힘 있는 문장으로 옮겨라.
- end: 명제가 **착지**하는 가장 힘 있는 문장 — 축복 선언·"~줄로 믿습니다"·아멘 직전 문장·명제 재선언·반전의 한마디.
  그 문장을 읽고 "그래서?"가 떠오르면 미완이다 — 결론이 나온 문장까지 포함하라. 반대로 착지 뒤의 "자, 그러면",
  "다음으로", 부연은 절대 넣지 마라. 예화만 있고 적용이 없는 곳에서 끝내지 마라.
- 길이: 40~60초 목표, 온전한 이야기·크레센도는 90초까지. 설정 없이 명제 한 문장만 뗀 15~20초 조각은 미달
  (크레센도+축복 착지가 붙은 25초부터 허용). 넘치면 끝을 당기지 말고 **앞의 도입·중복 해설을 잘라라.**
- 한 컷 = 명제 하나. 컷끼리 같은 예화·같은 문장을 반복하지 마라(겹치면 더 강한 쪽 하나만).
- appeal: 이 컷이 만드는 감정 — 위로 / 선언 / 뜨끔 / 감동 / 재미. 후보 절반 이상이 같은 appeal이면 나머지를
  놓친 것이니(특히 뜨끔·감동·재미) 찌르는 대목·예화 클라이맥스·웃긴 대목을 다시 찾아라.

## 명제처럼 보이지만 명제가 아닌 것 — 반드시 걸러내라 (실패 실측)
문장이 "~은 ~이다" 형태로 단호해도 **감정·갈등·적용이 없는 용어 정의/분류**는 명제가 아니라 그냥 해설이다.
청중이 "아, 그렇구나" 하고 끝나지 "내 얘기다/찔린다/위로된다"가 안 생기면 후보에서 빼라. 실패 실측(이 규칙이
없어서 실제로 나온 지루한 후보들 — 절대 이런 식으로 뽑지 마라):
- "회사 비전은 비전 아닙니다": '환상'과 '꿈'의 사전적 차이를 설명하는 대목. 갈등도 적용도 없다 → 제외.
- "환상이라고 다 하나님 아닙니다": 가짜 계시를 분별해야 한다는 **같은 주제를 두 번** 뽑음(다른 명제가 이미
  이단·기도원장 얘기로 이 주제를 다뤘다면 겹치는 쪽은 버려라).
- 예화가 있어도 **결말이 "~한 삶을 사는 것이다" 식 해설로 끝나면** 웃기거나 뭉클한 채로 안 끝난다 → 결말을
  감정이 남는 문장(놀람·웃음·뭉클함·찔림)으로 다시 잡거나 후보에서 빼라.
판별식: 이 명제를 시청자에게 그대로 들려줬을 때 "그렇군" 이상의 반응(웃음/뭉클/찔림/위로/도전)이 실제로
있는가? 없으면 아무리 문장이 단호해도 버려라. **명제 지도에 이런 정의·분류 문장이 섞여 있으면 지도에서
아예 빼고 그 자리를 다른 진짜 명제로 채워라** — 개수를 못 채워도 상관없다. 후보 수보다 재미가 먼저다.

## 절대 제외
- 정치·특정 국가/민족/정당/정권/이념/전쟁을 다루거나 미화하는 구간, "역사적·국가적 사건 = 하나님의 직접 개입/섭리"
  비약(예: 소련 대사가 배탈로 회의에 빠져 대한민국이 살았다 → 섭리). 개인의 영적 진리가 아니면 제외.
- 본문 해설·강의만 있고 명제가 없는 것.
{extra}
## 개수
{min_clips}~{max_clips}개. 25분 설교에서 3~4개만 내는 건 대개 명제 지도를 제대로 안 만든 것이지만, 위 "명제처럼
보이지만 아닌 것"을 걸러내고 나니 5~6개뿐이어도 그게 정직한 결과라면 그대로 내라 — **재미없는 해설로 개수를
채우는 것보다 적은 게 낫다.** 강한 순으로 정렬하라(0번째가 가장 강력).

## 문장 전사본
{format_sentences(sentences)}

## 출력 형식
다른 설명 없이 아래 JSON 배열만 출력하라(```json 코드블록). 번호는 위 S번호의 **정수**만.
```json
[
  {{"core": 123, "start": 120, "end": 131, "appeal": "위로",
    "thesis": "명제를 한 줄로(전사본 표현 그대로)",
    "why": "왜 이 대목이 이 설교의 알맹이인지 한 문장"}}
]
```
"""


# ---------------------------------------------------------------------------
# 3. 결정론적 검증·보정
# ---------------------------------------------------------------------------
_STRUCT_START = re.compile(
    r"^(자[,.]?\s*(그러면|그럼|그래서)?|그러면|그럼|다음으로|다음은|첫째|둘째|셋째|넷째|첫\s*번째|두\s*번째|세\s*번째|"
    r"네\s*번째|마지막으로|끝으로)[,\s]"
)
_CONNECTOR_END_WORDS = {
    "그런데", "근데", "그래서", "그러니까", "그니까", "그리고", "그러나", "왜냐하면", "왜냐면", "그러면", "그럼",
    "자", "또", "즉", "다시", "특히", "이제", "그", "저", "이", "어", "음", "에",
}
_CONNECTOR_START = re.compile(
    r"^(그래서|그러니까|그니까|그런데|근데|그리고|그러나|그러면|그럼|그래도|그렇기\s*때문에|왜냐하면|이것도|이런|그런|저런|"
    r"이게|그게|이거|그거|이건|그건|여기서|거기서|또|즉|다시\s*말하면)(\s|,|$)")
_INCOMPLETE_SUFFIX = re.compile(r"(는데|인데|한데|았는데|었는데|지만|라서|어서|아서|니까|으니까|려고|으려고|면서|으면서|고|며|든지|거든요|는데요|면|으면|하면|다가)$")
_STRONG_LANDING = re.compile(r"(축복합니다|축복하십니다|축원합니다|줄로\s*믿습니다|줄\s*믿습니다|믿습니다|바랍니다|바라겠습니다|소망합니다|원합니다|되시기를|되기를|아멘)")
_LANDING_FOLLOW_SEC = 8.0
_SHORT_TARGET_SEC = 30.0       # 이보다 짧은 컷은 앞 설정을 보강
_SHORT_EXTEND_CAP_SEC = 45.0   # 보강해도 이 길이는 넘기지 않음


def _is_incomplete(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t.endswith(","):
        return True
    if t.endswith("?"):
        return True  # 질문으로 끝나면 답이 다음 문장에 있다
    last = re.sub(r"[.!?。]+$", "", t.split()[-1])
    if last in _CONNECTOR_END_WORDS:
        return True
    if not _TERMINAL_PUNCT.search(t) and _INCOMPLETE_SUFFIX.search(last):
        return True
    return False


def _starts_with_structure_marker(text: str) -> bool:
    return bool(_STRUCT_START.match(text.strip()))


def verify_and_fix(
    raw: dict,
    sentences: list[Sentence],
    min_sec: float,
    max_sec: float,
    hard_max_sec: float,
) -> tuple[dict | None, list[str]]:
    """모델의 컷(문장 번호)을 벤치마크 뼈대 기준으로 검증·보정한다.

    반환: (보정된 컷 dict 또는 None(탈락), 적용한 보정 로그)."""
    n = len(sentences)
    log: list[str] = []
    try:
        core = int(raw["core"]); start = int(raw["start"]); end = int(raw["end"])
    except (KeyError, TypeError, ValueError):
        return None, ["번호 파싱 실패"]
    if not (0 <= core < n):
        return None, [f"core S{core} 범위 밖"]
    start = max(0, min(start, n - 1)); end = max(0, min(end, n - 1))
    if end < start:
        start, end = end, start
    # (1) 핵심 문장 포함
    if core < start:
        log.append(f"start S{start}→S{core} (핵심 문장 포함)"); start = core
    if core > end:
        log.append(f"end S{end}→S{core} (핵심 문장 포함)"); end = core

    def dur(a: int, b: int) -> float:
        return sentences[b].end - sentences[a].start

    # (2) 첫 문장이 구조 표지("둘째,", "자, 그러면")면 떼어낸다
    while start < core and _starts_with_structure_marker(sentences[start].text):
        log.append(f"start S{start} 구조 표지 제거 → S{start+1}"); start += 1
    # (3) 끝이 미완(접속형·질문·쉼표)이면 착지까지 확장(최대 4문장, 상한 내)
    steps = 0
    while _is_incomplete(sentences[end].text) and end + 1 < n and steps < 4:
        if dur(start, end + 1) > hard_max_sec:
            break
        log.append(f"end S{end} 미완 → S{end+1}"); end += 1; steps += 1
    # (4) 바로 다음 문장이 축복·믿습니다·아멘 착지면 벤치마크처럼 그것까지 포함
    if end + 1 < n:
        nxt = sentences[end + 1]
        if (
            _STRONG_LANDING.search(nxt.text)
            and not _STRONG_LANDING.search(sentences[end].text)
            and not _starts_with_structure_marker(nxt.text)
            and nxt.start - sentences[end].end <= _LANDING_FOLLOW_SEC
            and dur(start, end + 1) <= hard_max_sec
        ):
            log.append(f"end S{end} → S{end+1} (축복 착지 포함)"); end += 1
    # "아멘." 한 단어 문장이 바로 뒤에 붙어 있으면 포함(청중 아멘 직후가 자연스러운 끝점)
    if end + 1 < n and re.fullmatch(r"(>>\s*)?아멘[.!]?", sentences[end + 1].text.strip()) and dur(start, end + 1) <= hard_max_sec:
        end += 1
    # (5) 길이 상한: 앞에서 자른다(핵심 문장은 유지)
    while dur(start, end) > hard_max_sec and start < core:
        start += 1
    if dur(start, end) > hard_max_sec:
        return None, log + [f"길이 {dur(start,end):.0f}초 > 상한 {hard_max_sec:.0f}초, 핵심 문장 유지 불가 → 탈락"]
    if dur(start, end) > hard_max_sec * 0.999 and start == core:
        pass
    # (6) 짧은 컷은 앞쪽 설정을 보강한다. 벤치마크 최단이 42초인데 모델은 명제 문장 근처만 24~25초로
    #     뚝 떼는 경향이 있다(실측 1GM: "우리를 보실 때 이뻐 죽겠어" 24초). 바로 앞 문장이 축복 착지·
    #     구조 표지·"아멘"(=앞 생각의 끝)이 아니면 30초가 될 때까지(최대 45초) 앞으로 넓힌다.
    #     단, 새 첫 문장이 "그래서/그런데/이런…"처럼 앞을 전제하는 접속어·지시어로 시작하면 거기서
    #     멈추지 않고 더 앞의 깨끗한 문장을 찾되, 못 찾으면 모델이 준 시작을 그대로 둔다
    #     (실측: "하나님이 버리셨느냐?"가 "그래서 우리가 하나님의 마음을 가져야 돼"로 밀린 사고).
    probe = start
    committed = start
    steps = 0
    while dur(committed, end) < _SHORT_TARGET_SEC and probe > 0 and steps < 5:
        prev = sentences[probe - 1]
        if (
            _starts_with_structure_marker(prev.text)
            or _STRONG_LANDING.search(prev.text)
            or re.fullmatch(r"아멘[.!]?", prev.text.strip())
            or dur(probe - 1, end) > _SHORT_EXTEND_CAP_SEC
        ):
            break
        probe -= 1; steps += 1
        if not _CONNECTOR_START.match(prev.text.strip()):
            committed = probe
    if committed != start:
        log.append(f"start S{start} → S{committed} (짧은 컷 앞 설정 보강)"); start = committed
    if dur(start, end) < min_sec * 0.8:
        return None, log + [f"길이 {dur(start,end):.0f}초 < 하한 → 탈락"]
    fixed = dict(raw); fixed.update({"core": core, "start": start, "end": end})
    return fixed, log


def dedupe_cuts(cuts: list[dict], sentences: list[Sentence], overlap_ratio: float = 0.5) -> list[dict]:
    """겹치는 컷은 앞(강한) 것만 남긴다."""
    kept: list[dict] = []
    for c in cuts:
        a0, a1 = sentences[c["start"]].start, sentences[c["end"]].end
        dup = False
        for k in kept:
            b0, b1 = sentences[k["start"]].start, sentences[k["end"]].end
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            if inter / max(1e-6, min(a1 - a0, b1 - b0)) >= overlap_ratio:
                dup = True; break
        if not dup:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# 4. 2차 패스 — 확정된 컷의 실제 대사만 보고 채점·캡션·키워드·제목 초안
# ---------------------------------------------------------------------------
def build_score_prompt(items: list[dict]) -> str:
    blocks = []
    for it in items:
        blocks.append(
            f"### 클립 {it['index']} ({it['duration']:.0f}초, appeal={it['appeal']})\n"
            f"핵심 문장: {it['core_line']}\n대사 전문:\n{it['text']}\n"
        )
    return f"""너는 교회 쇼츠 편집자다. 아래 클립들은 이미 경계가 확정된 쇼츠 후보다(각각 설교자의 실제 대사 전문).
클립마다 세부 축 채점과 게시 정보를 만들어라. **점수는 이 날것 대사 그대로**에 매겨라 — 네가 정리한 줄거리가 아니라
실제로 들리는 말이 스크롤을 멈추는가로.

세부 축(1~10 정수): core_score(설교자가 진짜 힘줘 말한 알맹이인가), hook(첫 문장만 따로 읽고 멈추게 하는가 — 배경
설명·중간을 툭 자른 느낌이면 3 이하), retention(전진감·죽은 구간 없음), emotion(감정 스파이크), relatability("내 얘기"),
payoff(끝이 힘 있게 착지), quotability(스샷 떠 공유할 한 문장). 눈금: 5=쓸 만함, 7=이 설교의 손꼽는 대목,
8=잘 되는 채널 상위 클립 수준, 9~10=채널 1위감(드묾). 정직하게 — 약하면 낮게.

title = 핵심 문장을 벤치마크 스타일로 다듬은 15~20자 구어체 한 줄(예: "기도하는 사람은 오염되지 않습니다").
insight = 이 클립이 시청자 삶의 어떤 문제에 어떻게 닿는가 한 줄. caption = 훅 한 문장. hashtags 3개.
keywords = 이 클립에 실제 등장하는 고유명사(성경 인물·지명·용어) 3~5개를 **올바른 철자**로(자막 정밀 전사의 철자
힌트로 쓴다. 전사본에 틀리게 적혀 있어도 바르게).

{''.join(blocks)}
## 출력 형식
다른 설명 없이 아래 JSON 배열만(```json 코드블록). index는 위 클립 번호.
```json
[
  {{"index": 0, "core_score": 8, "hook": 7, "retention": 7, "emotion": 8, "relatability": 8, "payoff": 8, "quotability": 7,
    "title": "…", "insight": "…", "caption": "…",
    "hashtags": ["#설교","#은혜","#힐링"], "keywords": ["…"]}}
]
```
출력은 짧게 — 글자 수가 곧 대기시간이다.
"""


# ---------------------------------------------------------------------------
# 5. 전체 흐름
# ---------------------------------------------------------------------------
def select_highlights_v2(
    transcript: Transcript,
    min_clips: int,
    max_clips: int,
    min_duration_sec: float,
    max_duration_sec: float,
    hard_max_duration_sec: float,
    model: str = "",
    score_model: str = "sonnet",
    thinking_tokens: int = 8192,
    score_thinking_tokens: int = 1024,
    extra_block: str = "",
    timeout_sec: int = 900,
    on_progress=None,
) -> list[Clip]:
    sentences = build_sentences(transcript)
    if not sentences:
        return []
    print(f"[v2] 문장 재조립: {len(transcript.segments)}조각 → {len(sentences)}문장", flush=True)

    def _prog(lo: float, hi: float):
        def f(frac: float, msg: str) -> None:
            if on_progress:
                on_progress(lo + (hi - lo) * max(0.0, min(1.0, frac)), msg)
        return f

    # --- 1차: 명제 지도 + 컷
    prompt1 = build_thesis_cut_prompt(sentences, transcript.duration_sec, min_clips, max_clips, extra_block)
    raw_cuts = _invoke_claude_json(
        prompt1, model=model, thinking_tokens=thinking_tokens, timeout_sec=timeout_sec,
        on_progress=_prog(0.0, 0.7), max_clips=max_clips,
    )
    print(f"[v2] 1차 패스: 컷 {len(raw_cuts)}개", flush=True)

    # --- 검증·보정
    fixed: list[dict] = []
    for i, rc in enumerate(raw_cuts):
        cut, log = verify_and_fix(rc, sentences, min_duration_sec, max_duration_sec, hard_max_duration_sec)
        if log:
            print(f"[v2] 검증 컷{i}: " + " / ".join(log), flush=True)
        if cut is not None:
            fixed.append(cut)
    fixed = dedupe_cuts(fixed, sentences)
    if not fixed:
        return []

    # --- 2차: 실제 대사 채점
    items = []
    for i, c in enumerate(fixed):
        s, e = sentences[c["start"]], sentences[c["end"]]
        text = " ".join(x.text for x in sentences[c["start"]: c["end"] + 1])
        items.append({
            "index": i, "duration": e.end - s.start, "appeal": c.get("appeal", ""),
            "core_line": sentences[c["core"]].text, "text": text,
        })
    if on_progress:
        on_progress(0.72, "확정된 구간을 채점하는 중...")
    scored: dict[int, dict] = {}
    try:
        raw_scores = _invoke_claude_json(
            build_score_prompt(items), model=score_model, thinking_tokens=score_thinking_tokens,
            timeout_sec=600, on_progress=_prog(0.72, 0.98), max_clips=len(items),
        )
        for r in raw_scores:
            try:
                scored[int(r.get("index"))] = r
            except (TypeError, ValueError):
                continue
    except Exception as exc:  # noqa: BLE001 - 채점 실패해도 컷은 살린다
        print(f"[v2] 2차 채점 실패(컷은 유지, 기본 점수): {exc}", flush=True)

    clips: list[Clip] = []
    for i, c in enumerate(fixed):
        s, e = sentences[c["start"]], sentences[c["end"]]
        r = scored.get(i, {})
        core_line = sentences[c["core"]].text
        computed = compute_scores(r if r else {"core_score": 6, "hook": 6, "retention": 6, "emotion": 6,
                                               "relatability": 6, "payoff": 6, "quotability": 6})

        def _f(k: str):
            try:
                return float(r[k]) if r.get(k) is not None else None
            except (TypeError, ValueError):
                return None

        clips.append(Clip(
            start=s.start, end=e.end,
            title=str(r.get("title") or c.get("thesis") or core_line).strip()[:40],
            caption=str(r.get("caption", "")).strip(),
            hashtags=[str(h) for h in (r.get("hashtags") or ["#설교", "#은혜", "#말씀"])],
            reason=str(c.get("why", "")).strip(),
            score=computed["score"], core_score=computed["core_score"], viral_score=computed["viral_score"],
            hook_score=_f("hook"), retention_score=_f("retention"), emotion_score=_f("emotion"),
            relatability_score=_f("relatability"), payoff_score=_f("payoff"), quotability_score=_f("quotability"),
            appeal=str(c.get("appeal", "")).strip(),
            hook_line=s.text, payoff_line=e.text, core_line=core_line,
            insight=str(r.get("insight", "")).strip(),
            anchored=True,  # 경계가 문장 시각으로 확정됨 — 렌더의 끝 스냅은 미세조정만
            title_candidates=[str(t).strip() for t in (r.get("title_candidates") or []) if str(t).strip()],
            keywords=[str(k).strip() for k in (r.get("keywords") or []) if str(k).strip()],
        ))
    return clips
