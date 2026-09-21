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
from dataclasses import dataclass, field

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
    words: list = field(default_factory=list)  # [(start, end, text)] — 첫 단어 말버릇 트림용
    reaction: str = ""  # 이 문장 중/직후의 청중 반응("웃음", "박수") — 자동자막의 [웃음] 토큰에서


_NOISE_TOKEN = re.compile(r"^\[[^\]]*\]$|^>>$|^>>\S*$")  # [노래] [음악] [한숨] >> 등
_REACTION_TOKEN = re.compile(r"^\[(웃음|박수|환호|웃음소리|박수 소리)\]$")
_TERMINAL_PUNCT = re.compile(r"[.?!。]$")
# 마침표 없이도 문장이 끝났다고 볼 수 있는 종결어미(단어 끝). 자동자막은 마침표가 대체로 있지만
# 빠지는 곳이 있어, 긴 문장(12초/25단어 초과)에서만 이 규칙으로 보조 분할한다.
_FINAL_ENDING = re.compile(r"(습니다|ㅂ니다|입니다|니다|세요|어요|아요|해요|예요|이에요|거예요|네요|죠|지요|까|나요|잖아요|랍니다|습니까|십시오)$")
_MAX_SENT_SEC = 12.0
_MAX_SENT_WORDS = 25
_GAP_SPLIT_SEC = 1.2
# 자동자막/whisper 단어 토큰에 마침표가 '가운데' 박힌 경우("돼.이", "아멘.에", "없습니다.에 에").
# 2026-09-18 실측(1GM): 이런 토큰이 14개인데 문장 분할이 토큰 '끝'의 구두점만 보니까 앞 문장과
# 뒤 문장이 한 문장으로 붙었다. 붙은 문장이 컷의 시작이 되면 "그래서 우리가 하나님의 마음을
# 가져야 돼. 이 지역에도 많은 영혼들이…"처럼 접속어로 시작하는 훅이 나온다(hook 3점 원인).
_GLUED_TOKEN = re.compile(r"^(.*?[.?!。])(\S+)$")


def _split_glued_token(start: float, end: float, text: str) -> list[tuple[float, float, str]]:
    """'돼.이' 같은 토큰을 '돼.' + '이'로 나누고 시각은 글자 수 비례로 배분한다."""
    m = _GLUED_TOKEN.match(text)
    if not m:
        return [(start, end, text)]
    head, tail = m.group(1), m.group(2)
    if not tail.strip() or len(head) < 2:
        return [(start, end, text)]
    if head[-2].isdigit() and tail[0].isdigit():  # "10.5" 같은 소수는 그대로
        return [(start, end, text)]
    total = max(1, len(head) + len(tail))
    mid = start + (end - start) * (len(head) / total)
    return [(start, mid, head)] + _split_glued_token(mid, end, tail)


def build_sentences(transcript: Transcript) -> list[Sentence]:
    """단어 시각으로 전사본을 문장 단위로 재조립한다.

    분할 규칙(우선순위): 문장 부호(. ? !) → 단어 사이 1.2초 이상 침묵 → 너무 길면 종결어미.
    단어 시각이 없는 세그먼트는 세그먼트 텍스트를 한 덩어리로 쓴다."""
    words: list[tuple[float, float, str]] = []
    reactions: list[tuple[float, str]] = []
    for seg in transcript.segments:
        if seg.words:
            for w in seg.words:
                t = (w.text or "").replace(">>", "").strip()
                if not t:
                    continue
                if _NOISE_TOKEN.match(t):
                    # 청중 반응([웃음]·[박수])은 잡음이 아니라 '장면이 먹혔다'는 실측 신호다 — 시각을 기억해 두었다가
                    # 그 문장에 표식으로 붙인다(벤치마크 상위 클립 자막에도 [웃음]이 찍혀 있음). [음악]·[노래]·[한숨]은 버린다.
                    if _REACTION_TOKEN.match(t):
                        reactions.append((float(w.start), t.strip("[]")))
                    continue
                words.extend(_split_glued_token(float(w.start), float(w.end), t))
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
            sentences.append(Sentence(len(sentences), cur[0][0], max(w[1] for w in cur), text, list(cur)))
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
    # 청중 반응을 그 시각을 품은(또는 직전) 문장에 표식으로 붙인다(텍스트는 건드리지 않음 — 형식 출력에서만 보임).
    for t, label in reactions:
        target = None
        for sent in sentences:
            if sent.start <= t:
                target = sent
            else:
                break
        if target is not None:
            target.reaction = label
    return sentences


# 첫 문장 맨 앞의 말버릇·추임새 단어("그래서", "그니까", "근데", "어", "자", "예"). 문장이 최소 단위라
# "그래서 베드로가 … 쿠오바디스 영화의 한 장면에 보면 …"처럼 설정이 든 긴 문장은 통째로 버릴 수 없다 →
# 단어 시각을 이용해 그 단어(0.2~0.5초)만 잘라내고 다음 단어부터 클립을 시작한다(2026-09-18, 실측 1GM).
_LEAD_TRIM_WORDS = {
    "그래서", "그러니까", "그니까", "그런데", "근데", "그리고", "그러나", "그러면", "그럼", "그래도", "그래가지고",
    "자", "어", "응", "음", "에", "아", "예", "네", "그", "저", "이제", "또", "뭐", "인제",
}
_LEAD_TRIM_MAX_WORDS = 2


def trim_lead_words(sent: Sentence) -> tuple[float, str]:
    """문장 앞의 말버릇 단어를 떼고 (실제 시작 시각, 남은 텍스트)를 돌려준다. 못 떼면 원래 값."""
    ws = list(sent.words)
    if len(ws) < 4:
        return sent.start, sent.text
    k = 0
    while k < _LEAD_TRIM_MAX_WORDS and k < len(ws) - 3 and re.sub(r"[.,!?]+$", "", ws[k][2]) in _LEAD_TRIM_WORDS:
        k += 1
    if k == 0:
        return sent.start, sent.text
    return ws[k][0], " ".join(w[2] for w in ws[k:]).strip()


def _fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def format_sentences(sentences: list[Sentence]) -> str:
    return "\n".join(
        f"S{s.idx} [{_fmt_ts(s.start)}] {s.text}" + (f"  ← [청중 {s.reaction}]" if s.reaction else "")
        for s in sentences
    )


# ---------------------------------------------------------------------------
# 2. 1차 패스 — '쇼츠 요소' 지도 + 컷 (문장 번호로만 경계 지정, 점프컷 포함)
# ---------------------------------------------------------------------------
BENCHMARK_BLOCK = """## 벤치마크 — 잘 되는 교회 쇼츠 채널의 조회수 상위 클립 실측
상위 클립의 공통 뼈대: **구체(장면·실화·대사·숫자·질문)로 시작** → 설교자가 한 문장으로 못 박은 **명제가 착지**
(축복·"~줄로 믿습니다"·아멘 직전·명제 재선언·반전의 한마디)에서 딱 끊는다. 실제 예(조회수·길이):
- 「담임목사의 군생활 꿀팁」(1만, 64초, 재미): 이등병 때 "야 김하나, 에프킬라 뿌려" 재연 → "필수 요소는 소리가 아닙니다"
  → 착지 "신의 필수 조건은 하나님의 능력입니다."
- 「목사님의 해외 출국 전 장바구니」(6.9천, 76초, 뜨끔): "저도 해외를 가게 되니까 자꾸 뭘 찾게 되냐면요" → 목베개 7개
  자학 유머 → 착지 "우리는 다 갖고 있어도 뭔가를 더 원합니다 … 하나님께서 이것을 원하시느냐."
- 「엄마는 교회 가자 vs 안 가겠다는 아들」(6.2천, 68초, 뜨끔): 소파 대사 재연 → 착지 "사람은 그렇게 쉽게 변화되지 않습니다."
- 「딱 맞아떨어지면 무조건 하나님의 뜻일까?」(채널 1위, 153초, 교훈): 목사 간증 단골 레퍼토리(쌀독 긁는 소리) 재연 →
  반전 "딱 맞아떨어진다고 다 하나님의 뜻은 아닙니다" → 끝 "성도의 기준은 상황이 아닙니다." 이야기가 온전하면 길어도 터진다.
- 「아무것도 보이지 않을 때」(69초, 위로): "내가 지금 어려운 상황인데 내 옆에 도와줄 사람이 하나도 없고…" → 요나 이야기를
  압축 → 끝 "지금 보이지 않는다고 해서 없다는 말이 아닙니다 … 반드시 있을 줄로 믿습니다."
- 「교회 사는 딸에게 부모가 하는 잔소리」(4.5천, 감동 간증): "새벽에 들어가도 엄마 아빠는 주무시고 계셨어요 … '네가 가봤자
  교회고 만나봤자 교회 친구인데'."
- 「규칙보다 구원이 먼저입니다」(59초, 교훈): "규칙이 신앙을 지배하면 안 됩니다. 은혜가 우리를 붙잡아야 합니다." → 십계명은
  구원 뒤에 주신 것 → 끝 "좀 부족해도 연약해도 완벽하지 않아도 … 계명을 주신 줄로 믿습니다."
**아래(1천~2천)**: "참된 자유는 그리스도만 주십니다", "우리가 먼저 하나님을 사랑한 게 아닙니다" — 문장은 옳고 단호하지만
구체가 없고 시청자 자신의 상황이 안 떠오르는 **선언만 있는 클립**.
공통점: (1) 시청자는 **성경을 모르는 일반 대중까지** 포함한다 — 위 상위 클립은 전부 성경 지식 없이도 자기 얘기로
들린다(군생활·장바구니·엄마와 아들·부모 잔소리). 성경 인물 이야기(베드로가 어디로 갔다, 요나가 도망쳤다)가 주된
내용인 클립은 교인 밖에서는 공감을 못 얻는다. 첫 문장은 인물 소개·본문 설명·연도·지명 나열이 아니라 **상황/대사/질문/직격**이다. (2) 한 클립 = 한 감정·한 명제. (3) 끝은 가장
힘 있는 문장에서 딱 끊는다. (4) 40~60초가 주류, 온전한 이야기면 90초까지 괜찮다. (5) 위 상위 클립은 재미만이 아니다 —
뜨끔·교훈·위로·감동이 골고루 있고, 공통점은 **구체가 있고 시청자가 자기 얘기로 느낀다**는 것이다."""


def build_thesis_cut_prompt(
    sentences: list[Sentence],
    video_duration_sec: float,
    min_clips: int,
    max_clips: int,
    extra_block: str = "",
    hard_max_sec: float = 90.0,
    max_span_sec: float = 180.0,
) -> str:
    extra = f"\n{extra_block}\n" if extra_block else ""
    return f"""너는 조회수가 잘 나오는 교회 쇼츠 채널의 수석 편집자다. 아래는 {video_duration_sec/60:.0f}분 설교를
**문장 단위**로 정리한 전사본이다(각 줄 = 문장 하나, 앞의 S번호가 문장 번호, [분:초]는 시작 시각).
이번 작업은 딱 하나다: **이 설교에서 쇼츠로 떴을 때 사람이 멈춰 보게 되는 '쇼츠 요소'가 있는 대목을 전부 찾고, 대목마다
컷(시작·끝 문장 번호, 필요하면 중간에 들어낼 문장)을 정하는 것.** 채점·제목·캡션은 다른 단계가 한다.
사고는 전부 "어느 대목이 시청자에게 실제 반응(웃음·뭉클·찔림·위로·'아, 그래서였구나')을 일으키는가"에 써라.

{BENCHMARK_BLOCK}

## 1단계 — 쇼츠 요소 지도
설교 전체를 읽고 아래 다섯 요소가 있는 대목을 **빠짐없이** 나열하라. 다섯 요소는 동등하다 — 재미만 찾지 마라.
- **재미**: 유머·자학·흉내·대사 재연(전사에 [웃음]이 있으면 그 직전 문장들이 핵심).
- **감동**: 실화·간증·희생·눈물겨운 헌신, 구체 숫자·고유명사·대사가 있는 이야기("식권밖에 없어요", "보증금 가져왔어요").
- **뜨끔**: 시청자 자신의 삶을 정면으로 찌르는 직격·질문("인생의 연조가 저절로 깊어지는 건 아니지 않습니까?").
- **교훈(통찰)**: 뻔한 권면이 아니라 **관점을 뒤집는 한마디**("딱 맞아떨어진다고 다 하나님의 뜻은 아닙니다", "교회를
  다시 지어라 → 알고 보니 지역 사람들에게 물어보라"). 판별: 듣고 나서 "아 그렇구나"가 아니라 "아, 그래서였구나 / 내가
  거꾸로 알고 있었네"가 생기는가. 교회 안 다니는 사람이 들어도 자기 삶에 적용되는 **일반적인 교훈**이 가장 좋다.
- **위로**: 지금 힘든 사람에게 그대로 들려주고 싶은 한 대목("지금 보이지 않는다고 없는 게 아닙니다").
요소마다 그 대목이 **착지하는 명제 문장**(설교자가 한 문장으로 못 박은 결론, 없으면 펀치라인)을 짝지어라.
장면(재연·실화)이 없어도 명제가 위 판별을 통과하면 후보다. 반대로 문장이 단호해도 **감정·갈등·적용이 없는 용어
정의/분류/본문 해설**("환상과 꿈의 차이는…")은 요소가 아니다 — 시청자가 "그렇군" 하고 넘긴다.
25~40분 설교면 보통 6~12개다. 4개 미만이면 못 찾은 것이니 본문 해설 사이에 툭 튀어나온 일상 언어·숫자·대사·질문·
반전을 다시 훑어라. 한 대목에 요소가 둘 이상이면(감동+교훈) 더 강한 쪽을 appeal로 적어라.

## 2단계 — 대목마다 컷 잡기 (문장 번호로)
- core: 착지하는 명제 문장 번호(없으면 펀치라인). 반드시 start~end 안에 있어야 한다.
- start = **훅 문장**: 시청자가 첫 3초에 듣는 문장이다. 상황 제시·대사의 첫 줄·질문·직격·반전 예고 중 하나여야 한다.
  **훅으로 금지**: 연도·날짜·지명·인명 나열("2006년 12월에 성전 부지를 매입하고…", "복음을 들고 태평양을 건너 대서양을
  건너…"), 인물·배경 소개("○○가 ~했는데"), "오늘 본문은", 30단어 넘는 긴 문장, 앞 문맥을 전제하는 "그래서/이것도/둘째,"
  시작. 그런 문장이 설정에 필요해 보여도 **그 뒤의 첫 구체 문장으로 옮겨라** — 예: "2006년 12월에…" 대신 바로 뒤의
  "전도하면 될 거라고 생각했어요."가 훅이다. 설정이 조금 빠져도 훅이 사는 쪽이 낫다.
- end: 명제가 **착지**하는 가장 힘 있는 문장 — 축복 선언·"~줄로 믿습니다"·아멘 직전 문장·명제 재선언·반전의 한마디.
  그 문장을 읽고 "그래서?"가 떠오르면 미완이다 — 결론이 나온 문장까지 포함하라. 착지 뒤의 "자, 그러면", "다음으로",
  부연은 절대 넣지 마라. 예화만 있고 적용이 없는 곳에서 끝내지 마라.
- **skip(점프컷)**: 온전한 이야기가 {hard_max_sec:.0f}초를 넘기면 앞이나 뒤를 잘라 반토막 내지 말고, **중간에서 빼도 흐름이
  안 깨지는 문장들**을 skip으로 지정해 들어내라. 빼도 되는 것: 같은 말 반복, 곁길·부연 설명, 슬라이드 넘기기("다음
  넘겨 보시죠"), 수치·연도 나열, 추임새 문장. 빼면 안 되는 것: 뒤 문장이 가리키는 내용("그 사람이…"의 '그 사람'이
  나온 문장), 반전의 전제, 대사의 앞뒤. **자연스러움이 최우선**: skip 앞 문장은 말이 끝난 문장이어야 하고, skip 뒤
  문장은 "그래서/그런데/그러니까"로 시작하지 않아야 한다. 각 skip은 [첫 문장, 끝 문장] 번호 쌍이고 start·end·core는
  뺄 수 없다. 들어내고 남는 길이가 {hard_max_sec:.0f}초 이내면 원본 구간은 {max_span_sec:.0f}초까지 잡아도 된다.
  {hard_max_sec:.0f}초 이내 컷에도 죽은 구간(슬라이드 설명·반복)이 있으면 skip으로 빼라 — 단 빼는 게 억지스러우면 안 빼는 게 낫다.
- 길이(skip 제외 후): 40~60초 목표, 온전한 이야기·크레센도는 {hard_max_sec:.0f}초까지. 설정 없이 명제 한 문장만 뗀 15~20초 조각은
  미달(크레센도+축복 착지가 붙은 25초부터 허용).
- 한 컷 = 한 명제. 컷끼리 같은 예화·같은 문장을 반복하지 마라(겹치면 더 강한 쪽 하나만). 단, 이 규칙은 "다른 주제를
  섞지 마라"는 뜻이지 **이어지는 한 이야기를 두 토막으로 내라는 뜻이 아니다.** 설정 → 전환 → 착지가 연달아 이어지면
  그건 컷 하나다(길면 skip으로 줄여라). 반으로 쪼개면 앞토막은 착지가 없고 뒤토막은 설정이 없어 둘 다 죽는다(실측).
- appeal: 재미 / 감동 / 뜨끔 / 교훈 / 위로 중 하나. 후보 절반 이상이 같은 appeal이면 나머지 요소를 놓친 것이니 다시 찾아라.
- scene: 이 컷의 내용을 한 줄로(구체가 드러나게. 예: "밥 식권을 헌금함에 넣은 청년, 목사가 아직 책상에 보관 중").

## 절대 제외
- 정치·특정 국가/민족/정당/정권/이념/전쟁을 다루거나 미화하는 구간, "역사적·국가적 사건 = 하나님의 직접 개입/섭리"
  비약(예: 소련 대사가 배탈로 회의에 빠져 대한민국이 살았다 → 섭리). 개인의 영적 진리가 아니면 제외.
- **남을 비판·경계·폭로하는 대목**: 이단·사이비·타종교·타교단·다른 목회자/기도원/특정 집단·직군을 두고 "저들은
  가짜다/조심하라/그 유래는 이렇다"고 말하는 구간 전부(예: 이단 교주의 창시 일화, "○○ 사람들 조심하세요",
  "그거 점쟁이지 뭐예요"). 이야기가 아무리 구체적이고 흥미로워도 시청자에게 남는 감정이 '남 욕·경계'라 공유되지
  않고 채널이 논쟁에 끌려 들어간다(실측 실패). 경고형 명제("~조심해야 됩니다", "~은 가짜다")가 착지인 컷도 제외.
- **성경 인물 이야기가 주된 내용인 컷**(베드로·요나·모세·다윗·요셉·바울 등 인물의 행적을 따라가는 대목, "○○가
  어디로 가서 무엇을 했다"): 성경을 모르는 시청자에겐 남의 옛날이야기라 공감이 안 생긴다(사용자 지시). 성경 구절·
  인물이 **한두 문장 근거로만** 스쳐 가고 클립이 그것 없이도 서면 괜찮지만, 인물 이야기를 들어내면 클립이 무너지면
  제외다. 대신 그 대목이 착지하는 **일반적인 삶의 교훈**만 따로 서는지 보고, 서면 그 부분만 컷으로 잡아라.
- 본문 해설·강의만 있고 요소가 없는 것.
{extra}
## 개수
{min_clips}~{max_clips}개. 위 다섯 요소를 다 훑고도 5~6개뿐이면 그게 정직한 결과다 — 해설로 개수를 채우지 마라.
강한 순으로 정렬하라(0번째가 가장 강력).

## 문장 전사본
{format_sentences(sentences)}

## 출력 형식
다른 설명 없이 아래 JSON 배열만 출력하라(```json 코드블록). 번호는 위 S번호의 **정수**만. skip은 없으면 빈 배열.
```json
[
  {{"core": 123, "start": 118, "end": 131, "skip": [[121, 122], [127, 127]], "appeal": "감동",
    "scene": "내용 한 줄",
    "thesis": "착지 명제를 한 줄로(전사본 표현 그대로)",
    "why": "왜 이 대목에서 사람이 멈추는지 한 문장"}}
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
    r"이게|그게|이거|그거|이건|그건|여기서|거기서|또|즉|다시\s*말하면)(이|그|저)?(\s|,|$)")
_INCOMPLETE_SUFFIX = re.compile(r"(는데|인데|한데|았는데|었는데|지만|라서|어서|아서|니까|으니까|려고|으려고|면서|으면서|고|며|든지|거든요|는데요|면|으면|하면|다가)$")
_STRONG_LANDING = re.compile(r"(축복합니다|축복하십니다|축원합니다|줄로\s*믿습니다|줄\s*믿습니다|믿습니다|바랍니다|바라겠습니다|소망합니다|원합니다|되시기를|되기를|아멘)")
_LANDING_FOLLOW_SEC = 8.0
_SHORT_TARGET_SEC = 30.0       # 이보다 짧은 컷은 앞 설정을 보강
_SHORT_EXTEND_CAP_SEC = 45.0   # 보강해도 이 길이는 넘기지 않음
# 추임새만 있는 문장("예.", "응.", "어.", "아멘.", "할렐루야.") — 컷의 첫 문장으로는 죽은 1~2초.
_FILLER_ONLY = re.compile(r"^(>>\s*)?(예|네|응|어|음|에|아|자|그|아멘|할렐루야|그렇죠|그죠)[.!?,]*$")
# "에 에 환상과…", "어 그런데…". 단 '예/네'는 관형사·일상어로도 쓰여("네 이웃을 사랑하라")
# 맨 앞 한 번만으로는 추임새로 보지 않는다 — 문장부호가 붙거나 추임새가 이어질 때만 인정.
_FILLER_PREFIX = re.compile(
    r"^(>>\s*)?("
    r"(예|네|응|어|음|에|아)[.,!]\s*"
    r"|(응|어|음|에|아)\s+"
    r"|(예|네)\s+(?=(예|네|응|어|음|에|아)[\s.,!])"
    r")+"
)
_CONNECTOR_LOOKBACK = 3        # 접속어 시작을 고칠 때 뒤로 살펴볼 문장 수
_CONNECTOR_LOOKBACK_SEC = 20.0 # 뒤로 넓혀도 이만큼까지만
_TRANSITION_MAX_WORDS = 8      # 이보다 긴 접속어 문장은 '전환 추임새'가 아니라 내용 문장 → 앞으로 옮겨 버리지 않는다
_MERGE_GAP_SEC = 20.0          # 이 간격 이내로 붙은 두 컷은(사이에 대지 전환·착지가 없으면) 한 흐름으로 본다


# 훅으로 죽는 문장(2026-09-21 실측 zvsOYhjdmks): "2006년도 12월에 이곳에 성전 부지를 매입하고…"(연도 시작),
# "복음을 들고 태평양을 건너 대서양을 건너 시베리아 횡단 철도를 타고…"(29초짜리 한 문장). 접속어 교정(7)이
# "그리고 한 2년 동안…"을 고치려다 한 문장 앞의 연도 문장으로 옮겼다 — '접속어만 아니면 깨끗하다'고 봤기 때문.
_WEAK_HOOK_YEAR = re.compile(r"^\D{0,4}\d{4}\s*년")
_WEAK_HOOK_MAX_WORDS = 30
_WEAK_HOOK_MIN_DIGIT_GROUPS = 3
_WEAK_HOOK_LOOKAHEAD = 3  # 약한 훅을 고칠 때 앞(미래)으로 살펴볼 문장 수


def _is_weak_hook(text: str) -> bool:
    """첫 3초에 들리면 스크롤을 못 멈추는 문장: 연도로 시작, 숫자 나열, 너무 긴 문장."""
    t = text.strip()
    if not t:
        return True
    if len(t.split()) > _WEAK_HOOK_MAX_WORDS:
        return True  # 29초짜리 한 문장은 끝이 물음표여도 첫 3초가 죽는다(실측 "복음을 들고 태평양을 건너…")
    if t.endswith("?"):
        return False  # 짧은 질문은 훅이 된다
    if _WEAK_HOOK_YEAR.match(t):
        return True
    if len(re.findall(r"\d+", t)) >= _WEAK_HOOK_MIN_DIGIT_GROUPS:
        return True
    return False


def _is_dirty_start(text: str) -> bool:
    """첫 문장으로 두면 '중간을 툭 자른' 훅이 되는 문장(접속어·지시어·구조 표지·추임새 시작)."""
    t = text.strip()
    if not t or _FILLER_ONLY.match(t) or _FILLER_PREFIX.match(t):
        return True
    return bool(_CONNECTOR_START.match(t) or _starts_with_structure_marker(t))


def _is_clean_start(text: str) -> bool:
    """시작을 옮길 때 '후보'로 삼아도 되는 문장인가 — 더럽지 않고, 착지도 아니고, 조각도 아니고, 약한 훅도 아님.
    (현재 시작이 고칠 대상인지는 _is_dirty_start/_is_weak_hook로 본다: "놀라운 신분이에요." 같은 짧은 선언은
    시작으로 이미 괜찮으므로 건드리지 않는다.)"""
    t = text.strip()
    if _is_dirty_start(t) or _is_weak_hook(t):
        return False
    if _STRONG_LANDING.search(t) or re.fullmatch(r"(>>\s*)?아멘[.!]?", t):
        return False
    # "피신하십시오." "권유를 합니다." 같은 1~2단어 조각은 시작으로 세울 수 없다(질문은 짧아도 훅이 된다).
    if len(t.split()) < 3 and not t.endswith("?"):
        return False
    return True


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


# ---- 점프컷(skip) ----------------------------------------------------------
# 2026-09-21 사용자: "너무 긴 부분은 과감하게 자르고 붙여도 된다, 자연스럽기만 하다면." 그전까지 컷은 연속 구간
# 하나뿐이라 3분짜리 이야기(교회 땅 이야기)는 앞을 잘라 반전의 전제를 잃거나 통째로 탈락했다. 이제 모델이
# "빼도 흐름이 안 깨지는 문장" 구간(skip)을 지정하고, 여기서 자연스러움을 검증한 뒤 렌더의 keep_ranges로 넘긴다.
def _parse_skips(raw_skip, n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in raw_skip or []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if a > b:
            a, b = b, a
        a = max(0, a); b = min(n - 1, b)
        if a <= b:
            out.append((a, b))
    return _merge_ranges(out)


def _merge_ranges(rs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rs = sorted(rs)
    out: list[tuple[int, int]] = []
    for a, b in rs:
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _clip_skips(skips: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    """start·end 문장은 뺄 수 없다 → skip을 (start, end) 안쪽으로 자른다."""
    out = []
    for a, b in skips:
        a2, b2 = max(a, start + 1), min(b, end - 1)
        if a2 <= b2:
            out.append((a2, b2))
    return out


def _sanitize_skips(
    skips: list[tuple[int, int]], sentences: list[Sentence], start: int, end: int, core: int, log: list[str],
) -> list[tuple[int, int]]:
    """자연스럽지 않은 skip은 버린다: 핵심 문장을 품음 / 앞 문장이 말이 안 끝남 / 뒤 문장이 접속어로 시작."""
    out: list[tuple[int, int]] = []
    for a, b in _clip_skips(skips, start, end):
        if a <= core <= b:
            log.append(f"skip S{a}~S{b} 핵심 문장 포함 → 무시"); continue
        before, after = sentences[a - 1], sentences[b + 1]
        if _is_incomplete(before.text) and not before.text.strip().endswith("?"):
            log.append(f"skip S{a}~S{b} 앞 문장 미완('{before.text[-12:]}') → 무시"); continue
        if _CONNECTOR_START.match(after.text.strip()) or _FILLER_PREFIX.match(after.text.strip()):
            log.append(f"skip S{a}~S{b} 뒤 문장 접속어 시작('{after.text[:12]}') → 무시"); continue
        out.append((a, b))
    return _merge_ranges(out)


def kept_runs(start: int, end: int, skips: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """start~end에서 skip을 뺀 '남는 문장 구간'(inclusive) 목록."""
    runs: list[tuple[int, int]] = []
    cur = start
    for a, b in _clip_skips(skips, start, end):
        if a > cur:
            runs.append((cur, a - 1))
        cur = b + 1
    if cur <= end:
        runs.append((cur, end))
    return runs


def _eff_dur(sentences: list[Sentence], start: int, end: int, skips: list[tuple[int, int]]) -> float:
    """skip을 들어낸 뒤 실제 남는 길이(초)."""
    return sum(sentences[b].end - sentences[a].start for a, b in kept_runs(start, end, skips))


def verify_and_fix(
    raw: dict,
    sentences: list[Sentence],
    min_sec: float,
    max_sec: float,
    hard_max_sec: float,
    max_span_sec: float | None = None,
) -> tuple[dict | None, list[str]]:
    """모델의 컷(문장 번호)을 벤치마크 뼈대 기준으로 검증·보정한다.

    길이 규칙은 전부 skip(점프컷)을 들어낸 뒤의 '실제 남는 길이'로 재고, 원본 구간(start~end)은 max_span_sec까지 허용.
    반환: (보정된 컷 dict 또는 None(탈락), 적용한 보정 로그). 보정된 컷의 skip은 검증을 통과한 것만 남는다."""
    n = len(sentences)
    log: list[str] = []
    if max_span_sec is None:
        max_span_sec = hard_max_sec * 2
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
    # (0) 점프컷: 자연스럽지 않은 skip은 여기서 미리 버려 길이 계산에 끼지 않게 한다
    skips = _sanitize_skips(_parse_skips(raw.get("skip"), n), sentences, start, end, core, log)
    if skips:
        log.append("skip " + ", ".join(f"S{a}~S{b}" for a, b in skips))

    def span(a: int, b: int) -> float:
        return sentences[b].end - sentences[a].start

    def dur(a: int, b: int) -> float:
        return _eff_dur(sentences, a, b, skips)

    # (2) 첫 문장이 구조 표지("둘째,", "자, 그러면")·추임새("예.", "응.")면 떼어낸다
    while start < core and (
        _starts_with_structure_marker(sentences[start].text) or _FILLER_ONLY.match(sentences[start].text.strip())
    ):
        log.append(f"start S{start} 구조 표지/추임새 제거 → S{start+1}"); start += 1
    # 끝이 "예."/"응." 같은 추임새면 뗀다("아멘."은 착지라 유지)
    while end > core and _FILLER_ONLY.match(sentences[end].text.strip()) and not re.search(r"아멘", sentences[end].text):
        log.append(f"end S{end} 추임새 제거 → S{end-1}"); end -= 1
    # (3) 끝이 미완(접속형·질문·쉼표)이면 착지까지 확장(최대 4문장, 상한 내)
    steps = 0
    while _is_incomplete(sentences[end].text) and end + 1 < n and steps < 4:
        if dur(start, end + 1) > hard_max_sec or span(start, end + 1) > max_span_sec:
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
            and dur(start, end + 1) <= hard_max_sec and span(start, end + 1) <= max_span_sec
        ):
            log.append(f"end S{end} → S{end+1} (축복 착지 포함)"); end += 1
    # "아멘." 한 단어 문장이 바로 뒤에 붙어 있으면 포함(청중 아멘 직후가 자연스러운 끝점)
    if (
        end + 1 < n and re.fullmatch(r"(>>\s*)?아멘[.!]?", sentences[end + 1].text.strip())
        and dur(start, end + 1) <= hard_max_sec and span(start, end + 1) <= max_span_sec
    ):
        end += 1
    # (5) 길이 상한: 앞에서 자른다(핵심 문장은 유지). 남는 길이(skip 제외)와 원본 구간 둘 다 본다.
    while (dur(start, end) > hard_max_sec or span(start, end) > max_span_sec) and start < core:
        start += 1
    if dur(start, end) > hard_max_sec:
        return None, log + [f"길이 {dur(start,end):.0f}초 > 상한 {hard_max_sec:.0f}초, 핵심 문장 유지 불가 → 탈락"]
    if span(start, end) > max_span_sec:
        return None, log + [f"원본 구간 {span(start,end):.0f}초 > 상한 {max_span_sec:.0f}초 → 탈락"]
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
        if not _CONNECTOR_START.match(prev.text.strip()) and not _is_weak_hook(prev.text):
            committed = probe
    if committed != start:
        log.append(f"start S{start} → S{committed} (짧은 컷 앞 설정 보강)"); start = committed
    # (7) 첫 문장이 접속어·지시어("그래서/그니까/그러나/이런…")로 시작하면 앞 문맥을 전제하는
    #     '중간을 툭 자른' 훅이다(2026-09-18 실측 1GM: 6개 중 3개가 이렇게 시작, hook 3~4점).
    #     모델에게 하지 말라고 써 놔도 실행마다 다르게 나오므로 여기서 결정론적으로 고친다:
    #     먼저 뒤로 최대 3문장(20초) 안에서 깨끗한 문장을 찾되 축복 착지·아멘·구조 표지(=앞 생각의
    #     끝/새 대지)를 넘어가진 않는다. 못 찾으면 앞으로(핵심 문장까지) 첫 깨끗한 문장으로 옮긴다.
    if _is_dirty_start(sentences[start].text):
        moved = None
        nearest_clean = None
        for k in range(1, _CONNECTOR_LOOKBACK + 1):
            j = start - k
            if j < 0:
                break
            sj = sentences[j]
            if (
                _starts_with_structure_marker(sj.text)
                or _STRONG_LANDING.search(sj.text)
                or re.fullmatch(r"(>>\s*)?아멘[.!]?", sj.text.strip())
                or sentences[start].start - sj.start > _CONNECTOR_LOOKBACK_SEC
                or dur(j, end) > hard_max_sec or span(j, end) > max_span_sec
            ):
                break
            if _is_clean_start(sj.text):
                if nearest_clean is None:
                    nearest_clean = j
                # 질문 문장("베드로가 어디 가서 순교합니까?")은 벤치마크형 훅이라 창 안에 있으면 그쪽을 택한다
                if sj.text.strip().endswith("?"):
                    moved = j
                    break
        if moved is None:
            moved = nearest_clean
        # 앞으로 옮기는 건 첫 문장이 짧은 전환("그래서 우리가 하나님의 마음을 가져야 돼.")일 때만.
        # "그래서 베드로가 … 쿠오바디스 영화의 한 장면에 보면 …"처럼 긴 내용 문장은 '그래서'가
        # 말버릇일 뿐 설정이 들어 있어 잘라내면 이야기가 무너진다(실측 1GM 쿠오바디스).
        if moved is None and len(sentences[start].text.split()) <= _TRANSITION_MAX_WORDS:
            for j in range(start + 1, core + 1):
                if _is_clean_start(sentences[j].text) and dur(j, end) >= min_sec * 0.8:
                    moved = j
                    break
        if moved is not None:
            log.append(f"start S{start} 접속어 시작 → S{moved} (깨끗한 첫 문장)"); start = moved
    # (8) 첫 문장이 약한 훅(연도 시작·숫자 나열·30단어 넘는 긴 문장)이면 앞(미래)으로 최대 3문장 안에서
    #     더 나은 훅을 찾는다: 질문 > 6단어 이상의 깨끗한 문장 > 첫 깨끗한 문장. 설정이 조금 빠져도 훅이 사는
    #     쪽이 낫다(실측 zvs: "복음을 들고 태평양을 건너…" 29초 문장 → "와서 1년도 안 돼서 돌아가신 선교사님들…").
    if _is_weak_hook(sentences[start].text) and start < core:
        best = None
        for j in range(start + 1, min(core, start + _WEAK_HOOK_LOOKAHEAD) + 1):
            tj = sentences[j].text.strip()
            if not _is_clean_start(tj) or dur(j, end) < min_sec * 0.8:
                continue
            if tj.endswith("?"):
                best = j; break
            if best is None or (len(sentences[best].text.split()) < 6 <= len(tj.split())):
                best = j
        if best is not None:
            log.append(f"start S{start} 약한 훅(연도/나열/장문) → S{best}"); start = best
    skips = _sanitize_skips(skips, sentences, start, end, core, log)
    if dur(start, end) < min_sec * 0.8:
        return None, log + [f"길이 {dur(start,end):.0f}초 < 하한 → 탈락"]
    fixed = dict(raw); fixed.update({"core": core, "start": start, "end": end, "skip": [list(s) for s in skips]})
    return fixed, log


def merge_adjacent_cuts(
    cuts: list[dict], sentences: list[Sentence], hard_max_sec: float, gap_sec: float = _MERGE_GAP_SEC,
) -> tuple[list[dict], list[str]]:
    """한 흐름을 둘로 쪼갠 컷을 합친다(강한 쪽의 core·appeal을 유지).

    2026-09-18 실측(1GM): "우리를 보실 때 이뻐 죽겠어"(33초)와 "하나님이 버리셨느냐? 그럴 수
    없느니라"(34초)는 설정→반전→착지가 이어지는 하나의 흐름인데, '한 컷 = 명제 하나' 규칙 때문에
    모델이 둘로 나눠 냈고 각각은 훅이 죽은 반쪽이 됐다. 두 컷이 8초 이내로 붙어 있고, 앞 컷이
    축복·아멘 착지로 끝나지 않았으며(=생각이 아직 안 끝남), 합쳐도 상한 이내면 하나로 만든다.
    50% 이상 겹치는 컷은 같은 장면의 중복이라 여기서 합치지 않고 dedupe_cuts가 강한 쪽만 남긴다."""
    cuts = [dict(c) for c in cuts]
    log: list[str] = []

    def _t(c: dict) -> tuple[float, float]:
        return sentences[c["start"]].start, sentences[c["end"]].end

    changed = True
    while changed:
        changed = False
        for i in range(len(cuts)):
            for j in range(i + 1, len(cuts)):
                a, b = cuts[i], cuts[j]
                first, second = (a, b) if _t(a)[0] <= _t(b)[0] else (b, a)
                f0, f1 = _t(first); s0, s1 = _t(second)
                inter = max(0.0, min(f1, s1) - max(f0, s0))
                if inter / max(1e-6, min(f1 - f0, s1 - s0)) >= 0.5:
                    continue  # 중복 → dedupe 담당
                if s0 - f1 > gap_sec:
                    continue
                end_text = sentences[first["end"]].text
                if _STRONG_LANDING.search(end_text) or re.fullmatch(r"(>>\s*)?아멘[.!]?", end_text.strip()):
                    continue  # 앞 컷이 착지로 끝남 = 다른 생각
                # 두 컷 사이의 다리 문장에 대지 전환("마지막으로")·축복 착지·아멘이 있으면 다른 생각
                bridge = sentences[first["end"] + 1: second["start"]]
                if any(
                    _starts_with_structure_marker(x.text) or _STRONG_LANDING.search(x.text)
                    or re.fullmatch(r"(>>\s*)?아멘[.!]?", x.text.strip())
                    for x in bridge
                ):
                    continue
                new_start, new_end = first["start"], max(first["end"], second["end"])
                # 두 컷의 점프컷(skip)은 합집합으로 가져간다 — 길이는 skip을 들어낸 뒤 실제 남는 길이로 잰다.
                merged_skips = _merge_ranges(
                    [tuple(s) for s in (first.get("skip") or [])] + [tuple(s) for s in (second.get("skip") or [])]
                )
                if _eff_dur(sentences, new_start, new_end, merged_skips) > hard_max_sec:
                    # 합치면 넘칠 때: 뒤 컷의 핵심 문장 이후 가장 늦은 착지(축복·믿습니다·아멘)까지로 끝을 당겨
                    # 상한에 맞춘다(실측 1GM: 모델이 뒤 컷을 마지막 축복 기도까지 늘려 102초가 됐지만
                    # "…은혜 베푸신 줄 믿습니다. 아멘."에서 끊으면 77초로 한 흐름이 온전히 들어간다).
                    late_core = max(first["core"], second["core"])
                    trimmed = None
                    for cand in range(new_end - 1, late_core - 1, -1):
                        if _eff_dur(sentences, new_start, cand, merged_skips) > hard_max_sec:
                            continue
                        tx = sentences[cand].text
                        if _STRONG_LANDING.search(tx) or re.fullmatch(r"(>>\s*)?아멘[.!]?", tx.strip()):
                            trimmed = cand
                            break
                    if trimmed is None:
                        continue
                    new_end = trimmed
                merged = dict(a)  # 강한 쪽(a=i)의 core/appeal/thesis 유지
                merged["start"], merged["end"] = new_start, new_end
                merged["skip"] = [list(s) for s in _clip_skips(merged_skips, new_start, new_end)]
                if b.get("why"):
                    merged["why"] = f"{a.get('why', '')} + {b['why']}".strip(" +")
                log.append(
                    f"컷 S{a['start']}~S{a['end']} + S{b['start']}~S{b['end']} → S{new_start}~S{new_end} (한 흐름 병합)"
                )
                cuts[i] = merged
                del cuts[j]
                changed = True
                break
            if changed:
                break
    return cuts, log


# 남을 비판·경계·폭로하는 주제(이단·사이비·타종교·점쟁이…)의 표지어. 프롬프트에 "절대 제외"라고 써 놔도
# 모델은 이야기가 구체적이면 "재미"로 뽑아 올린다(실측 2026-09-20, 1GM: 기도원장 점쟁이/이단 클립과 몰몬경
# 금판 클립이 1·2위 — 사용자 "감동도 재미도 없구만"). 그래서 여기서 결정론적으로 거른다.
# 전사 오타까지 잡는다(몰몽경/몰경 = 몰몬경).
_POLEMIC_RE = re.compile(
    r"이단|사이비|몰몬|몰몽|몰경|신천지|여호와의\s*증인|통일교|안식교|구원파|하나님의\s*교회|점쟁이|무당|굿을|"
    r"혹세\s*무민|거짓\s*계시|거짓\s*선지자|미혹|교주|사교|포교"
)
_POLEMIC_MIN_SENTENCES = 2  # 컷 안에서 표지어가 든 문장이 이만큼이면 주제 자체가 타자 비판이다


def polemic_reason(cut: dict, sentences: list[Sentence]) -> str:
    """컷이 '남 비판·경계' 주제면 그 근거 문자열을, 아니면 빈 문자열을 돌려준다.

    판정: (a) 핵심 문장(core)·모델이 적은 장면/명제 요약에 표지어가 있거나,
          (b) 컷 본문에서 표지어가 든 문장이 2개 이상."""
    try:
        core = sentences[int(cut["core"])].text
        body = sentences[int(cut["start"]): int(cut["end"]) + 1]
    except (KeyError, TypeError, ValueError, IndexError):
        return ""
    m = _POLEMIC_RE.search(core)
    if m:
        return f"핵심 문장에 '{m.group(0)}'"
    for key in ("scene", "thesis"):
        m = _POLEMIC_RE.search(str(cut.get(key) or ""))
        if m:
            return f"{key}에 '{m.group(0)}'"
    hits = [s for s in body if _POLEMIC_RE.search(s.text)]
    if len(hits) >= _POLEMIC_MIN_SENTENCES:
        words = sorted({_POLEMIC_RE.search(s.text).group(0) for s in hits})
        return f"본문 {len(hits)}문장에 {'/'.join(words)}"
    return ""


def drop_polemic_cuts(cuts: list[dict], sentences: list[Sentence]) -> tuple[list[dict], list[str]]:
    """타자 비판 주제 컷을 제외한다. 전부 제외돼 후보가 1개 이하로 남으면(설교 자체가 그 주제)
    제외 대신 맨 뒤로 보내고 표시만 남긴다 — 결과 0개보다는 낫다."""
    kept, dropped, log = [], [], []
    for c in cuts:
        why = polemic_reason(c, sentences)
        if why:
            dropped.append({**c, "polemic": why})
            log.append(f"S{c['start']}~S{c['end']} 타자 비판 주제 제외 ({why})")
        else:
            kept.append(c)
    if len(kept) < 2 and dropped:
        log.append(f"남는 후보 {len(kept)}개 → 제외 대신 뒤로 보냄")
        kept = kept + dropped
    return kept, log


# 성경 인물 이야기가 주된 내용인 컷의 표지(2026-09-21 사용자: "성경 인물 이야기는 빼라. 성경 모르는 일반 대중에겐
# 베드로가 어디로 갔다 같은 내용은 공감이 안 된다 — 과감하게 버리고 일반적인 교훈을 뽑아라"). 프롬프트의
# '절대 제외'만으로는 모델이 이야기가 구체적이면 계속 뽑아 올리므로(이단 필터와 같은 실측) 여기서 결정론적으로
# 거른다. '예수/하나님/그리스도'는 인물 서사가 아니라 신앙 언어라 표지에서 뺀다. "요한복음 3장"처럼 책 이름
# 인용은 이야기가 아니므로 뒤에 복음/서/기/계시록이 붙으면 제외.
_BIBLE_NAME_RE = re.compile(
    r"(베드로|요나|모세|다윗|아브라함|아브람|요셉|바울|사울|엘리야|엘리사|야곱|이삭|솔로몬|노아|다니엘|에스더|룻|보아스|"
    r"기드온|삼손|여호수아|갈렙|사무엘|느헤미야|에스라|욥(?!바)|이사야|예레미야|에스겔|호세아|마리아|마르다|나사로|삭개오|"
    r"니고데모|유다|빌립|스데반|바나바|디모데|요한|야고보|안드레|막달라|골리앗|가룟|라합|므낫세|"
    r"히스기야|여로보암|르호보암|아합|이세벨|나아만|게하시|발람|미리암|아론|라헬|"
    r"이스라엘\s*백성|바리새인|사두개인|제자들)(?!복음|서|기|계시록|일서|이서|삼서|전서|후서)"
)
_BIBLE_STORY_MIN_SENTENCES = 3
_BIBLE_STORY_MIN_RATIO = 0.25


def bible_story_reason(cut: dict, sentences: list[Sentence]) -> str:
    """컷이 '성경 인물 이야기'가 주된 내용이면 근거 문자열을, 아니면 빈 문자열을 돌려준다.

    판정: (a) 핵심 문장·모델의 장면/명제 요약에 인물 이름이 있거나,
          (b) 남는 본문(skip 제외)에서 인물 이름이 든 문장이 3개 이상이고 25% 이상."""
    try:
        core = sentences[int(cut["core"])].text
        runs = kept_runs(int(cut["start"]), int(cut["end"]), [tuple(s) for s in (cut.get("skip") or [])])
        body = [s for a, b in runs for s in sentences[a: b + 1]]
    except (KeyError, TypeError, ValueError, IndexError):
        return ""
    m = _BIBLE_NAME_RE.search(core)
    if m:
        return f"핵심 문장에 '{m.group(1)}'"
    for key in ("scene", "thesis"):
        m = _BIBLE_NAME_RE.search(str(cut.get(key) or ""))
        if m:
            return f"{key}에 '{m.group(1)}'"
    hits = [s for s in body if _BIBLE_NAME_RE.search(s.text)]
    if len(hits) >= _BIBLE_STORY_MIN_SENTENCES and len(hits) / max(1, len(body)) >= _BIBLE_STORY_MIN_RATIO:
        names = sorted({_BIBLE_NAME_RE.search(s.text).group(1) for s in hits})
        return f"본문 {len(hits)}/{len(body)}문장에 {'/'.join(names[:4])}"
    return ""


def drop_bible_story_cuts(cuts: list[dict], sentences: list[Sentence]) -> tuple[list[dict], list[str]]:
    """성경 인물 이야기 컷을 제외한다. 남는 후보가 1개 이하면(설교 전체가 인물 강해) 제외 대신 맨 뒤로 보내고
    표시만 남긴다 — 결과 0개보다는 낫다(점수 상한 30으로 항상 맨 아래)."""
    kept, dropped, log = [], [], []
    for c in cuts:
        why = bible_story_reason(c, sentences)
        if why:
            dropped.append({**c, "bible_story": why})
            log.append(f"S{c['start']}~S{c['end']} 성경 인물 이야기 제외 ({why})")
        else:
            kept.append(c)
    if len(kept) < 2 and dropped:
        log.append(f"남는 후보 {len(kept)}개 → 제외 대신 뒤로 보냄")
        kept = kept + dropped
    return kept, log


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

시청자는 **성경을 모르는 일반 대중까지** 포함한다. 성경 지식 없이도 자기 얘기로 들리는가가 모든 축의 전제다.
세부 축(1~10 정수): core_score(**쇼츠 요소의 세기** — 재미(웃음)·감동(실화·희생)·뜨끔(직격)·교훈(관점을 뒤집는
한마디)·위로 중 하나가 구체(장면·실화·대사·숫자·질문)와 함께 있고 명제로 착지하는가. 구체 없이 "~입니다" 선언만이면
5 이하, 뻔한 권면("기도하세요")도 5 이하, 관점을 뒤집는 일반적 교훈이면 장면이 없어도 7 이상 가능),
hook(첫 문장만 따로 읽고 멈추게 하는가 — 구체 상황·대사·질문·직격이면 높게, 교리 선언·배경 설명·연도·나열·중간을 툭 자른
느낌이면 3 이하), retention(전진감·죽은 구간 없음), emotion(웃음·뭉클·찔림·위로·"아, 그래서였구나"의 실제 반응 —
옳기만 하고 반응이 없으면 낮게), relatability("내 얘기" — 교회 안 다니는 사람도 그런가), payoff(끝이 힘 있게 착지),
quotability(스샷 떠 공유할 한 문장). 눈금: 5=쓸 만함, 7=이 설교의 손꼽는 대목, 8=잘 되는 채널 상위 클립 수준,
9~10=채널 1위감(드묾). 정직하게 — 약하면 낮게.
주된 내용이 남(이단·사이비·타종교·다른 목회자·특정 집단)을 비판·경계·폭로하는 것이면 emotion·relatability·core_score
모두 3 이하 — 남 욕·경계는 흥미로워도 시청자 자신의 감정(웃음·뭉클·위로)이 아니고 공유되지 않는다.
주된 내용이 성경 인물의 행적 이야기(베드로가 어디로 갔다, 요나가 도망쳤다)면 relatability·core_score 4 이하 —
성경 모르는 시청자에겐 남의 옛날이야기다.

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


def _skips_of(cut: dict) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in (cut.get("skip") or [])]


def cut_keep_ranges(cut: dict, sentences: list[Sentence]) -> list[list[float]]:
    """컷의 남는 문장 구간을 절대초 [start, end] 목록으로. 자동자막 문장은 시각이 조금씩 겹치므로
    뒤 구간의 시작이 앞 구간의 끝보다 앞서면 밀어서 겹치지 않게 한다(렌더 select 필터는 겹침을 못 다룬다)."""
    out: list[list[float]] = []
    for a, b in kept_runs(int(cut["start"]), int(cut["end"]), _skips_of(cut)):
        s0, e0 = float(sentences[a].start), float(sentences[b].end)
        if out and s0 < out[-1][1] + 0.05:
            s0 = out[-1][1] + 0.05
        if e0 - s0 > 0.1:
            out.append([round(s0, 3), round(e0, 3)])
    return out


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
    debug_path=None,
    max_span_sec: float = 180.0,
) -> list[Clip]:
    """max_span_sec: 점프컷(skip)을 들어내기 전 원본 구간의 상한. 남는 길이는 hard_max_duration_sec 이내여야 한다.
    debug_path가 있으면 1차 원시 컷·검증 로그·병합 결과를 JSON으로 남긴다(선정 품질 불만이
    왔을 때 '모델이 뭘 줬고 검증이 뭘 바꿨는지'를 사후에 볼 수 있게 — 예전엔 아무것도 안 남았다)."""
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
    prompt1 = build_thesis_cut_prompt(
        sentences, transcript.duration_sec, min_clips, max_clips, extra_block,
        hard_max_sec=hard_max_duration_sec, max_span_sec=max_span_sec,
    )
    raw_cuts = _invoke_claude_json(
        prompt1, model=model, thinking_tokens=thinking_tokens, timeout_sec=timeout_sec,
        on_progress=_prog(0.0, 0.7), max_clips=max_clips,
    )
    print(f"[v2] 1차 패스: 컷 {len(raw_cuts)}개", flush=True)

    # --- 검증·보정
    fixed: list[dict] = []
    verify_logs: list[dict] = []
    for i, rc in enumerate(raw_cuts):
        cut, log = verify_and_fix(
            rc, sentences, min_duration_sec, max_duration_sec, hard_max_duration_sec, max_span_sec=max_span_sec,
        )
        if log:
            print(f"[v2] 검증 컷{i}: " + " / ".join(log), flush=True)
        verify_logs.append({"raw": rc, "fixed": cut, "log": log})
        if cut is not None:
            fixed.append(cut)
    fixed, merge_log = merge_adjacent_cuts(fixed, sentences, hard_max_duration_sec)
    for m in merge_log:
        print(f"[v2] 병합: {m}", flush=True)
    fixed, polemic_log = drop_polemic_cuts(fixed, sentences)
    for m in polemic_log:
        print(f"[v2] 주제 필터: {m}", flush=True)
    fixed, bible_log = drop_bible_story_cuts(fixed, sentences)
    for m in bible_log:
        print(f"[v2] 인물 이야기 필터: {m}", flush=True)
    fixed = dedupe_cuts(fixed, sentences)
    if debug_path is not None:
        try:
            import json
            from pathlib import Path
            dbg = {
                "sentences": [{"idx": s.idx, "start": s.start, "end": s.end, "text": s.text} for s in sentences],
                "raw_cuts": raw_cuts, "verify": verify_logs, "merge_log": merge_log, "polemic_log": polemic_log,
                "bible_log": bible_log,
                "final_cuts": [{**c, "start_sec": sentences[c["start"]].start, "end_sec": sentences[c["end"]].end,
                                "keep_ranges": cut_keep_ranges(c, sentences),
                                "eff_sec": _eff_dur(sentences, c["start"], c["end"], _skips_of(c))}
                               for c in fixed],
            }
            Path(debug_path).write_text(json.dumps(dbg, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - 디버그 저장 실패는 선정에 영향 없음
            print(f"[v2] 디버그 저장 실패: {exc}", flush=True)
    if not fixed:
        return []

    # --- 2차: 실제 대사 채점
    items = []
    for i, c in enumerate(fixed):
        runs = kept_runs(c["start"], c["end"], _skips_of(c))
        # 점프컷으로 들어낸 문장은 채점 대상에서도 뺀다(실제로 들리는 대사만). 이음새는 ' … '로 표시.
        text = " … ".join(" ".join(x.text for x in sentences[a: b + 1]) for a, b in runs)
        items.append({
            "index": i, "duration": _eff_dur(sentences, c["start"], c["end"], _skips_of(c)),
            "appeal": c.get("appeal", ""), "core_line": sentences[c["core"]].text, "text": text,
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
        clip_start, hook_text = trim_lead_words(s)  # "그래서 베드로가…" → "베드로가…"부터
        computed = compute_scores(r if r else {"core_score": 6, "hook": 6, "retention": 6, "emotion": 6,
                                               "relatability": 6, "payoff": 6, "quotability": 6})
        if c.get("polemic") or c.get("bible_story"):
            # 후보 부족으로 살려 둔 타자 비판/성경 인물 이야기 컷: 점수 상한을 걸어 항상 맨 아래
            computed["score"] = min(int(computed["score"]), 30)
        # 점프컷: 남길 구간(절대초). 첫 구간의 시작은 말버릇 트림을 반영한다. 한 구간뿐이면 빈 목록(=전체 사용).
        keep_abs = cut_keep_ranges(c, sentences)
        if len(keep_abs) > 1:
            keep_abs[0] = [max(keep_abs[0][0], clip_start), keep_abs[0][1]]
        else:
            keep_abs = []

        def _f(k: str):
            try:
                return float(r[k]) if r.get(k) is not None else None
            except (TypeError, ValueError):
                return None

        clips.append(Clip(
            start=clip_start, end=e.end,
            title=str(r.get("title") or c.get("thesis") or core_line).strip()[:40],
            caption=str(r.get("caption", "")).strip(),
            hashtags=[str(h) for h in (r.get("hashtags") or ["#설교", "#은혜", "#말씀"])],
            reason=str(c.get("why", "")).strip(),
            score=computed["score"], core_score=computed["core_score"], viral_score=computed["viral_score"],
            hook_score=_f("hook"), retention_score=_f("retention"), emotion_score=_f("emotion"),
            relatability_score=_f("relatability"), payoff_score=_f("payoff"), quotability_score=_f("quotability"),
            appeal=str(c.get("appeal", "")).strip(),
            hook_line=hook_text, payoff_line=e.text, core_line=core_line,
            insight=str(r.get("insight", "")).strip(),
            anchored=True,  # 경계가 문장 시각으로 확정됨 — 렌더의 끝 스냅은 미세조정만
            title_candidates=[str(t).strip() for t in (r.get("title_candidates") or []) if str(t).strip()],
            keywords=[str(k).strip() for k in (r.get("keywords") or []) if str(k).strip()],
            keep_ranges=keep_abs,
        ))
    return clips
