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

from src.highlights import Clip, QuotaExceededError, _invoke_claude_json
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
    # 2026-09-22 실측(zvsOYhjdmks): 훅이 "이렇게 연수원 가면서 …"로 시작해 '이렇게'가 앞 이야기를 가리켰다.
    # 클립 첫 문장 맨 앞에 오는 이 부사는 언제나 앞 문맥을 가리키는 되받이라 잘라내는 게 맞다.
    "이렇게", "그렇게", "저렇게",
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
- **skip(점프컷) — 재미없는 부분을 도려내는 핵심 기능**: 잘 되는 채널의 상위 쇼츠를 원본 설교와 대조해 보면
  18/26개가 이렇게 만들어졌다. 원본 80~520초에서 **중간을 여러 군데 들어내** 60초 안팎으로 압축한다(남기는 비율
  중앙값 66%, 이음새는 한 클립에 1~7곳). 단 **안 빼도 60초 안에 들면 추임새("응.", "어.") 말고는 빼지 마라** —
  길이가 넘칠 때 줄이는 도구다(실측 실패: 59초짜리에서 착지의 근거 문장을 '부연'으로 빼 버림). 빼야 할 것: 같은 말 반복, 곁길·부연("이거는 사실 ~만이 목적이 아니라"), 슬라이드 넘기기,
  수치·연도 나열, 늘어지는 설명. 남길 것: 상황 제시 → 대사·장면 → 반전 → 착지 명제.
  **절대 조건은 '누가 봐도 짜깁기한 티가 나지 않고, 의미·맥락이 자연스럽게 이어질 것'이다.** 실측으로 확인된 기준:
   (1) skip 바로 **앞 문장은 말이 끝난 문장**이어야 한다(쉼표·"~하는데"로 끝나면 안 됨). 실제 이음새 39곳 전부가 지켰다.
   (2) 이어붙인 **뒤 문장이 가리키는 대상이 남은 내용 안에 있어야** 한다. "그 사장님이…"라고 이어지는데 사장님이
       소개된 문장을 빼 버리면 안 된다. 이게 '어색한 짜깁기'의 진짜 원인이다.
   (3) 뒤 문장이 "그런데/근데/그래서"로 시작하는 건 **괜찮다** — 실제 이음새의 21%가 그 모양이고 자연스럽다.
       다만 "둘째,/세 번째로"처럼 앞 항목을 전제하는 순서 표지로 이어지면 안 된다.
   (4) 대사·반전은 앞뒤를 붙여서 통으로 남겨라. 빼면 웃음이나 반전이 죽는다.
   (5) **장면의 구체(누가 무엇을 어떻게 했다·숫자·대사)는 재미 그 자체라 빼지 마라.** 빼는 건 해설·반복·곁길이다.
       네가 scene·why에 적은 내용이 들어 있는 문장은 절대 빼면 안 된다(실측 실패: why에 "직접 요리해 먹는다"를 적어
       놓고 "시장 봐서 요리를 직접 합니다"를 뺐다). 착지를 떠받치는 약속·은혜 선언("회개하는 자를 받아 주신다")도
       부연이 아니다. 그래도 길면 해설 쪽(종교적 의미 풀이, 같은 말 되풀이)을 빼라.
   (6) 줄일 때 **가장 먼저 빼는 것은 성경 인물 이야기의 세부**(○○가 어디로 가서 무엇을 했다는 행적, 본문 낭독,
       시대 배경)다 — 시청자는 성경을 모르는 일반 대중이라 그 세부는 남의 옛날이야기다(사용자 지시). 인물이 한 문장
       근거로만 스치는 건 두되, 행적을 따라가는 여러 문장은 통째로 들어내고 그 뒤의 적용·대사·착지를 남겨라.
       반대로 **핵심(core)과 그 착지, 반전 대사는 절대 skip 안에 넣지 마라** — 실측에서 핵심을 건너뛴 컷이 나왔다.
  각 skip은 [첫 문장, 끝 문장] 번호 쌍이고 start·end·core는 뺄 수 없다. 들어내고 남는 길이가 {hard_max_sec:.0f}초
  이내면 원본 구간은 {max_span_sec:.0f}초까지 잡아도 된다. {hard_max_sec:.0f}초 이내 컷에도 늘어지는 구간이 있으면 빼서
  밀도를 올려라 — 단 위 5개 조건 중 하나라도 어기면 빼지 않는 쪽이 낫다.
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
# 2026-09-22 시도했다가 **되돌린** 규칙: 끝이 강한 착지가 아니면 뒤 14문장/75초 안의 마지막 착지까지 늘리기.
# 정답지로 재 보니 평균 소재 커버가 67%→61%로 떨어지고, 늘어난 길이 때문에 (5)가 시작을 핵심 문장 너머로
# 밀어 클립 하나가 통째로 탈락했다(HQMyIBO42-s 26%→0%). 모델이 중간 명제에서 끊는 문제는 실재하지만
# 결정론적 확장으로는 못 고친다 — 프롬프트 쪽에서 풀 것. 같은 아이디어를 다시 넣지 말 것.
# 2026-09-22 정답지 A/B로 30→45, 45→60 상향: 적중 6/7·커버 69%는 그대로인데 40초 미만 후보가 4개→2개로 줄었다.
# 벤치마크 상위 쇼츠 실측 길이가 59~111초라 30초대 후보는 '설정 없이 명제만' 쪽에 가깝다.
_SHORT_TARGET_SEC = 45.0       # 이보다 짧은 컷은 앞 설정을 보강
_SHORT_EXTEND_CAP_SEC = 60.0   # 보강해도 이 길이는 넘기지 않음
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
    """첫 문장으로 두면 '중간을 툭 자른' 훅이 되는 문장(접속어·지시어·구조 표지로 시작).

    말버릇 추임새("아 하나님께서 원하시면 나도 원하죠.")는 여기서 버리지 않는다 — 렌더 직전
    `trim_lead_words`가 그 단어(0.2~0.5초)만 잘라내고 나머지를 훅으로 쓴다. 예전엔 추임새 접두도
    '더러움'으로 보고 문장을 통째로 건너뛰어, 크레센도의 설정 문장이 날아갔다(2026-09-22 실측
    mjg0tcadzjY: "아 하나님께서 원하시면 나도 원하죠." → 이어지는 "여러분, 그렇지 않습니다."부터
    시작돼 반박의 대상이 사라짐). 문장 전체가 추임새인 것("예.", "아멘.")은 여전히 버린다."""
    t = text.strip()
    if not t or _FILLER_ONLY.match(t):
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


# 이음새 판정 규칙은 **실제로 터진 쇼츠의 이음새 39곳**(정답지 26개 중 점프컷 18개)을 대조해 정한 것이다
# (2026-09-22). 사용자 요구: "누가 봐도 짜깁기해서 어색하게 잘리면 안 되고, 의미·맥락상 자연스럽게 이어져야 한다."
#   · 앞 문장 완결: 실제 이음새 39곳 전부가 지킴(위반 0) → 엄격히 유지.
#   · 지시어가 가리키는 대상이 남은 내용 안에 있을 것: 37/39가 지킴 → 이게 '어색한 짜깁기'의 진짜 신호다.
#   · 뒤 문장이 접속어면 무조건 금지: **틀린 규칙이었다.** 실제 이음새 8곳(21%)이 "그런데/근데/그래서/자,"로
#     시작하는데 전부 자연스럽다 — 접속어가 가리키는 건 잘려나간 부분이 아니라 '남아 있는 앞 문장'이기 때문.
#     그래서 접속어 금지는 없애고, 대상이 사라지는 경우만 막는다.
#   · 순서 표지(둘째/셋째)는 앞 항목이 잘리면 말이 안 되므로 계속 금지("자,"·"그러면"은 담화 표지라 허용).
_ORDINAL_START = re.compile(r"^(둘째|셋째|넷째|다섯째|두\s*번째|세\s*번째|네\s*번째|다음으로|마지막으로)[,\s]")
# "그 놀라운 능력이" — 지시어 + 명사. "그거/이게" 같은 대명사는 명사가 없어 대상 대조를 못 하므로 제외한다.
_DEMONSTRATIVE_REF = re.compile(r"^(?:>>\s*)?(이|그|저)\s+(\S{2,})")
_PARTICLE_TAIL = re.compile(r"(은|는|이|가|을|를|에|에서|의|도|만|과|와|로|으로|께서|한테|에게|이나|나)$")
# "그 아십니까" 처럼 지시어가 대상을 가리키지 않는 관용 표현(실제 이음새에서 오탐이던 것).
_DEMONSTRATIVE_IDIOM = re.compile(r"^(?:>>\s*)?(이|그|저)\s+(아십니까|아세요|뭐냐|뭡니까|누구|왜|어떻게)")


def _seam_problem(before: Sentence, after: Sentence, kept_before: str) -> str:
    """이음새(앞 문장 → 잘라낸 뒤 이어지는 문장)가 어색하면 이유를, 자연스러우면 빈 문자열을 돌려준다."""
    bt, at = before.text.strip(), after.text.strip()
    if _is_incomplete(bt) and not bt.endswith("?"):
        return f"앞 문장이 안 끝남('…{bt[-12:]}')"
    if _FILLER_ONLY.match(at):
        return f"뒤 문장이 추임새뿐('{at[:10]}')"
    if _ORDINAL_START.match(at):
        return f"뒤 문장이 순서 표지('{at[:10]}') — 앞 항목이 잘리면 말이 안 된다"
    m = _DEMONSTRATIVE_REF.match(at)
    if m and not _DEMONSTRATIVE_IDIOM.match(at):
        noun = _PARTICLE_TAIL.sub("", m.group(2))
        if len(noun) >= 2 and noun not in kept_before:
            return f"뒤 문장이 잘려나간 대상을 가리킴('{m.group(0)}')"
    return ""


def _sanitize_skips(
    skips: list[tuple[int, int]], sentences: list[Sentence], start: int, end: int, core: int, log: list[str],
) -> list[tuple[int, int]]:
    """자연스럽지 않은 skip은 버린다(위 규칙). 앞쪽 skip부터 차례로 확정하며, '남은 내용'은 그때까지
    확정된 skip을 뺀 텍스트로 계산한다 — 그래야 지시어의 대상이 실제로 남아 있는지 정확히 본다."""
    out: list[tuple[int, int]] = []
    for a, b in _clip_skips(skips, start, end):
        if a <= core <= b:
            log.append(f"skip S{a}~S{b} 핵심 문장 포함 → 무시"); continue
        kept_before = " ".join(
            s.text for i0, i1 in kept_runs(start, a - 1, out) for s in sentences[i0: i1 + 1]
        )
        why = _seam_problem(sentences[a - 1], sentences[b + 1], kept_before)
        if why:
            log.append(f"skip S{a}~S{b} {why} → 무시"); continue
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


# 들어내면 안 되는 skip 되살리기(2026-09-27 실신고 "재밌는 부분·핵심을 잘랐다", 1GM 실측).
#   · 땅끝 마을: 안 잘라도 59초인데 '밀도'용으로 "하나님은 버리지 아니하시고 회개하는 자를 탁 받아 주시고…"(착지의
#     근거)를 뺐다. → 안 빼도 max_duration_sec 안에 들면 추임새 말고는 빼지 않는다.
#   · 호텔밥: 모델 스스로 why에 "직접 요리해 먹는다"를 이유로 적어 놓고 그 문장(S214)을 뺐다. → scene·why·thesis에 쓴
#     구체어가 들어 있는 문장은 되살린다(상한 안에서).
_GENERIC_TOKENS = {
    "하나님", "하나님이", "예수님", "주님", "우리", "우리가", "여러분", "사람", "사람들", "그리고", "그래서", "그런데",
    "이것", "그것", "때문", "정말", "모든", "하는", "있는", "없는", "것이", "것을", "거예요", "합니다", "입니다",
}


def _content_stems(text: str) -> set[str]:
    out: set[str] = set()
    for tok in re.findall(r"[가-힣A-Za-z0-9%]+", text or ""):
        stem = _PARTICLE_TAIL.sub("", tok) if len(tok) > 2 else tok
        stem = re.sub(r"(해|하|했|합|하는|해요|합니다|한다|하고|해서|먹는다|먹고)$", "", stem) or stem
        if len(stem) >= 2 and tok not in _GENERIC_TOKENS and stem not in _GENERIC_TOKENS:
            out.add(stem)
    return out


def _restore_costly_skips(
    raw: dict, skips: list[tuple[int, int]], sentences: list[Sentence], start: int, end: int,
    max_sec: float, hard_max_sec: float, log: list[str],
) -> list[tuple[int, int]]:
    if not skips:
        return skips

    def _is_filler(i: int) -> bool:
        return bool(_FILLER_ONLY.match(sentences[i].text.strip())) or len(sentences[i].text.split()) <= 1

    def _filler_only_skips(sk: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for a, b in sk:
            run: int | None = None
            for i in range(a, b + 1):
                if _is_filler(i):
                    run = i if run is None else run
                elif run is not None:
                    out.append((run, i - 1)); run = None
            if run is not None:
                out.append((run, b))
        return out

    # (a) 안 잘라도 목표 길이 안 → 추임새만 빼고 전부 되살린다
    if _eff_dur(sentences, start, end, []) <= max_sec + 1.0:
        kept = _filler_only_skips(skips)
        if kept != skips:
            log.append(f"skip 되살림: 안 잘라도 {_eff_dur(sentences, start, end, []):.0f}초(≤{max_sec:.0f}) — 추임새만 뺀다")
        return kept
    # (b) 모델이 scene·why·thesis에 쓴 구체어가 든 문장은 되살린다(상한 안에서만, 문장 단위)
    key = _content_stems(" ".join(str(raw.get(k) or "") for k in ("scene", "why", "thesis")))
    if not key:
        return skips
    cur = list(skips)
    for a, b in skips:
        for i in range(a, b + 1):
            kept_text = " ".join(
                sentences[x].text for r0, r1 in kept_runs(start, end, cur) for x in range(r0, r1 + 1)
            )
            hit = sorted(t for t in key if t in sentences[i].text and t not in kept_text)
            if not hit:
                continue
            trial: list[tuple[int, int]] = []
            for c0, c1 in cur:
                if c0 <= i <= c1:
                    if c0 <= i - 1:
                        trial.append((c0, i - 1))
                    if i + 1 <= c1:
                        trial.append((i + 1, c1))
                else:
                    trial.append((c0, c1))
            if _eff_dur(sentences, start, end, trial) <= hard_max_sec:
                log.append(f"S{i} 되살림: 장면/이유의 핵심어 {hit[:3]} 가 이 문장에만 있음")
                cur = trial
            else:
                log.append(f"S{i}에 핵심어 {hit[:3]} 있으나 되살리면 {hard_max_sec:.0f}초 초과 — 유지")
    return _merge_ranges(cur)


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
    skips = _restore_costly_skips(raw, skips, sentences, start, end, max_sec, hard_max_sec, log)
    # 하한은 config min_duration_sec 그대로 적용한다(예전 0.8배 여유는 30초대 반쪽 후보를 통과시켰다 —
    # 2026-09-22 정답지 A/B: 하한을 엄격히 해도 적중 6/7·커버 70% 유지, 40초 미만 후보만 2개→1개로 줄었다).
    if dur(start, end) < min_sec:
        return None, log + [f"길이 {dur(start,end):.0f}초 < 하한 {min_sec:.0f}초 → 탈락"]
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
# 앞에 한글이 붙어 있으면 이름이 아니다 — "필요한/중요한"의 '요한', "자유다/이유다"의 '유다' 오탐 방지
# (2026-09-21 실측: 6.9천 조회수 '장바구니' 쇼츠가 "반드시 필요한 사람인데"의 '요한'으로 걸렸다).
_BIBLE_NAME_RE = re.compile(
    r"(?<![가-힣])(베드로|요나|모세|다윗|아브라함|아브람|요셉|바울|사울|엘리야|엘리사|야곱|이삭|솔로몬|노아|다니엘|에스더|룻|보아스|"
    r"기드온|삼손|여호수아|갈렙|사무엘|느헤미야|에스라|욥(?!바)|이사야|예레미야|에스겔|호세아|마리아|마르다|나사로|삭개오|"
    r"니고데모|유다|빌립|스데반|바나바|디모데|요한|야고보|안드레|막달라|골리앗|가룟|라합|므낫세|"
    r"히스기야|여로보암|르호보암|아합|이세벨|나아만|게하시|발람|미리암|아론|라헬|"
    # '제자들'은 뺐다: 특정 인물 서사가 아니라 일반 명사로 더 많이 쓰이고("우리도 제자들처럼"),
    # 4,338회로 검증된 쇼츠가 이것 때문에 제외됐다(2026-09-22).
    r"이스라엘\s*백성|바리새인|사두개인)(?!복음|서|기|계시록|일서|이서|삼서|전서|후서)"
)
_BIBLE_STORY_MIN_SENTENCES = 3
_BIBLE_STORY_MIN_RATIO = 0.30
_BIBLE_HOOK_SENTENCES = 2   # 클립의 첫 이 문장이 '훅'이다 — 여기서 인물이 나오면 시청자에겐 옛날이야기로 들린다


def bible_story_reason(cut: dict, sentences: list[Sentence]) -> str:
    """컷이 '성경 인물 이야기'가 주된 내용이면 근거 문자열을, 아니면 빈 문자열을 돌려준다.

    판정(2026-09-22 실측으로 재설계): **클립이 인물로 시작하는가**가 핵심이다.
      (a) 핵심 문장(착지 명제)에 인물 이름 → 인물 서사
      (b) 훅(남는 본문의 첫 2문장)에 인물 이름 → 시청자가 첫 3초에 옛날이야기로 인식
      (c) 남는 본문의 30% 이상, 3문장 이상에 인물 이름 → 내내 인물 이야기
    **모델이 쓴 요약문(scene/thesis)은 더 이상 보지 않는다.** 5,200회로 검증된 "아무것도 보이지 않을 때"가
    요약에 '요나'가 들어갔다는 이유로 제외됐다 — 실제 클립은 "도와줄 사람이 하나도 없는 상황"으로 시작하고
    요나는 근거로 잠깐 나올 뿐이었다(본문 비중 14%). 요약은 모델의 말일 뿐 시청자가 듣는 내용이 아니다."""
    try:
        core = sentences[int(cut["core"])].text
        runs = kept_runs(int(cut["start"]), int(cut["end"]), [tuple(s) for s in (cut.get("skip") or [])])
        body = [s for a, b in runs for s in sentences[a: b + 1]]
    except (KeyError, TypeError, ValueError, IndexError):
        return ""
    m = _BIBLE_NAME_RE.search(core)
    if m:
        return f"핵심 문장에 '{m.group(1)}'"
    for s in body[:_BIBLE_HOOK_SENTENCES]:
        m = _BIBLE_NAME_RE.search(s.text)
        if m:
            return f"훅에 '{m.group(1)}'"
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


def dedupe_cuts(cuts: list[dict], sentences: list[Sentence], overlap_ratio: float = 0.3) -> list[dict]:
    """겹치는 컷은 앞(강한) 것만 남긴다.

    기준 0.5→0.3 (2026-09-22 정답지 A/B): 0.5에서는 16초를 공유하는 두 후보(35초·82초)가 둘 다 살아남아
    검토 화면에 사실상 같은 소재가 두 장 떴다. 0.3으로 내려도 적중·커버는 그대로였다."""
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
# 4-1. 나열형 교훈(첫째·둘째·셋째) 전용 컷
# ---------------------------------------------------------------------------
# 사용자 요구(2026-09-27, "매우 중요"): 설교에서 "첫째, 둘째, 셋째"로 교훈이 나오면 그건 무조건 쇼츠에
# 넣는다. 단 재미없는 부분은 날려 알찬 60초 이내로. 일반 컷 규칙(한 컷=한 명제, 구조 표지로 시작 금지,
# 순서 표지 앞 skip 금지, 길면 앞부터 깎기)은 전부 이 구조를 쪼개거나 버리는 쪽으로 작동하므로,
# 나열은 결정론적으로 찾아내 별도 패스로 만들고 일반 검증·필터·중복제거를 거치지 않는다.
_ENUM_ORD = [
    (1, r"첫\s*째|첫\s*번\s*째"), (2, r"둘\s*째|두\s*번\s*째"), (3, r"셋\s*째|세\s*번\s*째"),
    (4, r"넷\s*째|네\s*번\s*째"), (5, r"다섯\s*째|다섯\s*번\s*째"),
]
# 순서 표지는 문장 어디에 있어도 센다 — 실측(2026-09-27, 설교 17편): "성령이 오시면 첫째 권능을 받아요",
# "전도의 원리 첫 번째는", "오늘 세 가지로 … 첫째는 뭐죠?"처럼 1번 항목은 문장 중간에서 나오는 게 보통이라
# 문장 첫머리만 보면 대부분 놓쳤다. 오탐은 '1→2→3 순서로 이어질 것'과 아래 명사 수식 제외로 거른다.
_ENUM_BOUND = r"(?:^|(?<=[\s,.?!>]))"
# "둘째 아들"·"첫째 날"·"세 번째 주일" 같은 명사 수식은 나열이 아니다.
_ENUM_NOT = re.compile(
    r"^\s*(아들|딸|날|형|언니|오빠|누나|동생|아이|자녀|손자|손녀|며느리|사위|부인|아내|주일|주|시간|해|달|줄|칸|장|절|편지|권)"
    r"(?:[이가은는을를의에도과와]|\s|[.,?!]|$)"
)
_ENUM_LAST = re.compile(
    r"^(?:>>\s*)?(?:(?:자|그리고|또|이제)[,.]?\s+)?(마지막으로|마지막\s*(?:세|네|다섯)?\s*번째로?|끝으로)"
)
_ENUM_FIRST_FALLBACK = re.compile(r"^(?:>>\s*)?(?:자[,.]?\s+)?(먼저|우선)[,\s]")
ENUM_BUDGET_SEC = 58.0          # 60초 이내(렌더 무음 제거로 더 줄지만 여유를 둔다) — 항목 2개 기준
_ENUM_MAX_GAP_SEC = 900.0       # 항목 사이 최대 간격(대지 설교는 항목 하나가 10분을 넘기도 한다)
# 2026-09-29 실신고 "나열 점프컷이 너무 에바(과하다)". 실측(IOSxqPI2nUc): 8분(S50~S131, 483초)짜리 2항목 나열을
# **한 문장씩 7개**(7.5초마다 점프, 원본의 11%만 남음)로 떼어 53초를 만들었다 — "첫 번째는 ~하나님입니다 →
# (5분 건너뜀) → 그런데 하나님께서는 이 연약한 한 여인의 손을 …(시스라·야엘 소개는 잘림) → …". 목차 낭독이지
# 쇼츠가 아니다. 1GM도 같은 모양(708초→46초, 5조각). 벤치마크 점프컷은 원본의 40~88%를 남기고 조각 하나가
# '상황→대사→착지'다. 원인 3가지와 대책:
#   1) 프롬프트가 "표지 + 가장 선명한 1~2문장"을 요구 → 문장 한 줄씩 띄엄띄엄. → 항목마다 **연속 블록**(2~5문장,
#      12~22초)을 통째로 남기게 하고, 항목당 구간 ≤2(표지+블록). 성경 인물 줄거리는 통째로 건너뛰고 적용 대목을 잡는다.
#   2) 예산 58초가 항목 수와 무관 → 3항목이면 항목당 10초, 표지 말고는 남을 게 없다. → 항목 3개부터 20초씩 가산,
#      hard_max(90)까지(벤치마크 상위 쇼츠 59~111초). 2항목은 여전히 60초 이내.
#   3) 나열 패스는 일반 검증(_sanitize_skips)을 우회해 이음새 규칙이 하나도 안 걸렸다 → _enum_fix_seams:
#      단독 문장 조각 확장, 잘려나간 대상을 가리키는 시작 되돌리기, 안 끝난 문장으로 조각이 끝나면 늘리기.
_ENUM_ITEM_EXTRA_SEC = 20.0     # 항목 3개부터 항목마다 더 주는 길이
_ENUM_MIN_BLOCK_SENTENCES = 2   # 표지가 아닌 조각의 최소 문장 수
_ENUM_MIN_BLOCK_SEC = 10.0      # 표지가 아닌 조각의 최소 길이
_ENUM_EXTEND_CAP = 3            # 조각을 앞/뒤로 늘릴 때 최대 문장 수


def _enum_budget(n_items: int, base: float = ENUM_BUDGET_SEC, hard_max: float = 90.0) -> float:
    """항목 수에 따른 남는 길이 예산: 2항목 58초, 3항목 78초, 4항목부터 hard_max."""
    if n_items <= 2:
        return min(base, hard_max)
    return min(hard_max, base + _ENUM_ITEM_EXTRA_SEC * (n_items - 2))


def _enum_number(text: str) -> int:
    """문장에 나오는 첫 순서 표지의 번호(1~5), 문장 첫머리 '마지막으로'면 -1, 없으면 0."""
    t = text.strip()
    best: tuple[int, int] | None = None  # (위치, 번호)
    for n, pat in _ENUM_ORD:
        for m in re.finditer(_ENUM_BOUND + r"(?:" + pat + r")", t):
            rest = t[m.end():]
            if not rest.startswith("로") and _ENUM_NOT.match(rest):
                continue
            if best is None or m.start() < best[0]:
                best = (m.start(), n)
            break
    if best:
        return best[1]
    if _ENUM_LAST.match(t):
        return -1
    return 0


def find_enumerations(sentences: list[Sentence]) -> list[tuple[list[int], bool]]:
    """'첫째 → 둘째 → (셋째…)'로 이어지는 항목 표지 문장 id 묶음들 (묶음, 1번 항목 표지를 찾았는가).
    1번은 전사에서 자주 뭉개지므로 '둘째 → 셋째'로 시작하는 묶음도 받고, 1번은 바로 앞의 '먼저'로
    보충하거나(찾으면) 못 찾으면 모델에게 찾게 한다."""
    groups: list[tuple[list[int], bool]] = []
    cur: list[int] = []
    last_n = 0

    def _close():
        if len(cur) < 2:
            return
        marks = list(cur)
        has_first = _enum_number(sentences[marks[0]].text) == 1
        if not has_first:
            # 2번 앞에서 "먼저 …"로 시작하는 문장을 1번으로 보충(2→3 간격 이내, 최소 5분)
            span = max(300.0, sentences[marks[1]].start - sentences[marks[0]].start)
            i = marks[0] - 1
            while i >= 0 and sentences[marks[0]].start - sentences[i].start <= span:
                if _ENUM_FIRST_FALLBACK.match(sentences[i].text.strip()):
                    marks.insert(0, i); has_first = True
                    break
                i -= 1
        groups.append((marks, has_first))

    for s in sentences:
        n = _enum_number(s.text)
        if n == 0:
            continue
        gap_ok = bool(cur) and s.start - sentences[cur[-1]].start <= _ENUM_MAX_GAP_SEC
        if cur and gap_ok and n == last_n + 1:
            cur.append(s.idx); last_n = n
        elif cur and gap_ok and n == last_n and s.start - sentences[cur[-1]].start < 90:
            continue  # "첫째, 기도입니다. 첫째로 기도는…" 같은 되풀이
        elif cur and gap_ok and n == -1 and last_n >= 2 and len(cur) >= 2 and s.start - sentences[cur[-1]].start <= max(
            180.0, 1.5 * (sentences[cur[-1]].start - sentences[cur[0]].start) / (len(cur) - 1)
        ):  # 설교 맺음말의 "마지막으로 기도하겠습니다"를 항목으로 잘못 붙이지 않게 간격을 본다
            cur.append(s.idx); last_n += 1
            _close(); cur = []; last_n = 0
        elif n in (1, 2):
            _close(); cur = [s.idx]; last_n = n
        else:
            _close(); cur = []; last_n = 0
    _close()
    return groups


def _build_enum_prompt(
    sentences: list[Sentence], lo: int, hi: int, marks: list[int], budget: float, has_first: bool = True,
) -> str:
    base = 1 if has_first else 2
    mark_no = {m: k + base for k, m in enumerate(marks)}
    n_items = len(marks) + (0 if has_first else 1)
    first_note = "" if has_first else (
        "\n※ 1번 항목의 표지(첫째/첫 번째)는 전사에서 잡히지 않았다. ★항목2 앞에서 1번 항목이 시작되는 문장을\n"
        "  직접 찾아 그 문장과 설명 1~2문장을 반드시 keep에 넣어라(\"first\"에 그 문장 id).\n"
    )
    body = "\n".join(
        f"S{s.idx} [{_fmt_ts(s.start)}] ({s.end - s.start:.0f}초)"
        + (f" ★항목{mark_no[s.idx]}" if s.idx in mark_no else "") + f" {s.text}"
        for s in sentences[lo: hi + 1]
    )
    max_runs = 2 * n_items + 2
    return f"""아래는 한국 교회 설교 전사본 일부다. 설교자가 교훈을 {n_items}가지로 나열한다(★항목 표시 문장이
각 항목의 시작).{first_note} 이 나열 전체를 쇼츠 하나로 만든다 — 모든 항목이 반드시 들어가야 하고, 남는 길이 합계는
{budget:.0f}초 이내여야 한다(각 문장 앞 괄호가 그 문장 길이). 시청자는 교회를 안 다니는 사람도 포함한 일반 대중.

## 절대 규칙 — 문장을 한 줄씩 띄엄띄엄 뽑지 말 것
지난 실패: 8분짜리 나열에서 한 문장씩 7개를 떼어 53초를 만들었더니 목차 낭독이 됐다("첫 번째는 ~하나님입니다 →
(5분 건너뜀) → 그런데 하나님께서는 이 연약한 한 여인의 손을… → (건너뜀) → …"). 시청자는 그 여인이 누군지도 모른 채
표지만 듣는다. 잘 된 쇼츠의 점프컷은 조각 하나하나가 '상황 → 대사 → 착지'로 이어지는 덩어리다.
- 항목마다 **연속 블록 하나**(끊기지 않은 2~5문장, 12~22초)를 통째로 남긴다. 블록 = 그 항목에서 시청자가 "내 얘기다"
  하는 대목: "나는 나이가 적어, 나는 실력이 없어…" 같은 나열, 청중에게 던지는 질문, 일상 대사 재연, 찌르는 한 줄과 그 앞뒤.
- ★표지 문장이 블록과 떨어져 있으면 [표지 문장] + [블록] 두 구간. 항목당 구간은 최대 2개, 전체 구간 수 ≤ {max_runs}.
  표지 문장에 항목 이름이 없으면(예: "둘째로요.") 바로 뒤 이름 문장까지 표지로 친다.
- 성경 인물 줄거리(인물 이름이 나오는 서사, 본문 낭독, 시대 배경 설명)는 **통째로** 건너뛰고 그 뒤에 오는 적용 대목을
  잡아라. 줄거리 속 한 문장만 떼어 오면("그 여인의 손을 사용하셨다") 이름을 모르는 시청자에겐 소음이다.
- 예산이 넘치면 블록을 쪼개지 말고 도입·마무리를 먼저 빼라. 그래도 넘치면 각 블록을 2~3문장까지만 줄여라.
- 이음새: 각 구간의 마지막 문장은 끝난 문장이어야 하고, 구간의 첫 문장이 잘려나간 대상을 가리키는 "그 ○○/이 ○○"로
  시작하면 안 된다(그 대상이 나오는 앞 문장부터 구간을 시작하라).

## 구성
- 도입(선택, 1~2문장): 이 나열이 무엇에 대한 답인지 여는 질문·문제 제기("어떻게 하면 ~할까요?", "~하는 세 가지").
  없으면 생략하고 ★항목1부터 시작.
- 항목 1~{n_items}: 위 규칙대로 표지 + 연속 블록. 항목마다 비슷한 분량으로.
- 마무리(선택, 1문장): 전체를 묶는 착지 문장이 있으면.
- 날릴 것: 성경 본문 낭독·배경 설명·같은 말 반복·추임새·광고·인사·"여러분 그렇죠?" 류.
- 블록으로 잡으면 안 되는 것: 이단·타종교·점쟁이·특정 집단을 비판·경계하는 대목("그거 점쟁이지 뭐예요", "조심하세요"),
  정치·국가·전쟁 이야기. 그 항목의 다른 대목(적용·질문·일상 대사)을 찾아라.

입력:
{body}

출력은 반드시 ```json ... ``` 코드블록 안의 JSON 배열(원소 하나)만:
[{{"keep": [[첫 문장 id, 끝 문장 id], ...], "first": 1번 항목 시작 문장 id, "core": 가장 핵심 문장 id, "thesis": "나열 전체의 주제 한 줄",
  "appeal": "교훈", "why": "왜 이 문장들을 골랐는지 한 줄"}}]
keep은 남길 문장 구간(양끝 포함) 목록, 시간순. 출력은 짧게.
※ ★표시가 설교자의 교훈·요점 나열이 아니라 단순 서수(사진 번호, 사건 순서, 인용문 속 "둘째도 겸손")라면
  컷을 만들지 말고 [{{"not_enum": true}}] 만 출력하라."""


def _runs_of(ids: set[int]) -> list[tuple[int, int]]:
    """문장 id 집합 → 연속 구간(inclusive) 목록, 시간순."""
    s = sorted(ids)
    runs: list[tuple[int, int]] = []
    for i in s:
        if runs and runs[-1][1] == i - 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


def _enum_fix_seams(
    keep_ids: set[int], marks: list[int], sentences: list[Sentence], lo: int, hi: int, log: list[str],
) -> set[int]:
    """나열 컷의 조각(연속 구간)들에 일반 컷과 같은 이음새 규칙을 결정론적으로 적용한다(2026-09-29).
      · 표지가 아닌 조각이 한 문장/10초 미만이면 뒤로 늘린다(한 줄씩 떼어 온 '목차 낭독' 방지).
      · 조각의 첫 문장이 잘려나간 대상을 가리키면("그 여인", "이 사람") 대상이 나오는 앞 문장까지 되돌린다.
        못 되돌리면 표지 없는 조각은 버린다.
      · 조각의 마지막 문장이 안 끝났으면(쉼표·접속형 어미) 끝나는 문장까지 늘린다.
    표지 문장은 순서 표지("둘째는…")로 시작하는 게 정상이므로 그 검사만 면제한다."""
    keep = set(keep_ids)
    mark_set = set(marks)

    def _dur(a: int, b: int) -> float:
        return sentences[b].end - sentences[a].start

    def _run_has_mark(a: int, b: int) -> bool:
        return any(a <= m <= b for m in marks)

    # (0) 성경 인물 줄거리 조각 버리기(2026-09-29 사용자: "성경 인물 세부 이야기는 가급적 생략"). 표지 없는 조각의
    #     절반 이상 문장에 인물 이름이 나오면 줄거리다 — 이름을 모르는 시청자에겐 소음. 예산은 _enum_fill_budget이 채운다.
    for a, b in _runs_of(keep):
        if _run_has_mark(a, b) or b - a + 1 < 2:
            continue
        named = sum(1 for i in range(a, b + 1) if _BIBLE_NAME_RE.search(sentences[i].text))
        if named * 2 >= b - a + 1:
            keep.difference_update(range(a, b + 1))
            log.append(f"조각 S{a}~S{b} 버림: 성경 인물 줄거리({named}/{b - a + 1}문장에 인물 이름)")
            continue
        # 남 비판·경계 대목("그거 점쟁이지 뭐예요", 이단·몰몬)은 일반 컷에서 제외하는 것과 같은 이유로 블록에서도 뺀다
        pm = next((_POLEMIC_RE.search(sentences[i].text) for i in range(a, b + 1)
                   if _POLEMIC_RE.search(sentences[i].text)), None)
        if pm:
            keep.difference_update(range(a, b + 1))
            log.append(f"조각 S{a}~S{b} 버림: 남 비판·경계 표지어 '{pm.group(0)}'")

    for _ in range(3):  # 늘리기/되돌리기가 서로를 건드릴 수 있어 몇 번 돌려 안정시킨다
        changed = False
        runs = _runs_of(keep)
        for k, (a, b) in enumerate(runs):
            # (1) 표지 없는 단독 조각 → 뒤로 늘리기
            if not _run_has_mark(a, b) and (b - a + 1 < _ENUM_MIN_BLOCK_SENTENCES or _dur(a, b) < _ENUM_MIN_BLOCK_SEC):
                nb = b
                while (nb - b < _ENUM_EXTEND_CAP and nb + 1 <= hi
                       and (nb - a + 1 < _ENUM_MIN_BLOCK_SENTENCES or _dur(a, nb) < _ENUM_MIN_BLOCK_SEC)):
                    nb += 1
                if nb > b:
                    keep.update(range(b + 1, nb + 1)); changed = True
                    log.append(f"조각 S{a}~S{b} 한 줄뿐 → S{nb}까지 늘림")
                    b = nb
            # (2) 첫 문장이 잘려나간 대상을 가리킴 → 앞으로 되돌리기
            if k > 0 and a not in mark_set:
                pa, pb = runs[k - 1]
                kept_before = " ".join(sentences[i].text for i in sorted(keep) if i < a)
                na = a
                why = _seam_problem(sentences[pb], sentences[na], kept_before)
                steps = 0
                while why and steps < _ENUM_EXTEND_CAP and na - 1 > pb:
                    na -= 1; steps += 1
                    why = _seam_problem(sentences[pb], sentences[na], kept_before)
                if not why and na < a:
                    keep.update(range(na, a)); changed = True
                    log.append(f"조각 S{a} 시작이 잘린 대상을 가리켜 S{na}부터로 되돌림")
                elif why and not _run_has_mark(a, b):
                    keep.difference_update(range(a, b + 1)); changed = True
                    log.append(f"조각 S{a}~S{b} 버림: {why}")
                    continue
            # (3) 마지막 문장이 안 끝남 → 뒤로 늘리기. 중간 조각은 질문으로 끝나도 되지만(이음새 규칙과 동일)
            #     클립의 맨 끝 조각이 질문으로 끝나면 답 없이 끝나는 것이라 안 끝난 것으로 본다.
            is_last = k == len(runs) - 1

            def _open(t: str, last: bool = is_last) -> bool:
                return _is_incomplete(t) and (last or not t.endswith("?"))

            bt = sentences[b].text.strip()
            if _open(bt):
                nb = b
                while nb - b < _ENUM_EXTEND_CAP and nb + 1 <= hi:
                    nb += 1
                    if not _open(sentences[nb].text.strip()):
                        break
                if nb > b and not _open(sentences[nb].text.strip()):
                    keep.update(range(b + 1, nb + 1)); changed = True
                    log.append(f"조각 끝 S{b} 미완('…{bt[-10:]}') → S{nb}까지 늘림")
        if not changed:
            break
    return keep


_ENUM_FILL_SLACK_SEC = 4.0      # 예산에 이만큼도 못 미치면 블록을 늘려 채운다
_ENUM_BLOCK_MAX_SEC = 26.0      # 블록 하나가 이보다 길어지게는 안 늘린다(항목 간 균형)
_ENUM_FILL_CAP = 4              # 블록당 최대 늘리는 문장 수


def _enum_fill_budget(
    keep_ids: set[int], marks: list[int], sentences: list[Sentence], hi: int, budget: float, log: list[str],
) -> set[int]:
    """모델이 예산을 크게 남기면(2026-09-29 실측: 58초 예산에 43초, 블록마다 13초) 각 블록을 뒤로 한 문장씩
    돌아가며 늘려 예산을 채운다 — "나는 나이가 적어, 가진 게 없어, 말을 잘 못해"에서 끊긴 나열을 "실력이 없어,
    경험이 부족해"까지 잇는 식. 표지만 있는 조각은 모델이 일부러 점프한 것이라 안 늘리고, 착지(축원·아멘)로
    끝난 블록, 다음 문장이 순서 표지·구조 표지·추임새면 멈춘다."""
    keep = set(keep_ids)
    mark_set = set(marks)
    grown: dict[int, int] = {}  # 조각 시작 → 늘린 문장 수

    def _total() -> float:
        return sum(sentences[b].end - sentences[a].start for a, b in _runs_of(keep))

    if _total() >= budget - _ENUM_FILL_SLACK_SEC:
        return keep
    progressed = True
    while progressed:
        progressed = False
        for a, b in _runs_of(keep):
            if all(i in mark_set for i in range(a, b + 1)):
                continue  # 표지만 있는 조각
            last = sentences[b].text.strip()
            # 질문으로 끝난 블록은 답이 다음 문장에 있으니 길이·횟수 상한을 넘어서라도 한 문장 더 잇는다
            question = last.endswith("?")
            if not question and (grown.get(a, 0) >= _ENUM_FILL_CAP
                                 or sentences[b].end - sentences[a].start >= _ENUM_BLOCK_MAX_SEC):
                continue
            if _STRONG_LANDING.search(last) or re.fullmatch(r"(>>\s*)?아멘[.!]?", last):
                continue
            nxt = b + 1
            if nxt > hi or nxt in keep:
                continue
            t = sentences[nxt].text.strip()
            if _FILLER_ONLY.match(t) or _enum_number(t) != 0 or _STRUCT_START.match(t):
                continue
            if _total() + (sentences[nxt].end - sentences[b].end) > budget:
                continue
            keep.add(nxt); grown[a] = grown.get(a, 0) + 1; progressed = True
            log.append(f"블록 S{a}~S{b} → S{nxt}까지 늘려 예산 채움")
    return keep


def _fit_enum_keep(
    keep_ids: set[int], marks: list[int], sentences: list[Sentence], budget: float, log: list[str],
    extra_protected: set[int] | None = None,
) -> set[int]:
    """모든 항목 표지 포함을 강제하고, 합계가 budget을 넘으면 가장 긴 항목의 꼬리부터 덜어낸다.
    덜어낸 뒤 조각 끝이 안 끝난 문장이면 그 문장도 같이 덜어낸다(이음새 유지).
    extra_protected: 표지 외에 덜어내면 안 되는 문장(핵심 문장, thesis·why의 구체어가 든 문장) —
    2026-09-29 사용자 "건너뛰기할 때 핵심도 건너뛴다"에 대한 결정론적 보호."""
    protected: set[int] = set(extra_protected or ())
    for m in marks:
        protected.add(m)
        # "둘째로요." 처럼 표지만 있고 항목 이름이 없으면 다음 문장까지 보호한다.
        if len(re.sub(r"\s", "", sentences[m].text)) < 10 and m + 1 < len(sentences):
            protected.add(m + 1)
    missing = protected - keep_ids
    if missing:
        log.append("누락 항목 표지 강제 포함: " + ",".join(f"S{i}" for i in sorted(missing)))
    keep = set(keep_ids) | protected
    keep = {i for i in keep if i in protected or not _FILLER_ONLY.match(sentences[i].text.strip())}

    def _block(i: int) -> int:  # 문장이 속한 항목(도입=0, k번째 항목=k)
        return sum(1 for m in marks if m <= i)

    def _total() -> float:  # 실제 keep_ranges와 같은 셈: 연속 문장 묶음마다 첫 시작~끝 끝
        ids = sorted(keep)
        tot, a = 0.0, None
        for k, i in enumerate(ids):
            a = i if a is None else a
            if k + 1 == len(ids) or ids[k + 1] != i + 1:
                tot += sentences[i].end - sentences[a].start
                a = None
        return tot

    while _total() > budget:
        by_block: dict[int, float] = {}
        for i in keep:
            by_block[_block(i)] = by_block.get(_block(i), 0.0) + sentences[i].end - sentences[i].start
        victim = None
        for b in sorted(by_block, key=lambda b: -by_block[b]):
            cands = [i for i in keep if _block(i) == b and i not in protected]
            if cands:
                # 항목은 꼬리(부연)부터, 도입(b=0)은 머리부터 — 도입은 1번 항목에 붙은 문장이 이어짐을 살린다
                victim = min(cands) if b == 0 else max(cands)
                break
        if victim is None:
            log.append(f"표지 문장만으로 {_total():.0f}초 — 더 줄일 수 없음")
            break
        keep.discard(victim)
        log.append(f"S{victim} 덜어냄(길이 초과)")
        # 꼬리를 덜어낸 조각의 새 끝이 안 끝난 문장이면 그것도 덜어낸다 — 안 그러면 "…인데" 하고 점프한다
        j = victim - 1
        while j in keep and j not in protected and _is_incomplete(sentences[j].text.strip()) \
                and (j == max(keep) or not sentences[j].text.strip().endswith("?")):
            keep.discard(j)
            log.append(f"S{j} 덜어냄(앞 문장이 미완으로 남아)")
            j -= 1
    return keep


def build_enumeration_cuts(
    sentences: list[Sentence], model: str = "", thinking_tokens: int = 4096,
    timeout_sec: int = 600, budget: float = ENUM_BUDGET_SEC, max_groups: int = 2,
    hard_max_sec: float = 90.0,
) -> tuple[list[dict], list[dict]]:
    """나열 묶음마다 컷 dict(일반 컷과 같은 모양: core/start/end/skip + enum=True)를 만든다. (컷들, 디버그 로그)
    budget은 항목 2개 기준이고 항목 수에 따라 _enum_budget으로 늘어난다(hard_max_sec까지)."""
    cuts: list[dict] = []
    logs: list[dict] = []
    base_budget = budget
    for marks, has_first in find_enumerations(sentences)[:max_groups]:
        marks = list(marks)
        n_items = len(marks) + (0 if has_first else 1)
        budget = _enum_budget(n_items, base_budget, hard_max_sec)
        log: list[str] = [("" if has_first else "(1번 표지 없음) ") + "항목 표지: "
                          + " / ".join(f"S{m} {sentences[m].text[:20]}" for m in marks)
                          + f" / 예산 {budget:.0f}초({n_items}항목)"]
        lo = max(0, marks[0] - 12)
        if not has_first:  # 1번 항목을 모델이 찾을 수 있게 2→3 간격만큼 앞을 더 보여 준다
            back = max(240.0, sentences[marks[1]].start - sentences[marks[0]].start)
            while lo > 0 and sentences[marks[0]].start - sentences[lo - 1].start <= back:
                lo -= 1
        hi = marks[-1]
        while hi + 1 < len(sentences) and hi - marks[-1] < 40 and sentences[hi + 1].start - sentences[marks[-1]].start < 240:
            hi += 1
        keep_ids: set[int] = set()
        meta: dict = {}
        try:
            raw = _invoke_claude_json(
                _build_enum_prompt(sentences, lo, hi, marks, budget, has_first), model=model,
                thinking_tokens=thinking_tokens, timeout_sec=timeout_sec, max_clips=1,
            )
            meta = raw[0] if raw and isinstance(raw[0], dict) else {}
            if meta.get("not_enum"):
                log.append("모델 판정: 교훈 나열 아님 → 컷 안 만듦")
                print("[v2] 나열 컷: " + " / ".join(log), flush=True)
                logs.append({"marks": marks, "raw": meta, "log": log, "cut": None})
                continue
            if not has_first:
                try:
                    f0 = int(meta.get("first"))
                    if lo <= f0 < marks[0]:
                        marks.insert(0, f0)
                        log.append(f"모델이 찾은 1번 항목: S{f0} {sentences[f0].text[:20]}")
                except (TypeError, ValueError):
                    log.append("1번 항목을 찾지 못함")
            for rng in meta.get("keep") or []:
                try:
                    a, b = int(rng[0]), int(rng[1])
                except (TypeError, ValueError, IndexError):
                    continue
                keep_ids.update(i for i in range(max(lo, min(a, b)), min(hi, max(a, b)) + 1))
        except QuotaExceededError:
            raise  # 한도 소진은 최소 구성으로 덮지 않는다 — 사용자에게 진짜 원인을 보여야 한다
        except Exception as exc:  # noqa: BLE001 - 모델이 실패해도 표지 문장+직후 문장으로 최소 구성
            log.append(f"모델 실패 → 표지+직후 문장으로 구성: {exc}")
            for m in marks:
                keep_ids.update({m, min(m + 1, len(sentences) - 1)})
        # 핵심 문장은 모델이 keep에 안 넣었어도 강제로 넣고 덜어내기에서 보호한다.
        core = meta.get("core")
        try:
            core = int(core)
            if not (lo <= core <= hi):
                core = marks[0]
        except (TypeError, ValueError):
            core = marks[0]
        if core not in keep_ids:
            log.append(f"핵심 문장 S{core}이 keep에 없음 → 강제 포함")
            keep_ids.add(core)
        keep = _enum_fix_seams(keep_ids, marks, sentences, lo, hi, log)
        keep = _enum_fill_budget(keep, marks, sentences, hi, budget, log)
        # thesis·why에 적은 구체어가 든 문장은 덜어내기에서 보호(모델이 이유로 쓴 문장을 스스로 빼는 실수 방지)
        key = _content_stems(" ".join(str(meta.get(k) or "") for k in ("thesis", "why")))
        key_ids = {i for i in keep if key and sum(1 for t in key if t in sentences[i].text) >= 2}
        keep = _fit_enum_keep(keep, marks, sentences, budget, log, extra_protected={core} | key_ids)
        ids = sorted(keep)
        start, end = ids[0], ids[-1]
        skips: list[tuple[int, int]] = []
        for a, b in zip(ids, ids[1:]):
            if b > a + 1:
                skips.append((a + 1, b - 1))
        if core not in keep:
            core = marks[0]
        cut = {
            "core": core, "start": start, "end": end, "skip": [list(x) for x in skips],
            "appeal": "교훈",  # 모델이 여기에 문장을 써 넣기도 해서 고정한다
            "thesis": str(meta.get("thesis") or sentences[marks[0]].text[:40]),
            "why": f"나열형 교훈 {len(marks)}가지 전부 포함 — " + str(meta.get("why") or ""),
            "enum": True,
        }
        log.append(f"완성: S{start}~S{end}, {len(ids)}문장, {_eff_dur(sentences, start, end, skips):.0f}초")
        print("[v2] 나열 컷: " + " / ".join(log), flush=True)
        cuts.append(cut)
        logs.append({"marks": marks, "raw": meta, "log": log, "cut": cut})
    return cuts, logs


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
    # 나열형 교훈(첫째·둘째·셋째)은 무조건 한 컷(위 4-1). 일반 검증·필터·중복제거를 거치지 않고 맨 앞에 둔다.
    enum_logs: list[dict] = []
    try:
        if on_progress:
            on_progress(0.70, "첫째·둘째·셋째 나열 교훈을 모으는 중...")
        enum_cuts, enum_logs = build_enumeration_cuts(
            sentences, model=model, budget=min(ENUM_BUDGET_SEC, float(max_duration_sec) - 2.0),
            hard_max_sec=float(hard_max_duration_sec),
        )
        fixed = enum_cuts + fixed
    except QuotaExceededError:
        raise
    except Exception as exc:  # noqa: BLE001 - 나열 컷 실패로 일반 선정을 잃지 않는다
        print(f"[v2] 나열 컷 실패: {exc}", flush=True)
    if debug_path is not None:
        try:
            import json
            from pathlib import Path
            dbg = {
                "sentences": [{"idx": s.idx, "start": s.start, "end": s.end, "text": s.text} for s in sentences],
                "raw_cuts": raw_cuts, "verify": verify_logs, "merge_log": merge_log, "polemic_log": polemic_log,
                "bible_log": bible_log, "enum_log": enum_logs,
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
        # 2차 채점이 실패/누락된 클립: 예전엔 전 축 6점(=60점)으로 채웠는데, 정직하게 채점된
        # 클립 상당수가 60점 미만이라 **채점 못 한 클립이 2~3위로 올라가는 순위 역전**이 났다
        # (main.py가 score 내림차순으로 정렬). 이제 명시적으로 바닥 점수를 주고 맨 뒤로 보낸다.
        if r:
            computed = compute_scores(r)
        else:
            computed = compute_scores({"core_score": 1, "hook": 1, "retention": 1, "emotion": 1,
                                       "relatability": 1, "payoff": 1, "quotability": 1})
            print(f"[v2] 클립 {i} 채점 결과 없음 — 최하위로 보냄", flush=True)
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
