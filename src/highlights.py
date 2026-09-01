"""전사본을 로컬 Claude Code에 전달해 하이라이트 구간을 선정한다.

핵심 원칙 (중요, 함부로 바꾸지 말 것):
  하이라이트 판단은 전적으로 Claude가 전사본을 "읽고 이해"해서 내린다.
  audio_peaks의 오디오 에너지 힌트는 프롬프트에 참고용으로만 곁들이며,
  "에너지가 높은 구간 = 하이라이트"라는 기계적 매칭으로 대체해서는 안 된다.
  (배경: 실제 판단 기준은 설교 맥락상 완결되고 울림 있는 메시지인지 여부다.)

두 가지 모드를 지원한다:
  - mode="auto": `claude -p` 서브프로세스를 직접 호출해 완전 자동으로 clips.json 생성
  - mode="manual": 프롬프트 파일만 저장해두고 대기. 사용자가 이 Claude Code 세션에서
    직접 "하이라이트 골라줘"라고 요청하면, 그 세션(나)이 clips.json을 작성한다.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.audio_peaks import PeakHint, format_hints_for_prompt
from src.scoring import compute_scores
from src.transcribe import Transcript


@dataclass
class Clip:
    start: float
    end: float
    title: str          # 훅 오버레이용 짧은 한 줄
    caption: str         # SNS 게시글 본문
    hashtags: list[str]
    reason: str          # 왜 이 구간을 선정했는지 (검토 UI에 표시)
    # 바이럴/핵심 두 축의 분리 채점 (1~10)과 그 교집합 통합 점수(0~100).
    # 기본값 None은 이 필드가 없던 기존 clips.json과도 호환된다.
    score: Optional[float] = None         # 100점 만점 통합 점수(진짜 뜰 확률=핵심×바이럴 교집합). UI 표기/필터 기준
    core_score: Optional[float] = None    # 설교의 진짜 핵심에 얼마나 근접한가 (1~10)
    viral_score: Optional[float] = None   # 스크롤을 멈추고 저장·공유하고 싶은가 (1~10). scoring.py가 세부축에서 계산
    # viral을 이루는 세부 축(1~10). 모델이 이걸 매기면 scoring.py가 viral_score/score를 결정론적으로 계산한다.
    # (예전 clips.json에는 없으므로 기본 None으로 하위호환.)
    hook_score: Optional[float] = None          # 첫 1~2초 훅
    retention_score: Optional[float] = None     # 전진감/죽은 구간 없음
    emotion_score: Optional[float] = None        # 감정 스파이크
    relatability_score: Optional[float] = None   # 공감 "내 얘기"
    payoff_score: Optional[float] = None         # 마무리 펀치라인 착지
    quotability_score: Optional[float] = None    # 인용각
    # 이 클립의 장면 유형("재미"/"감동"/"뜨끔")과 실제 첫/끝 문장(전사본 그대로 인용).
    # 선정 품질 장치: 모델이 '설명 구간'이 아니라 '장면'을 뽑도록 강제하고, hook 점수를
    # 요약이 아닌 실제 첫 문장에 매기게 한다. 검토 UI 상세에도 표시된다.
    # hook_line/payoff_line은 동시에 '경계 앵커'다: 시스템이 이 인용문을 전사본 단어열에서
    # 문자열 매칭으로 찾아 start/end를 그 문장의 실제 발화 시각으로 확정한다
    # (main.anchor_clip_to_quotes). 숫자 추측 + 종결어미 휴리스틱만으로 경계를 잡던 시절
    # "말이 안 끝났는데 뚝 끊기는" 사고가 반복된 것의 구조적 해결책.
    appeal: str = ""
    hook_line: str = ""
    payoff_line: str = ""
    # 이 클립이 시청자에게 주는 새 관점 한 줄(뻔한 권면이면 후보 탈락 규칙과 연동).
    insight: str = ""
    # 인용문 앵커링이 실제로 성공했는지. 성공한 클립은 렌더 단계의 끝 스냅을 미세 조정
    # 수준으로만 허용한다(앵커가 이미 '생각의 완결' 지점이므로).
    anchored: bool = False
    # 대안 제목 후보(편집기에서 한 번에 골라 바꿀 수 있게). title 외에 5개 정도.
    title_candidates: list[str] = field(default_factory=list)
    # 이 클립에 등장하는 고유명사(성경 인물/지명/용어). 최종 자막 정밀 재전사(large-v3)에
    # initial_prompt 힌트로 넣어 유튜브 자동자막이 틀리는 이름(룻·보아스·기드온 등)의 철자를 고정한다.
    keywords: list[str] = field(default_factory=list)
    # 웹 UI의 위치 편집 툴에서 드래그로 조정한, 기본 위치 대비 픽셀 오프셋(렌더 해상도 기준).
    # 기본값 0은 기존 clips.json(이 필드가 없는)과도 호환된다.
    title_offset_x: float = 0.0
    title_offset_y: float = 0.0
    caption_offset_x: float = 0.0
    caption_offset_y: float = 0.0
    # 웹 자막 편집기에서 사용자가 직접 수정/확정한 자막 라인 목록.
    # 각 원소: {"start": 절대초, "end": 절대초, "text": "표시할 자막"}.
    # 비어 있으면(기본) 렌더 시 정밀 재전사로 자막을 만들고, 값이 있으면 재전사를 건너뛰고
    # 이 라인들을 그대로 화면 자막으로 쓴다(사용자 편집이 최우선).
    caption_overrides: list = field(default_factory=list)
    # 화면 처리 모드(편집기에서 선택). ""=config 기본값(fit). "fit"=풀 화면(원본 안 자름),
    # "cover"=화면 확대(세로 꽉 채우되 좌우/하단 잘림).
    fill_mode: str = ""
    # 제목/자막 글꼴 스타일(편집기에서 선택). 빈 문자열/0 이면 config 기본값을 쓴다.
    title_font: str = ""       # ASS Fontname(폰트 내부 family 이름)
    title_size: int = 0        # px (0=기본)
    title_align: str = ""      # "left" | "center" | "right"
    title_spacing: float = 0.0  # 자간(px)
    caption_font: str = ""
    caption_size: int = 0
    caption_align: str = ""
    caption_spacing: float = 0.0
    # 편집기에서 사용자가 영상 구간을 직접 정했으면 True. 이 경우 렌더 시 문장 끝 자동 확장/스냅을
    # 하지 않고 사용자가 정한 start/end를 그대로 쓴다.
    trimmed: bool = False


def build_prompt(
    transcript: Transcript,
    peak_hints: list[PeakHint],
    min_clips: int,
    max_clips: int,
    min_duration_sec: int,
    max_duration_sec: int,
    categories: list[str],
    video_duration_sec: float,
    feedback_block: str = "",
    transcript_is_cleaned: bool = False,
) -> str:
    transcript_text = transcript.to_plain_text_with_timestamps()
    hints_text = format_hints_for_prompt(peak_hints)
    categories_text = "\n".join(f"  - {c}" for c in categories)
    feedback_section = f"\n{feedback_block}\n" if feedback_block else ""
    # 노트북LM 등으로 '다듬어진' 텍스트로 선정할 때, 매끈함에 속아 점수를 올리는 것을 막는다.
    # (실측: 같은 설교를 자동자막→정리된 붙여넣기로 바꾸자 평균 score가 70→78로 부풀었다.
    #  글이 깔끔해진 건 편집 덕이지 전달이 좋아서가 아니다 = 프롬프트가 경고하는 '요약 함정'.)
    cleaned_section = (
        """
### ⚠️ 이 전사본은 '정리·교정된' 텍스트다 (채점에 매우 중요)
이 전사본은 노트북LM 등으로 다듬어져 실제 발화보다 문장이 매끄럽다. 실제 오디오엔
더듬거림·군더더기("음…", "그래서 이제")·반복·말끊김이 그대로 있다. **텍스트가 깔끔하다는
이유로 hook·retention·payoff를 올리지 마라 — 그건 편집 덕이지 전달이 좋아서가 아니다.**
점수는 '실제로 그 초에 귀에 들리는 소리' 기준으로 매겨라. 다듬어진 문장이 매끈해 보일수록
'요약 함정'을 의심하고 오히려 한 단계 낮춰라. (이 경로에서는 8·9가 특히 남발되기 쉽다.)
"""
        if transcript_is_cleaned
        else ""
    )

    return f"""너는 조회수가 잘 나오는 교회 쇼츠를 만드는 최고의 편집자다. {video_duration_sec/60:.0f}분 설교 전체에서
"이 부분만큼은 사람들이 끝까지 보고, 저장하고, 공유할 것"이라 확신하는 진짜 알맹이만 골라낸다.

아래는 {video_duration_sec/60:.0f}분 길이 설교 영상의 전체 전사본이다 (타임스탬프 [HH:MM:SS] 포함).

## 채점 방식 (중요 — 너는 세부 축만 매기고, 통합 점수는 시스템이 계산한다)
좋은 쇼츠는 두 축이 **동시에** 높아야 한다. 하나만 높으면 실패다.
  (A) **설교의 핵심(core)** — 설교자가 진짜 힘줘 말한 알맹이인가? (알맹이 없는 자극 = 낚시, 실패)
  (B) **바이럴(viral)** — 스크롤을 멈추고, 끝까지 보고, 저장·공유하고 싶은가?
너는 아래 축들을 **각각 1~10으로 정직하게** 매기기만 하면 된다. viral_score와 통합 score(0~100)는
시스템이 정해진 공식(가중 기하평균 + 훅 관문)으로 계산하니 **네가 통합 점수를 지어내지 마라.**

### 점수 눈금 (냉정하게 — 대부분은 낮다. 8·9 남발 금지)
지금까지 **모든 후보가 7점 이상으로 뭉쳐 나오는 게 최대 문제**다. 실제로 명장면은 드물다. 아래 기준으로 짜게 매겨라.
  - **1~3**: 평범/약함 (설교엔 좋지만 스크롤 멈추게 하진 못함). **대다수 구간이 여기다.**
  - **4~5**: 쓸 만하지만 특출나지 않음. 애매하면 여기(5)에 두고 위로 올리지 마라.
  - **6~7**: 확실히 강함. 이 설교에서 손에 꼽는 축.
  - **8**: 전체 설교에서 1~2개 나올까 말까 한 정말 강한 지점.
  - **9~10**: "이건 무조건 터진다" 싶은 예외적 명장면. 웬만하면 주지 마라.
평균이 5 근처가 정상이다. 확신이 없으면 **무조건 낮은 쪽**으로. (한 후보의 여러 축이 전부 7~8이면 십중팔구 과대평가다.)
  - core_score (1~10): 설교의 진짜 핵심에 얼마나 근접한가.
  - **hook (1~10): 첫 1~2초 훅. 가장 중요. 이게 낮으면(6 미만) 시스템이 viral을 통째로 깎는다.**
    **반드시 hook_line(클립 첫 문장을 전사본에서 그대로 옮긴 것)만 보고 매겨라** — 네가 정리한
    줄거리가 아니라 실제로 들리는 첫 마디가 스크롤을 멈추는가. 배경 설명·밋밋한 도입·설교
    중간을 툭 자른 느낌이면 3 이하다. 그런 후보는 hook_line이 강한 문장이 되도록 시작점을 옮겨라.
  - retention (1~10): 전진감 있고 죽은 구간이 없는가.
  - emotion (1~10): 감정 스파이크(전율·감동·뜨끔·위로)가 있는가.
  - relatability (1~10): 안 믿는 일반 시청자도 "이거 내 얘기"로 느끼는가.
  - payoff (1~10): 마지막이 펀치라인/울림으로 깔끔히 착지하는가.
  - quotability (1~10): 스샷 떠서 공유할 인용각 문장이 있는가.
점수는 **후보를 버리는 필터가 아니라 우선순위 도구**다. 최종 취사선택은 사람(편집기 UI)이 한다.
그러니 억지로 개수를 채우지도, 약한 걸 감추지도 말고 **정직하게** 매겨라(약하면 낮게).
{cleaned_section}
{feedback_section}

## 실제로 잘 뜨는 숏폼의 핵심 로직 (일반 + 교회 계정 공통 분석 → 이 기준으로 viral_score를 매겨라)
숏폼(틱톡/릴스/쇼츠)이 터지는 원리는 정해져 있다. 아래를 얼마나 만족하는지가 곧 viral_score다.
1. **첫 1~2초 훅이 전부다.** 클립 첫 문장이 궁금증 격차(open loop)·통념 파괴·대담한 선언·강한 호명 중 하나로
   스크롤을 물리적으로 멈춰야 한다. 밋밋하게 시작하면(배경 설명, "오늘 본문은~") 3초 안에 이탈 → 죽음.
2. **리텐션(끝까지 보게 하는 힘)이 랭킹을 만든다.** 전진감이 있고 죽은 구간이 없어야. 늘어지면 감점.
3. **감정 스파이크가 공유를 만든다.** 전율·감동·뜨끔함·위로 — 감정이 확 올라오는 지점일수록 고득점.
4. **"이거 완전 내 얘기"(공감)** — 일반 시청자가 자기 상황(불안·번아웃·관계·죄책감·외로움)으로 느끼면 저장·댓글 폭발.
5. **깔끔한 페이오프.** 열어둔 궁금증/긴장이 마지막에 만족스럽게 닫혀야 한다. 흐지부지 끝 = 공유 0.
6. **한 클립 = 한 메시지.** 뾰족한 하나. 여러 요점을 욱여넣으면 약해진다.
7. **인용각 한마디.** 스크린샷 떠서 공유하고 싶은 문장이 들어있으면 강력하다.
교회 계정 특화로 특히 잘 터지는 것: 세상 통념을 뒤집는 대담한 진리, 아플 때 위로가 되는 재해석,
"지금 ~로 힘든 사람 이거 들어라" 직접 호명, 생생한 간증/비유, 설교자의 감정 고조가 해결된 진리로 착지하는 크레센도.

## 무엇을 뽑는가 — '설명'이 아니라 '장면'을 뽑아라 (가장 중요, 지금까지의 최대 실패 원인)
지금까지의 실패는 **"조리 있게 잘 설명한 구간"을 뽑아온 것**이다. 논리가 완결되고 교훈이
명확해도, 보는 사람이 웃지도·울컥하지도·뜨끔하지도 않으면 그 클립은 그냥 '좋은 말씀'이고
아무도 안 본다. 모든 후보는 아래 셋 중 **최소 하나가 실제 대사에 뚜렷이 존재**해야 하며,
어느 쪽인지 appeal 필드에 표기하라:
  - **재미**: 청중이 실제로 웃(었)을 대목 — 자학 개그, 일상 흉내·대사 재연, 의외의 솔직함,
    빵 터지는 구체적 디테일("아내가 저한테 뭐라는 줄 아세요?")
  - **감동**: 울컥하는 순간 — 구체적 인물 이야기(간증·예화)의 클라이맥스, 설교자 자신의
    실패·눈물 고백, 아픈 사람에게 위로가 훅 들어오는 재해석
  - **뜨끔**: 찔리는 순간 — 시청자의 핑계·위선·회피를 콕 집는 직격, 통념을 뒤집는 선언
후보를 확정하기 전에 하나씩 물어라: **"안 믿는 20대가 이걸 보고 웃거나, 울컥하거나,
뜨끔하는가?"** 셋 다 아니면 다른 축 점수가 높아도 후보에서 빼라(개수가 줄어도 좋다).

### 뽑지 말 것 (아무리 '좋은 말씀'이어도 — 재미없는 클립이 나오는 전형)
- 교리·본문 배경의 **해설/강의** 구간. 설명이 조리 있을수록 오히려 함정이다(요약 함정과 동종).
- "우리는 ~해야 합니다" 식 일반 훈계·권면 — 구체적 장면 없이 당위만 있는 것.
- 요점 **선언**부("첫째, 감사하십시오") — 선언은 장면이 아니다. 장면은 선언 뒤에 이어지는
  예화·이야기 쪽에 있다.

## 절대 선정 금지 주제 (아무리 자극적/바이럴해도 무조건 제외)
이 채널은 **교회 말씀 채널**이다. 알맹이는 "지금 이 사람의 신앙과 삶에 닿는 영적 메시지"이지
역사·정치·국가 서사가 아니다. 아래에 해당하면 core/viral 점수와 무관하게 **후보에서 완전히 빼라.**
- **정치적·논쟁적 소지가 있는 주제**: 특정 국가·민족·정당·정권·이념(좌우, 남북, 반공 등)·전쟁·
  국가의 흥망을 다루거나 미화·정치적으로 해석하는 구간. 시청자를 편 가르게 만들 소지가 조금이라도 있으면 제외.
- **"역사적·국가적 사건 = 하나님의 직접 개입"식 논리 비약 금지**: "하나님이 (이 전쟁/이 나라)를
  이렇게 구하셨다", "이 우연은 사실 하나님의 섭리였다"처럼 특정 역사·정치·국가적 결과를 하나님의
  직접 개입으로 단정하는 주장. 교훈적으로 들려도 비약이고 어색하며 논쟁을 부른다.
  (예: 6·25 때 소련 대사가 배탈로 회의에 빠져 대한민국이 살았다 → 하나님의 섭리 → **이런 유형 금지.**)
- **판별 기준**: "이게 특정 국가·정치가 아니라 아무 나라 누구에게나 통하는 개인의 영적 진리인가?"
  아니라면 빼라. 위로·회개·구원·은혜·믿음·기도·관계·내면의 변화처럼 **보편적이고 개인적인** 것만 남겨라.

## 절대 원칙 (이걸 어기면 실패다)
- **양보다 질.** 억지로 개수를 채우지 마라. 진짜 강한 게 1개뿐이면 1개만 내도 된다.
  평범한 구간을 하나라도 끼워 넣느니, 최고만 내는 게 낫다. (최소 {min_clips}, 최대 {max_clips}개 범위에서
  '정말 뽑을 만한 것'만. 점수는 필터가 아니라 우선순위 도구이니, 낸 후보에는 약하면 약한 대로 정직한 점수를 매겨라.)
- **끝이 흐지부지되면 절대 안 된다 (최우선).** 쇼츠의 성패는 마지막 3초가 좌우한다.
  - 클립의 **마지막 문장 = 이 클립에서 가장 강한 펀치라인/울림/반전**이어야 한다. 여기서 딱 끝내라.
  - 펀치라인 뒤에 붙는 **꼬리를 반드시 잘라내라**: "자, 그러면", "다음으로", "제가 오늘 드리고 싶은 건",
    "그래서 우리가~ 해야 됩니다" 식의 전환어·부연·힘 빠지는 마무리·다음 포인트 도입부는 클립에 넣지 마라.
    이런 꼬리를 물고 있으면 감정이 툭 꺼져서 흐지부지된다.
  - 반대로 결정적 한마디가 나오기 **직전에** 끊는 것도 최악이다. 펀치라인은 온전히 포함하되, 그 뒤는 자른다.
- **완결성은 타협 불가.** 전체 설교를 안 본 사람이 이 클립만 보고도 "무슨 말인지 완전히 이해했고 울림이 있다"가 돼야 한다.
  - 시작: 앞 맥락 없이도 이해되는, 완결된 문장으로 시작. (예: "~라고 하는데" 처럼 앞이 잘린 채 시작 금지)
  - 중간에 "둘째,", "셋째," 같은 설교 구조어로 시작해서 앞 내용을 전제하면 안 된다.

## 훅으로 쓸 만한 첫 문장의 구체적 표현 결 (이런 식으로 시작하는 구간을 노려라)
- **통념 뒤집기**: "사실 기도는 ~가 아닙니다", "믿음은 ~라고 생각하지만 정반대입니다"
- **뜨끔한 지적**: "당신이 그렇게 지친 진짜 이유는", "우리가 착각하는 게 하나 있습니다"
- **직접 호명**: "지금 마음이 불안한 사람 있습니까? 이거 들으세요"
- **누구나 겪는 감정에 이름 붙이기**: 불안, 번아웃, 관계의 상처, 열등감, 죄책감
- **반전 구조**: 예상과 다른 결말/깨달음으로 끝나 여운이 남는 것

## 탐색 앵커: 전사본에서 '장면'이 숨어 있는 곳 (이 순서로 뒤져라)
1. **이야기/예화 구간**: 구체적 인물·시간·장소·대화가 등장하는 곳("제가 아는 집사님이…",
   "어느 날 아들이 저한테…"). 특히 **따옴표 대사가 오가는 재연**이 장면성이 가장 강하다.
   클립은 이야기 전체가 아니라 **긴장이 최고조에 달했다가 반전/깨달음으로 꺾이는 클라이맥스
   한 토막**만 잡아라(도입·배경은 잘라도 클라이맥스만으로 이해되게).
2. **웃음 표지**: 청중 웃음·반응이 언급되는 곳, 설교자가 스스로 웃거나 과장·자학하는 대목.
3. **설교자 1인칭 고백**: "저도 사실은…", "부끄러운 얘기 하나 하면…" — 실패담일수록 강하다.
4. **감정 크레센도**: 같은 말이 반복되며 고조되다가(오디오 에너지 힌트 참고) 진리 선언으로
   착지하는 곳 — 고조 시작부터가 아니라 착지 직전 30초쯤이 클립감이다.
5. **통념 파괴 선언**: "기도는 ~가 아닙니다" — 단, 선언 한 줄만 있으면 약하다. 선언에
   구체적 장면·근거가 바로 붙는 곳만.
- 설교 구조 표지("첫째/둘째/셋째")는 그 **뒤에 나오는 예화를 찾는 지도**로만 써라.
  표지 직후의 요점 선언 자체를 클립으로 잡는 건 위 '뽑지 말 것' 1순위다.
- 교훈 전체(수 분)를 담으라는 게 아니다 — 그 안에서 가장 뾰족한 {min_duration_sec}~{max_duration_sec}초 한 토막만.

## 제목(title) 작성법 — 조회수의 절반. 목표는 단 하나: **처음 보는 사람이 "뜨끔"해서 손가락을 멈춘다.**
title은 영상 맨 위에 고정되는 훅이다. 밋밋하면 아무도 안 누른다. 채널 주인이 여러 번 강조했다:
**제목만 잘 뽑아도 반은 먹고 간다. 안 궁금하고 안 누르고 싶은 제목이 지금까지의 실패다.**

### 이 채널이 원하는 결(반드시 이 결로 뽑아라): **"뜨끔·직접 지적"**
시청자가 방금 속으로 한 **생각·핑계·착각·미루는 행동**을 **콕 집어 2인칭으로 정면으로 던진다.**
"남 얘기"가 아니라 "지금 너 말이야"가 되게 만든다. 이게 손가락을 멈추게 하는 핵심 로직이다.
- **골드 예시(이 톤과 문장 감각을 그대로 배워라):**
  - "설교 들으며 '남편이 들었어야' 하셨죠?"
  - "그 말씀, 옆 사람 아니라 당신 겁니다"
  - "'나 들으라는 말 아니겠지' 했던 당신에게"
  - "당신이 지친 진짜 이유, 아무도 안 짚어줍니다"
  - "기도가 안 되는 게 아니라, 안 하고 있는 겁니다"
- **왜 먹히나:** 시청자의 방어심리(회피·변명·자기합리화·미룸)를 정확히 호명해 "어? 내 얘기잖아" 하고 멈추게 한다.

### 뽑는 법(이 순서대로)
1. 이 클립에서 **시청자가 뜨끔할 한 지점**을 먼저 찾아라 — 그들이 흔히 하는 핑계·착각, 외면하는 진실, 미루는 행동.
2. 그걸 **2인칭·현재·구체**로 정면으로 던져라. "~하셨죠?" / "그거, 당신 겁니다" / "~했던 당신에게" / "~가 아니라 ~하고 있는 겁니다" 같은 직격.
3. **막연하면 실패다.** 추상어 대신 **그 순간의 구체적 장면·대사·상황**(골드 예시처럼)을 넣어라.
4. 8개쯤 빠르게 써보고 그중 "처음 보는 사람 손가락을 가장 확실히 멈출 것" 하나를 title로 골라라.

### 넘지 말아야 할 선(짧게 — 어기면 폐기)
- **사이비·전도지 톤 금지.** "…성경은 이렇게 말합니다 / …그것이 기적입니다 / …하나님이 함께하십니다"처럼 답을 감추고
  훈계·통보하며 끝내는 결은 안 믿는 사람에겐 전단지로 읽힌다. 정면 지적은 하되 설교 결론을 그대로 옮기지 마라.
- **경건성 — 칼끝의 방향이 핵심.** 도발의 칼끝은 **시청자(우리의 회피·착각·게으름)**를 향하게 하라. 하나님·예수·성경·
  믿음·기도를 향하면 안 된다. 한 줄만 봐도 그것들을 부정/조롱하는 것처럼 읽히면 폐기("거짓말/사기/소용없다"류를 믿음에 붙이지 마라).
- **교회 용어·요약체 금지.** "미전도 종족·성화·복음화율" 등 안 믿는 사람이 모르는 말, "~는 ~입니다"식 목차형 요약 금지.
- **추상 명사 단독 금지.** "기적·은혜·축복·함께하심"만 덜렁 쓰지 말고 구체적 상황에 걸어라.
- **내용 요약 의문문 금지 (실제 반복된 실패).** "예수님은 왜 기도했을까요?"처럼 클립 내용을
  3인칭 질문으로 요약한 제목은 궁금하지도 뜨끔하지도 않다 — 설교 제목이지 훅이 아니다.
  제목엔 반드시 **시청자('당신')의 구체적 상황·핑계·행동**이 들어가야 한다. 판별법: 제목에서
  "당신(2인칭)"이 지워져도 말이 되면 실패다.
- 길이 15자 내외, 완결된 한 줄.

### 후보(title_candidates 3개)
전부 위 **"뜨끔·직접 지적"** 결로 뽑되, 서로 다른 각도로 변주하라(예: ①핑계·행동 호명 / ②반전 지적 / ③뜨끔한 질문).
밋밋한 요약이나 다른 결로 빠지지 마라. **title에는 4개(title+후보3) 중 가장 확실히 손가락을 멈출 1등**을 골라 넣어라
(초안을 그대로 넣지 말고, 4개를 다 쓴 뒤 재순위해서 1등 배치).

## 반드시 지킬 품질 규칙 (간단히 — 초 단위 자가검증은 하지 마라, 시간 낭비다)
전사본을 한 번 읽고 강한 구간을 바로 골라라. 아래만 지키면 된다. **각 후보를 초 단위로 다시 대조하며
길게 사고하지 마라** — 정밀한 경계는 사용자가 '만들기'를 누를 때 시스템이 맞춘다.
- **요약 함정(가장 중요)**: 점수를 **네가 머릿속으로 매끈하게 정리한 줄거리**가 아니라 **[start,end] 날것
  대사 그대로**에 매겨라. 매끈하게 바꿔야만 그럴듯해지면 → 실물은 군더더기·더듬거림투성이이니 viral을 낮춰라.
- **장면 검증**: 각 후보의 appeal(재미/감동/뜨끔)이 실제 대사 어디에 있는지 스스로 짚어봐라.
  "메시지가 좋다"는 답이 나오면 그건 장면이 없다는 뜻이다 — 빼라.
- **insight 검증 (뻔하면 탈락)**: insight엔 "이 클립을 본 사람이 새로 얻는 관점 한 줄"을 써라.
  써놓고 보니 "기도해야 한다/믿음을 가져라/감사하라" 같은 **뻔한 권면·요약이면 그 후보를 통째로
  버려라.** 안 믿는 20대가 봐도 "오, 그렇게 볼 수도 있구나" 싶은 **재해석·전복·구체적 진단**만
  통과다. (교회 안에서 핵심인 것과 밖에서 인사이트인 것은 다르다 — 밖 기준으로 판별하라.)
- **hook_line/payoff_line은 전사본에서 글자 그대로 복사하라 (매우 중요 — 재구성 금지).**
  시스템이 이 두 문장을 전사본에서 **문자열로 찾아** 클립의 실제 시작/끝 시각을 확정한다.
  네가 문장을 다듬거나 요약해서 쓰면 시스템이 못 찾아 경계가 어긋난다. 오탈자가 있어도 전사본
  표기 그대로 옮겨라.
  - hook_line = 클립이 시작하는 첫 문장. 따로 읽어봐서 설교 중간을 툭 자른 느낌이면 시작점을 옮겨라.
  - payoff_line = 클립의 **마지막 문장(생각이 완결되는 펀치라인)**. "그래서?"가 안 떠오르는,
    결론·반전·적용이 착지한 바로 그 문장이어야 한다. 이 문장 끝에서 영상이 잘린다고 생각하고 골라라.
- **금지 주제**: 정치·논쟁 소지, "역사·국가적 사건 = 하나님의 직접 개입/섭리" 비약은 점수 무관 **즉시 제외**
  (위 "절대 선정 금지 주제" 참조). 개인의 보편적 영적 메시지가 아니면 남기지 마라.
- **제목 자가검증(위 '뜨끔·직접 지적' 기준)**: title과 후보들을 한 줄씩 읽어 —
  (A) 시청자를 2인칭으로 콕 집어 뜨끔하게 하는가? "그래서 뭐?"·남 얘기·밋밋한 요약이면 다시 써라.
  (B) 사이비·전도지 톤인가? 권위 떡밥("성경은 이렇게 말합니다"류)·추상 명사 통보("그것이 기적입니다"류)면 다시 써라.
  (C) 도발의 칼끝이 하나님·믿음을 향하는가? "거짓말/사기/소용없다"류가 믿음에 붙으면 다시 써라(칼끝은 우리의 회피를 향하게).
- **끝은 '문장의 종결'이 아니라 '생각의 완결'이다 (가장 흔한 실패).** 평서문으로 문장이 끝났어도, 그 문장이
  기대를 걸어놓기만 했으면(원인만 말하고 결론 미도달 / 질문 던지고 답 없음 / "그런데·그래서"로 이어질 기세 /
  예화를 꺼내놓고 적용 없음) 시청자는 "어? 뭔가 더 나올 줄 알았는데" 하며 뚝 끊긴 느낌을 받는다.
  판별: 끝 문장을 읽고 "그래서?"가 떠오르면 미완이다 — **payoff(결론·반전·적용)가 나온 직후**까지 포함하라.
  길이가 걸리면 끝을 당기지 말고 **앞(도입·중복 설명)을 잘라서** 맞춰라. 꼬리 "자, 그러면…"은 붙이지 마라.
  start/end 자체는 어림값이어도 된다(정밀 경계는 시스템이 맞춤).

## 각 클립 형식 조건
- **길이**: 목표 {min_duration_sec}~{max_duration_sec}초. **2026 플랫폼 실측: 22~45초가 스위트스팟이고,
  60초를 넘기면 완주율(확산의 제1신호)이 무너진다.** 그래서 우선순위는:
  ① 완결이 안 되면 **늘리지 말고 더 짧게 완결되는 토막(문장 단위)을 다시 찾아라** — 같은 구간에서
  도입을 잘라내면 대부분 해결된다. ② 그래도 payoff 직전에 끊길 상황이면 그때만 {max_duration_sec}초를
  10~15초쯤 넘겨라(payoff 직전 끊김이 최악인 건 여전하다). 설교 한 단락(2~4분)을 통째로 잡는 건
  금지 — 하나의 뾰족한 메시지가 담긴 **한 토막**만. (상한을 크게 넘는 후보는 시스템이 통째로 버린다.)
- **서로 다른 메시지**: 클립끼리 같은 내용/같은 비유를 반복하지 말 것.

## 오디오 에너지 힌트 (참고용일 뿐. 이것만으로 판단하지 말 것 — 헛기침/잡음일 수도 있음)
{hints_text}

## 전사본
{transcript_text}

## 출력 형식
다른 설명 없이, 아래 JSON 배열만 출력하라 (```json 코드블록으로 감쌀 것).
**배열 순서 = 네가 판단한 강한 순 (0번째가 가장 강력). 통합 score/viral_score는 시스템이 세부 축에서 계산하므로 넣지 마라.**

```json
[
  {{
    "start": 123.4,
    "end": 175.0,
    "appeal": "재미 | 감동 | 뜨끔 중 하나 (이 클립의 장면 유형)",
    "hook_line": "클립 첫 문장 — 전사본에서 글자 그대로 복사 (시스템이 문자열로 찾아 시작 시각 확정)",
    "payoff_line": "클립 마지막 문장(생각이 완결되는 펀치라인) — 전사본에서 글자 그대로 복사 (끝 시각 확정)",
    "insight": "이 클립이 주는 새 관점 한 줄 (뻔한 권면이면 이 후보 자체를 빼라)",
    "core_score": 9,
    "hook": 9,
    "retention": 8,
    "emotion": 8,
    "relatability": 8,
    "payoff": 9,
    "quotability": 7,
    "title": "영상 맨 위 한 줄 제목 ('뜨끔·직접 지적' 결, 시청자를 2인칭으로 콕 집어, 15자 내외)",
    "title_candidates": ["후보1(핑계·행동 호명)", "후보2(반전 지적)", "후보3(뜨끔한 질문)"],
    "caption": "게시글 캡션 — 훅 한 문장만 (길게 쓰지 마라)",
    "hashtags": ["#설교", "#은혜", "#힐링"],
    "keywords": ["룻", "보아스", "나오미"],
    "reason": "선정 이유 + 펀치라인 인용을 한 문장으로"
  }}
]
```
**출력은 짧게 써라 — 출력 글자 수가 곧 사용자의 대기시간이다.** caption 1문장, reason 1문장,
hashtags 3개, keywords는 이 클립에 실제 등장하는 고유명사만 3~5개. 장황하게 쓰지 마라.
- **keywords**: 이 클립 구간에 등장하는 고유명사(성경 인물·지명·용어, 설교 주제어)를 정확한 철자로 적어라.
  최종 자막을 정밀 전사할 때 이 이름들의 철자를 고정하는 힌트로 쓴다(유튜브 자막이 룻→'루시', 기드온→'기도원'처럼
  틀리는 걸 막기 위함). 전사본에 틀리게 적혀 있어도 너는 맥락으로 올바른 표기를 알 것이니 바르게 적어라.
- 세부 축(core_score/hook/retention/emotion/relatability/payoff/quotability)은 모두 1~10 정수. 통합 score는 넣지 마라(시스템이 계산).
- start/end는 전사본 타임스탬프 기준 **초 단위 숫자**로 변환해서 적을 것. end는 반드시 펀치라인이 끝나는 지점이어야 한다.
"""


def _stream_delta(ev: dict) -> tuple[str, str] | None:
    """claude -p stream-json 한 줄(dict)에서 진행률용 델타를 뽑는다.

    반환: ("thinking"|"text", 텍스트조각) 또는 None(진행률과 무관한 이벤트).
    --include-partial-messages가 켜지면 raw API 이벤트가 {"type":"stream_event",
    "event":{...}} 형태로 래핑되어 온다. 형식이 낯설어도 조용히 None을 돌려
    선정 자체는 계속 굴러가게 한다(진행률은 장식, 파이프라인이 본체)."""
    e = ev.get("event") or {}
    if e.get("type") != "content_block_delta":
        return None
    d = e.get("delta") or {}
    if d.get("type") == "thinking_delta":
        return ("thinking", str(d.get("thinking") or ""))
    if d.get("type") == "text_delta":
        return ("text", str(d.get("text") or ""))
    return None


def _extract_json_array(text: str) -> list[dict]:
    """claude -p 응답 텍스트에서 JSON 배열만 뽑아낸다."""
    fenced = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    if not fenced:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"응답에서 JSON 배열을 찾을 수 없습니다:\n{text[:500]}")
        candidate = candidate[start : end + 1]

    return json.loads(candidate)


def _validate_and_build_clips(
    raw_clips: list[dict],
    video_duration_sec: float,
    min_duration_sec: int,
    max_duration_sec: int,
) -> list[Clip]:
    # 길이 상한: 프롬프트로 20~60초를 요구해도 모델이 이따금 설교 한 단락을 통째로 잡아
    # 4분짜리(예: 269초) '쇼츠'를 반환한다(실측: 16OEzTyLTao). 그런 구간은 시작 맥락도
    # 안 맞고 쇼츠도 아니므로, 완결을 위한 여유(1.5배)를 넘으면 후보에서 제외한다.
    # (한 클립이 나쁘다고 raise로 분석 전체를 죽이지 않고, 그 클립만 건너뛴다.)
    hard_max_duration = max_duration_sec * 1.5
    clips: list[Clip] = []
    for i, c in enumerate(raw_clips):
        start = float(c["start"])
        end = float(c["end"])
        if end <= start:
            print(f"[highlights] 클립 {i} 건너뜀: end({end})가 start({start}) 이하")
            continue
        start = max(0.0, start)
        end = min(video_duration_sec, end)
        duration = end - start
        if duration < min_duration_sec * 0.8:
            print(f"[highlights] 클립 {i} 건너뜀: 너무 짧음({duration:.1f}초)")
            continue
        if duration > hard_max_duration:
            print(
                f"[highlights] 클립 {i} 건너뜀: 너무 김({duration:.1f}초 > 상한 {hard_max_duration:.0f}초). "
                f"모델이 설교 단락을 통째로 잡았을 가능성 - 쇼츠로 부적합."
            )
            continue

        def _as_score(v) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        # 세부 축(1~10)으로부터 viral_score/score를 결정론적으로 계산한다(scoring.py).
        # 세부 축이 없는 예전 응답은 c의 viral_score/score를 존중(하위호환).
        computed = compute_scores(c)

        clips.append(
            Clip(
                start=start,
                end=end,
                title=str(c.get("title", "")).strip(),
                caption=str(c.get("caption", "")).strip(),
                hashtags=list(c.get("hashtags", [])),
                reason=str(c.get("reason", "")).strip(),
                score=computed["score"],
                core_score=computed["core_score"],
                viral_score=computed["viral_score"],
                hook_score=_as_score(c.get("hook")),
                retention_score=_as_score(c.get("retention")),
                emotion_score=_as_score(c.get("emotion")),
                relatability_score=_as_score(c.get("relatability")),
                payoff_score=_as_score(c.get("payoff")),
                quotability_score=_as_score(c.get("quotability")),
                appeal=str(c.get("appeal", "")).strip(),
                hook_line=str(c.get("hook_line", "")).strip(),
                payoff_line=str(c.get("payoff_line", "")).strip(),
                insight=str(c.get("insight", "")).strip(),
                title_candidates=[
                    str(t).strip() for t in c.get("title_candidates", []) if str(t).strip()
                ],
                keywords=[str(k).strip() for k in c.get("keywords", []) if str(k).strip()],
            )
        )
    return clips


def select_highlights_auto(
    transcript: Transcript,
    peak_hints: list[PeakHint],
    min_clips: int,
    max_clips: int,
    min_duration_sec: int,
    max_duration_sec: int,
    categories: list[str],
    timeout_sec: int = 900,
    feedback_block: str = "",
    model: str = "",
    transcript_is_cleaned: bool = False,
    thinking_tokens: int = 2048,
    on_progress=None,
) -> list[Clip]:
    """`claude -p` 서브프로세스를 호출해 자동으로 하이라이트를 선정한다.

    model: claude CLI에 넘길 모델 별칭(예: "sonnet"). 빈 값이면 CLI 기본 모델을 쓴다.
    이 단계는 전사본 읽고 판단하는 작업이라 Sonnet으로 충분하며, Opus보다 훨씬 저렴하다.

    on_progress(frac 0.0~0.99, message): 선정 내부의 '진짜' 진행률 콜백. 스트리밍 델타
    (생각/작성)를 받아 호출한다 — 예전엔 이 단계가 깜깜이라 가짜 시간 티커로만 바를 채워
    "진행바가 부정확하다"는 불만의 주원인이었다."""
    prompt = build_prompt(
        transcript=transcript,
        peak_hints=peak_hints,
        min_clips=min_clips,
        max_clips=max_clips,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        categories=categories,
        video_duration_sec=transcript.duration_sec,
        feedback_block=feedback_block,
        transcript_is_cleaned=transcript_is_cleaned,
    )

    claude_path = shutil.which("claude")
    if not claude_path:
        raise RuntimeError("claude CLI를 PATH에서 찾을 수 없습니다 (claude --version으로 설치 확인)")

    cmd = [
        claude_path, "-p",
        # 스트리밍 출력: 생각/작성 델타를 실시간으로 받아 진짜 진행률을 UI로 올린다.
        # (예전 --output-format json은 3~5분 내내 침묵 → 가짜 시간 티커로만 바를 채웠다.)
        # stream-json은 -p 모드에서 --verbose가 필수이고, 델타를 받으려면
        # --include-partial-messages도 필요하다.
        "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        # 기본 claude -p는 Claude Code 시스템 프롬프트+도구 정의(~37k 토큰: 실측 cacheRead
        # 28.5k+write 8.9k)를 통째로 실어 보낸다. 이 작업은 도구가 전혀 필요 없는 단발 텍스트
        # 분석이므로, 시스템 프롬프트를 갈아끼우고 MCP도 끊어 호출당 ~15k 토큰을 아낀다
        # (실측: 동일 미니 호출이 37k → 21.6k 토큰).
        "--system-prompt", "너는 교회 쇼츠 편집 전문가다. 지시받은 형식대로만 출력한다.",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        # 이 작업은 단발 텍스트 분석이라 도구가 전혀 필요 없다. 도구 정의를 아예 빼면
        # (1) 프롬프트가 더 가벼워지고 (2) 모델이 중간에 검색/파일읽기 같은 도구를 쓰며
        # 여러 턴을 도는 경로가 원천 차단된다(선정이 수 분씩 걸린 원인 후보).
        "--tools", "",
    ]
    if model:
        cmd += ["--model", model]  # 비우면 CLI 기본 모델(비쌀 수 있음). config에서 4.5로 고정.
    # 생각(thinking) 상한은 속도↔선정 품질의 트레이드오프. 실측(2026-09-01): 선정 199초의
    # 정체는 출력 생성 9,631토큰(생각 4096 + 후보 JSON ~5.5k)이었다 — 즉 생각 토큰은 초로
    # 직결된다. 품질 장치(장면 앵커·appeal·hook_line)는 프롬프트에 있으므로, 생각은 1024의
    # 2배인 2048로 절충한다. config highlights.thinking_tokens로 조절 가능.
    env = {**os.environ, "MAX_THINKING_TOKENS": str(max(512, int(thinking_tokens)))}

    # 한도 안내문 감지는 반드시 "호출이 실패한 경우"에만 쓴다. 성공 응답(JSON 결과)에도
    # 'resets' 같은 단어가 메타데이터로 들어올 수 있어, 성공 전체를 먼저 문자열 검사하면
    # 4~5분 걸려 성공한 분석을 한도 오류로 오판해 통째로 버리는 치명적 버그가 된다
    # (실측: subtype=success에 클립 JSON까지 있는 응답을 한도 도달로 폐기).
    def _quota_hint(text: str) -> bool:
        t = (text or "").lower()
        return any(s in t for s in (
            "session limit", "usage limit", "hit your", "rate limit", "quota",
            "too many requests", "출력 한도", "한도에 도달",
        ))

    def _raise_quota(raw: str):
        raise RuntimeError(
            "Claude 사용량(세션) 한도에 도달했습니다. 한도가 리셋된 뒤 다시 시도하거나, "
            "구독과 별개인 ANTHROPIC_API_KEY를 설정해 API 경로로 돌리세요.\n"
            f"원문: {raw.strip()[:300]}"
        )

    def _notify(frac: float, msg: str) -> None:
        if on_progress is None:
            return
        try:
            on_progress(min(0.99, max(0.0, frac)), msg)
        except Exception:  # noqa: BLE001 - 진행률 표시 실패가 선정을 죽이면 안 됨
            pass

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        env=env,
        # 프로젝트 폴더에서 실행하면 CLAUDE.md/메모리 등 프로젝트 컨텍스트까지 얹힌다 — 중립
        # 임시 폴더에서 실행해 순수 프롬프트만 보낸다.
        cwd=tempfile.gettempdir(),
    )
    timed_out = threading.Event()

    def _kill_on_timeout() -> None:
        timed_out.set()
        try:
            proc.kill()
        except OSError:
            pass

    watchdog = threading.Timer(timeout_sec, _kill_on_timeout)
    watchdog.start()
    # stderr는 별도 스레드로 비운다(파이프가 차면 프로세스가 멈추는 교착 방지).
    stderr_chunks: list[str] = []
    t_err = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read() or ""), daemon=True
    )
    t_err.start()

    # 진행률 추정 기준. 생각: thinking_tokens 예산 대비(한글 섞임 기준 토큰≈2.5자).
    # 작성: 실측 클립당 JSON ~1000자(다이어트 후) × max_clips. '"start"' 등장 횟수 = 몇 번째
    # 클립을 쓰는 중인지 — 사용자에게 "후보 3번째 작성 중"처럼 진짜 진행을 보여준다.
    think_budget_chars = max(1.0, float(thinking_tokens) * 2.5)
    expect_text_chars = max(1.0, float(max_clips) * 1000.0)
    thinking_chars = 0
    text_chars = 0
    clip_count = 0
    _CLIP_MARK = '"start"'
    scan_tail = ""
    text_parts: list[str] = []
    raw_stdout: list[str] = []
    result_event: dict | None = None
    # stdin 쓰기도 별도 스레드: 긴 프롬프트를 쓰는 동안 자식의 stdout 파이프가 차면
    # 서로 기다리는 교착이 이론상 가능하다 — 읽기(메인)와 쓰기(스레드)를 분리해 원천 차단.
    def _feed_stdin() -> None:
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except OSError:
            pass  # 자식이 먼저 죽은 경우: 아래 returncode/stderr 처리에서 원인이 드러난다

    threading.Thread(target=_feed_stdin, daemon=True).start()
    try:
        _notify(0.02, "AI가 설교 전사본을 읽는 중...")
        for line in proc.stdout:
            raw_stdout.append(line)
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                result_event = ev
                continue
            # CLI가 생각 토큰 추정치를 직접 알려준다(실측 확인: system/thinking_tokens
            # 이벤트의 estimated_tokens). 문자 수 환산보다 정확하므로 이걸 우선 쓴다.
            if ev.get("type") == "system" and ev.get("subtype") == "thinking_tokens":
                est_tok = float(ev.get("estimated_tokens") or 0)
                frac = 0.03 + 0.37 * min(1.0, est_tok / max(1.0, float(thinking_tokens)))
                _notify(frac, "AI가 설교 전체를 읽으며 구간을 고르는 중...")
                continue
            delta = _stream_delta(ev)
            if delta is None:
                continue
            kind, chunk = delta
            if kind == "thinking":
                thinking_chars += len(chunk)
                frac = 0.03 + 0.37 * min(1.0, thinking_chars / think_budget_chars)
                _notify(frac, "AI가 설교 전체를 읽으며 구간을 고르는 중...")
            else:
                text_parts.append(chunk)
                text_chars += len(chunk)
                # 청크 경계에 걸친 마커도 세도록 직전 꼬리를 붙여 검사하되, 꼬리 안에서
                # 이미 센 것은 빼서 이중 집계를 막는다.
                scan = scan_tail + chunk
                clip_count += scan.count(_CLIP_MARK) - scan_tail.count(_CLIP_MARK)
                scan_tail = scan[-(len(_CLIP_MARK) - 1):]
                frac = 0.40 + 0.59 * min(1.0, text_chars / expect_text_chars)
                msg = (
                    f"후보 작성 중... ({min(clip_count, max_clips)}번째 클립)"
                    if clip_count else "후보 작성 중..."
                )
                _notify(frac, msg)
        proc.wait()
    finally:
        watchdog.cancel()
        t_err.join(timeout=2.0)

    stdout_all = "".join(raw_stdout)
    stderr_all = "".join(stderr_chunks)
    if timed_out.is_set():
        raise RuntimeError(
            f"하이라이트 선정이 {timeout_sec}초를 넘겨 중단됐습니다. 잠시 후 다시 시도해 주세요."
        )
    if proc.returncode != 0:
        if _quota_hint(f"{stdout_all}\n{stderr_all}"):
            _raise_quota(stdout_all or stderr_all)
        raise RuntimeError(
            f"claude -p 실행 실패 (exit {proc.returncode}).\n"
            f"stderr: {stderr_all.strip()[:300]}\n"
            f"stdout: {stdout_all.strip()[:300]}"
        )
    outer = result_event
    if outer is None:
        # 구버전 CLI가 stream-json을 무시하고 통짜 JSON(예전 형식)을 냈을 가능성 폴백.
        try:
            outer = json.loads(stdout_all)
        except json.JSONDecodeError:
            if _quota_hint(f"{stdout_all}\n{stderr_all}"):
                _raise_quota(stdout_all or stderr_all)
            raise RuntimeError(
                "claude -p 응답에서 결과 이벤트를 찾지 못했습니다(한도/오류 안내문일 수 있음).\n"
                f"응답: {stdout_all.strip()[:300]}"
            )
    # 선정이 느릴 때 어디서 시간이 갔는지(모델/토큰/소요) 추적할 수 있게 서버 로그에 남긴다.
    # (실측 4분 20초짜리 호출의 내역을 알 수 없어 튜닝이 어림짐작이 됐던 문제 해결.)
    usage = outer.get("usage") or {}
    print(
        f"[highlights] claude -p 완료: model={model or '(cli기본)'} "
        f"duration_ms={outer.get('duration_ms')} api_ms={outer.get('duration_api_ms')} "
        f"turns={outer.get('num_turns')} "
        f"in={usage.get('input_tokens')} out={usage.get('output_tokens')} "
        f"cache_read={usage.get('cache_read_input_tokens')}",
        flush=True,
    )
    # result 이벤트에 텍스트가 비어 있으면 스트리밍으로 모아둔 델타로 폴백한다.
    result_text = outer.get("result", "") or "".join(text_parts)
    if outer.get("is_error") or outer.get("subtype") not in (None, "success"):
        if _quota_hint(result_text):
            _raise_quota(result_text)
        raise RuntimeError(f"claude -p 오류: {str(result_text)[:300]}")
    try:
        raw_clips = _extract_json_array(result_text)
    except ValueError:
        # 성공 형식이지만 result가 클립 배열이 아니라 한도 안내문인 경우(실측 존재).
        if _quota_hint(result_text):
            _raise_quota(result_text)
        raise
    return _validate_and_build_clips(
        raw_clips, transcript.duration_sec, min_duration_sec, max_duration_sec
    )


def save_prompt_for_manual_mode(prompt: str, output_path: Path) -> None:
    """manual 모드: 프롬프트를 파일로 저장해두고, 사용자가 Claude Code 세션에서 직접 요청하도록 안내."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")


# clips.json 접근 락. 이 파일은 "전체 읽기 → 수정 → 전체 덮어쓰기" 방식으로 다뤄지는데,
# 렌더 스레드(main.render_selected)·편집기 저장(web_app.save_clip_position)·재분석이 같은
# 프로세스의 다른 스레드에서 동시에 이 패턴을 밟으면 마지막 저장이 다른 쪽 수정을 통째로
# 덮어쓴다(실제 시나리오: 렌더 도는 몇 분 사이 편집기에서 저장한 자막·제목이 렌더 종료
# 시점의 낡은 사본에 밀려 소실). 읽기-수정-저장 구간은 반드시 이 락 안에서 수행할 것.
CLIPS_LOCK = threading.RLock()


def load_clips_json(path: Path) -> list[Clip]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Clip(**c) for c in data]


def save_clips_json(clips: list[Clip], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(c) for c in clips], ensure_ascii=False, indent=2), encoding="utf-8"
    )
