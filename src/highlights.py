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

## 탐색 앵커: "첫째/둘째/셋째" 구조 표지 (후보를 찾는 지도로 써라)
이 설교들은 대개 교훈을 "첫째, ~", "둘째, ~", "셋째, ~"로 나눠 전개한다. 각 표지 **직후**는
설교자가 그 교훈을 가장 압축해 선언하는 지점이라(완결 문장 + 바로 이어지는 가장 강한 예화/적용),
강한 후보가 숨어 있을 확률이 높다. 전사본에서 이 표지들을 찾아 그 뒤 구간을 반드시 한 번씩 살펴라.
- 단, 클립 자체가 "둘째,"라는 말로 시작하면 앞 내용을 전제하게 되니(위 완결성 규칙),
  경계는 표지 단어를 빼고 그 뒤의 완결된 선언 문장부터 잡아라.
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
- **길이**: 목표 {min_duration_sec}~{max_duration_sec}초. 단, **{max_duration_sec}초를 맞추려고 payoff 직전에
  끊는 게 최악이다** — 생각의 완결(위 규칙)이 우선이니, 완결에 필요하면 {max_duration_sec}초를 15~20초쯤
  넘겨도 된다(시스템 허용 상한 안). 그 이상 길어지면 끝을 자르지 말고 앞(도입·중복 설명)을 잘라 좁혀라.
  설교 한 단락(2~4분)을 통째로 잡는 건 금지 — 하나의 뾰족한 메시지가 담긴 **한 토막**만.
  (상한을 크게 넘는 후보는 시스템이 통째로 버린다.)
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
    "core_score": 9,
    "hook": 9,
    "retention": 8,
    "emotion": 8,
    "relatability": 8,
    "payoff": 9,
    "quotability": 7,
    "title": "영상 맨 위 한 줄 제목 ('뜨끔·직접 지적' 결, 시청자를 2인칭으로 콕 집어, 15자 내외)",
    "title_candidates": ["후보1(핑계·행동 호명)", "후보2(반전 지적)", "후보3(뜨끔한 질문)"],
    "caption": "유튜브/인스타/틱톡 게시글 캡션 (2~3문장, 첫 문장이 훅, 설교 맥락 살려서)",
    "hashtags": ["#설교", "#은혜", "..."],
    "keywords": ["룻", "보아스", "나오미", "맥추감사절"],
    "reason": "핵심 메시지 한 줄 + 마지막 펀치라인 짧게 인용 (2문장 이내로 짧게)"
  }}
]
```
- **keywords**: 이 클립 구간에 등장하는 고유명사(성경 인물·지명·용어, 설교 주제어)를 정확한 철자로 적어라.
  최종 자막을 정밀 전사할 때 이 이름들의 철자를 고정하는 힌트로 쓴다(유튜브 자막이 룻→'루시', 기드온→'기도원'처럼
  틀리는 걸 막기 위함). 전사본에 틀리게 적혀 있어도 너는 맥락으로 올바른 표기를 알 것이니 바르게 적어라.
- 세부 축(core_score/hook/retention/emotion/relatability/payoff/quotability)은 모두 1~10 정수. 통합 score는 넣지 마라(시스템이 계산).
- start/end는 전사본 타임스탬프 기준 **초 단위 숫자**로 변환해서 적을 것. end는 반드시 펀치라인이 끝나는 지점이어야 한다.
"""


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
) -> list[Clip]:
    """`claude -p` 서브프로세스를 호출해 자동으로 하이라이트를 선정한다.

    model: claude CLI에 넘길 모델 별칭(예: "sonnet"). 빈 값이면 CLI 기본 모델을 쓴다.
    이 단계는 전사본 읽고 판단하는 작업이라 Sonnet으로 충분하며, Opus보다 훨씬 저렴하다."""
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
        claude_path, "-p", "--output-format", "json",
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
    # 실측: 선정 지연의 대부분이 첫 토큰 전 '생각(thinking)'이다(Sonnet 5에서 261초).
    # 프롬프트가 이미 "한 번 읽고 바로 골라라"를 요구하므로 상한을 빡빡하게 잡는다.
    # 3000으로도 전사본 붙여넣기 경로가 4분 20초 걸려(실측 2026-08-23) 1024로 더 조인다 —
    # 채점 기준·형식이 프롬프트에 전부 명시돼 있어 긴 사고가 필요 없는 작업이다.
    env = {**os.environ, "MAX_THINKING_TOKENS": "1024"}
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=timeout_sec,
        env=env,
        # 프로젝트 폴더에서 실행하면 CLAUDE.md/메모리 등 프로젝트 컨텍스트까지 얹힌다 — 중립
        # 임시 폴더에서 실행해 순수 프롬프트만 보낸다.
        cwd=tempfile.gettempdir(),
    )
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

    if proc.returncode != 0:
        if _quota_hint(f"{proc.stdout}\n{proc.stderr}"):
            _raise_quota(proc.stdout or proc.stderr or "")
        raise RuntimeError(
            f"claude -p 실행 실패 (exit {proc.returncode}).\n"
            f"stderr: {(proc.stderr or '').strip()[:300]}\n"
            f"stdout: {(proc.stdout or '').strip()[:300]}"
        )
    try:
        outer = json.loads(proc.stdout)
    except json.JSONDecodeError:
        if _quota_hint(f"{proc.stdout}\n{proc.stderr}"):
            _raise_quota(proc.stdout or proc.stderr or "")
        raise RuntimeError(
            "claude -p 응답을 JSON으로 읽지 못했습니다(한도/오류 안내문일 수 있음).\n"
            f"응답: {(proc.stdout or '').strip()[:300]}"
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
    result_text = outer.get("result", "")
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


def load_clips_json(path: Path) -> list[Clip]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Clip(**c) for c in data]


def save_clips_json(clips: list[Clip], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(c) for c in clips], ensure_ascii=False, indent=2), encoding="utf-8"
    )
