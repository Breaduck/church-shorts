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
from src.lyrics_bugs import clean_lyric_lines, fetch_lyrics_from_bugs, normalize_title_key
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
    # 확인 팝업에서 클립을 '분할'하고 일부 조각을 지운 경우, 실제로 남길 구간들(절대초 [start,end]).
    # 비어 있으면 [start,end] 전체를 남긴다. 여러 개면 렌더가 무음 제거와 같은 방식으로 이어붙인다
    # (render._combine_keep → select/aselect 필터). start/end는 이 구간들의 바깥 경계와 일치시킨다.
    keep_ranges: list = field(default_factory=list)
    # 클립 종류. ""(기본)=설교 하이라이트(기존 동작 전부 그대로), "praise"=찬양 곡 통편집.
    # praise 클립은 렌더 시 정밀 재전사·자막·훅 배속·문장 스냅을 모두 건너뛰고
    # 곡 구간 그대로 + 상단 제목(곡 제목)만 넣는다.
    clip_type: str = ""
    # 업로드 찬양 클립 전용: 자막을 발화 싱크에 맞춰 단어별로 색이 바뀌는(카라오케) 효과로
    # 낼지 여부. 기본 False(정적인 흰 자막) — whisper 타이밍이 노래에서 부정확할 수 있어
    # 사용자가 "싱크 맞추기"로 직접 확인·저장한 클립에만 켠다(사용자 요청, 2026-09-05).
    caption_karaoke: bool = False
    # 자막에서 형광(강조)으로 칠할 핵심 단어들. 비어 있으면 keywords로 폴백한다.
    # 정확도 교정 패스(correct_sermon_captions)나 선정 단계가 채운다. 렌더에서
    # 이 단어가 자막에 나오면 노란 형광+볼드로 강조한다(config captions.highlight_* 참고).
    caption_highlights: list = field(default_factory=list)
    # 영어 자막 트랙(caption_overrides와 같은 {start,end,text} 구조, 텍스트만 영어).
    # '영어 자막' 버튼으로 번역해 채우고, 렌더 옵션(caption_lang="en")이면 이걸 쓴다.
    # 한국어(caption_overrides)는 그대로 보존 — "현재 것도 유지, 영어로도" 요청.
    caption_overrides_en: list = field(default_factory=list)
    # 재생 배속(1.0~2.0). 팝업에서 선택. 렌더는 자막까지 구운 완성본에 후처리로 적용
    # (setpts+atempo, 음정 유지)하므로 자막 싱크가 자동으로 함께 배속된다.
    playback_speed: float = 1.0
    # '복제' 직후 한 번만 True. 편집창(팝업/스튜디오)이 이 클립을 열 때 잘라낸 구간
    # 주변이 아니라 원본 영상 전체를 보여주게 하는 힌트다(사용자 요청: 복제본은 같은
    # 장면을 다시 다듬기보다 다른 구간을 고르는 용도로도 쓰이므로). 사용자가 그 편집창을
    # 한 번이라도 열면(preview_info 조회 시) 다시 False로 꺼서, 이후엔 보통 클립처럼
    # 트림된 구간 위주로 보이게 한다(일회성 힌트 — 매번 전체로 열리면 오히려 불편).
    show_full_source_once: bool = False
    # 배경 음악(선택). 스튜디오에서 업로드한 파일을 output/<video_id>/bgm/에 저장하고
    # 여기엔 참조만 남긴다. 없으면 None(기존 clips.json과 호환). 렌더는 이 파일을 클립
    # 길이에 맞춰 반복/트림하고 volume 배율로 원본 오디오와 믹싱한다(_add_sfx와 같은 패턴).
    bgm: Optional[dict] = None


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

## 제0원칙 — **대중적 보편성** (이게 흔들리면 나머지는 의미 없다)
이 쇼츠는 교인이 아니라 **성경을 전혀 모르는 일반 시청자**가 스크롤하다 멈추는 게 목표다.
"요셉 이야기", "베드로 이야기", "존 웨슬리가", "윌리엄 캐리는" — 이렇게 **인물 소개로 시작하는
클립은 성경에 관심 없는 사람이 0.5초 만에 넘긴다.** (실측: 우리 채널의 실패작 대부분이 hook_line이
성경·교회사 인물 이름으로 시작했다.) 뽑을 것은 **누구나 자기 얘기로 느끼는 보편적 삶의 교훈**이다:
외로움("가장 가까운 사람도 내 고난을 몰라준다"), 자기연민, 불안·우울, 백 없음·조건 없음, 실패,
비교·열등감, 관계·부부·자녀, 돈, 타협, 놓친 기회, 내려놓음, 지침. 성경 인물·본문은 그 보편 교훈의
**근거로 뒤에 붙는 것**이지 클립의 주어가 아니다.
- **판별식(보편성)**: "성경을 한 번도 안 읽어본 사람이 첫 문장만 듣고 '내 얘기네' 하는가?"
  아니면 hook은 3 이하, relatability도 3 이하다. 그런 후보는 시작점을 보편 문장 쪽으로 옮기거나 버려라.
- 참고(실측, 잘 되는 교회 쇼츠 채널의 조회수 상위 설교 클립 12개 분석): 길이 40~60초, 첫 문장은
  전부 **보편 질문 또는 보편 선언**("여러분, 왜 고난이 고난일까요?", "사람들은 자기 연민을 굉장히
  좋아합니다", "백이 없어도 괜찮아요", "신앙은 타협하는 게 아니라는 거예요", "이 세상의 모든 기회,
  그건 또 있어요"), 마지막 문장은 설교자의 **축복 선언·"~줄로 믿습니다"·아멘**으로 착지. 성경 인물
  이름으로 시작하는 클립은 상위권에 **하나도 없었다.** 한 클립 = 한 감정 = 한 문장짜리 메시지.

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

## 작업 순서 — 클립을 고르기 전에 '보편 주제 지도'부터 만들어라 (사고 순서 강제)
전사본을 읽으며 **먼저** 다음을 찾아 머릿속에 전부 나열하라(많을수록 좋다):
  1. 설교자가 **삶의 보편적 문제**(외로움·불안·자기연민·실패·비교·관계·돈·타협·지침·백 없음 등)를
     직접 건드리는 문장이 어디 있는가? 특히 **"여러분, 왜 ~일까요?" 식 보편 질문**, "~해도
     괜찮아요"·"~는 ~가 아닙니다" 식 **보편 선언**, "지금 ~한 사람" 식 **직접 호명** 문장을 전부 찾아라.
  2. 그 문장 뒤에서 설교자의 톤이 올라가 **축복 선언·"~줄로 믿습니다"·아멘**으로 착지하는 곳은 어디인가?
  3. 청중이 실제로 웃었거나 울컥했을 대목, 설교자의 1인칭 고백은 어디인가?
그 순간들이 이 설교의 '보편 주제 지도'다. 클립은 이 지도 위의 순간을 hook(첫 문장) 또는
payoff(끝 문장)로 잡는다. 30~40분 설교면 보통 **8~12개**는 나온다 — 지도가 3개에서 멈췄다면
성경 인물 서사 밑에 깔린 보편 문장을 놓친 것이니 다시 훑어라.

## 무엇을 뽑는가 — 보는 사람 마음에 **한 감정**이 확실히 생기는 토막
논리가 완결되고 교훈이 명확해도, 보는 사람 마음에 아무 감정도 안 생기면 그냥 '좋은 말씀'이고
아무도 안 본다. 모든 후보는 아래 중 **최소 하나가 실제 대사에 뚜렷이 존재**해야 하며,
어느 쪽인지 appeal 필드에 표기하라:
  - **위로**: 아픈 사람에게 훅 들어오는 말 — "가장 가까운 사람도 내 고난을 몰라준다", "백이
    없어도 괜찮아요", "내가 없는 것 때문에 안 되는 게 아니에요". **가장 흔하고 가장 잘 통하는 유형.**
  - **선언**: 짧고 단호한 한 줄 진리 — "신앙은 타협하는 게 아닙니다", "하나님이 주신 건 아무도
    못 빼앗아요". 한 문장짜리 20초 클립도 좋다.
  - **뜨끔**: 찔리는 순간 — 시청자의 핑계·위선·회피를 콕 집는 직격("사람들은 자기연민을 좋아합니다"),
    통념을 뒤집는 선언
  - **감동**: 울컥하는 순간 — 간증·예화의 클라이맥스, 설교자 자신의 실패·눈물 고백
  - **재미**: 청중이 실제로 웃(었)을 대목 — 자학 개그, 일상 흉내·대사 재연, 의외의 솔직함
후보를 확정하기 전에 하나씩 물어라: **"성경을 모르는 사람이 이걸 보고 위로받거나, 뜨끔하거나,
울컥하거나, 웃는가?"** 전부 아니면 다른 축 점수가 높아도 후보에서 빼라.

### 뽑지 말 것 (재미없는 클립이 나오는 전형)
- **인물 소개·성경 배경 설명으로 시작하는 구간** ("요셉이 애굽으로 팔려갔죠", "감리교 창시자
  존 웨슬리가 있습니다", "윌리엄 캐리는 근대 선교의 아버지라 불리는데"). 이야기 자체가 좋아도
  hook_line이 이거면 실패다 — 그 이야기가 착지하는 **보편 교훈 문장**에서 시작하거나, 인물 이름
  없이도 이해되는 클라이맥스 한 토막만 잡아라.
- **성경 지식을 전제해야 이해되는 구간** (인물·사건·용어를 이미 아는 사람만 알아듣는 것).
- 교리·본문 배경의 **해설/강의** 구간. 설명이 조리 있을수록 오히려 함정이다(요약 함정과 동종).
- 설교 구조 표지("첫째, ~하십시오")로 시작해 앞 내용을 전제하는 것.
※ 주의: "기도하면 어두움이 떠나갑니다"처럼 **뻔한 권면이어도 보편적 상처(불안·우울·외로움)에
정확히 닿으면 좋은 후보다.** 새로움·반전·인사이트가 없다는 이유로 버리지 마라 — 잘 되는 채널의
상위 클립 대부분이 '새로운 관점'이 아니라 **'내 얘기인 위로'**였다.

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
- **후보는 넓게, 점수는 정직하게.** 최종 취사선택은 사람이 편집기에서 하니, 후보를 미리 줄이지 마라.
  {min_clips}~{max_clips}개를 내되 **가능하면 {max_clips}개 가까이** 내라(30분 설교에서 2~3개만 내는 건 실패다 —
  보편 주제 지도를 다시 훑어라). 대신 점수는 정직하게: 약한 후보는 약한 대로 낮게 매겨 뒤로 보내라.
  단, 정치·역사 금지 주제와 '뽑지 말 것'은 개수를 위해 끼워 넣지 마라.
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

## 탐색 앵커: 전사본에서 클립감이 숨어 있는 곳 (이 순서로 뒤져라)
1. **보편 질문·보편 선언 문장**: "여러분, 왜 ~일까요?", "~해도 괜찮아요", "~는 ~가 아닙니다",
   "사람들은 ~을 좋아합니다" — 이 문장이 곧 hook_line이다. 여기서 시작해 그 문장이 착지하는
   축복 선언/아멘까지가 클립이다. **가장 먼저, 가장 많이 찾아라.**
2. **감정 크레센도 → 축복 착지**: 같은 말이 반복되며 고조되다가(오디오 에너지 힌트 참고)
   "~하시기를 축복합니다", "~줄로 믿습니다", "아멘"으로 착지하는 곳 — 고조 시작부터가 아니라
   착지 직전 20~50초가 클립감이다. 청중의 "아멘" 직후가 자연스러운 끝점이다.
3. **직접 호명**: "지금 ~한 분 계십니까", "~때문에 힘든 사람" — 자기 상황이면 무조건 멈춘다.
4. **설교자 1인칭 고백**: "저도 사실은…", "부끄러운 얘기 하나 하면…" — 실패담일수록 강하다.
5. **이야기/예화의 클라이맥스**: 구체적 대화가 오가는 재연 중 **긴장이 최고조에 달했다가
   깨달음으로 꺾이는 한 토막** — 단, 인물 이름·배경 없이도 이해되는 토막만. 성경 인물 서사는
   그 서사가 착지하는 보편 교훈 문장을 hook으로 세우고 서사는 근거로만 뒤에 붙여라.
6. **웃음 표지**: 청중 웃음·반응이 언급되는 곳, 설교자가 스스로 웃거나 과장·자학하는 대목.
- 설교 구조 표지("첫째/둘째/셋째")는 그 **뒤에 나오는 보편 문장·예화를 찾는 지도**로만 써라.
- 교훈 전체(수 분)를 담으라는 게 아니다 — 그 안에서 가장 뾰족한 {min_duration_sec}~{max_duration_sec}초 한 토막만.
  **한 문장 선언 + 축복 착지로 끝나는 15~25초 클립도 훌륭한 후보다.** 길이를 채우려고 앞에 설명을 붙이지 마라.

## 제목(title) — 여기서는 **초안**만 쓴다 (최종 제목은 별도 단계가 클립 대사만 보고 다시 뽑는다)
title 1개 + title_candidates 4개. 규칙 셋만 지켜라:
- **결론을 제목에 쓰지 않는다** (열린 고리). 제목만 읽고 클립의 답을 알 수 있으면 폐기.
- **이 클립 고유의 구체물 1개** (실제 대사 토막·상황·숫자). 다른 설교에 얹어도 말이 되면 폐기.
- **보는 사람 자신의 얘기**여야 한다. "목사가 임종 때 남긴 말"처럼 남 얘기 티저는 안 누른다 —
  그 장면이 시청자의 어떤 상처·핑계를 건드리는지로 바꿔 써라.
15자 내외 구어체, 교회 용어 금지, 칼끝은 하나님이 아니라 우리의 회피를 향하게.

## 반드시 지킬 품질 규칙
**내용 판단(어느 순간이 진짜 인사이트인가, 제목이 정말 누르고 싶은가)에는 사고를 아끼지 마라 —
품질이 대기시간보다 우선이다.** 단, 초 단위 경계 대조에는 시간을 쓰지 마라: 정밀한 경계는
사용자가 '만들기'를 누를 때 시스템이 hook_line/payoff_line 문자열로 맞춘다.
- **요약 함정(가장 중요)**: 점수를 **네가 머릿속으로 매끈하게 정리한 줄거리**가 아니라 **[start,end] 날것
  대사 그대로**에 매겨라. 매끈하게 바꿔야만 그럴듯해지면 → 실물은 군더더기·더듬거림투성이이니 viral을 낮춰라.
- **감정 검증**: 각 후보의 appeal(위로/선언/뜨끔/감동/재미)이 실제 대사 **어느 문장**에 있는지 스스로
  짚어봐라. "메시지가 좋다"만 나오고 문장을 못 짚으면 감정이 없다는 뜻이다 — 빼라.
- **보편성 검증 (가장 중요)**: hook_line을 따로 읽어라. 성경·교회사 인물 이름, 본문 배경 설명,
  교회 용어로 시작하면 성경 모르는 사람은 바로 넘긴다 → 시작점을 보편 문장으로 옮기거나 버려라.
- **insight**: "이 클립이 시청자 삶의 어떤 문제에 어떻게 닿는가" 한 줄을 써라(예: "가장 가까운
  사람도 내 고난을 몰라준다는 외로움을 정면으로 인정해주는 위로"). 뻔한 권면이어도 **보편적 상처에
  정확히 닿으면 통과**다. 탈락은 "성경 지식이 있어야만 의미가 생기는 것"뿐이다.
- **hook_line/payoff_line은 전사본에서 글자 그대로 복사하라 (매우 중요 — 재구성 금지).**
  시스템이 이 두 문장을 전사본에서 **문자열로 찾아** 클립의 실제 시작/끝 시각을 확정한다.
  네가 문장을 다듬거나 요약해서 쓰면 시스템이 못 찾아 경계가 어긋난다. 오탈자가 있어도 전사본
  표기 그대로 옮겨라.
  - hook_line = 클립이 시작하는 첫 문장. 따로 읽어봐서 설교 중간을 툭 자른 느낌이면 시작점을 옮겨라.
  - payoff_line = 클립의 **마지막 문장(생각이 완결되는 펀치라인)**. "그래서?"가 안 떠오르는,
    결론·반전·적용이 착지한 바로 그 문장이어야 한다. 이 문장 끝에서 영상이 잘린다고 생각하고 골라라.
- **금지 주제**: 정치·논쟁 소지, "역사·국가적 사건 = 하나님의 직접 개입/섭리" 비약은 점수 무관 **즉시 제외**
  (위 "절대 선정 금지 주제" 참조). 개인의 보편적 영적 메시지가 아니면 남기지 마라.
- **제목**: 결론 유출·범용 템플릿·남 얘기 티저면 다시 써라(초안이지만 대충 쓰지 마라 — 별도 단계 실패 시 이게 그대로 나간다).
- **끝은 '문장의 종결'이 아니라 '생각의 완결'이다 (가장 흔한 실패).** 평서문으로 문장이 끝났어도, 그 문장이
  기대를 걸어놓기만 했으면(원인만 말하고 결론 미도달 / 질문 던지고 답 없음 / "그런데·그래서"로 이어질 기세 /
  예화를 꺼내놓고 적용 없음) 시청자는 "어? 뭔가 더 나올 줄 알았는데" 하며 뚝 끊긴 느낌을 받는다.
  판별: 끝 문장을 읽고 "그래서?"가 떠오르면 미완이다 — **payoff(결론·반전·적용)가 나온 직후**까지 포함하라.
  길이가 걸리면 끝을 당기지 말고 **앞(도입·중복 설명)을 잘라서** 맞춰라. 꼬리 "자, 그러면…"은 붙이지 마라.
  start/end 자체는 어림값이어도 된다(정밀 경계는 시스템이 맞춤).

## 각 클립 형식 조건
- **길이: 완결이 항상 길이보다 우선이다.** 목표 {min_duration_sec}~{max_duration_sec}초.
  잘 되는 채널의 상위 클립은 40~60초가 주류지만, **보편 선언 한 문장 + 축복 착지로 끝나는 15~25초
  클립도 좋다** — 짧다고 앞에 설명을 붙여 늘리지 마라. 60초를 좀 넘겨도 괜찮다 — **최악은 긴 클립이
  아니라 생각이 완결되기 전에 중간에서 뚝 끊긴 클립이다.** 길이를 목표에 맞추려고 예화의 허리나
  payoff 직전을 자르지 마라. 우선순위:
  ① 하나의 생각(긴장 → 반전/깨달음 → 착지)이 온전히 담기는 경계를 먼저 잡아라.
  ② 그게 {max_duration_sec}초를 넘으면 **끝을 당기지 말고 도입·중복 설명을 앞에서 잘라라.**
  ③ 그래도 넘으면 {max_duration_sec}초를 15~20초까지 넘겨도 된다(완결 시에만).
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
    "appeal": "위로 | 선언 | 뜨끔 | 감동 | 재미 중 하나 (이 클립이 만드는 감정)",
    "hook_line": "클립 첫 문장 — 전사본에서 글자 그대로 복사 (시스템이 문자열로 찾아 시작 시각 확정)",
    "payoff_line": "클립 마지막 문장(생각이 완결되는 펀치라인) — 전사본에서 글자 그대로 복사 (끝 시각 확정)",
    "insight": "이 클립이 시청자 삶의 어떤 문제에 어떻게 닿는가 한 줄 (성경 모르는 사람 기준)",
    "core_score": 9,
    "hook": 9,
    "retention": 8,
    "emotion": 8,
    "relatability": 8,
    "payoff": 9,
    "quotability": 7,
    "title": "영상 맨 위 한 줄 제목 초안 (결론 숨김·클립 고유 구체물·시청자 자신의 얘기, 15자 내외)",
    "title_candidates": ["후보1", "후보2", "후보3", "후보4"],
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
    raw_clips = _invoke_claude_json(
        prompt, model=model, thinking_tokens=thinking_tokens,
        timeout_sec=timeout_sec, on_progress=on_progress, max_clips=max_clips,
    )
    return _validate_and_build_clips(
        raw_clips, transcript.duration_sec, min_duration_sec, max_duration_sec
    )


def _clip_excerpt(transcript: Transcript, start: float, end: float, pad: float = 3.0) -> str:
    """클립 구간(±pad초)의 전사 문장만 이어붙인다 — 제목 단계에 '그 클립의 실제 대사'만 보여주기 위함."""
    parts = [
        seg.text.strip()
        for seg in transcript.segments
        if seg.end > start - pad and seg.start < end + pad and seg.text.strip()
    ]
    return " ".join(parts)


def build_title_prompt(items: list[dict]) -> str:
    """제목 전용 프롬프트. items: [{index, excerpt, hook_line, payoff_line, insight, appeal, draft_title, draft_candidates}]

    왜 별도 단계인가(2026-09-07): 선정 프롬프트 안에서 제목은 클립당 15개 필드 중 하나, 10개
    클립·경계·채점과 사고 예산을 나눠 쓰는 '곁다리'였다. 실측 결과물이 (1) "교회 키운 목사,
    임종 때 남긴 말"·"구두닦이가 벽에 붙여둔 지도" 같은 **남 얘기 티저**(궁금하긴 한데 내 일이
    아님), (2) "지금 포기하고 싶은 당신에게" 같은 **아무 설교에나 붙는 콜아웃**, (3) "죽으면
    그만이라는 착각" 같은 **결론 노출**로 뭉쳐 나와 사용자가 GPT로 다시 돌리는 지경이었다.
    이 단계는 클립 대사만 놓고 제목 하나에 사고를 다 쓴다."""
    blocks = []
    for it in items:
        cands = " / ".join(it.get("draft_candidates") or [])
        blocks.append(
            f"### 클립 {it['index']}  (감정 유형: {it.get('appeal') or '-'})\n"
            f"- 첫 문장: {it.get('hook_line') or '-'}\n"
            f"- 끝 문장(펀치라인): {it.get('payoff_line') or '-'}\n"
            f"- 이 클립이 닿는 시청자 문제: {it.get('insight') or '-'}\n"
            f"- 초안 제목(참고만, 대개 약함): {it.get('draft_title') or '-'} / {cands}\n"
            f"- 클립 대사 전문:\n{it.get('excerpt') or '-'}\n"
        )
    clips_text = "\n".join(blocks)
    return f"""너는 한국 유튜브 쇼츠 제목만 10년 쓴 카피라이터다. 아래 각 클립에 대해 **보는 사람이 안 누르고는
못 배기는 제목 5개**를 뽑는다. 이 제목은 세로 영상 맨 위에 클립 내내 고정 표시되는 훅이다.

## 지금까지 우리 제목이 안 눌리는 이유 (실제 실패작 — 이 유형이면 폐기)
1. **남 얘기 티저**: "교회 키운 목사, 임종 때 남긴 말", "척추 망가져 침대에 누운 선생이 쓴 책",
   "구두닦이가 벽에 붙여둔 지도", "베드로를 돌아서게 한 한마디", "13년 억울하게 산 그가 한 말".
   궁금하긴 한데 **내 일이 아니다.** 사람은 자기 이익·자기 위협·자기 비밀에만 손가락을 멈춘다.
   누군지도 모르는 목사·선생·구두닦이의 이야기는 0.3초 만에 넘긴다.
2. **아무 설교에나 붙는 콜아웃**: "지금 포기하고 싶은 당신에게", "돈·실적에 무시당한 당신에게",
   "아무도 날 안 봐준다는 당신에게". 정보량 0. 위로 예고편이지 궁금증이 아니다.
3. **결론 노출·설교 요지**: "죽으면 그만이라는 착각", "기도 안 하면 다 그림의 떡",
   "하나님이 널 그냥 두시는 이유". 답을 다 말했거나 훈계라 볼 이유가 없다.
4. 후보 5개가 전부 같은 결("~당신에게" 3개) — 고를 게 없다.

## 미치도록 궁금한 제목의 3요소 — **셋이 동시에** 걸려야 한다
- **나(자기 관련성)**: 읽는 순간 "나?" 소리가 나야 한다. 시청자의 상처·핑계·비밀·습관·관계·돈·몸.
  3인칭 인물은 제목에서 빼고, 그 인물이 건드리는 **시청자 자신의 상황**으로 바꿔라.
  (예: 요셉 13년 억울함 → "억울한 일, 왜 하필 나한테만 올까")
- **격차(정보 구멍)**: 답이 있다고 알려주되 답은 숨긴다. '이것'·'한 가지'·'진짜 이유'·'한마디'·숫자·
  "~하는 사람 특징"·"~하면 안 되는 이유". 구멍이 **좁고 구체적**일수록 강하다("비결" 같은 뭉뚱그림은 약함).
- **긴장(모순·금기·위협·반전)**: 상식과 부딪히거나("잘 풀리는 게 더 위험"), 금지하거나("절대 ~하지 마세요"),
  들키거나("아무한테도 말 안 한 그 생각"), 손해를 걸거나("이거 모르면 평생 ~").

## 잘 먹히는 장치 (클립마다 서로 다른 장치를 골라 5개를 만들어라)
① **속마음 들킴** — 시청자가 속으로만 한 말을 1인칭으로 받아쓴다: "나만 힘든 거 같을 때", "괜히 기도했네"
② **모순 선언** — 상식을 뒤집는 한 줄: "문제를 없애달라고 기도하지 마라", "술술 풀리는 인생이 무서운 이유"
③ **금기·경고** — 흔한 행동을 하지 말라고만: "이 말 하는 순간 관계 끝납니다", "이런 기도 절대 하지 마세요"
④ **콕 집는 진단** — 뭐가 문제인지 안 알려주고 있다고만: "교만의 증거가 뭔지 아세요?", "지친 진짜 이유는 따로 있어요"
⑤ **좁은 격차** — 숫자·'한 가지'·'한마디': "하나님이 사람 볼 때 보는 딱 하나", "부부싸움 멈추는 유일한 방법"
⑥ **대상 콜아웃 + 반전** — 호명하되 뒤에 예상 밖 꼬리: "믿지 않는 자녀 둔 부모, 이거 하지 마세요"
   (호명만 하고 끝나는 "~한 당신에게"는 폐기)
⑦ **미완성 대사** — 클립의 결정적 문장을 중간에서 끊는다: "'너 그러다…' 그다음 말이" — 단, 그 대사가 시청자 얘기일 때만
⑧ **질문 던지기** — 시청자의 상처를 겨냥한 질문: "왜 고난이 고난일까요?", "왜 하필 나한테만 이럴까"
(참고: 잘 되는 교회 쇼츠 채널의 상위 클립은 "왜 고난이 고난일까요?" "백이 없어도 괜찮아요!" "딱 맞아떨어지면
무조건 하나님의 뜻일까?"처럼 **클립 첫 문장이 그대로 제목**인 경우가 많았다 — 그 첫 문장이 보편 질문/모순
선언이면 인용 그대로가 최고의 후보다. 단 축복·결론 문장 인용은 안 된다.)

## 작업 순서 (클립마다)
1. 대사 전문을 읽고 **이 클립이 시청자의 어떤 상처/핑계/통념을 건드리는지** 한 줄로 적어라(성경 인물 이름 없이).
2. 위 장치 중 서로 다른 것으로 **10개** 드래프트.
3. **클릭 시뮬레이션**: 교회 안 다니는 27세 직장인 / 40대 주부 / 교회 청년 셋이 스크롤하다 이 제목을 봤다고
   상상하고, 각각 멈출지 1~10으로 매겨라. 셋 중 둘 이상 8 미만이면 버려라. "나?" 소리가 안 나면 8이 될 수 없다.
4. 살아남은 것 중 **장치가 서로 다른 5개**를 고르고, 1등을 title로.
5. 폐기 검사: 결론 노출? 다른 설교에 얹어도 말이 됨? 남 얘기? 교회 용어(성화·복음화·미전도)? 전도지 톤?
   하나님·믿음·기도를 조롱하는 것으로 읽힘? — 하나라도 걸리면 그 후보는 빼고 다시.

## 형식
- 15자 내외(최대 20자), 구어체, 완결된 한 줄. 따옴표·물음표·말줄임표 사용 가능. 이모지 금지.
- 생활 명사(부부·자녀·돈·출근·잠·카톡·시댁·월급) 하나가 들어가면 강해진다.

{clips_text}

## 출력 형식
다른 설명 없이 아래 JSON 배열만 출력하라 (```json 코드블록). 클립 index마다 정확히 하나씩.
```json
[
  {{"index": 0, "title": "1등 제목", "title_candidates": ["2등", "3등", "4등", "5등"]}}
]
```
"""


def refine_titles(
    clips: list[Clip],
    transcript: Transcript,
    model: str = "",
    thinking_tokens: int = 3072,
    timeout_sec: int = 600,
    on_progress=None,
) -> int:
    """선정된 클립들의 title/title_candidates를 제목 전용 패스로 다시 뽑아 덮어쓴다.

    반환: 제목이 교체된 클립 수. 호출·파싱 실패는 예외로 올리므로 호출자가 try로 감싸
    초안 제목으로 폴백한다(제목 때문에 파이프라인이 죽지 않게)."""
    if not clips:
        return 0
    items = []
    for i, c in enumerate(clips):
        items.append({
            "index": i,
            "excerpt": _clip_excerpt(transcript, c.start, c.end),
            "hook_line": c.hook_line,
            "payoff_line": c.payoff_line,
            "insight": c.insight,
            "appeal": c.appeal,
            "draft_title": c.title,
            "draft_candidates": list(c.title_candidates or []),
        })
    prompt = build_title_prompt(items)
    raw = _invoke_claude_json(
        prompt, model=model, thinking_tokens=thinking_tokens,
        timeout_sec=timeout_sec, on_progress=on_progress, max_clips=len(clips),
    )
    replaced = 0
    for r in raw:
        try:
            idx = int(r.get("index"))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(clips)):
            continue
        title = str(r.get("title", "")).strip()
        cands = [str(t).strip() for t in (r.get("title_candidates") or []) if str(t).strip()]
        if not title:
            continue
        c = clips[idx]
        # 초안 제목도 후보 꼬리에 남겨 사용자가 편집기에서 비교·복구할 수 있게 한다.
        old = [c.title] + list(c.title_candidates or [])
        merged = []
        for t in cands + old:
            if t and t != title and t not in merged:
                merged.append(t)
        c.title = title
        c.title_candidates = merged[:6]
        replaced += 1
    print(f"[highlights] 제목 전용 패스: {replaced}/{len(clips)}개 교체", flush=True)
    return replaced


def _invoke_claude_json(
    prompt: str,
    model: str = "",
    thinking_tokens: int = 2048,
    timeout_sec: int = 900,
    on_progress=None,
    max_clips: int = 6,
    allowed_tools: str = "",
) -> list[dict]:
    """`claude -p`에 프롬프트를 보내 JSON 배열 응답을 받아 파싱한다.

    select_highlights_auto(설교)와 select_praise_songs(찬양)가 공유하는 실행부 —
    스트리밍 진행률/워치독/한도 감지/JSON 추출까지 동일하게 처리한다.

    allowed_tools: 빈 문자열(기본)이면 도구를 전부 끈다(단발 텍스트 분석 최적화).
    "WebSearch"처럼 지정하면 그 도구만 허용·자동승인한다 — 가사 확인처럼 모델 기억만으론
    부족해 실제 인터넷 검색이 필요한 경우용(사용자 요청 2026-09-05)."""
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
    ]
    if allowed_tools:
        # 지정한 도구만 허용 + 자동 승인(-p 모드는 대화형 승인 불가). WebSearch면 모델이
        # 실제 인터넷 검색으로 사실 확인(정식 가사 등)을 하고 답한다 — 턴이 몇 번 늘지만
        # 기억 기반 오답(비슷한 다른 곡 가사 등)을 크게 줄인다.
        cmd += ["--tools", allowed_tools, "--allowedTools", allowed_tools]
    else:
        # 이 작업은 단발 텍스트 분석이라 도구가 전혀 필요 없다. 도구 정의를 아예 빼면
        # (1) 프롬프트가 더 가벼워지고 (2) 모델이 중간에 검색/파일읽기 같은 도구를 쓰며
        # 여러 턴을 도는 경로가 원천 차단된다(선정이 수 분씩 걸린 원인 후보).
        cmd += ["--tools", ""]
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
        _notify(0.02, "AI가 전사본을 읽는 중...")
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
                _notify(frac, "AI가 전사본 전체를 읽으며 구간을 고르는 중...")
                continue
            delta = _stream_delta(ev)
            if delta is None:
                continue
            kind, chunk = delta
            if kind == "thinking":
                thinking_chars += len(chunk)
                frac = 0.03 + 0.37 * min(1.0, thinking_chars / think_budget_chars)
                _notify(frac, "AI가 전사본 전체를 읽으며 구간을 고르는 중...")
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
        return _extract_json_array(result_text)
    except ValueError:
        # 성공 형식이지만 result가 클립 배열이 아니라 한도 안내문인 경우(실측 존재).
        if _quota_hint(result_text):
            _raise_quota(result_text)
        raise


def build_praise_prompt(
    transcript: Transcript,
    min_duration_sec: int,
    max_duration_sec: int,
    video_duration_sec: float,
) -> str:
    """전체 예배 실황 전사본에서 '찬양 곡' 각각의 구간과 제목을 찾는 프롬프트.

    설교 하이라이트 선정과 달리 바이럴 채점이 아니라 '구간 분할 + 곡 식별'이 과제다.
    노래는 자동 전사가 가사를 띄엄띄엄 받아적으므로, 반복 후렴/시적 표현/짧은 줄 패턴으로
    노래 구간을 알아보게 하고, 설교·기도·광고·멘트는 제외하도록 명시한다."""
    transcript_text = transcript.to_plain_text_with_timestamps()
    return f"""너는 교회 예배 실황 영상을 편집하는 전문가다. 아래는 약 {video_duration_sec/60:.0f}분짜리 '전체 예배 실황'의
전사본이다 (타임스탬프 [HH:MM:SS] 포함). 이 영상에는 회중 찬양(다같이 부르는 찬양), 성가대/특송 찬양,
설교, 기도, 성경봉독, 광고·멘트가 섞여 있다.

## 과제
'찬양 곡' 하나하나를 각각 정확히 하나의 구간으로 찾아라. 곡별로 세로 영상(쇼츠 형태)으로 잘라 올릴 것이다.

## 전사본에서 노래를 알아보는 법 (중요)
- 노래(찬양)는 자동 전사가 가사를 띄엄띄엄·부정확하게 받아적는다. 반복되는 후렴 구절, 시적 표현
  ("주님", "은혜", "찬양", "영광" 등), 짧고 리듬감 있는 줄이 이어지면 노래일 가능성이 크다.
- 말(설교/기도/광고)은 문장이 산문적이고 논리가 이어진다. "다음 찬양은…", "다 같이 일어나서" 같은
  멘트는 곡 사이의 안내이지 곡이 아니다.
- 같은 곡 안에서 가사 인식이 몇십 초씩 비는 구간(간주)이 있어도 한 곡으로 묶어라.
  반대로 서로 다른 곡을 하나로 합치지 마라(후렴 가사가 완전히 달라지면 새 곡이다).

## 각 곡의 필드
- "start": 그 곡의 가사가 처음 들리는 시각(초, 숫자). 전주는 시스템이 자동으로 몇 초 앞을 붙이니 가사 기준으로.
- "end": 마지막 가사가 끝나는 시각(초, 숫자).
- "title": 곡 제목. 가사로 곡을 알아볼 수 있으면 정확한 원제("주 은혜임을" 등)를 써라.
  확신이 없으면 대표 가사 한 소절을 제목으로 써라(예: "내 영혼이 은총 입어").
- "song_type": "다같이" | "성가대" | "특송" 중 하나. 성가대 곡은 보통 설교 직전에 있고 화음 합창이다. 애매하면 "다같이".
- "first_line": 구간 첫머리에 실제로 들리는 가사(전사본에서 그대로 인용) — 경계 검증용.
- "last_line": 구간 끝에 실제로 들리는 가사(전사본에서 그대로 인용).
- "confidence": 이 구간이 정말 한 곡의 찬양이라는 확신(1~10).
- "caption": 업로드용 한 줄 설명(곡 제목 + 예배 맥락, 담백하게).
- "hashtags": 5개 내외(#찬양 #ccm #교회 등 + 곡 관련).
- "reason": 왜 이 경계인지 한 문장.

## 규칙
- 곡 길이는 보통 {min_duration_sec}~{max_duration_sec}초다. {min_duration_sec}초보다 훨씬 짧은 조각은 멘트/간주로 보고 제외하라.
- 설교·기도·성경봉독·광고는 절대 곡으로 포함하지 마라.
- 곡을 빠뜨리지 마라: 예배 앞부분 경배와 찬양(보통 여러 곡 연속)과 성가대 찬양을 모두 찾아라.
- 시간순(start 오름차순)으로 정렬해서 출력하라.

전사본:
{transcript_text}

출력은 반드시 ```json ... ``` 코드블록 안의 JSON 배열만. 다른 설명은 쓰지 마라."""


def _build_praise_clips(
    raw: list[dict],
    video_duration_sec: float,
    min_duration_sec: int,
    max_duration_sec: int,
    pad_start_sec: float = 4.0,
    pad_end_sec: float = 6.0,
) -> list[Clip]:
    """찬양 곡 감지 결과를 Clip 목록으로 변환한다.

    - 가사 기준 경계에 전주/여운 패딩을 붙이되, 이웃 곡 경계는 침범하지 않는다.
    - trimmed=True: 렌더의 문장 끝 스냅/자동 확장은 노래에 무의미하므로 경계 그대로 쓴다.
    - score는 채점 개념이 없으므로 None, 배열 순서는 시간순(곡 순서)이다."""
    items: list[dict] = []
    for i, c in enumerate(raw):
        try:
            start = max(0.0, float(c["start"]))
            end = min(video_duration_sec, float(c["end"]))
        except (KeyError, TypeError, ValueError):
            print(f"[praise] 곡 {i} 건너뜀: start/end 파싱 실패")
            continue
        dur = end - start
        if dur <= 0 or dur < min_duration_sec * 0.5:
            print(f"[praise] 곡 {i} 건너뜀: 너무 짧음({dur:.0f}초)")
            continue
        if dur > max_duration_sec * 1.5:
            print(f"[praise] 곡 {i} 건너뜀: 너무 김({dur:.0f}초) - 여러 곡을 합쳤을 가능성")
            continue
        items.append({**c, "start": start, "end": end})
    items.sort(key=lambda c: c["start"])

    clips: list[Clip] = []
    for i, c in enumerate(items):
        # 전주/여운 패딩: 앞 곡 끝·뒤 곡 시작을 넘지 않는 선에서 붙인다.
        prev_end = items[i - 1]["end"] if i > 0 else 0.0
        next_start = items[i + 1]["start"] if i + 1 < len(items) else video_duration_sec
        start = max(prev_end, c["start"] - pad_start_sec)
        end = min(next_start, c["end"] + pad_end_sec, video_duration_sec)
        song_type = str(c.get("song_type", "") or "다같이").strip()
        # 모델이 hashtags를 배열 대신 "#찬양 #교회" 문자열로 줄 때가 있다(실측) —
        # 문자열을 그대로 이터레이션하면 글자 하나하나가 태그가 되므로 공백 분리로 방어.
        raw_tags = c.get("hashtags", [])
        if isinstance(raw_tags, str):
            raw_tags = raw_tags.split()
        clips.append(
            Clip(
                start=start,
                end=end,
                title=str(c.get("title", "")).strip() or f"찬양 {i+1}",
                caption=str(c.get("caption", "")).strip(),
                hashtags=[str(h).strip() for h in raw_tags if str(h).strip()],
                reason=f"[{song_type}] " + str(c.get("reason", "")).strip(),
                appeal=song_type,
                hook_line=str(c.get("first_line", "")).strip(),
                payoff_line=str(c.get("last_line", "")).strip(),
                trimmed=True,
                clip_type="praise",
            )
        )
    return clips


def select_praise_songs(
    transcript: Transcript,
    min_duration_sec: int,
    max_duration_sec: int,
    pad_start_sec: float = 4.0,
    pad_end_sec: float = 6.0,
    model: str = "",
    thinking_tokens: int = 4096,
    timeout_sec: int = 900,
    on_progress=None,
) -> list[Clip]:
    """전체 예배 실황 전사본에서 찬양 곡별 구간을 자동 감지한다(claude -p)."""
    prompt = build_praise_prompt(
        transcript=transcript,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        video_duration_sec=transcript.duration_sec,
    )
    raw = _invoke_claude_json(
        prompt, model=model, thinking_tokens=thinking_tokens,
        timeout_sec=timeout_sec, on_progress=on_progress,
        max_clips=10,  # 진행률 표기용 예상 곡 수(경배와찬양 4~6곡 + 성가대 1~2곡)
    )
    return _build_praise_clips(
        raw, transcript.duration_sec, min_duration_sec, max_duration_sec,
        pad_start_sec=pad_start_sec, pad_end_sec=pad_end_sec,
    )


def correct_praise_lyrics(
    songs: list[dict],
    model: str = "",
    thinking_tokens: int = 2048,
    timeout_sec: int = 600,
    on_progress=None,
) -> dict[int, str]:
    """whisper가 받아적은 노래 가사(오인식 다수)를 정식 가사로 교정한다.

    songs: [{"index": i, "title": 곡제목, "raw_text": whisper 원문(줄바꿈 없이 이어붙인 텍스트)}]
    반환: {index: 교정된 가사 전체 텍스트(자막 한 줄 분량마다 \n으로 줄바꿈)}

    이전 버전은 "출력 줄 수가 입력 줄 수와 정확히 같아야만" 교정을 적용했는데, whisper는
    노래를 숨쉬는 지점(휴지) 기준으로 줄을 들쭉날쭉 나눠서 이 조건이 실전에서 거의 항상
    깨졌다 — 결과적으로 교정이 사실상 안 먹히는 사고였다(실사용 신고: "자막 정확도 너무
    안좋음"). 이제 모델이 줄 구조를 자유롭게 다시 나누게 하고, 시간은 whisper 줄 경계가
    아니라 문자 수 비례로 배분한다(_distribute_lines_by_chars) — 사용자 결정: "싱크는
    안 맞아도 되니 가사 정확도가 우선"이므로 이 트레이드오프가 맞다.

    배경: whisper는 회중 찬양(음악+합창)을 심하게 오인식한다(실측: "금면류관을 드려서
    만유의 주 찬양" → "금멸 육아를 들여서 마녀의 주 찬양"). 찬송가/유명 CCM 가사는
    모델이 이미 알고 있으므로, 별도 크롤링·사전 학습 없이 지식 기반 교정이 가능하다."""
    payload = json.dumps(songs, ensure_ascii=False, indent=1)
    prompt = f"""너는 한국 교회 찬송가·CCM 가사 전문가다. 아래는 음성인식(whisper)이 회중 찬양(노래)을
받아적은 원문이다(줄바꿈 없이 이어붙인 텍스트, 노래 전사라 오인식이 많다).
(실제 예: "금면류관을 드려서 만유의 주 찬양"이 "금멸 육아를 들여서 마녀의 주 찬양"으로 오인식됨)

각 곡의 제목과 whisper 원문(raw_text)이 주어진다. 그 곡을 안다면(찬송가 번호나 가사로 식별)
raw_text가 담고 있는 절(들)의 '정식 가사'로 복원하라.

## 규칙 (모두 중요)
- raw_text에 실제로 담긴 절만 복원하라 — raw_text에 없는 절(예: 안 부른 3절)을 추가하지 마라.
  오인식 때문에 원문과 달라 보여도, 발음 유사성과 노래 구조(후렴 반복 등)로 몇 절의 어느
  소절인지 추론해서 순서대로 채워라.
- 출력은 자막 한 줄에 어울리는 짧은 소절 단위로 줄바꿈(\\n)하라 — 한 줄은 대략 8~16자.
  whisper 원문의 줄 구분은 신경 쓰지 마라(숨쉬는 지점 기준이라 노래 소절과 안 맞는다) —
  가사 자체의 자연스러운 소절 단위로 새로 나눠라.
- 곡을 전혀 모르겠으면 raw_text를 최대한 자연스럽게 다듬어서(명백한 오인식만 고쳐서) 출력하라
  — 빈 값을 반환하지 마라(자막이 아예 없는 것보다 원문이라도 있는 게 낫다).
- 노래가 아닌 멘트·기도가 raw_text에 섞여 있으면 자연스럽게 생략하거나 다듬어라.

입력:
{payload}

출력은 반드시 ```json ... ``` 코드블록 안의 JSON 배열만:
[{{"index": 0, "lyrics": "첫 줄\\n둘째 줄\\n..."}}, ...]"""
    raw = _invoke_claude_json(
        prompt, model=model, thinking_tokens=thinking_tokens,
        timeout_sec=timeout_sec, on_progress=on_progress, max_clips=len(songs),
    )
    out: dict[int, str] = {}
    for item in raw:
        try:
            idx = int(item["index"])
            text = str(item.get("lyrics", "")).strip()
        except (TypeError, ValueError):
            continue
        if text:
            out[idx] = text
    return out


def _distribute_lines_by_chars(lines: list[str], start: float, end: float) -> list[dict]:
    """교정된 가사 줄들을 [start,end] 구간에 문자 수 비례로 시간을 배분한다.

    whisper의 원래 줄 경계(숨쉬는 지점)와 모델이 새로 나눈 줄 경계가 다를 수 있어(그래서
    correct_praise_lyrics의 '줄 수 일치' 강제를 없앴다), 정확한 단어별 발화 시각 대신
    문자 수 비례로 근사한다. 사용자 결정: 가사 정확도가 싱크 정밀도보다 우선."""
    lines = [ln for ln in lines if ln]
    if not lines:
        return []
    total_chars = sum(len(ln) for ln in lines) or 1
    span = max(0.1, end - start)
    out: list[dict] = []
    cursor = start
    for ln in lines:
        dur = span * (len(ln) / total_chars)
        out.append({"start": round(cursor, 2), "end": round(cursor + dur, 2), "text": ln})
        cursor += dur
    out[-1]["end"] = round(end, 2)
    return out


def fetch_praise_lyrics_by_titles(
    titles: list[str],
    model: str = "",
    thinking_tokens: int = 2048,
    timeout_sec: int = 600,
    on_progress=None,
) -> dict[int, list[str]]:
    """곡 제목만으로 '정식 가사'를 가져온다(모델 지식 기반, 전사 불필요).

    사용자 요청(2026-09-05): 직접 찍어 올린 찬양은 whisper 전사 정확도가 너무 낮다
    ("금면류관"→"금멸 육아"). 곡 제목을 알면 전사를 통째로 건너뛰고, 유명 찬송가·CCM의
    정식 가사(모델이 이미 앎)를 자막으로 넣는다. 인터넷 크롤링 없이 지식 기반으로 충분하다는
    것이 앞선 실측 결론(project_praise_mode 메모).

    titles: 사용자가 입력한 곡 제목들(입력 순서 = 영상 재생 순서로 가정).
    반환: {index: [자막 한 줄, ...]}  — 모델이 모르는 곡은 결과에서 빠진다.

    2026-09-05: WebSearch를 허용해 모델이 '기억'이 아니라 실제 인터넷 검색으로 정식
    가사를 확인하게 했다(사용자 요청: "인터넷에 검색해서 가져와" — 기억 기반은 비슷한
    다른 곡/절 혼동 오답이 있었음). 검색 실패 시엔 기억 기반으로라도 쓰게 프롬프트에 명시.
    """
    songs = [
        {"index": i, "title": t.strip()}
        for i, t in enumerate(titles)
        if t and t.strip()
    ]
    if not songs:
        return {}
    # 제목별 가사 캐시: 찬양은 매주 같은 곡이 반복되므로, 한 번 검색한 곡은 다시 검색하지
    # 않는다(사용자 신고 2026-09-06: "제목만 넣으면 바로 나와야 하는 것 아니냐" — 남은
    # 대기시간의 전부가 WebSearch 가사 확인 1~2분이었다). 캐시 히트면 그 곡은 즉시,
    # 전 곡 히트면 claude 호출 자체를 건너뛴다. 키는 공백 정규화한 제목.
    cache_path = Path("output") / "lyrics_cache.json"
    try:
        _cache: dict[str, list[str]] = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _cache = {}

    # 키는 공백·기호를 전부 뺀 제목(2026-09-07: "다가도록"/"다 가도록"처럼 띄어쓰기만 다른
    # 입력이 캐시를 빗나가 매번 재검색됐고, 그 재검색이 엉뚱한 가사를 물어왔다).
    _ckey = normalize_title_key

    def _save_cache() -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError:
            pass  # 캐시 저장 실패는 치명적이지 않음(다음에 다시 검색하면 됨)

    cached_out: dict[int, list[str]] = {}
    pending = []
    for s in songs:
        hit = _cache.get(_ckey(s["title"]))
        if hit:
            # 예전 버전이 절 번호("1. ")째로 저장해 둔 캐시도 여기서 한 번 더 정리한다.
            cached_out[s["index"]] = clean_lyric_lines(list(hit))
        else:
            pending.append(s)
    if not pending:
        if on_progress:
            on_progress(1.0, "가사 캐시 재사용 (검색 생략)")
        return cached_out

    # 1차: 벅스에서 코드로 직접 가져온다(결정적 — 모델이 검색을 건너뛰거나 지어내는 일이
    # 없다). 제목 자리에 벅스 트랙 URL/번호를 넣으면 그 트랙 가사를 그대로 쓴다.
    still = []
    for s in pending:
        if on_progress:
            on_progress(0.2, f"벅스에서 '{s['title']}' 가사 찾는 중...")
        try:
            lines, meta = fetch_lyrics_from_bugs(s["title"])
        except Exception as e:  # noqa: BLE001 - 네트워크 문제면 모델 폴백
            print(f"[praise] 벅스 조회 실패 '{s['title']}': {e}")
            lines, meta = [], {}
        if lines:
            cached_out[s["index"]] = lines
            _cache[_ckey(s["title"])] = lines
        else:
            print(f"[praise] {meta.get('reason', '벅스에서 못 찾음')} → 모델 검색 폴백")
            still.append(s)
    _save_cache()
    if not still:
        if on_progress:
            on_progress(1.0, "벅스 가사 확정")
        return cached_out
    songs = still
    payload = json.dumps(songs, ensure_ascii=False, indent=1)
    prompt = f"""너는 한국 교회 찬송가·CCM(복음성가) 가사 전문가다. 아래 곡 제목들의 '정식 가사'를 써라.

## 먼저 할 일: 벅스(music.bugs.co.kr)에서 가사 확인 (필수)
가사 출처는 **벅스(music.bugs.co.kr)로 고정**한다 — 사용자 확인(2026-09-06): 이 사이트 찬송
가사가 가장 정확하다. 네 기억이나 다른 사이트는 같은 제목의 '다른 곡' 가사를 내놓는 오답이
잦았다(실측: "주님 부활했네"에 전혀 다른 곡 가사가 나옴).

각 곡마다 반드시 이 순서로 하라:
1) WebSearch로 `site:music.bugs.co.kr <곡 제목> 가사` 를 검색해 그 곡의 벅스 트랙 페이지를
   찾는다(URL 형태: https://music.bugs.co.kr/track/1503357).
2) 그 트랙 페이지를 WebFetch로 열어 '가사' 영역의 실제 가사를 **그대로** 읽어 쓴다.
   기억으로 고쳐 쓰지 마라 — 조사 하나(예: "주를"/"주님", "깨뜨셨네"/"깨뜨렸네")까지 벅스 표기를 따른다.
3) 같은 제목의 곡이 여러 개면 한국 교회에서 회중이 부르는 찬송가/CCM 버전을 고른다.
벅스에서 끝내 못 찾은 경우에만 다른 출처를 쓰되, 그래도 확신이 없으면 빈 문자열로 두라.

## 규칙 (모두 중요)
- 각 곡의 널리 불리는 정식 가사를 그대로 쓴다(찬송가 번호로 주어지면 그 장 가사).
- 회중이 실제로 부르는 분량(보통 1절 + 후렴 + 2절 정도)을 순서대로 쓴다. 절이 여러 개면
  1절→후렴→2절→(후렴) 순으로, 실제 예배에서 부르는 흐름대로 이어서 써라.
- **한 줄 = 실제로 함께 부르는 한 소절(한 문장/한 호흡)**로 줄바꿈(\\n)하라. 짧은 조각으로
  더 쪼개지 마라. 예시(찬송가 88장):
  내 진정 사모하는 친구가 되시는
  구주 예수님은 아름다와라
  산 밑에 백합화요 빛나는 새벽별
  주님 형언 할 길 아주 없도다
  처럼 한 소절을 통째로 한 줄에 둔다(대략 12~20자, 한 줄이 조금 길어도 쪼개지 마라).
- 제목만으로 곡을 특정할 수 없으면(동명이곡 등) 가장 널리 알려진 곡의 가사를 쓴다.
- 정말 모르는 곡이면 그 곡의 lyrics는 빈 문자열("")로 둔다(지어내지 마라).

입력(곡 제목):
{payload}

출력은 반드시 ```json ... ``` 코드블록 안의 JSON 배열만:
[{{"index": 0, "lyrics": "첫 줄\\n둘째 줄\\n..."}}, ...]"""
    raw = _invoke_claude_json(
        prompt, model=model, thinking_tokens=thinking_tokens,
        timeout_sec=timeout_sec, on_progress=on_progress, max_clips=len(songs),
        allowed_tools="WebSearch,WebFetch",  # 벅스 트랙 페이지를 찾아(WebSearch) 열어(WebFetch) 가사를 그대로 옮긴다
    )
    out: dict[int, list[str]] = dict(cached_out)
    title_by_index = {s["index"]: s["title"] for s in songs}
    for item in raw:
        try:
            idx = int(item["index"])
            text = str(item.get("lyrics", "")).strip()
        except (TypeError, ValueError):
            continue
        lines = clean_lyric_lines(text.split("\n"))
        if lines:
            out[idx] = lines
            if idx in title_by_index:
                _cache[_ckey(title_by_index[idx])] = lines
    _save_cache()
    return out


def correct_sermon_captions(
    lines: list[str],
    context: str = "",
    model: str = "",
    thinking_tokens: int = 2048,
    timeout_sec: int = 600,
    on_progress=None,
) -> tuple[list[str], list[str]]:
    """설교 자막 초안(whisper)을 문맥·성경지식으로 교정하고 강조어를 뽑는다.

    사용자 요청(2026-09-05): 인스타 레퍼런스처럼 '뜻을 이해해 오타 없는' 자막을 원함.
    whisper large-v3도 성경 고유명사·구어체를 이따금 틀린다(룻→루시, 보아스→보아즈 등).
    Claude가 뜻을 이해해 명백한 오인식만 고치고, 각 줄의 핵심 단어(강조 대상)를 뽑는다.

    핵심 제약: **줄 수와 순서를 그대로 유지**한다(각 줄의 시간축이 1:1로 매핑되므로).
    줄을 합치거나 나누지 않는다 — 내용(단어)만 고친다.

    lines: 자막 줄 텍스트 목록(시간 순).
    반환: (교정된 줄 목록(입력과 같은 길이), 강조 키워드 목록)
    """
    lines = [str(l or "") for l in lines]
    if not lines:
        return [], []
    numbered = [{"i": i, "text": t} for i, t in enumerate(lines)]
    payload = json.dumps(numbered, ensure_ascii=False, indent=1)
    ctx = f"\n이 클립의 주제/제목(참고): {context}\n" if context.strip() else ""
    prompt = f"""너는 한국 교회 설교 자막 교정 전문가다. 아래는 음성인식(whisper)이 설교를 받아적은
자막 줄들이다(대체로 정확하나 성경 고유명사·구어체에서 이따금 오인식이 있다).
{ctx}
## 할 일 (각 줄마다)
1) 명백한 오인식만 고쳐라(뜻이 통하게). 특히 성경 인물/지명/용어의 철자
   (예: 보아즈→보아스, 루시→룻, 기도원→기드온 류)를 문맥으로 바로잡아라.
2) 그 줄에서 '가장 강조하고 싶은 핵심 단어' 0~2개를 골라 "hl"에 넣어라
   (설교 메시지가 실린 명사·동사. 조사는 빼고 어간만: "은혜","십자가","사랑").

## 규칙 (매우 중요)
- 줄 수와 순서를 입력과 정확히 똑같이 유지하라(합치기·나누기·삭제 금지). i를 그대로 붙여라.
- 뜻이 이미 자연스러운 줄은 text를 원문 그대로 두라(억지 교정 금지). 문장을 새로 쓰지 마라.
- 확신이 없으면 원문을 유지하라(추측으로 바꾸지 마라).

입력:
{payload}

출력은 반드시 ```json ... ``` 코드블록 안의 JSON 배열만:
[{{"i": 0, "text": "교정된 줄", "hl": ["핵심어"]}}, ...]"""
    raw = _invoke_claude_json(
        prompt, model=model, thinking_tokens=thinking_tokens,
        timeout_sec=timeout_sec, on_progress=on_progress, max_clips=1,
    )
    corrected = list(lines)
    highlights: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item["i"])
            txt = str(item.get("text", "")).strip()
        except (TypeError, ValueError, KeyError):
            continue
        if 0 <= idx < len(corrected) and txt:
            corrected[idx] = txt
        for h in (item.get("hl") or []):
            hs = str(h).strip()
            if hs and hs.lower() not in seen:
                seen.add(hs.lower())
                highlights.append(hs)
    return corrected, highlights


def translate_captions_to_english(
    lines: list[str],
    context: str = "",
    model: str = "",
    thinking_tokens: int = 1024,
    timeout_sec: int = 600,
    on_progress=None,
) -> list[str]:
    """자막 줄들을 영어로 번역한다(줄 수·순서 1:1 유지, 시간축 그대로 매핑).

    사용자 요청(2026-09-05): "영어로도 버튼 누르면 나오면 좋겠다". 쇼츠 자막이므로
    직역보다 짧고 자연스러운 영어(구어체)로, 각 줄 길이를 원문과 비슷하게 유지한다.
    설교/찬양 맥락이라 성경 용어는 통용 영어 표기(grace, cross, salvation 등)를 쓴다."""
    lines = [str(l or "") for l in lines]
    if not lines:
        return []
    numbered = [{"i": i, "text": t} for i, t in enumerate(lines)]
    payload = json.dumps(numbered, ensure_ascii=False, indent=1)
    ctx = f"\n맥락(참고): {context}\n" if context.strip() else ""
    prompt = f"""너는 한국 교회 설교/찬양 자막을 영어로 옮기는 전문 번역가다. 아래 자막 줄들을
영어로 번역하라(쇼츠 자막용).
{ctx}
## 규칙
- 줄 수와 순서를 입력과 정확히 똑같이 유지하라(합치기·나누기·삭제 금지). i를 그대로 붙여라.
- 직역이 아니라 짧고 자연스러운 구어체 영어로. 각 줄은 화면 한 줄에 맞게 간결하게.
- 성경/신앙 용어는 통용 영어(grace, the cross, salvation, faith, the Lord 등)로.
- 문장부호는 최소화(자막이라). 각 줄은 대략 원문과 비슷한 분량으로.

입력:
{payload}

출력은 반드시 ```json ... ``` 코드블록 안의 JSON 배열만:
[{{"i": 0, "text": "English line"}}, ...]"""
    raw = _invoke_claude_json(
        prompt, model=model, thinking_tokens=thinking_tokens,
        timeout_sec=timeout_sec, on_progress=on_progress, max_clips=1,
    )
    out = list(lines)
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item["i"])
            txt = str(item.get("text", "")).strip()
        except (TypeError, ValueError, KeyError):
            continue
        if 0 <= idx < len(out) and txt:
            out[idx] = txt
    return out


def guess_praise_title_from_snippet(
    snippet_text: str,
    model: str = "",
    thinking_tokens: int = 1024,
    timeout_sec: int = 300,
    on_progress=None,
) -> tuple[str, int]:
    """짧게 훑은 whisper 전사 조각(부정확·짧음)에서 찬양 곡 제목을 추정한다.

    사용자 요청(2026-09-05): "그냥 제목만 파악해서 검색해서 자막 확보되잖아" — 사용자가
    제목을 안 넣어도, 영상 전체를 통째로 정밀 전사하는 대신 짧은 구간만 빠르게 훑어 제목을
    추정하면 곧장 정식 가사 경로(fetch_praise_lyrics_by_titles)로 갈 수 있다. whisper가
    노래를 심하게 오인식해도 후렴 반복·발음 유사성으로 Claude가 알아맞히는 경우가 많다.

    반환: (제목 또는 빈 문자열, confidence 1~10). 확신 없으면 빈 문자열 — 호출자가 안전하게
    기존 전체 분석(다곡 예배 실황 대응)으로 폴백해야 한다."""
    text = (snippet_text or "").strip()
    if not text:
        return "", 0
    prompt = f"""너는 한국 교회 찬송가·CCM 전문가다. 아래는 음성인식(whisper)이 찬양(회중 노래) 일부를
받아적은 원문 조각이다(부정확할 수 있다 — 노래 전사라 오인식이 흔하다).

원문 조각:
{text[:2000]}

이 조각만으로 어떤 찬양(찬송가 또는 CCM)인지 알아맞혀라. 발음 유사성과 후렴 반복 패턴으로
추론해도 좋다. 확신이 없으면 title을 빈 문자열로 두라(억지로 맞추지 마라 — 틀린 제목으로
엉뚱한 가사가 나가는 게 가장 나쁘다).

출력은 반드시 ```json ... ``` 코드블록 안의 JSON 배열(원소 하나)만:
[{{"title": "곡 제목 또는 빈 문자열", "confidence": 1~10}}]"""
    raw = _invoke_claude_json(
        prompt, model=model, thinking_tokens=thinking_tokens,
        timeout_sec=timeout_sec, on_progress=on_progress, max_clips=1,
    )
    if not raw or not isinstance(raw[0], dict):
        return "", 0
    title = str(raw[0].get("title", "") or "").strip()
    try:
        conf = int(raw[0].get("confidence", 0) or 0)
    except (TypeError, ValueError):
        conf = 0
    return title, conf


def _build_praise_clips_from_titles(
    titles: list[str],
    lyrics_by_idx: dict[int, list[str]],
    video_duration_sec: float,
) -> list[Clip]:
    """곡 제목 입력 기반으로 (전사 없이) 찬양 클립을 만든다.

    - 영상을 곡 순서대로 나눈다. 여러 곡이면 각 곡의 가사 분량(글자 수)에 비례해 시간을
      나눈다(대략치 — 사용자가 편집기에서 경계·시간을 조정할 수 있다).
    - 각 곡의 가사 줄은 그 구간에 글자 수 비례로 배분한다(_distribute_lines_by_chars).
      전사가 없어 단어별 발화 시각을 알 수 없으므로 '정식 가사(정확한 텍스트)'를 우선하고
      싱크는 근사한다 — 사용자 결정: "그냥 흰색 자막이 쭉 떠 있으면 된다".
    - caption_karaoke=False: 업로드 찬양은 파란색 단어 강조(카라오케) 없이 정적 흰 자막이
      기본(사용자 요청 2026-09-05). 전사가 없어 단어 싱크 자체가 불가능하기도 하다."""
    n = len(titles)
    if n == 0 or video_duration_sec <= 0:
        return []
    lyrics = {i: lyrics_by_idx.get(i, []) for i in range(n)}
    weights = [max(1, sum(len(ln) for ln in lyrics[i])) for i in range(n)]
    total_w = sum(weights) or 1
    clips: list[Clip] = []
    cursor = 0.0
    for i, title in enumerate(titles):
        is_last = i == n - 1
        span = video_duration_sec * (weights[i] / total_w)
        start = cursor
        end = video_duration_sec if is_last else min(video_duration_sec, cursor + span)
        cursor = end
        lines = lyrics[i]
        overrides = _distribute_lines_by_chars(lines, start, end) if lines else []
        clips.append(
            Clip(
                start=round(start, 2),
                end=round(end, 2),
                title=title.strip() or f"찬양 {i+1}",
                caption="",
                hashtags=[],
                reason="[입력 제목] 곡 제목 기반 정식 가사 자막(전사 없음)",
                appeal="다같이",
                trimmed=True,
                clip_type="praise",
                caption_overrides=overrides,
                caption_karaoke=False,
            )
        )
    return clips


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
