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
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.audio_peaks import PeakHint, format_hints_for_prompt
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
    viral_score: Optional[float] = None   # 스크롤을 멈추고 저장·공유하고 싶은가 (1~10)
    # 웹 UI의 위치 편집 툴에서 드래그로 조정한, 기본 위치 대비 픽셀 오프셋(렌더 해상도 기준).
    # 기본값 0은 기존 clips.json(이 필드가 없는)과도 호환된다.
    title_offset_x: float = 0.0
    title_offset_y: float = 0.0
    caption_offset_x: float = 0.0
    caption_offset_y: float = 0.0


def build_prompt(
    transcript: Transcript,
    peak_hints: list[PeakHint],
    min_clips: int,
    max_clips: int,
    min_duration_sec: int,
    max_duration_sec: int,
    categories: list[str],
    video_duration_sec: float,
) -> str:
    transcript_text = transcript.to_plain_text_with_timestamps()
    hints_text = format_hints_for_prompt(peak_hints)
    categories_text = "\n".join(f"  - {c}" for c in categories)

    return f"""너는 조회수가 잘 나오는 교회 쇼츠를 만드는 최고의 편집자다. {video_duration_sec/60:.0f}분 설교 전체에서
"이 부분만큼은 사람들이 끝까지 보고, 저장하고, 공유할 것"이라 확신하는 진짜 알맹이만 골라낸다.

아래는 {video_duration_sec/60:.0f}분 길이 설교 영상의 전체 전사본이다 (타임스탬프 [HH:MM:SS] 포함).

## 네가 뽑아야 하는 건 "교집합"이다 (가장 중요)
좋은 쇼츠는 두 가지가 **동시에** 만족돼야 한다. 하나만 높으면 실패다.
  (A) **설교의 핵심** — 설교자가 진짜 힘줘 말한 알맹이인가? (알맹이 없는 자극 = 낚시, 실패)
  (B) **바이럴** — 스크롤을 멈추고, 끝까지 보고, 저장·공유하고 싶은가? (핵심이지만 지루하면 = 안 터짐, 실패)
각 후보에 두 축을 **따로** 매긴다: core_score(A, 1~10), viral_score(B, 1~10).
그리고 이 둘의 **교집합**을 0~100점 통합 점수 `score`로 환산한다 = "이게 실제로 뜰 확률".
  - score는 두 축이 **모두** 높을 때만 높다. 한 축이라도 낮으면 score도 확 떨어진다(곱셈적 사고).
  - 기준선: core·viral 둘 다 9면 score 90+, 둘 다 8이면 80~85, 한쪽이라도 7 이하로 처지면 80 미만.
**score 80점 미만은 아예 출력하지 마라.** 80점 넘는 게 3개뿐이면 3개만, 하나도 없으면 그렇게 말고 가장 근접한 걸 내되 솔직한 점수를 매겨라.

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

## 절대 원칙 (이걸 어기면 실패다)
- **양보다 질.** 억지로 개수를 채우지 마라. core/viral 둘 다 7점 이상이 3개뿐이면 3개만 내라.
  평범한 구간을 하나라도 끼워 넣느니, 최고만 3개 내는 게 낫다. (최소 {min_clips}, 최대 {max_clips}개)
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

## 작업 순서 (반드시 이 순서로 사고할 것)
1단계) 이 설교의 **핵심 주제 한 줄**과, 설교자가 밀어붙인 **핵심 메시지 2~4개**를 정리한다.
2단계) 각 메시지를 강력하고 완결되게 담은 실제 후보 구간을 전사본에서 여러 개 찾는다.
3단계) 각 후보에 core_score(1~10)와 viral_score(1~10)를 냉정하게 매기고, 둘의 교집합으로 score(0~100)를 낸다.
   **score 80점 이상만 남긴다.**
4단계) **자가검증** — 남긴 각 후보에 대해:
   (a) 첫 문장을 그대로 떠올려라 → "이게 1~2초 안에 스크롤을 멈추게 하나?" 아니면 시작점을 옮겨라.
   (b) 마지막 문장을 그대로 떠올려라 → "이게 가장 강한 마무리인가? 뒤에 힘 빠지는 꼬리가 붙어있진 않나?"
       꼬리가 있으면 end를 펀치라인 끝으로 당겨라.
5단계) **재고** — 남은 것 중 가장 약한 후보 하나를 냉정하게 탈락 후보로 재검토한다. 확신 없으면 빼라.
6단계) 각 구간의 정확한 start/end 타임스탬프를 완결된 문장 경계(특히 end=펀치라인 끝)에 맞춰 확정한다.

## 각 클립 형식 조건
- **길이**: 최종 {min_duration_sec}~{max_duration_sec}초. 완결을 위해 조금 길어지는 건 괜찮지만, 완결됐으면 늘리지 말고 끝내라.
- **서로 다른 메시지**: 클립끼리 같은 내용/같은 비유를 반복하지 말 것.

## 오디오 에너지 힌트 (참고용일 뿐. 이것만으로 판단하지 말 것 — 헛기침/잡음일 수도 있음)
{hints_text}

## 전사본
{transcript_text}

## 출력 형식
다른 설명 없이, 아래 JSON 배열만 출력하라 (```json 코드블록으로 감쌀 것).
**score 80점 미만은 배열에 넣지 마라. 배열 순서 = score 높은 순 (0번째가 가장 강력한 클립).**

```json
[
  {{
    "start": 123.4,
    "end": 175.0,
    "score": 88,
    "core_score": 9,
    "viral_score": 8,
    "title": "영상 맨 위에 고정될 한 줄 제목 (호기심/공감 유발, 15자 내외)",
    "caption": "유튜브/인스타/틱톡 게시글 캡션 (2~3문장, 설교 맥락 살려서)",
    "hashtags": ["#설교", "#은혜", "..."],
    "reason": "담은 핵심 메시지 + core/viral/score 근거 + 마지막 문장을 그대로 인용하고 왜 그게 강한 마무리인지"
  }}
]
```
- score는 0~100 정수. 80 미만은 출력 금지.
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
    clips: list[Clip] = []
    for i, c in enumerate(raw_clips):
        start = float(c["start"])
        end = float(c["end"])
        if end <= start:
            raise ValueError(f"클립 {i}: end({end})가 start({start})보다 작거나 같습니다")
        start = max(0.0, start)
        end = min(video_duration_sec, end)
        duration = end - start
        if duration < min_duration_sec * 0.8:
            raise ValueError(f"클립 {i}: 길이({duration:.1f}초)가 최소 길이에 비해 너무 짧습니다")

        def _as_score(v) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        clips.append(
            Clip(
                start=start,
                end=end,
                title=str(c.get("title", "")).strip(),
                caption=str(c.get("caption", "")).strip(),
                hashtags=list(c.get("hashtags", [])),
                reason=str(c.get("reason", "")).strip(),
                score=_as_score(c.get("score")),
                core_score=_as_score(c.get("core_score")),
                viral_score=_as_score(c.get("viral_score")),
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
) -> list[Clip]:
    """`claude -p` 서브프로세스를 호출해 자동으로 하이라이트를 선정한다."""
    prompt = build_prompt(
        transcript=transcript,
        peak_hints=peak_hints,
        min_clips=min_clips,
        max_clips=max_clips,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        categories=categories,
        video_duration_sec=transcript.duration_sec,
    )

    claude_path = shutil.which("claude")
    if not claude_path:
        raise RuntimeError("claude CLI를 PATH에서 찾을 수 없습니다 (claude --version으로 설치 확인)")

    proc = subprocess.run(
        [claude_path, "-p", "--output-format", "json"],
        input=prompt,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p 실행 실패 (exit {proc.returncode}):\n{proc.stderr}")

    outer = json.loads(proc.stdout)
    result_text = outer.get("result", "")
    raw_clips = _extract_json_array(result_text)
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
