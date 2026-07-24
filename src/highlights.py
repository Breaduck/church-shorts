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

    return f"""너는 조회수가 잘 나오는 교회 쇼츠를 만드는 최고의 편집자다. 26분 설교 전체에서
"이 부분만큼은 사람들이 끝까지 보고, 저장하고, 공유할 것"이라 확신하는 진짜 알맹이만 골라낸다.

아래는 {video_duration_sec/60:.0f}분 길이 설교 영상의 전체 전사본이다 (타임스탬프 [HH:MM:SS] 포함).

## 절대 원칙 (이걸 어기면 실패다)
- **양보다 질.** 억지로 개수를 채우지 마라. 정말 강력한 구간이 3개뿐이면 3개만 내라.
  평범한 구간을 하나라도 끼워 넣느니, 최고만 3개 내는 게 낫다. (최소 {min_clips}, 최대 {max_clips}개)
- **완결성은 타협 불가.** 각 클립은 그 클립만 봐도 처음부터 끝까지 하나의 완성된 이야기/메시지여야 한다.
  전체 설교를 안 본 사람이 이 클립만 보고도 "아, 무슨 말인지 완전히 이해했고 울림이 있다"가 되어야 한다.
  - 시작: 앞 맥락 없이도 이해되는, 완결된 문장으로 시작. (예: "~라고 하는데" 처럼 앞이 잘린 채 시작 금지)
  - 끝: 결론/핵심/반전이 완전히 떨어진 뒤에 끝낼 것. 결정적 한마디가 나오기 직전에 끊으면 최악이다.
  - 중간에 "둘째,", "셋째," 같은 설교 구조어로 시작해서 앞 내용을 전제하면 안 된다.
- **진짜 핵심인가?** "그냥 괜찮은 구간"과 "이 설교의 심장"은 다르다. 설교자가 가장 힘줘 말한 것,
  가장 기억에 남는 비유/일화/명언, 일반인도 공감하거나 뜨끔할 메시지 — 그런 것만.

## 작업 순서 (반드시 이 순서로 사고할 것)
1단계) 먼저 이 설교의 **핵심 주제 한 줄**과, 설교자가 밀어붙인 **핵심 메시지 2~4개**를 머릿속으로 정리한다.
2단계) 그 메시지를 가장 강력하고 완결되게 담고 있는 실제 구간을 전사본에서 찾는다.
   후보를 여러 개 떠올린 뒤, "이 클립만 보면 완결되는가? 스크롤을 멈추게 하는가? 공유하고 싶은가?"를
   냉정하게 자문해 통과한 것만 남긴다.
3단계) 각 구간의 정확한 start/end 타임스탬프를 완결된 문장 경계에 맞춰 확정한다.

## 각 클립 형식 조건
- **길이**: 최종적으로 {min_duration_sec}~{max_duration_sec}초 분량. 이 안에 '완결된 하나의 이야기'가 들어가야 하므로,
  너무 짧게(20초 남짓) 잘라서 이야기가 덜 끝나게 하지 말 것. 완결을 위해 조금 길어지는 건 괜찮다.
- **훅(첫 문장)**: 클립 첫 문장이 그 자체로 궁금증/충격/공감을 유발해 스크롤을 멈추게 할 것.
- **서로 다른 메시지**: 클립끼리 같은 내용/같은 비유를 반복하지 말 것.

## 오디오 에너지 힌트 (참고용일 뿐. 이것만으로 판단하지 말 것 — 헛기침/잡음일 수도 있음)
{hints_text}

## 전사본
{transcript_text}

## 출력 형식
다른 설명 없이, 아래 JSON 배열만 출력하라 (```json 코드블록으로 감쌀 것). reason에는 1단계에서
파악한 핵심 메시지 중 무엇을 담았고 왜 완결되는지를 구체적으로 적어라.
**배열 순서 = 바이럴 예상 순위 (0번째가 가장 강력할 것으로 예상되는 클립).**

```json
[
  {{
    "start": 123.4,
    "end": 175.0,
    "title": "영상 맨 위에 고정될 한 줄 제목 (호기심/공감 유발, 15자 내외)",
    "caption": "유튜브/인스타/틱톡 게시글 캡션 (2~3문장, 설교 맥락 살려서)",
    "hashtags": ["#설교", "#은혜", "..."],
    "reason": "이 설교의 어떤 핵심 메시지를 담았는지 + 왜 클립만 봐도 완결되는지 (구체적으로)"
  }}
]
```
start/end는 전사본 타임스탬프 기준 **초 단위 숫자**로 변환해서 적을 것.
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
        clips.append(
            Clip(
                start=start,
                end=end,
                title=str(c.get("title", "")).strip(),
                caption=str(c.get("caption", "")).strip(),
                hashtags=list(c.get("hashtags", [])),
                reason=str(c.get("reason", "")).strip(),
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
