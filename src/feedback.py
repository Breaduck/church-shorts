"""피드백 루프: 실제로 올린 클립의 성과를 기록하고, 다음 선정 때 프롬프트에 주입한다.

왜 필요한가:
  예측 점수(score)는 '이 채널에서 실제로 뜰 확률'과 캘리브레이션돼 있지 않았다.
  즉 80점이 진짜 80점이라는 근거가 없었다. 이 모듈은 사람이 올린 클립의 실제 성과
  (조회수/저장/공유/체감등급)를 파일에 누적하고, 다음 하이라이트 선정 프롬프트에
  "예측 → 실제" 사례로 되먹여 모델이 채널 특성에 맞게 채점을 보정하도록 한다.

저장 위치: 프로젝트 루트의 feedback.json (영상 단위가 아니라 채널 전체 학습용).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.fsutil import atomic_write_text

FEEDBACK_PATH = Path("feedback.json")

# 사람이 빠르게 매기는 체감 등급.
RATING_HIT = "hit"    # 잘 됨(터짐)
RATING_OK = "ok"      # 평범
RATING_FLOP = "flop"  # 망함
RATING_LABELS = {RATING_HIT: "HIT ✅", RATING_OK: "보통", RATING_FLOP: "FLOP ❌"}


@dataclass
class PerformanceRecord:
    video_id: str
    clip_index: int
    title: str
    hook_text: str = ""          # 클립 첫 문장(훅). 어떤 훅이 먹혔는지 학습용.
    start: float = 0.0
    end: float = 0.0
    # 예측(선정 당시 값)
    predicted_score: Optional[float] = None
    predicted_core: Optional[float] = None
    predicted_viral: Optional[float] = None
    predicted_subscores: dict = field(default_factory=dict)
    # 실제 성과(사람이 나중에 입력)
    views: Optional[int] = None
    retention_pct: Optional[float] = None   # 평균 조회율(%)
    saves: Optional[int] = None
    shares: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    rating: str = RATING_OK
    notes: str = ""
    logged_at: str = ""
    # 스튜디오 지속률 CSV 분석(src/retention.py analyze 결과). 첫 3초 이탈·급락 지점과 그때 하던 말.
    retention_analysis: dict = field(default_factory=dict)
    # 채널 자동 추적(src/channel_tracking.py)이 찾은 실제 업로드 영상 id.
    youtube_video_id: str = ""
    # rating을 사람이 아니라 채널 중앙값 배수로 자동 판정했는지. 사람이 한 번 저장하면 False가 돼
    # 이후 자동 추적이 rating을 건드리지 않는다.
    auto_rated: bool = False
    # 게시일(YYYYMMDD)과 조회수의 채널 중앙값 대비 배수 — 갓 올린 영상을 근거에서 빼고, 날것 조회수 대신 배수로 본다.
    upload_date: str = ""
    view_ratio: Optional[float] = None


def load_feedback(path: Path = FEEDBACK_PATH) -> list[PerformanceRecord]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        # 깨진 파일 하나가 모든 선정을 멈추면 안 된다(성과 피드백은 참고 자료일 뿐).
        print(f"[feedback] {path} 읽기 실패, 피드백 없이 진행: {e}", flush=True)
        return []
    records = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        # 알 수 없는 키는 버리고 아는 필드만 취해 스키마 변화에 견고하게.
        known = {k: v for k, v in d.items() if k in PerformanceRecord.__dataclass_fields__}
        records.append(PerformanceRecord(**known))
    return records


def save_feedback(records: list[PerformanceRecord], path: Path = FEEDBACK_PATH) -> None:
    atomic_write_text(path, json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2))


def upsert_feedback(record: PerformanceRecord, path: Path = FEEDBACK_PATH) -> None:
    """(video_id, clip_index) 기준으로 있으면 갱신, 없으면 추가."""
    if not record.logged_at:
        record.logged_at = datetime.now().isoformat(timespec="seconds")
    records = load_feedback(path)
    for i, r in enumerate(records):
        if r.video_id == record.video_id and r.clip_index == record.clip_index:
            records[i] = record
            break
    else:
        records.append(record)
    save_feedback(records, path)


def _fmt_metrics(r: PerformanceRecord) -> str:
    parts = []
    if r.views is not None:
        parts.append(f"조회 {r.views:,}")
    if r.retention_pct is not None:
        parts.append(f"조회율 {r.retention_pct:.0f}%")
    if r.saves is not None:
        parts.append(f"저장 {r.saves:,}")
    if r.shares is not None:
        parts.append(f"공유 {r.shares:,}")
    if r.likes is not None:
        parts.append(f"좋아요 {r.likes:,}")
    if r.retention_analysis:
        from src.retention import summarize
        parts.append("이탈: " + summarize(r.retention_analysis))
    return "·".join(parts) if parts else "지표없음"


MIN_AGE_DAYS_FOR_EVIDENCE = 7   # 이보다 갓 올린 영상은 근거로 쓰지 않는다(아래 _usable 주석)
MIN_RECORDS_FOR_BLOCK = 4       # 이보다 적으면 블록을 아예 넣지 않는다 — 사례 한두 개는 교훈이 아니라 잡음


def _age_days(r: PerformanceRecord, now: datetime) -> Optional[int]:
    if not r.upload_date:
        return None
    try:
        return (now - datetime.strptime(r.upload_date, "%Y%m%d")).days
    except ValueError:
        return None


def _usable(r: PerformanceRecord, now: datetime) -> bool:
    """프롬프트 근거로 쓸 수 있는 기록인가.

    2026-11-15 실측으로 걸러낸 세 가지 잡음:
    - 찬양 클립(예측 점수 없음)이 설교 선정 프롬프트에 섞여 들어갔다("당신은 영광의 왕 [예측 -] = FLOP").
    - 올린 지 하루 된 영상의 조회수 145가 '보통'이라는 등급을 달고 석 달 된 영상의 1,100과 나란히 실렸다.
    - 자동 판정은 28일을 기다렸는데, 이 채널 쇼츠는 조회수가 이틀 안에 굳는다(1주일 뒤 재측정: 15개 중
      14개가 변화 1% 미만) → 7일이면 충분하다."""
    if r.predicted_score is None or r.views is None:
        return False
    age = _age_days(r, now)
    if age is None:
        return not r.auto_rated  # 게시일을 모르면 사람이 직접 매긴 기록만 믿는다
    return age >= MIN_AGE_DAYS_FOR_EVIDENCE


def format_feedback_for_prompt(
    records: list[PerformanceRecord], max_examples: int = 12, purpose: str = "select",
    now: Optional[datetime] = None,
) -> str:
    """'예측 → 실제 성과' 블록. purpose="select"는 1차(소재·시작 문장 고르기), "score"는 2차(채점)용 머리말.

    조회수는 날것이 아니라 **채널 중앙값 대비 배수**로 보여준다(채널 전체가 1천 회 안팎이라 날것 숫자는
    '다 비슷하게 낮다'로만 읽힌다). 쓸 만한 기록이 MIN_RECORDS_FOR_BLOCK 미만이면 빈 문자열."""
    now = now or datetime.now()
    usable = [r for r in records if _usable(r, now)]
    if len(usable) < MIN_RECORDS_FOR_BLOCK:
        return ""

    def _ratio(r: PerformanceRecord) -> Optional[float]:
        return r.view_ratio if r.view_ratio else None

    ranked = sorted(usable, key=lambda r: (_ratio(r) or 0.0, r.views or 0), reverse=True)
    if len(ranked) > max_examples:  # 위·아래 끝이 교훈이다 — 가운데를 덜어낸다
        half = max_examples // 2
        ranked = ranked[:half] + ranked[-(max_examples - half):]

    if purpose == "score":
        intro = [
            "## 이 채널에 실제로 올린 클립의 성과 (채점 눈금을 여기에 맞춰라)",
            "아래는 과거 클립의 [당시 예측 점수·hook 점수 / 첫 문장] → 실제 조회수(채널 중앙값 대비 배수)다.",
            "배수가 낮은 클립과 닮은 시작·소재에는 hook·retention을 낮게, 높은 클립과 닮았으면 높게 매겨라.",
        ]
    else:
        intro = [
            "## 이 채널에 실제로 올린 클립의 성과 (소재와 시작 문장을 고를 때 참고)",
            "아래는 과거 클립의 [당시 예측 점수·hook 점수 / 첫 문장] → 실제 조회수(채널 중앙값 대비 배수)다.",
            "쇼츠는 첫 노출 묶음에서 시청자가 넘겨 버리면 노출이 끊긴다 — 배수가 낮은 클립은 '초반에 넘겨진' 클립이다.",
            "그런 클립과 닮은 시작 문장·소재는 피하고, 배수가 높은 쪽과 닮은 대목을 우선하라.",
        ]
    lines = intro + [""]
    for r in ranked:
        hook_s = (r.predicted_subscores or {}).get("hook")
        pred = f"예측 {int(r.predicted_score)}" + (f"·hook {int(hook_s)}" if hook_s is not None else "")
        ratio = _ratio(r)
        perf = f"조회 {r.views:,}" + (f"(중앙값의 {ratio:.1f}배)" if ratio else "")
        if r.likes is not None:
            perf += f"·좋아요 {r.likes:,}"
        if r.retention_pct is not None:
            perf += f"·조회율 {r.retention_pct:.0f}%"
        if r.retention_analysis:
            from src.retention import summarize
            perf += "·이탈: " + summarize(r.retention_analysis)
        label = "" if r.rating == RATING_OK else f" = {RATING_LABELS.get(r.rating, r.rating)}"
        title = " ".join((r.title or "제목없음").split())
        hook = " ".join((r.hook_text or "").split())[:40]
        note = f" — {r.notes.strip()}" if r.notes.strip() else ""
        lines.append(f'- "{title}" [{pred} / 첫 문장 "{hook}"] → {perf}{label}{note}')

    lessons = []
    # 예측이 실제를 갈랐는가: 예측 점수 상위 절반과 하위 절반의 실제 배수를 비교해 그대로 적는다(없는 교훈은 지어내지 않는다).
    with_ratio = [r for r in usable if _ratio(r)]
    if len(with_ratio) >= 6:
        by_pred = sorted(with_ratio, key=lambda r: r.predicted_score or 0)
        k = len(by_pred) // 2
        lo, hi = by_pred[:k], by_pred[-k:]
        lo_m = sum(_ratio(r) for r in lo) / len(lo)
        hi_m = sum(_ratio(r) for r in hi) / len(hi)
        if hi_m >= lo_m * 1.5:
            cut = int(max(r.predicted_score or 0 for r in lo))
            lessons.append(
                f"- 예측 {cut}점 이하였던 클립은 실제로 평균 {lo_m:.1f}배, 그보다 높았던 클립은 평균 {hi_m:.1f}배였다 — "
                "예측이 낮게 나온 결(첫 문장이 순서 표지·교리 선언·본문 설명)은 실제로도 초반에 넘겨졌다."
            )
        elif lo_m >= hi_m * 1.5:
            lessons.append(
                f"- 예측이 높았던 클립(평균 {hi_m:.1f}배)보다 낮았던 클립(평균 {lo_m:.1f}배)이 실제로 더 잘 됐다 — "
                "지금 채점 기준이 이 채널과 거꾸로다. 위 사례에서 잘 된 쪽의 공통점을 다시 보라."
            )
    ra = [r for r in usable if r.retention_analysis]
    leaky = [r for r in ra if (r.retention_analysis.get("hook_leak") or 0) >= 40]
    if leaky:
        ex = "; ".join(f'"{(r.retention_analysis.get("hook_said") or r.hook_text)[:40]}"' for r in leaky[:3])
        lessons.append(f"- 첫 3초에 40% 이상 스와이프된 클립이 {len(leaky)}건: {ex} — 이런 첫 문장으로 시작하는 컷은 피하라.")
    cliff_lines = [c.get("said") for r in ra for c in (r.retention_analysis.get("cliffs") or []) if c.get("said")]
    if cliff_lines:
        lessons.append("- 시청자가 급히 빠져나간 직전 문장들(이런 이음새·곁가지는 컷에서 빼라): "
                       + " / ".join(f'"{t[:40]}"' for t in cliff_lines[:5]))
    if lessons:
        lines += ["", "### 이 기록에서 실제로 확인된 것:"] + lessons
    lines.append("")
    return chr(10).join(lines)
