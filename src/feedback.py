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


def load_feedback(path: Path = FEEDBACK_PATH) -> list[PerformanceRecord]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for d in data:
        # 알 수 없는 키는 버리고 아는 필드만 취해 스키마 변화에 견고하게.
        known = {k: v for k, v in d.items() if k in PerformanceRecord.__dataclass_fields__}
        records.append(PerformanceRecord(**known))
    return records


def save_feedback(records: list[PerformanceRecord], path: Path = FEEDBACK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


def _record_signal(r: PerformanceRecord) -> float:
    """예측과 실제가 얼마나 어긋났는지(=학습 가치)를 대략 점수화. 클수록 교훈적."""
    if r.rating == RATING_HIT:
        base = 3.0
    elif r.rating == RATING_FLOP:
        base = 3.0
    else:
        base = 0.5
    # 예측이 높았는데 flop, 낮았는데 hit이면 특히 교훈적 → 가중.
    pred = r.predicted_score or 50
    if r.rating == RATING_FLOP and pred >= 80:
        base += 2.0
    if r.rating == RATING_HIT and pred < 80:
        base += 2.0
    return base


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
    return "·".join(parts) if parts else "지표없음"


def format_feedback_for_prompt(
    records: list[PerformanceRecord], max_examples: int = 14
) -> str:
    """선정 프롬프트에 넣을 '예측 → 실제 성과' 캘리브레이션 블록을 만든다.
    학습 가치(예측-실제 괴리)가 큰 사례를 우선 노출한다. 기록이 없으면 빈 문자열."""
    if not records:
        return ""

    ranked = sorted(records, key=_record_signal, reverse=True)[:max_examples]

    lines = [
        "## 이 채널의 실제 성과 피드백 (예측을 현실에 보정하라 — 매우 중요)",
        "아래는 과거에 실제로 올린 클립들의 [예측 점수 → 실제 성과]다.",
        "네 채점이 현실과 어긋났던 패턴을 학습해, 이번 채점(특히 viral 세부 축)에 반영하라.",
        "예측이 높았는데 FLOP이면 그런 결의 클립을 낮게, 낮았는데 HIT이면 그런 결을 높게 조정하라.",
        "",
    ]
    for r in ranked:
        pred = f"예측 {int(r.predicted_score)}" if r.predicted_score is not None else "예측 -"
        sub = r.predicted_subscores or {}
        sub_str = ""
        if sub:
            sub_str = " (" + "/".join(
                f"{k[:3]}{int(v)}" for k, v in sub.items() if v is not None
            ) + ")"
        label = RATING_LABELS.get(r.rating, r.rating)
        title = (r.title or r.hook_text or "제목없음").strip()
        note = f" — {r.notes.strip()}" if r.notes.strip() else ""
        lines.append(f'- "{title}" [{pred}{sub_str}] → {_fmt_metrics(r)} = {label}{note}')

    # 간단한 집계 교훈.
    hits = [r for r in records if r.rating == RATING_HIT]
    flops = [r for r in records if r.rating == RATING_FLOP]
    over = [r for r in flops if (r.predicted_score or 0) >= 80]
    under = [r for r in hits if (r.predicted_score or 0) < 80]
    lessons = []
    if over:
        lessons.append(f"- 예측 80+였는데 실제로 망한 클립이 {len(over)}건 있었다. 네 기준이 과대평가하는 결이 있으니 냉정하게.")
    if under:
        lessons.append(f"- 예측 80 미만이었는데 실제로 터진 클립이 {len(under)}건 있었다. 놓친 강점 결이 있으니 재검토하라.")
    if lessons:
        lines.append("")
        lines.append("### 도출된 채널 교훈:")
        lines.extend(lessons)

    lines.append("")
    return "\n".join(lines)
