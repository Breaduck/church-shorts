"""평가에서 나온 후보들의 '실물 품질'을 검사한다 (모델 호출 없음).

재현율(scripts/eval_selection.py)은 "정답 주제를 찾았나"만 본다. 실제로 쓸 수 있는 클립인지는 따로 봐야 한다:
훅이 첫 3초에 먹히는가, 점프컷 이음새가 자연스러운가, 길이가 벤치마크 범위인가, 같은 소재가 겹치지 않는가,
성경 인물 이야기가 새어 들어오지 않았는가.

사용법: venv/Scripts/python.exe scripts/audit_clips.py
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_selection import load_transcript  # noqa: E402
from src.main import load_config  # noqa: E402
from src.sermon_cut import (  # noqa: E402
    _BIBLE_NAME_RE,
    _CONNECTOR_START,
    _FILLER_ONLY,
    _eff_dur,
    _is_incomplete,
    _is_weak_hook,
    _skips_of,
    build_sentences,
    cut_keep_ranges,
    dedupe_cuts,
    drop_bible_story_cuts,
    drop_polemic_cuts,
    kept_runs,
    merge_adjacent_cuts,
    trim_lead_words,
    verify_and_fix,
)

BENCH_MIN, BENCH_MAX = 40.0, 115.0   # 벤치마크 상위 쇼츠 실측 길이 범위(59~111초)에 여유


def audit() -> int:
    issues: dict[str, list[str]] = {}
    lengths: list[float] = []
    appeals: dict[str, int] = {}
    seams = pieces = clips = 0

    def flag(kind: str, msg: str) -> None:
        issues.setdefault(kind, []).append(msg)

    for f in sorted(glob.glob("output/_eval/eval_*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        sents = build_sentences(load_transcript(d["video"]))
        # 저장된 모델 원시 컷을 **현재 코드**로 다시 검증·필터한다 — 규칙을 손볼 때마다 모델을 다시 부르지
        # 않고도 실물 품질을 재려고(2026-09-22). 그래서 eval JSON의 ours가 아니라 여기서 다시 만든다.
        h = load_config(Path("config.yaml"))["highlights"]
        fixed = []
        for v in d["verify"]:
            cut, _ = verify_and_fix(
                dict(v["raw"]), sents, float(h["min_duration_sec"]), float(h["max_duration_sec"]),
                float(h.get("hard_max_duration_sec", 90)), max_span_sec=float(h.get("max_span_sec", 180)),
            )
            if cut:
                fixed.append(cut)
        fixed, _ = merge_adjacent_cuts(fixed, sents, float(h.get("hard_max_duration_sec", 90)))
        fixed, _ = drop_polemic_cuts(fixed, sents)
        fixed, _ = drop_bible_story_cuts(fixed, sents)
        fixed = dedupe_cuts(fixed, sents)
        d["ours"] = [{
            "start": round(sents[c["start"]].start, 1), "end": round(sents[c["end"]].end, 1),
            "eff": round(_eff_dur(sents, c["start"], c["end"], _skips_of(c)), 1),
            "keep_ranges": cut_keep_ranges(c, sents), "appeal": c.get("appeal", ""),
            "scene": c.get("scene", ""), "hook": trim_lead_words(sents[c["start"]])[1][:70],
        } for c in fixed]
        by_start = {round(sents[c["start"]].start, 1): c for c in fixed}
        for o in d["ours"]:
            clips += 1
            lengths.append(o["eff"])
            appeals[o["appeal"]] = appeals.get(o["appeal"], 0) + 1
            tag = f"{d['video']} {o['start']:.0f}s"

            # 1) 훅 — 렌더와 같은 기준으로 본다(trim_lead_words가 "아/그래서" 같은 말버릇 단어를 잘라낸 뒤의 첫 문장)
            hook = o["hook"].strip()
            if _is_weak_hook(hook):
                flag("약한 훅(연도·숫자나열·장문)", f"{tag}: {hook[:60]}")
            if _CONNECTOR_START.match(hook) or _FILLER_ONLY.match(hook):
                flag("접속어·추임새로 시작", f"{tag}: {hook[:60]}")
            if len(hook.split()) <= 2 and not hook.endswith("?"):
                flag("훅이 조각 문장", f"{tag}: {hook[:60]}")

            # 2) 길이
            if not (BENCH_MIN <= o["eff"] <= BENCH_MAX):
                flag("길이가 벤치마크 밖(40~115초)", f"{tag}: {o['eff']:.0f}초")

            # 3) 점프컷 이음새
            c = by_start.get(round(o["start"], 1))
            if c:
                runs = kept_runs(c["start"], c["end"], _skips_of(c))
                pieces += len(runs)
                for prev, nxt in zip(runs, runs[1:]):
                    seams += 1
                    before, after = sents[prev[1]], sents[nxt[0]]
                    # 물음표로 끝나고 점프하는 건 '질문 → 답'이라 오히려 자연스럽다(_sanitize_skips와 같은 예외).
                    if _is_incomplete(before.text) and not before.text.strip().endswith("?"):
                        flag("이음새: 앞 문장이 안 끝남", f"{tag}: …{before.text[-28:]} || {after.text[:28]}…")
                    if _CONNECTOR_START.match(after.text.strip()):
                        flag("이음새: 뒤 문장이 접속어", f"{tag}: …{before.text[-28:]} || {after.text[:28]}…")
                # 4) 끝
                if _is_incomplete(sents[c["end"]].text):
                    flag("끝 문장이 미완", f"{tag}: …{sents[c['end']].text[-40:]}")
                body = " ".join(s.text for a, b in runs for s in sents[a: b + 1])
                names = sorted({m.group(1) for m in _BIBLE_NAME_RE.finditer(body)})
                if len(names) >= 2:
                    flag("성경 인물이 여럿 등장", f"{tag}: {'/'.join(names)} | {o['scene'][:40]}")

        # 5) 후보끼리 겹침
        for i, a in enumerate(d["ours"]):
            for b in d["ours"][i + 1:]:
                ov = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
                if ov > 5:
                    flag("후보끼리 겹침", f"{d['video']}: {a['start']:.0f}~{a['end']:.0f} vs {b['start']:.0f}~{b['end']:.0f} ({ov:.0f}초)")

    print(f"후보 {clips}개 | 평균 {statistics.mean(lengths):.0f}초 "
          f"(최단 {min(lengths):.0f} / 최장 {max(lengths):.0f}) | 이음새 {seams}곳")
    print("감정 분포: " + ", ".join(f"{k or '없음'} {v}" for k, v in sorted(appeals.items(), key=lambda x: -x[1])))
    total = sum(len(v) for v in issues.values())
    print(f"\n지적 {total}건" + (" — 없음" if not total else ""))
    for kind, msgs in sorted(issues.items(), key=lambda x: -len(x[1])):
        print(f"\n[{kind}] {len(msgs)}건")
        for m in msgs[:6]:
            print("   " + m)
        if len(msgs) > 6:
            print(f"   … 외 {len(msgs)-6}건")
    return total


if __name__ == "__main__":
    audit()
