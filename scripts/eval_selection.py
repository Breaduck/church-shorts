"""선정 엔진 평가 하네스 — '정답지'로 채점한다.

배경(2026-09-21): "재밌는 부분/교훈 있는 부분을 못 잡는다"는 불만이 반복됐는데, 그때마다 프롬프트만 고치고
결과는 감으로 판단했다. 성과 데이터(feedback.json)는 0건이라 검증 수단이 없었다.
→ 실제로 터진 벤치마크 채널(@onlylord) 쇼츠를 **원본 설교 전사본에 역으로 정렬**해 "사람이 고른 정답 구간"을
만들고(scripts/build_ground_truth.py), 우리 엔진이 그 구간을 잡는지 재현율로 잰다.

사용법:
    venv/Scripts/python.exe scripts/eval_selection.py            # 정답지 있는 설교 전부
    venv/Scripts/python.exe scripts/eval_selection.py NTHrx-w8hnk  # 한 편만

판정: 정답 쇼츠가 실제로 남긴 소재(원본 구간 - 편집자가 들어낸 구간)를 우리 컷이 50% 이상 덮으면 '잡았다'(hit).
결과는 output/_eval/eval_<video>.json 과 표로 출력.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import load_config  # noqa: E402
from src.sermon_cut import (  # noqa: E402
    Sentence,
    build_sentences,
    build_thesis_cut_prompt,
    cut_keep_ranges,
    dedupe_cuts,
    drop_bible_story_cuts,
    drop_polemic_cuts,
    merge_adjacent_cuts,
    verify_and_fix,
    _eff_dur,
    _skips_of,
)
from src.highlights import _invoke_claude_json  # noqa: E402
from src.youtube_captions import parse_json3_to_transcript  # noqa: E402

EVAL_DIR = Path("output/_eval")
CAP_DIR = EVAL_DIR / "captions"
HIT_RATIO = 0.5


def load_transcript(video_id: str):
    path = CAP_DIR / f"{video_id}.ko.json3"
    if not path.exists():
        raise SystemExit(f"자막 없음: {path} (scripts/build_ground_truth.py 먼저 실행)")
    tr = parse_json3_to_transcript(path, 0.0)
    if tr.segments:
        tr.duration_sec = tr.segments[-1].end
    return tr


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def truth_kept(t: dict) -> list[tuple[float, float]]:
    """정답 쇼츠가 실제로 **남긴** 구간(원본 구간에서 편집자가 들어낸 cuts를 뺀 것).

    원본 구간으로 재현율을 재면 불공정하다: 편집자가 192초에서 68초만 남긴 클립을 우리가
    그 68초와 똑같이 잡아도 원본 대비 겹침은 35%밖에 안 나온다. 실제 쇼츠에 들어간 소재를
    얼마나 담았는지로 재야 한다."""
    kept: list[tuple[float, float]] = []
    cur = t["start"]
    for a, b in t.get("cuts") or []:
        if a > cur:
            kept.append((cur, a))
        cur = max(cur, b)
    if cur < t["end"]:
        kept.append((cur, t["end"]))
    return kept


def coverage(ours: list[dict], kept: list[tuple[float, float]]) -> tuple[float, int | None]:
    """우리 후보 하나가 정답이 남긴 소재를 얼마나 덮는가(0~1)와 그 후보 번호."""
    total = sum(b - a for a, b in kept) or 1e-6
    best, best_cov = None, 0.0
    for i, o in enumerate(ours):
        ov = sum(overlap((o["start"], o["end"]), k) for k in kept)
        if ov / total > best_cov:
            best, best_cov = i, ov / total
    return best_cov, best


def run_one(video_id: str, truths: list[dict], cfg: dict, model: str) -> dict:
    h = cfg["highlights"]
    tr = load_transcript(video_id)
    sentences = build_sentences(tr)
    print(f"[eval] {video_id}: {len(tr.segments)}조각 → {len(sentences)}문장, {tr.duration_sec/60:.0f}분", flush=True)

    hard_max = float(h.get("hard_max_duration_sec", 90))
    max_span = float(h.get("max_span_sec", 180))
    prompt = build_thesis_cut_prompt(
        sentences, tr.duration_sec, int(h["min_clips"]), int(h["max_clips"]),
        hard_max_sec=hard_max, max_span_sec=max_span,
    )
    raw_cuts = _invoke_claude_json(
        prompt, model=model, thinking_tokens=int(h.get("thinking_tokens", 8192)),
        timeout_sec=1200, max_clips=int(h["max_clips"]),
    )
    fixed, logs = [], []
    for rc in raw_cuts:
        cut, log = verify_and_fix(
            rc, sentences, float(h["min_duration_sec"]), float(h["max_duration_sec"]), hard_max,
            max_span_sec=max_span,
        )
        logs.append({"raw": rc, "fixed": cut, "log": log})
        if cut:
            fixed.append(cut)
    fixed, _ = merge_adjacent_cuts(fixed, sentences, hard_max)
    fixed, _ = drop_polemic_cuts(fixed, sentences)
    fixed, _ = drop_bible_story_cuts(fixed, sentences)
    fixed = dedupe_cuts(fixed, sentences)

    ours = []
    for c in fixed:
        ours.append({
            "start": round(sentences[c["start"]].start, 1),
            "end": round(sentences[c["end"]].end, 1),
            "eff": round(_eff_dur(sentences, c["start"], c["end"], _skips_of(c)), 1),
            "keep_ranges": cut_keep_ranges(c, sentences),
            "appeal": c.get("appeal", ""),
            "scene": c.get("scene", ""),
            "hook": sentences[c["start"]].text[:70],
        })

    results = []
    for t in truths:
        kept = truth_kept(t)
        cov, best = coverage(ours, kept)
        results.append({
            "short": t["short"], "views": t["views"], "title": t["title"],
            "truth": [t["start"], t["end"]], "truth_kept_sec": round(sum(b - a for a, b in kept), 1),
            "short_dur": t["short_dur"], "hit": cov >= HIT_RATIO,
            "coverage": round(cov, 2), "matched": best,
            "ours_eff": ours[best]["eff"] if best is not None else 0.0,
            "matched_span": [ours[best]["start"], ours[best]["end"]] if best is not None else [],
            "matched_pieces": len(ours[best]["keep_ranges"]) if best is not None else 0,
            "matched_hook": ours[best]["hook"] if best is not None else "",
        })
    return {"video": video_id, "ours": ours, "truths": results, "verify": logs}


def main() -> None:
    gt = json.loads((EVAL_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    by_sermon: dict[str, list[dict]] = {}
    for short_id, g in gt.items():
        by_sermon.setdefault(g["sermon"], []).append({**g, "short": short_id})

    targets = sys.argv[1:] or list(by_sermon)
    cfg = load_config(Path("config.yaml"))
    model = str(cfg["highlights"].get("model", "opus"))

    all_rows = []
    for vid in targets:
        out = run_one(vid, by_sermon[vid], cfg, model)
        (EVAL_DIR / f"eval_{vid}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n--- {vid}: 우리 후보 {len(out['ours'])}개")
        for o in out["ours"]:
            jc = f" 점프컷{len(o['keep_ranges'])}조각" if len(o["keep_ranges"]) > 1 else ""
            print(f"    {o['start']:.0f}~{o['end']:.0f} ({o['eff']:.0f}초{jc}) [{o['appeal']}] {o['hook'][:50]}")
        for r in out["truths"]:
            mark = "O 잡음" if r["hit"] else "X 놓침"
            print(f"    {mark} 소재 {r['coverage']:.0%} | 정답 {r['truth_kept_sec']:.0f}초→쇼츠 {r['short_dur']}초 · "
                  f"우리 {r['ours_eff']:.0f}초/{r['matched_pieces']}조각 | {r['views']}회 {r['title'][:28]}")
            all_rows.append(r)

    hits = sum(1 for r in all_rows if r["hit"])
    avg = sum(r["coverage"] for r in all_rows) / max(1, len(all_rows))
    print("")
    print(f"==== 정답 {len(all_rows)}개 중 {hits}개 잡음 "
          f"(재현율 {hits/max(1,len(all_rows)):.0%}), 평균 소재 커버 {avg:.0%}")


if __name__ == "__main__":
    main()
