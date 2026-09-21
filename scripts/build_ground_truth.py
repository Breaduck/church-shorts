"""벤치마크 채널(@onlylord)의 '터진 쇼츠'를 원본 설교에 역정렬해 정답지를 만든다.

결과: output/_eval/ground_truth.json — 쇼츠마다 {sermon, start, end, span, short_dur, views, title, cuts}
  cuts = 채널 편집자가 원본에서 **들어낸** 구간(점프컷) 목록. 2026-09-21 실측: 7개 중 5개가 점프컷을 썼고
  원본 80~192초를 59~111초로 압축(중앙값 63%). 우리 엔진이 '연속 구간 하나'만 내던 것이 구조적 차이였다.

사용법: venv/Scripts/python.exe scripts/build_ground_truth.py
  (쇼츠 설명란의 "일자/제목"으로 원본 설교를 사람이 찾아 PAIRS에 적는다. 자동자막을 yt-dlp로 받아 문자 단위
   difflib 정렬 → 시각 복원.)
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import yt_dlp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.youtube_captions import fetch_youtube_captions_json3, parse_json3_to_transcript  # noqa: E402

EVAL_DIR = Path("output/_eval")
CAP_DIR = EVAL_DIR / "captions"

# 원본 설교 id → 그 설교에서 나온 쇼츠 id들 (쇼츠 설명란 "일자 : 2026년 7월 15일 수요강해 제목 : …"로 찾음)
PAIRS: dict[str, list[str]] = {
    "NTHrx-w8hnk": ["HQMyIBO42-s", "pNULu_qodPU"],   # 07-15 수요강해 이 일을 위해 보내심을 받았노라
    "VXtJAJAMCUA": ["756zHqBHIIM", "2lvK7lbspgc"],   # 08-09 주일 낮 예기치 못한 하나님의 도우심
    "GZbRNcH4U1c": ["xH4iBAY3OHQ"],                  # 07-12 주일 낮 돌아가는 길은 없다
    "AVzWcGgsOPg": ["IwSGfR7Y2dc"],                  # 08-02 주일 낮 변화가 필요한 사람들에게
    "mjg0tcadzjY": ["Nwm0rZw4nd4"],                  # 08-19 수요강해 주님이 원하시는 것
    # "czqS-6O7XB8": ["Td5sm5sCgBA", "yRHtHK8yA0M"],  # 07-19 주일 낮 — 자막 정렬 실패(coverage 0.06) → 보류
}
MIN_BLOCK = 10   # 이 길이(문자) 이상 일치하는 블록만 신뢰
MIN_GAP = 25     # 이 길이(문자) 이상 통째로 빠진 곳만 '점프컷'으로 본다


def ensure_caption(vid: str, url: str) -> Path | None:
    p = CAP_DIR / f"{vid}.ko.json3"
    if p.exists():
        return p
    return fetch_youtube_captions_json3(url, CAP_DIR / f"{vid}.json3")


def charstream(vid: str) -> list[tuple[str, float]]:
    """(문자, 시각) 스트림 — 공백·문장부호 제거, 단어 안에서 시각 선형 보간."""
    t = parse_json3_to_transcript(CAP_DIR / f"{vid}.ko.json3", 0)
    out: list[tuple[str, float]] = []
    for s in t.segments:
        for w in s.words:
            txt = re.sub(r"[^\w가-힣]", "", w.text)
            for i, ch in enumerate(txt):
                out.append((ch, w.start + (w.end - w.start) * (i / max(1, len(txt)))))
    return out


def short_meta(short_id: str) -> dict:
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "noprogress": True}) as ydl:
        i = ydl.extract_info(f"https://www.youtube.com/shorts/{short_id}", download=False)
    return {"views": i.get("view_count") or 0, "dur": i.get("duration") or 0, "title": i.get("title") or ""}


def main() -> None:
    CAP_DIR.mkdir(parents=True, exist_ok=True)
    gt: dict[str, dict] = {}
    for sid, shorts in PAIRS.items():
        if not ensure_caption(sid, f"https://www.youtube.com/watch?v={sid}"):
            print("자막 없음(설교):", sid); continue
        S = charstream(sid); Sstr = "".join(c for c, _ in S)
        for sh in shorts:
            if not ensure_caption(sh, f"https://www.youtube.com/shorts/{sh}"):
                print("자막 없음(쇼츠):", sh); continue
            H = charstream(sh); Hstr = "".join(c for c, _ in H)
            sm = difflib.SequenceMatcher(None, Hstr, Sstr, autojunk=False)
            blocks = sorted((b for b in sm.get_matching_blocks() if b.size >= MIN_BLOCK), key=lambda b: b.b)
            if not blocks:
                print("정렬 실패:", sh); continue
            cov = sum(b.size for b in blocks) / max(1, len(Hstr))
            lo, hi = blocks[0].b, blocks[-1].b + blocks[-1].size
            cuts = []
            for a, b in zip(blocks, blocks[1:]):
                gs, ge = a.b + a.size, b.b
                if ge - gs >= MIN_GAP:
                    cuts.append([round(S[gs][1], 1), round(S[min(ge, len(S) - 1)][1], 1)])
            m = short_meta(sh)
            st, en = S[lo][1], S[min(hi, len(S) - 1)][1]
            gt[sh] = {
                "sermon": sid, "start": round(st, 1), "end": round(en, 1), "span": round(en - st, 1),
                "short_dur": m["dur"], "views": m["views"], "title": m["title"], "coverage": round(cov, 2),
                "cuts": cuts,
            }
            print(f"{sh} {m['views']:>6}회 쇼츠 {m['dur']:>4}초 ← 원본 {st:.0f}~{en:.0f} ({en-st:.0f}초, 일치 {cov:.0%}) "
                  f"점프컷 {len(cuts)}곳")
    (EVAL_DIR / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", EVAL_DIR / "ground_truth.json")


if __name__ == "__main__":
    main()
