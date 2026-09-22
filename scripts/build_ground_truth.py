"""벤치마크 채널(@onlylord)의 '터진 쇼츠'를 원본 설교에 역정렬해 정답지를 만든다.

결과: output/_eval/ground_truth.json — 쇼츠마다 {sermon, start, end, span, short_dur, views, title, cuts}
  cuts = 채널 편집자가 원본에서 **들어낸** 구간(점프컷). 2026-09-21 첫 실측: 7개 중 5개가 점프컷을 썼고
  원본 80~192초를 59~111초로 압축(중앙값 63%). 우리 엔진이 '연속 구간 하나'만 내던 것이 구조적 차이였다.

짝짓기는 자동이다: 쇼츠 설명란의 "일자 : 2026년 7월 15일 수요강해"에서 날짜를 뽑아 같은 날짜의 채널 영상을
모두 후보로 놓고, 자동자막을 문자 단위로 정렬해 **일치율이 가장 높은(≥MIN_COVERAGE) 영상**을 원본으로 확정한다.
사람이 짝을 적을 필요가 없고, 잘못 짝지으면 일치율이 낮아 자동으로 걸러진다.

사용법:
    venv/Scripts/python.exe scripts/build_ground_truth.py            # 상위 30개 쇼츠
    venv/Scripts/python.exe scripts/build_ground_truth.py 50         # 상위 50개
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
CHANNEL = "https://www.youtube.com/@onlylord"
MIN_BLOCK = 10          # 이 길이(문자) 이상 일치하는 블록만 신뢰
MIN_GAP = 25            # 이 길이(문자) 이상 통째로 빠진 곳만 '점프컷'으로 본다
MIN_COVERAGE = 0.80     # 쇼츠 대사의 이만큼이 원본에서 발견돼야 같은 설교로 인정
MIN_SHORT_DUR = 25      # 이보다 짧은 쇼츠는 설교 클립이 아니라 행사·예고편
MAX_SERMON_CANDIDATES = 6   # 같은 날짜 영상이 많으면 긴 것부터 이만큼만 시도

_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]
_DATE_RE = re.compile(r"일자\s*[:：]\s*(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")


def _ydl(opts: dict) -> yt_dlp.YoutubeDL:
    return yt_dlp.YoutubeDL({"quiet": True, "noprogress": True, "skip_download": True, **opts})


def list_channel(tab: str, limit: int) -> list[dict]:
    """채널 탭을 flat으로 훑는다(빠름). 캐시해 두고 재사용한다."""
    cache = EVAL_DIR / f"channel_{tab}.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        if len(rows) >= min(limit, 50):
            return rows
    with _ydl({"extract_flat": True, "playlistend": limit}) as ydl:
        info = ydl.extract_info(f"{CHANNEL}/{tab}", download=False)
    rows = [{"id": e.get("id"), "title": e.get("title") or "", "views": e.get("view_count") or 0,
             "dur": e.get("duration") or 0}
            for e in (info.get("entries") or []) if e.get("id")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def video_meta(vid: str, kind: str = "watch") -> dict:
    """제목·설명·길이·조회수(캐시)."""
    cache = EVAL_DIR / "meta" / f"{vid}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"https://www.youtube.com/shorts/{vid}" if kind == "shorts" else f"https://www.youtube.com/watch?v={vid}"
    try:
        with _ydl({}) as ydl:
            i = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 - 지워졌거나 지역 제한 → 건너뛴다
        print(f"  메타 실패 {vid}: {exc}")
        return {}
    m = {"id": vid, "title": i.get("title") or "", "desc": i.get("description") or "",
         "dur": i.get("duration") or 0, "views": i.get("view_count") or 0,
         "date": i.get("upload_date") or ""}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def ensure_caption(vid: str, kind: str = "watch") -> Path | None:
    p = CAP_DIR / f"{vid}.ko.json3"
    if p.exists():
        return p
    url = f"https://www.youtube.com/shorts/{vid}" if kind == "shorts" else f"https://www.youtube.com/watch?v={vid}"
    CAP_DIR.mkdir(parents=True, exist_ok=True)
    got = fetch_youtube_captions_json3(url, CAP_DIR / f"{vid}.json3")
    return got if got and got.exists() else None


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


def date_keys(y: int, mo: int, d: int) -> list[str]:
    """채널 영상 제목에 쓰이는 날짜 표기 변형들."""
    return [f"{mo:02d}/{d:02d}/{y}", f"{mo:02d}.{d:02d}.{y}", f"{mo}/{d}/{y}", f"{mo}.{d}.{y}",
            f"{_MONTHS[mo-1]} {d}, {y}", f"{_MONTHS[mo-1]} {d:02d}, {y}"]


def align(short_id: str, sermon_id: str) -> dict | None:
    """쇼츠 대사를 설교 자막에 정렬해 구간·점프컷·일치율을 낸다. 못 맞추면 None."""
    S = charstream(sermon_id); Sstr = "".join(c for c, _ in S)
    H = charstream(short_id); Hstr = "".join(c for c, _ in H)
    if len(Hstr) < 50 or len(Sstr) < 500:
        return None
    sm = difflib.SequenceMatcher(None, Hstr, Sstr, autojunk=False)
    blocks = sorted((b for b in sm.get_matching_blocks() if b.size >= MIN_BLOCK), key=lambda b: b.b)
    if not blocks:
        return None
    cov = sum(b.size for b in blocks) / len(Hstr)
    lo, hi = blocks[0].b, blocks[-1].b + blocks[-1].size
    cuts = []
    for a, b in zip(blocks, blocks[1:]):
        gs, ge = a.b + a.size, b.b
        if ge - gs >= MIN_GAP:
            cuts.append([round(S[gs][1], 1), round(S[min(ge, len(S) - 1)][1], 1)])
    st, en = S[lo][1], S[min(hi, len(S) - 1)][1]
    return {"sermon": sermon_id, "start": round(st, 1), "end": round(en, 1), "span": round(en - st, 1),
            "coverage": round(cov, 2), "cuts": cuts}


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    shorts = sorted(list_channel("shorts", 200), key=lambda r: -r["views"])[:top_n]
    videos = list_channel("videos", 400)
    existing = {}
    gt_path = EVAL_DIR / "ground_truth.json"
    if gt_path.exists():
        existing = json.loads(gt_path.read_text(encoding="utf-8"))

    gt: dict[str, dict] = dict(existing)
    for row in shorts:
        sh = row["id"]
        if sh in gt:
            continue
        m = video_meta(sh, "shorts")
        if not m or (m["dur"] or 0) < MIN_SHORT_DUR:
            continue
        dm = _DATE_RE.search(m["desc"])
        if not dm:
            print(f"{sh}: 설명란에 일자 없음 → 건너뜀 ({m['title'][:30]})")
            continue
        y, mo, d = int(dm.group(1)), int(dm.group(2)), int(dm.group(3))
        keys = date_keys(y, mo, d)
        cands = [v for v in videos if any(k in v["title"] for k in keys)]
        cands.sort(key=lambda v: -(v["dur"] or 0))
        cands = cands[:MAX_SERMON_CANDIDATES]
        if not cands:
            print(f"{sh}: {y}-{mo:02d}-{d:02d} 영상 없음 → 건너뜀")
            continue
        if not ensure_caption(sh, "shorts"):
            print(f"{sh}: 쇼츠 자막 없음 → 건너뜀")
            continue
        best = None
        for v in cands:
            if not ensure_caption(v["id"]):
                continue
            try:
                a = align(sh, v["id"])
            except Exception as exc:  # noqa: BLE001
                print(f"  정렬 오류 {v['id']}: {exc}")
                continue
            if a and (best is None or a["coverage"] > best["coverage"]):
                best = a
            if best and best["coverage"] >= 0.95:
                break
        if not best or best["coverage"] < MIN_COVERAGE:
            got = f"{best['coverage']:.0%}" if best else "없음"
            print(f"{sh}: 원본 못 찾음(최고 일치율 {got}) {m['title'][:30]}")
            continue
        gt[sh] = {**best, "short_dur": m["dur"], "views": m["views"], "title": m["title"]}
        print(f"{sh} {m['views']:>6}회 쇼츠 {m['dur']:>4}초 ← {best['sermon']} "
              f"{best['start']:.0f}~{best['end']:.0f} ({best['span']:.0f}초, 일치 {best['coverage']:.0%}) "
              f"점프컷 {len(best['cuts'])}곳 | {m['title'][:28]}")

    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    spans = [g["span"] for g in gt.values()]
    jump = [g for g in gt.values() if g["cuts"]]
    print(f"\n정답지 {len(gt)}개 (새로 {len(gt)-len(existing)}개). 점프컷 사용 {len(jump)}/{len(gt)}개, "
          f"원본 구간 {min(spans):.0f}~{max(spans):.0f}초")
    print("저장:", gt_path)


if __name__ == "__main__":
    main()
