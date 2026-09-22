"""점프컷 클립이 **실제 mp4로도** 제대로 나오는지 확인한다.

여태 점프컷은 전사 데이터 위에서만 검증했다(길이 계산·이음새 규칙). 실제 렌더에서 확인해야 할 것:
  1) 영상 길이 = 남길 구간의 합 (중간이 진짜로 빠졌는가)
  2) 자막이 압축된 타임라인으로 리매핑됐는가 (안 하면 뒤로 갈수록 자막이 밀린다)
  3) 이음새에서 오디오/영상이 깨지지 않는가 (프레임·샘플 수로 확인)

사용법:
    venv/Scripts/python.exe scripts/verify_jumpcut_render.py <video_id> [clip_index]
결과물: output/_eval/render_test/ (기존 클립은 절대 건드리지 않는다)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.highlights import clip_effective_duration, load_clips_json  # noqa: E402
from src.main import load_config  # noqa: E402
from src.render import render_clip  # noqa: E402
from src.youtube_captions import parse_json3_to_transcript  # noqa: E402

OUT = Path("output/_eval/render_test")


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,nb_frames,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    d = json.loads(out.stdout)
    res = {"duration": float(d["format"]["duration"])}
    for s in d.get("streams", []):
        res[s.get("codec_type", "?")] = float(s.get("duration") or 0)
    return res


def load_segments(video_dir: Path) -> list:
    """base 전사(유튜브 자동자막)를 세그먼트로. 자막 싱크 확인에 이걸 쓴다."""
    j3 = next(video_dir.glob("*.json3"), None)
    if j3:
        return parse_json3_to_transcript(j3, 0).segments
    tj = video_dir / "transcript.json"
    if tj.exists():
        from src.transcribe import Segment, Word
        data = json.loads(tj.read_text(encoding="utf-8"))
        return [Segment(start=s["start"], end=s["end"], text=s["text"],
                        words=[Word(**w) for w in s.get("words", [])])
                for s in data["segments"]]
    raise SystemExit("전사본을 찾을 수 없습니다")


def main() -> None:
    vid = sys.argv[1] if len(sys.argv) > 1 else "zvsOYhjdmks"
    want_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    video_dir = Path("output") / vid
    source = video_dir / "source.mp4"
    if not source.exists():
        raise SystemExit(f"원본 영상 없음: {source}")
    clips = load_clips_json(video_dir / "clips.json")
    cands = [(i, c) for i, c in enumerate(clips) if len(getattr(c, "keep_ranges", None) or []) > 1]
    if want_idx is not None:
        cands = [(want_idx, clips[want_idx])]
    if not cands:
        raise SystemExit("점프컷 클립이 없습니다 (keep_ranges가 2조각 이상인 후보 없음)")
    idx, clip = cands[0]
    cfg = load_config(Path("config.yaml"))
    segs = load_segments(video_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    kr = [[float(a), float(b)] for a, b in clip.keep_ranges]
    expect = sum(b - a for a, b in kr)
    print(f"클립 {idx}: {clip.title}")
    print(f"  원본 구간 {clip.start:.1f}~{clip.end:.1f} ({clip.end-clip.start:.1f}초)")
    print(f"  남길 구간 {len(kr)}조각 → {expect:.1f}초")
    for a, b in kr:
        print(f"    {a:.1f}~{b:.1f} ({b-a:.1f}초)")

    # 1) 점프컷 그대로 렌더
    jc_path = OUT / f"{vid}_clip{idx}_jumpcut.mp4"
    render_clip(source, segs, clip, jc_path, cfg["render"], cfg["captions"])
    jc = probe(jc_path)

    # 2) 비교용: 같은 클립을 통짜로 렌더(점프컷 없음)
    import copy
    whole = copy.deepcopy(clip)
    whole.keep_ranges = []
    wh_path = OUT / f"{vid}_clip{idx}_whole.mp4"
    render_clip(source, segs, whole, wh_path, cfg["render"], cfg["captions"])
    wh = probe(wh_path)

    pad = float(cfg["render"].get("end_pad_sec", 0) or 0)
    outro = cfg["render"].get("outro", {}) or {}
    outro_sec = float(outro.get("duration_sec", 0) or 0) if outro.get("enabled") else 0.0
    overhead = pad + outro_sec

    print(f"\n  점프컷 렌더: {jc['duration']:.1f}초 (기대 {expect + overhead:.1f}초 = 내용 {expect:.1f} + 여운/로고 {overhead:.1f})")
    print(f"  통짜 렌더  : {wh['duration']:.1f}초")
    print(f"  줄어든 양  : {wh['duration'] - jc['duration']:.1f}초 (빼려던 양 {clip.end - clip.start - expect:.1f}초)")

    ok = True
    if abs(jc["duration"] - (expect + overhead)) > 2.0:
        print("  [실패] 길이가 남길 구간 합과 다릅니다 — select 필터가 안 먹었거나 구간이 어긋남"); ok = False
    if wh["duration"] - jc["duration"] < 3.0:
        print("  [실패] 통짜 렌더와 길이 차이가 거의 없습니다 — 점프컷이 적용되지 않았습니다"); ok = False
    if abs(jc.get("video", 0) - jc.get("audio", 0)) > 1.5:
        print(f"  [실패] 영상({jc.get('video',0):.1f}s)과 오디오({jc.get('audio',0):.1f}s) 길이가 어긋납니다"); ok = False

    # 3) 자막 리매핑 확인: 마지막 자막이 영상 길이를 넘지 않아야 한다
    ass = jc_path.with_suffix(".ass")
    if not ass.exists():
        ass = next(OUT.glob(f"{vid}_clip{idx}_jumpcut*.ass"), None)
    if ass and ass.exists():
        import re as _re
        times = [m.group(1) for m in _re.finditer(r"^Dialogue:[^,]*,([^,]+),", ass.read_text(encoding="utf-8"), _re.M)]
        def sec(t: str) -> float:
            h, m_, s = t.split(":")
            return int(h) * 3600 + int(m_) * 60 + float(s)
        if times:
            last = max(sec(t) for t in times)
            print(f"  자막 마지막 시작 {last:.1f}초 / 영상 {jc['duration']:.1f}초")
            if last > jc["duration"] + 0.5:
                print("  [실패] 자막이 영상 길이를 넘습니다 — 압축 타임라인 리매핑 누락"); ok = False
    else:
        print("  (ass 파일이 정리돼 자막 시각은 확인 못 함)")

    print("\n결과:", "정상" if ok else "문제 있음")
    print("영상:", jc_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
