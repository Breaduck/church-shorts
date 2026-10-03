"""자막 자동 교정 — 추임새 제거 + 문맥 교정을 렌더·편집기 초안에 **기본으로** 건다(2026-09-29).

사용자 신고: "자막에 '응. 어' 같은 의성어가 들어간다", "발음 그대로 필터링 없이 들어간다 — 조금만 생각하면
'아 이 단어겠구나' 하는 걸 못 잡는다"(실측 IOSxqPI2nUc: '한글릇'→한 걸음, '성리'→승리, '더보라'→드보라,
'따라습니다'→따라 합니다). 예전엔 ✨ AI 자막 교정 **버튼**을 눌러야만 했고 안 누르면 whisper 원문 그대로 구워졌다.

구조:
  1) 결정론(항상): 줄 텍스트에서 추임새 토큰 제거(captions.strip_filler_tokens_text). 모델 없이도 '응. 어'는 사라진다.
  2) 모델(옵션, 기본 on — config captions.auto_correct): highlights.correct_sermon_captions(Sonnet)로 문맥 교정.
     줄 수·순서 1:1이라 시간축은 그대로. 결과는 output/<id>/caption_fix_cache.json에 (줄 텍스트+맥락) 해시로 캐시 —
     같은 초안이면 두 번 호출하지 않고, 편집기 초안과 렌더가 같은 캐시를 읽어 **미리보기 = 결과물**이 유지된다.
적용 지점: captions.build_ass(text_fixer=) — 렌더의 비편집 경로(main.render_selected가 clip.caption_text_fixer로 전달),
web_app._caption_lines_for_clip — 편집기 초안(정밀 캐시가 있을 때만 모델 호출; 자동자막 초안은 결정론만).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.fsutil import atomic_write_text
from src.captions import strip_filler_tokens_text

CACHE_NAME = "caption_fix_cache.json"


def _cache_path(video_dir: Path) -> Path:
    return Path(video_dir) / CACHE_NAME


def _load_cache(video_dir: Path) -> dict:
    p = _cache_path(video_dir)
    try:
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def _save_cache(video_dir: Path, cache: dict) -> None:
    try:
        atomic_write_text(_cache_path(video_dir), json.dumps(cache, ensure_ascii=False, indent=1))
    except OSError:
        pass


def _plausible(orig: str, fixed: str) -> bool:
    """교정 결과가 '같은 줄'로 보이는가 — 길이가 2배 이상 달라지면 모델이 줄을 새로 쓴 것이라 원문 유지."""
    lo, lf = len(orig.replace(" ", "")), len(fixed.replace(" ", ""))
    if lo == 0 or lf == 0:
        return True
    return 0.5 <= lf / lo <= 2.0


def autofix_caption_texts(
    video_dir: Path, texts: list[str], context: str = "", model: str = "",
    use_model: bool = True, aggressive_filler: bool = False, thinking_tokens: int = 1024,
    reference: str = "",
) -> tuple[list[str], list[str]]:
    """줄 텍스트 목록을 1:1로 교정해 (교정된 줄들, 강조어들)을 돌려준다. 빈 줄은 호출부가 뺀다."""
    det = [strip_filler_tokens_text(str(t or ""), aggressive_filler) for t in texts]
    if not use_model or not any(det):
        return det, []
    # 모델을 바꾸면(config auto_correct_model) 옛 모델 결과를 재사용하지 않게 키에 넣는다. 기본값("")일 땐
    # 예전 키 그대로라 기존 캐시가 살아 있다(불필요한 재호출 없음).
    key_parts = [det, context, reference] + ([model] if model else [])
    key = hashlib.sha1(json.dumps(key_parts, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    cache = _load_cache(video_dir)
    hit = cache.get(key)
    if isinstance(hit, dict) and isinstance(hit.get("lines"), list) and len(hit["lines"]) == len(det):
        return [str(x) for x in hit["lines"]], [str(h) for h in (hit.get("hl") or [])]
    from src.highlights import correct_sermon_captions
    from src.models import EXECUTION_MODEL

    try:
        corrected, hl = correct_sermon_captions(
            det, context=context, model=model or EXECUTION_MODEL, thinking_tokens=thinking_tokens,
            reference=reference,
        )
    except Exception as exc:  # noqa: BLE001 - 모델 실패(한도 등)는 결정론 결과로
        print(f"[caption_autofix] 모델 교정 실패 → 추임새 제거만 적용: {exc}", flush=True)
        return det, []
    out: list[str] = []
    for o, c in zip(det, corrected):
        c = strip_filler_tokens_text(str(c or "").strip(), aggressive_filler)
        out.append(c if _plausible(o, c) else o)
    cache[key] = {"lines": out, "hl": hl, "src": det, "context": context}
    _save_cache(video_dir, cache)
    changed = sum(1 for o, c in zip(det, out) if o.strip() != c.strip())
    print(f"[caption_autofix] {len(out)}줄 중 {changed}줄 교정, 강조어 {len(hl)}개", flush=True)
    return out, hl


def make_text_fixer(video_dir: Path, clip, cfg: dict, use_model: bool = True):
    """captions.build_ass(text_fixer=)에 넘길 콜러블. 설교 클립에만 의미가 있고 찬양(가사)은 None.
    use_model=False면 추임새 제거(결정론)만 한다."""
    caps = (cfg or {}).get("captions", {}) or {}
    if getattr(clip, "clip_type", "") == "praise":
        return None
    if not caps.get("auto_correct", True):
        use_model = False
    if not caps.get("strip_filler", True) and not use_model:
        return None
    context = " / ".join(
        s for s in (
            str(getattr(clip, "title", "") or ""), str(getattr(clip, "core_line", "") or ""),
            " ".join(str(k) for k in (getattr(clip, "keywords", None) or [])),
        ) if s
    )
    aggressive = bool(caps.get("aggressive_filler", False))
    model = str(caps.get("auto_correct_model", "") or "")
    # 두 번째 인식기: 유튜브 자동자막(transcript.json)의 같은 구간 텍스트. whisper와 단어가 다를 때 대조용.
    reference = _reference_text(Path(video_dir), clip) if use_model else ""

    def _fix(texts: list[str]) -> list[str]:
        fixed, _hl = autofix_caption_texts(
            Path(video_dir), list(texts), context=context, model=model,
            use_model=use_model, aggressive_filler=aggressive, reference=reference,
        )
        # 강조어(hl)는 캐시에만 남긴다 — 자동 패스가 caption_highlights를 채우면 줄마다 형광 강조가 켜져
        # 사용자가 고르지 않은 스타일 변화가 생긴다(✨ 버튼을 눌렀을 때만 반영).
        return fixed

    return _fix


def _reference_text(video_dir: Path, clip) -> str:
    """유튜브 자동자막(transcript.json)에서 클립 구간의 텍스트를 이어 붙인다. 없으면 빈 문자열."""
    try:
        from src.main import json_load_transcript

        tp = Path(video_dir) / "transcript.json"
        if not tp.exists():
            return ""
        segs = json_load_transcript(tp)["segments"]
        a, b = float(getattr(clip, "start", 0.0)), float(getattr(clip, "end", 0.0))
        parts = [(s.text or "").strip() for s in segs if s.end >= a - 1.0 and s.start <= b + 1.0]
        text = " ".join(p for p in parts if p)
        return text.replace(">>", " ")
    except Exception:  # noqa: BLE001 - 참조는 보조 정보라 실패해도 교정은 진행
        return ""
