"""사용 가능한 글꼴 레지스트리.

편집기에서 글씨체를 고를 수 있게, assets/fonts 와 '글씨체/' 폴더의 모든 ttf/otf를 스캔해
(1) 화면 표시용 이름, (2) libass가 실제로 찾는 family 이름(ASS Fontname), (3) 파일 경로를 뽑는다.

주의: 영상 렌더(libass)는 파일명이 아니라 폰트 내부의 family 이름으로 폰트를 찾는다. 그래서
fontTools로 name 테이블에서 family 이름을 추출해 ASS Fontname에 그대로 넣어야 글씨체가 실제로 바뀐다.
또한 libass가 폰트를 찾도록 모든 폰트 파일을 한 폴더(_allfonts)로 모아 fontsdir로 넘긴다.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fontTools.ttLib import TTFont

# 렌더 시 subtitles 필터의 fontsdir로 넘길 통합 폴더 (모든 폰트를 여기에 모은다)
CONSOLIDATED_DIR = Path("assets/_allfonts")
_SOURCE_DIRS = [Path("assets/fonts"), Path("글씨체")]

_registry_cache: list[dict] | None = None


def _decode_name(record) -> str:
    try:
        return record.toUnicode()
    except Exception:  # noqa: BLE001
        return ""


def _extract_names(font_path: Path) -> tuple[str, str]:
    """(ASS용 family 이름, 표시용 이름)을 반환. family는 영문 우선, 표시는 한글 우선."""
    try:
        tt = TTFont(str(font_path), fontNumber=0, lazy=True)
        name = tt["name"]
    except Exception:  # noqa: BLE001 - 손상/특이 폰트는 파일명으로 폴백
        stem = font_path.stem
        return stem, stem

    family_en = family_any = display_ko = ""
    for rec in name.names:
        if rec.nameID not in (1, 16):  # 1=Family, 16=Typographic Family
            continue
        s = _decode_name(rec).strip()
        if not s:
            continue
        family_any = family_any or s
        # platformID 3=Windows. langID 0x409=English(US), 0x412=Korean
        if rec.platformID == 3 and rec.langID == 0x409 and rec.nameID == 1:
            family_en = family_en or s
        if rec.platformID == 3 and rec.langID == 0x412:
            display_ko = display_ko or s
    try:
        tt.close()
    except Exception:  # noqa: BLE001
        pass
    family = family_en or family_any or font_path.stem
    display = display_ko or family
    return family, display


def _iter_font_files():
    for base in _SOURCE_DIRS:
        if not base.exists():
            continue
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            for fp in base.rglob(ext):
                if fp.name.startswith("."):  # macOS AppleDouble(._*) 등 메타데이터 파일 제외
                    continue
                yield fp


def get_font_registry(force: bool = False) -> list[dict]:
    """[{name(표시), family(ASS Fontname), file(파일명), path}] 목록. 표시이름 기준 정렬.
    같은 family가 여러 파일이면 하나만 남긴다(대표 파일)."""
    global _registry_cache
    if _registry_cache is not None and not force:
        return _registry_cache

    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)
    by_family: dict[str, dict] = {}
    for fp in _iter_font_files():
        family, display = _extract_names(fp)
        # 통합 폴더로 복사(libass fontsdir용). 파일명 충돌 방지 위해 필요시 부모명 접두.
        dest = CONSOLIDATED_DIR / fp.name
        if dest.exists() and dest.stat().st_size != fp.stat().st_size:
            dest = CONSOLIDATED_DIR / f"{fp.parent.name}_{fp.name}"
        try:
            if not dest.exists():
                shutil.copy2(fp, dest)
        except Exception:  # noqa: BLE001
            pass
        # 대표 폰트: Bold/Regular 등 여러 웨이트 중 첫 번째만 목록에 (family 단위로 하나)
        if family not in by_family:
            by_family[family] = {"name": display, "family": family, "file": dest.name, "path": str(dest)}

    registry = sorted(by_family.values(), key=lambda d: d["name"])
    _registry_cache = registry
    return registry


def default_font_dir_for_ass() -> str:
    """subtitles 필터 fontsdir에 넘길, 이스케이프된 통합 폴더 경로."""
    get_font_registry()  # 통합 폴더 보장
    return str(CONSOLIDATED_DIR.resolve()).replace("\\", "/").replace(":", "\\:")


_metrics_cache: dict[str, tuple | None] = {}


def _get_font_metrics(family: str):
    """family의 (unitsPerEm, cmap, hmtx)를 캐싱해서 반환. 없으면 None."""
    if family in _metrics_cache:
        return _metrics_cache[family]
    entry = next((f for f in get_font_registry() if f["family"] == family), None)
    metrics = None
    if entry is not None:
        try:
            tt = TTFont(entry["path"], fontNumber=0, lazy=True)
            metrics = (tt["head"].unitsPerEm, tt.getBestCmap(), tt["hmtx"])
        except Exception:  # noqa: BLE001 - 측정 실패 시 호출자가 근사치로 폴백
            metrics = None
    _metrics_cache[family] = metrics
    return metrics


def measure_text_width_px(text: str, family: str, font_size_px: float) -> float | None:
    """실제 폰트 파일의 글리프 advance width로 텍스트 픽셀 폭을 정확히 잰다.

    브라우저 미리보기(@font-face로 같은 폰트 파일을 쓴다)와 렌더(libass)가 같은 파일
    기준으로 폭을 재게 되어, 글자당 평균폭을 추정하던 예전 방식(폰트마다 실제 폭이
    달라 추정이 어긋남 — 예: Pretendard용으로 맞춘 비율을 Gmarket Sans에 그대로 쓰면
    미리보기와 실제 렌더 결과 크기가 달라 보이는 문제)보다 훨씬 정확하다.
    측정 불가(폰트 못 찾음 등)면 None — 호출자가 근사치 공식으로 폴백해야 한다."""
    m = _get_font_metrics(family)
    if m is None or not text:
        return None
    upm, cmap, hmtx = m
    total = 0.0
    for ch in text:
        gname = cmap.get(ord(ch))
        try:
            aw = hmtx[gname][0] if gname else upm * 0.55
        except KeyError:
            aw = upm * 0.55
        total += aw
    return total * (font_size_px / upm)
