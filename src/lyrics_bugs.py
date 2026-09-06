"""벅스(music.bugs.co.kr)에서 곡 가사를 '코드로' 직접 가져온다 — 모델 검색에 의존하지 않는다.

배경(2026-09-07): 곡 제목 → 정식 가사 경로를 claude -p + WebSearch/WebFetch에 맡겼더니,
프롬프트에 "벅스 고정"을 못박아도 모델이 검색을 건너뛰거나 기억으로 지어낸 가사를 내놓는
경우가 있었다(실측: '나의 갈 길 다 가도록'에 존재하지 않는 2절 가사). 같은 제목을 띄어쓰기만
다르게 넣으면 캐시도 빗나가 매번 다시 검색했다. 그래서 검색→트랙 페이지→<xmp> 가사 추출을
전부 결정적인 HTTP 코드로 바꾸고, 모델은 벅스에서 정말 못 찾았을 때의 폴백으로만 남긴다.

공개 함수:
- fetch_lyrics_from_bugs(title) -> (lines, meta) : 제목(또는 벅스 트랙 URL/번호)으로 가사
- normalize_title_key(title) : 캐시 키(공백·기호 무시)
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_TRACK_URL = "https://music.bugs.co.kr/track/{tid}"
_SEARCH_URL = "https://music.bugs.co.kr/search/track?q={q}"
_TRACK_REF_RE = re.compile(r"bugs\.co\.kr/track/(\d+)")
# 절 번호("1.", "2절")·"(후렴)" 같은 표지 줄 — 찬송가 등록본에 흔하고 자막이 아니다.
_MARKER_LINE_RE = re.compile(
    r"^[\(\[]?\s*(\d+\s*[.절)\]]?|후렴|반복|coda|verse\s*\d*|chorus)\s*[\)\]]?$", re.I
)


def _get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept-Language": "ko"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def normalize_title_key(title: str) -> str:
    """캐시/비교용 키: 공백·구두점·대소문자 차이를 무시한다('다가도록'=='다 가도록')."""
    return re.sub(r"[\s\W_]+", "", (title or "").lower())


def _strip_hymn_prefix(title: str) -> str:
    """'찬송가 384장 나의 갈 길 다 가도록' → '나의 갈 길 다 가도록' (검색어 정제)."""
    t = re.sub(r"(새?찬송가|통일찬송가|CCM)\s*", " ", title, flags=re.I)
    t = re.sub(r"\b\d+\s*장\b", " ", t)
    t = re.sub(r"[\(\[].*?[\)\]]", " ", t)  # 괄호 안 부연(영문 제목 등)
    return " ".join(t.split()).strip()


def parse_track_ref(text: str) -> int | None:
    """입력이 벅스 트랙 URL이나 트랙 번호면 그 id를 돌려준다(제목 대신 직접 지정용)."""
    m = _TRACK_REF_RE.search(text or "")
    if m:
        return int(m.group(1))
    s = (text or "").strip()
    if s.isdigit() and len(s) >= 5:
        return int(s)
    return None


def fetch_track_lyrics(track_id: int) -> tuple[list[str], dict]:
    """트랙 페이지의 <xmp> 가사 영역을 줄 단위로 돌려준다(빈 줄 제거, HTML 엔티티 해제)."""
    page = _get(_TRACK_URL.format(tid=track_id))
    m = re.search(r"<xmp[^>]*>(.*?)</xmp>", page, re.S)
    title_m = re.search(r"<title>(.*?)</title>", page, re.S)
    meta = {
        "track_id": track_id,
        "url": _TRACK_URL.format(tid=track_id),
        "page_title": html.unescape(title_m.group(1).strip()) if title_m else "",
    }
    if not m:
        return [], meta
    text = html.unescape(m.group(1))
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n")]
    return [ln for ln in lines if ln and not _MARKER_LINE_RE.match(ln)], meta


def search_tracks(query: str) -> list[dict]:
    """벅스 곡 검색 결과(첫 페이지)의 트랙 목록: [{track_id,title,artist,album}]."""
    page = _get(_SEARCH_URL.format(q=urllib.parse.quote(query)))
    out: list[dict] = []
    for m in re.finditer(r'<tr[^>]*trackId="(\d+)"[^>]*rowType="track"[^>]*>(.*?)</tr>', page, re.S):
        tid, body = int(m.group(1)), m.group(2)
        t = re.search(r'<p class="title"[^>]*>.*?title="([^"]*)"', body, re.S)
        a = re.search(r'<p class="artist"[^>]*>.*?title="([^"]*)"', body, re.S)
        al = re.search(r'class="album"[^>]*title="([^"]*)"', body, re.S)
        out.append({
            "track_id": tid,
            "title": html.unescape(t.group(1)) if t else "",
            "artist": html.unescape(a.group(1)) if a else "",
            "album": html.unescape(al.group(1)) if al else "",
        })
    return out


_HYMN_ARTIST_RE = re.compile(r"찬송가|찬양대|합창|성가|Unknown|교회|워십|worship", re.I)


def _rank(cands: list[dict], query: str) -> list[dict]:
    """제목이 정확히 같은 트랙(공백 무시)을 앞으로, 그 안에서는 찬송가/찬양대류 아티스트를
    먼저(회중이 부르는 원곡 가사일 확률이 높다), 나머지는 벅스 검색 순서를 유지한다.
    부분 일치(제목에 검색어 포함)는 그 다음, 그 밖은 뺀다(다른 곡 가사 방지)."""
    qk = normalize_title_key(query)
    exact, partial = [], []
    for c in cands:
        ck = normalize_title_key(c["title"])
        if ck == qk:
            exact.append(c)
        elif qk and (qk in ck or ck in qk) and len(ck) >= max(4, len(qk) // 2):
            partial.append(c)
    exact.sort(key=lambda c: 0 if _HYMN_ARTIST_RE.search(c.get("artist", "")) else 1)
    return exact + partial


def _lyrics_score(lines: list[str]) -> float:
    """같은 곡의 여러 등록본 중 '한 줄 = 한 소절(12~20자)'에 가까운 것을 고른다. 커버곡은
    한 소절을 반으로 쪼개 올린 경우가 많아 자막이 잘게 끊긴다."""
    if not lines:
        return -1e9
    avg = sum(len(ln) for ln in lines) / len(lines)
    return -abs(avg - 16.0)

def fetch_lyrics_from_bugs(
    title: str, max_tries: int = 6, log=print,
) -> tuple[list[str], dict]:
    """제목(또는 벅스 트랙 URL/번호)으로 가사를 가져온다. 못 찾으면 ([], {'reason': ...}).

    검색 결과 중 제목이 맞는 트랙을 순서대로 열어 가사가 있는 첫 트랙을 쓴다(벅스에는 가사
    미등록 트랙이 많다). 검색어는 원문 → 찬송가 접두 제거본 순으로 시도한다."""
    tid = parse_track_ref(title)
    if tid is not None:
        lines, meta = fetch_track_lyrics(tid)
        meta["query"] = title
        return lines, meta

    queries: list[str] = []
    for q in (title.strip(), _strip_hymn_prefix(title)):
        if q and q not in queries:
            queries.append(q)
    tried = 0
    for q in queries:
        try:
            cands = _rank(search_tracks(q), q)
        except Exception as e:  # noqa: BLE001 - 네트워크 실패는 폴백(모델)으로
            log(f"[bugs] 검색 실패 '{q}': {e}")
            continue
        log(f"[bugs] '{q}' 후보 {len(cands)}개: " + ", ".join(
            f"{c['track_id']}({c['title']}/{c['artist']})" for c in cands[:max_tries]))
        found: list[tuple[float, list[str], dict]] = []
        for c in cands[:max_tries]:
            tried += 1
            try:
                lines, meta = fetch_track_lyrics(c["track_id"])
            except Exception as e:  # noqa: BLE001
                log(f"[bugs] 트랙 {c['track_id']} 열기 실패: {e}")
                continue
            if not lines:
                continue
            meta.update({"query": q, "artist": c["artist"], "album": c["album"], "title": c["title"]})
            found.append((_lyrics_score(lines), lines, meta))
            if len(found) >= 3:  # 가사 있는 등록본 3개까지만 비교(요청 수 절약)
                break
        if found:
            found.sort(key=lambda t: -t[0])
            _, lines, meta = found[0]
            log(f"[bugs] 가사 확정: {meta['url']} ({meta['title']} / {meta['artist']}) {len(lines)}줄")
            return lines, meta
    return [], {"reason": f"벅스에서 '{title}' 가사를 찾지 못함 (후보 {tried}개 확인)"}
