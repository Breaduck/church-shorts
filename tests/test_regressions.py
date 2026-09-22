"""재발 버그 회귀 테스트.

여기 있는 테스트는 전부 **실제로 한 번 이상 터졌던 사고**를 그대로 못 박은 것이다.
(설교 자막·선정 파이프라인은 순수 함수가 많은데 테스트가 하나도 없어서, 같은 버그가
얼굴만 바꿔 반복됐다: "끝이 뚝 끊긴다" 3가지 다른 원인, 자막 위치 부호 반전,
1줄 폭 분절 실패, 무음 제거 후 자막 밀림 등.)

새 버그를 고칠 때는 여기에 그 상황을 재현하는 테스트를 먼저 추가할 것.

    venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.captions import (
    CaptionLine,
    _clamp_lines_non_overlap,
    _remap_after_silence_removal,
    _y_position,
    chunk_words_into_lines,
    shift_ass_times,
)
from src.highlights import Clip, load_clips_json, save_clips_json
from src.models import ALLOWED_MODELS, ANALYSIS_MODEL, EXECUTION_MODEL, sanitize
from src.render import _combine_keep, _intersect_intervals
from src.scoring import compute_scores
from src.sermon_cut import (
    Sentence,
    _is_incomplete,
    _starts_with_structure_marker,
    build_sentences,
    dedupe_cuts,
    merge_adjacent_cuts,
    trim_lead_words,
    verify_and_fix,
)
from src.transcribe import Segment, Transcript, Word


def W(a: float, b: float, t: str) -> Word:
    return Word(start=a, end=b, text=t)


def S(idx: int, a: float, b: float, t: str) -> Sentence:
    return Sentence(idx=idx, start=a, end=b, text=t)


# ───────────────────────── 문장 재조립 (선정 v2 1단계) ─────────────────────────

def test_build_sentences_splits_on_punctuation():
    """마침표/물음표에서 문장이 갈려야 한다. 여기가 어긋나면 v2의 모든 문장 번호가 밀린다."""
    tr = Transcript(language="ko", duration_sec=10.0, segments=[
        Segment(start=0.0, end=6.0, text="", words=[
            W(0.0, 0.5, "하나님은"), W(0.5, 1.0, "사랑이십니다."),
            W(1.2, 1.7, "여러분"), W(1.7, 2.3, "믿으십니까?"),
            W(2.5, 3.0, "저는"), W(3.0, 3.6, "믿습니다."),
        ]),
    ])
    sents = build_sentences(tr)
    assert [s.text for s in sents] == [
        "하나님은 사랑이십니다.", "여러분 믿으십니까?", "저는 믿습니다.",
    ]
    assert sents[0].start == 0.0 and sents[0].end == 1.0
    assert [s.idx for s in sents] == [0, 1, 2]


def test_build_sentences_splits_on_long_pause():
    """마침표가 없어도 1.2초 이상 침묵이면 문장을 나눈다(자동자막에 구두점이 빠지는 경우)."""
    tr = Transcript(language="ko", duration_sec=10.0, segments=[
        Segment(start=0.0, end=8.0, text="", words=[
            W(0.0, 0.4, "그래서"), W(0.4, 0.9, "제가"), W(0.9, 1.4, "말씀드립니다"),
            W(4.0, 4.4, "여러분"), W(4.4, 5.0, "감사합시다"),
        ]),
    ])
    sents = build_sentences(tr)
    assert len(sents) == 2
    assert sents[1].text.startswith("여러분")


# ───────── "끝이 뚝 끊긴다" 재발 방지: 미완 문장 판정 + 끝 확장 ─────────

@pytest.mark.parametrize("text", [
    "그래서 우리가 기도할 때,",          # 쉼표로 끝남
    "여러분 이거 아십니까?",             # 질문 → 답이 다음 문장
    "하나님이 우리를 사랑하시는데",       # 접속형 어미(-는데)
])
def test_is_incomplete_detects_dangling_end(text):
    assert _is_incomplete(text) is True


@pytest.mark.parametrize("text", [
    "하나님은 우리를 사랑하십니다.",
    "그것이 복입니다.",
])
def test_is_incomplete_accepts_complete_sentence(text):
    assert _is_incomplete(text) is False


def test_verify_and_fix_extends_past_incomplete_ending():
    """끝 문장이 미완이면 착지까지 확장해야 한다 — '이름을 그래서'처럼 말 중간에
    끊긴 실측 사고의 재발 방지."""
    sents = [
        S(0, 0.0, 5.0, "오늘 본문을 보겠습니다."),
        S(1, 5.0, 12.0, "하나님이 우리를 부르신 이유는 분명합니다."),
        S(2, 12.0, 20.0, "그것은 우리가 복의 통로가 되게 하시려고,"),   # 미완(쉼표+-려고)
        S(3, 20.0, 28.0, "우리를 통해 이웃을 살리시려는 것입니다."),
    ]
    cut, log = verify_and_fix({"core": 1, "start": 0, "end": 2}, sents, 15.0, 60.0, 90.0)
    assert cut is not None
    assert cut["end"] == 3, f"미완 끝이 확장되지 않았다: {log}"


def test_verify_and_fix_keeps_core_sentence_inside():
    """핵심 문장(core)이 구간 밖이면 구간을 넓혀 반드시 포함해야 한다."""
    sents = [S(i, i * 6.0, i * 6.0 + 6.0, f"문장{i}입니다.") for i in range(8)]
    cut, _ = verify_and_fix({"core": 5, "start": 0, "end": 2}, sents, 10.0, 60.0, 90.0)
    assert cut is not None
    assert cut["start"] <= 5 <= cut["end"]


def test_verify_and_fix_strips_structure_marker_start():
    """'둘째,' 같은 구조 표지로 시작하면 떼어낸다(훅이 죽으므로)."""
    assert _starts_with_structure_marker("둘째, 기도해야 합니다.")
    sents = [
        S(0, 0.0, 4.0, "둘째, 우리는 기도해야 합니다."),
        S(1, 4.0, 25.0, "기도는 하나님과의 대화이기 때문입니다."),
        S(2, 25.0, 45.0, "그래서 기도하는 사람은 반드시 응답을 받습니다."),
    ]
    cut, log = verify_and_fix({"core": 2, "start": 0, "end": 2}, sents, 15.0, 60.0, 90.0)
    assert cut is not None
    assert cut["start"] == 1, f"구조 표지가 안 떨어졌다: {log}"


def test_verify_and_fix_rejects_core_out_of_range():
    cut, log = verify_and_fix({"core": 99, "start": 0, "end": 1}, [S(0, 0, 5, "가.")], 5, 60, 90)
    assert cut is None and log


def test_dedupe_cuts_drops_overlapping():
    """겹치는 컷은 앞의 것만 남아야 한다(같은 장면이 후보에 2번 뜨는 문제)."""
    sents = [S(i, i * 10.0, i * 10.0 + 10.0, f"문장{i}.") for i in range(10)]
    cuts = [
        {"core": 1, "start": 0, "end": 4},
        {"core": 2, "start": 1, "end": 4},   # 위와 거의 완전히 겹침 → 탈락
        {"core": 8, "start": 7, "end": 9},   # 안 겹침 → 유지
    ]
    kept = dedupe_cuts(cuts, sents)
    assert [k["start"] for k in kept] == [0, 7]


def test_build_sentences_splits_glued_token():
    """'돼.이'처럼 토큰 가운데 마침표가 박힌 자동자막 토큰은 두 문장으로 갈라져야 한다.
    2026-09-18 실측(1GM): 이 토큰 14개 때문에 문장이 붙어 '그래서 우리가 …' 접속어 훅이 나왔다."""
    tr = Transcript(language="ko", duration_sec=10.0, segments=[Segment(
        start=0.0, end=10.0, text="",
        words=[W(0.0, 1.0, "마음을"), W(1.0, 2.0, "가져야"), W(2.0, 3.0, "돼.이"), W(3.0, 4.0, "이"),
               W(4.0, 5.0, "지역에도"), W(5.0, 6.0, "있습니다.")],
    )])
    sents = build_sentences(tr)
    assert [x.text for x in sents] == ["마음을 가져야 돼.", "이 이 지역에도 있습니다."]
    assert sents[1].start < 3.0 <= sents[1].end  # 뒤 조각 시각이 앞당겨져 있어야 한다


def test_verify_and_fix_repairs_connector_start_backward():
    """첫 문장이 '그래서/그니까'로 시작하면 바로 앞의 깨끗한 문장으로 시작을 옮긴다(hook 3점 원인)."""
    sents = [
        S(0, 0.0, 6.0, "하나님이 우리를 눈여겨 보시는 거예요."),
        S(1, 6.0, 12.0, "그래서 우리가 하나님의 마음을 가져야 돼."),
        S(2, 12.0, 30.0, "이 지역에도 예수 모르는 사람이 있습니다."),
        S(3, 30.0, 40.0, "하나님이 버리셨느냐? 그럴 수 없느니라."),
    ]
    cut, log = verify_and_fix({"core": 3, "start": 1, "end": 3}, sents, 15.0, 60.0, 90.0)
    assert cut is not None and cut["start"] == 0, log


def test_verify_and_fix_repairs_connector_start_forward_after_amen():
    """앞이 '아멘'(생각의 끝)이라 뒤로 못 가면 앞으로 첫 깨끗한 문장으로 옮긴다 — 앞 대지를 끌어오면 안 된다."""
    sents = [
        S(0, 0.0, 6.0, "은혜 베푸신 줄 믿습니다."),
        S(1, 6.0, 7.0, "아멘."),
        S(2, 7.0, 12.0, "그러니까 이게 중요한 거예요."),
        S(3, 12.0, 30.0, "기도하는 사람은 세상에 오염되지 않습니다."),
        S(4, 30.0, 45.0, "예수님은 인기에 오염되지 않으셨어요."),
    ]
    cut, log = verify_and_fix({"core": 3, "start": 2, "end": 4}, sents, 15.0, 60.0, 90.0)
    assert cut is not None and cut["start"] == 3, log


def test_verify_and_fix_strips_filler_start():
    """'예.', '응.' 같은 추임새 문장이 첫 문장이면 뗀다."""
    sents = [
        S(0, 0.0, 1.0, "예."),
        S(1, 1.0, 20.0, "성도는 거룩해야 합니다."),
        S(2, 20.0, 40.0, "거룩은 구별입니다."),
    ]
    cut, log = verify_and_fix({"core": 2, "start": 0, "end": 2}, sents, 15.0, 60.0, 90.0)
    assert cut is not None and cut["start"] == 1, log


def test_trim_lead_words_cuts_connector_from_first_sentence():
    """문장이 최소 단위라 못 버리는 긴 설정 문장은 첫 단어 '그래서'만 단어 시각으로 잘라낸다(쿠오바디스 실측)."""
    tr = Transcript(language="ko", duration_sec=10.0, segments=[Segment(
        start=0.0, end=10.0, text="",
        words=[W(0.0, 0.4, "그래서"), W(0.4, 1.0, "베드로가"), W(1.0, 2.0, "로마에서"), W(2.0, 3.0, "죽임을"),
               W(3.0, 4.0, "당하는데"), W(4.0, 5.0, "영화를"), W(5.0, 6.0, "보면.")],
    )])
    sent = build_sentences(tr)[0]
    start, text = trim_lead_words(sent)
    assert start == 0.4 and text.startswith("베드로가")
    short = build_sentences(Transcript(language="ko", duration_sec=3.0, segments=[Segment(
        start=0.0, end=3.0, text="", words=[W(0, 1, "그래서"), W(1, 2, "믿습니다."), W(2, 3, "아멘.")])]))[0]
    assert trim_lead_words(short) == (short.start, short.text)  # 너무 짧으면 건드리지 않는다


def test_merge_adjacent_cuts_joins_one_flow():
    """설정→반전→착지가 이어지는 한 흐름을 둘로 쪼갠 컷은 합쳐야 한다(1GM '이뻐 죽겠어'+'그럴 수 없느니라')."""
    sents = [S(i, i * 10.0, i * 10.0 + 10.0, f"문장{i}입니다.") for i in range(10)]
    cuts = [{"core": 1, "start": 0, "end": 3, "appeal": "위로", "why": "A"},
            {"core": 5, "start": 4, "end": 6, "appeal": "선언", "why": "B"}]
    merged, log = merge_adjacent_cuts(cuts, sents, 90.0)
    assert len(merged) == 1 and merged[0]["start"] == 0 and merged[0]["end"] == 6
    assert merged[0]["core"] == 1 and merged[0]["appeal"] == "위로"  # 강한 쪽 유지


def test_merge_adjacent_cuts_respects_landing_and_limit():
    """앞 컷이 축복 착지로 끝났거나 합치면 상한을 넘기면 합치지 않는다."""
    sents = [S(i, i * 10.0, i * 10.0 + 10.0, f"문장{i}입니다.") for i in range(12)]
    sents[3] = S(3, 30.0, 40.0, "되시기를 축복합니다.")
    cuts = [{"core": 1, "start": 0, "end": 3}, {"core": 5, "start": 4, "end": 6}]
    merged, _ = merge_adjacent_cuts(cuts, sents, 90.0)
    assert len(merged) == 2
    cuts = [{"core": 1, "start": 0, "end": 5}, {"core": 8, "start": 6, "end": 10}]  # 합치면 110초
    merged, _ = merge_adjacent_cuts(cuts, sents, 90.0)
    assert len(merged) == 2


# ───────────────────────── 자막 줄바꿈 (1줄 폭 규칙) ─────────────────────────

def test_chunk_words_respects_max_units():
    """자막은 화면에 무조건 1줄이어야 한다. max_units를 넘는 줄이 나오면
    libass가 멋대로 2줄로 랩핑한다(실측 불만) — 폭 상한이 지켜지는지 확인."""
    words = [W(i * 0.4, i * 0.4 + 0.35, "가나다라") for i in range(12)]
    max_units = 8.0  # 한글 1자 = 1unit 기준이면 8자 = 2단어
    lines = chunk_words_into_lines(words, max_words_per_line=4, max_units=max_units)
    assert lines
    for ln in lines:
        text = "".join(w.text for w in ln.words)
        assert len(text) <= max_units + 4, f"폭 상한 초과: {text!r}"
    # 단어가 하나도 사라지면 안 된다
    assert sum(len(ln.words) for ln in lines) == len(words)


def test_chunk_words_preserves_all_words():
    """어떤 설정에서도 단어가 유실되면 안 된다(자막이 통째로 빠지는 사고)."""
    words = [W(i * 0.5, i * 0.5 + 0.4, f"단어{i}") for i in range(23)]
    lines = chunk_words_into_lines(words, max_words_per_line=4)
    got = [w.text for ln in lines for w in ln.words]
    assert got == [w.text for w in words]


def test_clamp_lines_non_overlap():
    """자막 줄이 겹쳐 뜨면 화면에 두 줄이 동시에 보인다 — 끝을 다음 시작 앞으로 잘라야."""
    lines = [
        CaptionLine(start=0.0, end=5.0, words=[W(0.0, 5.0, "가")]),
        CaptionLine(start=2.0, end=7.0, words=[W(2.0, 7.0, "나")]),
    ]
    out = _clamp_lines_non_overlap(lines)
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start + 1e-6, f"자막이 겹친다: {a.end} > {b.start}"


# ───────────────── 무음 제거 후 시간 리매핑 (자막 밀림 사고) ─────────────────

def test_remap_after_silence_removal():
    """무음 제거로 영상이 압축되면 자막 시각도 같은 비율로 당겨져야 한다.
    이 리매핑이 빠져서 뒤로 갈수록 자막이 밀린 실측 사고가 있었다."""
    keep = [(0.0, 10.0), (15.0, 25.0)]   # 10~15초(5초)를 버림
    assert _remap_after_silence_removal(0.0, keep) == pytest.approx(0.0)
    assert _remap_after_silence_removal(5.0, keep) == pytest.approx(5.0)
    # 버려진 구간 뒤는 정확히 5초 당겨진다
    assert _remap_after_silence_removal(20.0, keep) == pytest.approx(15.0)
    # 단조 증가여야 한다(뒤로 튀면 자막 순서가 뒤집힌다)
    ts = [_remap_after_silence_removal(t, keep) for t in (0, 3, 9, 16, 20, 24)]
    assert ts == sorted(ts)


def test_intersect_intervals():
    assert _intersect_intervals([(0.0, 10.0)], [(2.0, 4.0), (6.0, 12.0)]) == [(2.0, 4.0), (6.0, 10.0)]
    assert _intersect_intervals([(0.0, 1.0)], [(5.0, 6.0)]) == []


def test_combine_keep_returns_none_for_whole_clip():
    """자를 게 없으면 None(빠른 경로). 여기서 빈 리스트를 돌려주면 select 필터가
    아무것도 안 남겨 영상이 0초가 된다."""
    assert _combine_keep(30.0, 100.0, None, None) is None
    assert _combine_keep(30.0, 100.0, [[100.0, 130.0]], None) is None


def test_combine_keep_converts_user_ranges_to_relative():
    """keep_ranges는 절대초, 내부는 상대초. 이 변환이 어긋나면 엉뚱한 구간이 남는다."""
    out = _combine_keep(30.0, 100.0, [[105.0, 115.0], [120.0, 125.0]], None)
    assert out == [(5.0, 15.0), (20.0, 25.0)]


# ───────────────────────── 자막 세로 위치 (부호 규약) ─────────────────────────

def test_y_position_bottom_is_below_center():
    """bottom이 center보다 아래(=y가 큼)여야 한다. 부호가 뒤집혀 자막이 위로 간 사고가 있었다.
    (config가 지원하는 값은 "bottom" | "center"뿐 — "top"은 없다.)"""
    res = (1080, 1920)
    y_center = _y_position(res, "center", 0.2)
    y_bottom = _y_position(res, "bottom", 0.2)
    assert y_bottom > y_center, "자막 bottom이 center보다 위에 있다 — 부호 뒤집힘"
    assert 0 < y_center < res[1] and 0 < y_bottom < res[1]
    # 하단 안전영역(플랫폼 UI) 안으로 들어가면 안 된다
    assert y_bottom < res[1] * (1 - 0.2) + 1


# ───────────────────────── ASS 시간 이동 (실제결과 미리보기) ─────────────────

def test_shift_ass_times_moves_and_drops():
    ass = (
        "[Events]\n"
        "Dialogue: 0,0:00:05.00,0:00:07.00,Cap,,0,0,0,,안녕\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Cap,,0,0,0,,지나감\n"
    )
    out = shift_ass_times(ass, -4.4)
    assert "0:00:00.60,0:00:02.60" in out      # 5.0→0.6로 이동
    assert "지나감" not in out                  # 끝이 0 이하 → 제거
    assert "[Events]" in out                    # 헤더는 보존


def test_shift_ass_times_clamps_negative_start():
    ass = "Dialogue: 0,0:00:01.00,0:00:09.00,Cap,,0,0,0,,걸침\n"
    out = shift_ass_times(ass, -3.0)
    assert out.startswith("Dialogue: 0,0:00:00.00,0:00:06.00,")


# ───────────────────────── 점수 계산 (결정론) ─────────────────────────

def test_compute_scores_is_multiplicative():
    """한 축이 낮으면 통합 점수가 확 떨어져야 한다(평균이 아니라 곱셈적)."""
    allhigh = compute_scores({"core_score": 9, "hook": 9, "retention": 9, "emotion": 9,
                              "relatability": 9, "payoff": 9, "quotability": 9})
    lowcore = compute_scores({"core_score": 2, "hook": 9, "retention": 9, "emotion": 9,
                              "relatability": 9, "payoff": 9, "quotability": 9})
    assert allhigh["score"] > lowcore["score"] * 1.5


def test_compute_scores_hook_gate():
    """hook이 낮으면 나머지가 아무리 좋아도 viral이 깎여야 한다(첫 2초가 죽은 클립)."""
    good = compute_scores({"core_score": 8, "hook": 9, "retention": 8, "emotion": 8,
                           "relatability": 8, "payoff": 8, "quotability": 8})
    weak = compute_scores({"core_score": 8, "hook": 2, "retention": 8, "emotion": 8,
                           "relatability": 8, "payoff": 8, "quotability": 8})
    assert weak["viral_score"] < good["viral_score"]
    assert 0 <= weak["score"] <= 100


def test_compute_scores_is_deterministic():
    r = {"core_score": 7, "hook": 6, "retention": 8, "emotion": 5,
         "relatability": 7, "payoff": 6, "quotability": 4}
    assert compute_scores(dict(r)) == compute_scores(dict(r))


# ───────────────────────── clips.json 저장/로드 ─────────────────────────

def _clip(**kw) -> Clip:
    base = dict(start=10.0, end=40.0, title="제목", caption="본문", hashtags=["#a"], reason="이유")
    base.update(kw)
    return Clip(**base)


def test_clips_json_roundtrip_preserves_edits():
    """편집 필드(자막·트림·스타일)가 저장→로드에서 유실되면 사용자의 작업이 날아간다."""
    d = Path(tempfile.mkdtemp()) / "clips.json"
    c = _clip(
        caption_overrides=[{"start": 11.0, "end": 13.0, "text": "고친 자막"}],
        trimmed=True, caption_box=True, caption_box_radius=55.0,
        keep_ranges=[[10.0, 20.0], [30.0, 40.0]], playback_speed=1.2,
    )
    save_clips_json([c], d)
    got = load_clips_json(d)[0]
    assert got.caption_overrides[0]["text"] == "고친 자막"
    assert got.trimmed is True and got.caption_box is True
    assert got.caption_box_radius == 55.0
    assert got.keep_ranges == [[10.0, 20.0], [30.0, 40.0]]
    assert got.playback_speed == 1.2


def test_clips_json_ignores_unknown_fields():
    """구버전으로 롤백하거나 필드를 정리해도 기존 작업물이 열려야 한다.
    예전 Clip(**c)는 모르는 키 하나에 TypeError로 전부 못 열었다."""
    d = Path(tempfile.mkdtemp()) / "clips.json"
    save_clips_json([_clip()], d)
    raw = json.loads(d.read_text(encoding="utf-8"))
    raw[0]["field_from_the_future"] = {"nested": 1}
    d.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    assert load_clips_json(d)[0].title == "제목"


def test_save_clips_json_is_atomic_and_keeps_backup():
    """중간에 죽어도 작업물이 통째로 날아가면 안 된다: tmp 잔여물 없음 + .bak 보존."""
    d = Path(tempfile.mkdtemp()) / "clips.json"
    save_clips_json([_clip(title="첫번째")], d)
    save_clips_json([_clip(title="두번째")], d)
    assert load_clips_json(d)[0].title == "두번째"
    assert d.with_name(d.name + ".bak").exists()
    assert not d.with_name(d.name + ".tmp").exists()
    assert json.loads(d.with_name(d.name + ".bak").read_text(encoding="utf-8"))[0]["title"] == "첫번째"


# ───────────────────────── 모델 ID ─────────────────────────

def test_sanitize_rejects_arbitrary_model_string():
    """UI에서 온 임의 문자열이 그대로 claude --model로 넘어가면 안 된다."""
    assert sanitize("; rm -rf /") == ""
    assert sanitize("") == ""
    assert sanitize(ANALYSIS_MODEL) == ANALYSIS_MODEL
    assert sanitize(f"  {EXECUTION_MODEL}  ") == EXECUTION_MODEL


def test_model_policy_analysis_is_opus_execution_is_sonnet():
    """모델 정책: 분석/선정=Opus, 자막 실행(교정·번역·가사)=Sonnet.
    한 곳에서만 정의되므로 여기서 정책 자체를 못 박아 둔다."""
    assert "opus" in ANALYSIS_MODEL
    assert "sonnet" in EXECUTION_MODEL
    assert ANALYSIS_MODEL in ALLOWED_MODELS and EXECUTION_MODEL in ALLOWED_MODELS


def _lines_text(words, wpl=3, max_units=13.5):
    return [" ".join(w.text for w in ln.words)
            for ln in chunk_words_into_lines(words, max_words_per_line=wpl, max_units=max_units)]


def test_chunk_keeps_modifier_with_its_noun():
    """2026-09-21 신고: "의미가 통하는 데까지 끊어야 하는데 이상하게 잘린다".
    창이 3단어로 고정돼 '우리 / 교회를', '이런 / 식으로'처럼 꾸미는 말과 꾸밈받는
    말이 갈라졌다 — 폭이 허락하면 몇 단어 더 봐서 의미 경계에서 끊어야 한다."""
    words = [W(i * 0.4, i * 0.4 + 0.35, t) for i, t in enumerate(
        ["박", "목사님이", "우리", "교회를", "부임하셨을", "때"])]
    for line in _lines_text(words):
        assert not line.endswith("우리"), f"관형어 뒤에서 끊겼다: {line!r}"


def test_chunk_keeps_dependent_noun_and_obligation():
    """'가까우신 / 거로', '해야 / 할'처럼 앞말에 붙어야 뜻이 사는 자리에서 끊지 않는다."""
    words = [W(i * 0.4, i * 0.4 + 0.35, t) for i, t in enumerate(
        ["우리", "교회가", "해야", "할", "일이", "있다는", "거예요"])]
    for line in _lines_text(words):
        assert not line.endswith("해야"), f"의무 표현이 갈라졌다: {line!r}"
    words = [W(i * 0.4, i * 0.4 + 0.35, t) for i, t in enumerate(
        ["굉장히", "가까우신", "거로", "미국을", "가니까"])]
    for line in _lines_text(words):
        assert not line.endswith("가까우신"), f"의존명사 앞에서 끊겼다: {line!r}"


# ─────────────────── 싱크 맞추기 뒤 빈 칸 정리 ───────────────────

def test_tighten_caption_lines_fills_gaps_and_drops_empty():
    """2026-09-21 신고: "싱크 맞추기 하면 빈 칸도 자동으로 없애고 시간이 딱딱 박혀야".
    빈 줄 삭제 + 줄 사이 빈 칸 메우기(앞 줄 끝을 늘림 — 뒤 줄을 앞당기면 '선행 싱크'가
    되므로 절대 금지) + 겹침 제거."""
    from types import SimpleNamespace

    from src.web_app import _tighten_caption_lines

    clip = SimpleNamespace(start=10.0, end=20.0)
    lines = [
        {"start": 10.4, "end": 11.0, "text": "첫 줄"},
        {"start": 13.0, "end": 14.0, "text": "  "},          # 빈 줄 → 삭제
        {"start": 14.0, "end": 15.0, "text": "둘째 줄"},
        {"start": 14.5, "end": 16.0, "text": "셋째 줄"},      # 겹침
    ]
    out, dropped, filled = _tighten_caption_lines(lines, clip)
    assert dropped == 1
    assert [x["text"] for x in out] == ["첫 줄", "둘째 줄", "셋째 줄"]
    assert out[0]["start"] == 10.0            # 앞 빈 칸(0.4초)은 첫 줄을 당겨 메움
    assert out[-1]["end"] == 20.0             # 뒤는 클립 끝까지
    starts = [x["start"] for x in out]
    assert starts[1] == 14.0 and starts[2] == 14.5   # 시작은 절대 앞당기지 않는다
    for a, b in zip(out, out[1:]):
        assert abs(a["end"] - b["start"]) < 0.06, f"빈 칸/겹침이 남았다: {a} {b}"
    assert filled >= 1


# ---------------------------------------------------------------------------
# 2026-09-21 선정 개편: 점프컷(skip→keep_ranges) · 약한 훅 · 성경 인물 필터
#
# 배경: "재밌는/교훈 있는 부분을 못 잡는다"가 반복됐는데 매번 프롬프트만 고치고 감으로 판정했다.
# 이제 정답지(output/_eval/ground_truth.json — 실제로 터진 명성교회 쇼츠를 원본 설교에 역정렬한 것)가
# 있으므로, **검증된 상위 클립을 우리 필터가 죽이지 않는가**를 모델 호출 없이 못 박는다.
# ---------------------------------------------------------------------------
from src.main import _sync_keep_ranges  # noqa: E402
from src.sermon_cut import (  # noqa: E402
    Sentence,
    _BIBLE_NAME_RE,
    _eff_dur,
    _is_weak_hook,
    bible_story_reason,
    cut_keep_ranges,
    kept_runs,
    verify_and_fix,
)

GROUND_TRUTH = Path("output/_eval/ground_truth.json")


def _sent(idx: int, start: float, end: float, text: str) -> Sentence:
    return Sentence(idx, start, end, text)


@pytest.mark.parametrize(
    "text,expected",
    [
        # 실측 사고: 6.9천 조회수 '장바구니' 쇼츠가 "반드시 필요한 사람인데"의 '요한'에 걸려 제외될 뻔했다.
        ("그거 외에도 반드시 필요한 사람인데", None),
        ("중요한 것은 이것입니다", None),
        ("우리가 원하는 자유다", None),
        ("요한복음 3장 16절", None),   # 책 이름은 인물 서사가 아니다
        ("욥바에서 환상을 봤어요", None),   # 지명(욥바)이지 욥이 아니다
        ("신학교에서 배운 것", None),      # '에서'는 조사
        ("엘리베이터도 하나 없는", None),
        ("베드로가 로마를 떠났습니다", "베드로"),
        ("요나가 니느웨에 가기 싫어", "요나"),
        ("사도 요한이 기록하기를", "요한"),
    ],
)
def test_bible_name_regex_no_false_positives(text: str, expected: str | None) -> None:
    m = _BIBLE_NAME_RE.search(text)
    assert (m.group(1) if m else None) == expected


@pytest.mark.skipif(not GROUND_TRUTH.exists(), reason="정답지 없음 (scripts/build_ground_truth.py)")
def test_proven_hit_shorts_survive_filters() -> None:
    """실제로 터진 쇼츠(5천~7천 조회수) 전부가 성경 인물 필터를 통과해야 한다.

    필터를 세게 만들 때마다 이 테스트로 '검증된 승자를 죽이는지'를 먼저 본다."""
    from src.sermon_cut import build_sentences
    from src.youtube_captions import parse_json3_to_transcript

    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    cap_dir = Path("output/_eval/captions")
    cache: dict[str, list[Sentence]] = {}
    dropped = []
    for short_id, g in gt.items():
        path = cap_dir / f"{g['sermon']}.ko.json3"
        if not path.exists():
            pytest.skip(f"자막 없음: {path}")
        sents = cache.setdefault(g["sermon"], build_sentences(parse_json3_to_transcript(path, 0)))
        idx = [s.idx for s in sents if s.start >= g["start"] - 1 and s.end <= g["end"] + 1]
        assert idx, f"{short_id}: 정답 구간을 문장에 매핑하지 못함"
        cut = {"core": idx[len(idx) // 2], "start": idx[0], "end": idx[-1], "scene": "", "thesis": ""}
        why = bible_story_reason(cut, sents)
        if why:
            dropped.append(f"{short_id}({g['views']}회): {why}")
    assert not dropped, "검증된 상위 클립이 필터에 걸렸다:\n" + "\n".join(dropped)


def test_weak_hook_rejects_year_and_long_sentences() -> None:
    # 실측 zvsOYhjdmks: "2006년도 12월에 이곳에 성전 부지를 매입하고…"가 훅이 돼 hook 3점.
    assert _is_weak_hook("2006년도 12월에 이곳에 성전 부지를 매입하고 성전을 건축하는")
    # 실측 S65: 40단어짜리 한 문장이 29초 동안 이어진다 — 끝이 물음표여도 첫 3초가 죽는다
    assert _is_weak_hook("복음을 " + "들고 태평양을 건너 " * 13 + "보셨습니까?")
    assert _is_weak_hook("1971년 7월 중순에 상경하셔서")
    assert not _is_weak_hook("전도하면 될 거라고 생각했어요.")
    assert not _is_weak_hook("인생의 연조가 저절로 깊어지는 것은 아니지 않습니까?")


def _story_sentences() -> list[Sentence]:
    """설정 → (빼도 되는 부연) → 펀치 → 착지. 문장마다 10초."""
    texts = [
        "여러분 이런 경험 있으시죠?",              # 0 훅
        "제가 그때 참 힘들었습니다.",               # 1
        "이거는 사실 부연 설명입니다.",             # 2 (skip 대상)
        "덧붙이자면 그렇다는 겁니다.",              # 3 (skip 대상)
        "제 친구가 저한테 이렇게 말하더라고요.",       # 4 펀치
        "우리는 다 갖고 있어도 더 원합니다.",        # 5 core
        "그런 은혜가 있기를 축복합니다.",           # 6 착지
    ]
    return [_sent(i, i * 10.0, i * 10.0 + 9.5, t) for i, t in enumerate(texts)]


def test_jump_cut_removes_middle_and_keeps_length() -> None:
    sents = _story_sentences()
    raw = {"core": 5, "start": 0, "end": 6, "skip": [[2, 3]], "appeal": "교훈"}
    cut, log = verify_and_fix(raw, sents, 20, 60, 90, max_span_sec=180)
    assert cut is not None, log
    assert [list(s) for s in cut["skip"]] == [[2, 3]]
    assert kept_runs(cut["start"], cut["end"], [(2, 3)]) == [(0, 1), (4, 6)]
    # 원본 69.5초(0.0~69.5) → 가운데 두 문장을 들어내면 19.5 + 29.5 = 49.0초
    assert round(_eff_dur(sents, cut["start"], cut["end"], [(2, 3)]), 1) == 49.0
    ranges = cut_keep_ranges(cut, sents)
    assert len(ranges) == 2 and ranges[0][1] < ranges[1][0]


def test_jump_cut_blocks_dangling_reference() -> None:
    """'어색한 짜깁기'의 진짜 신호: 이어붙인 뒤 문장이 **잘려나간 것**을 가리킬 때.

    2026-09-22, 실제로 터진 쇼츠의 이음새 39곳을 대조한 결과 37곳(95%)이 이 조건을 지켰다."""
    sents = _story_sentences()
    sents[4] = _sent(4, 40.0, 49.5, "그 사장님이 저한테 이렇게 말하더라고요.")
    sents[2] = _sent(2, 20.0, 29.5, "그때 어떤 사장님을 만났습니다.")
    # '사장님'이 잘려나가는 구간 안에만 있으면 → skip 무시
    cut, log = verify_and_fix({"core": 5, "start": 0, "end": 6, "skip": [[2, 3]]}, sents, 20, 60, 90)
    assert cut is not None and cut["skip"] == [], log
    # '사장님'이 남는 구간(S1)에 이미 나왔으면 → 이어붙여도 가리킬 대상이 있으므로 허용
    sents[1] = _sent(1, 10.0, 19.5, "제가 그때 어떤 사장님을 만났습니다.")
    cut2, log2 = verify_and_fix({"core": 5, "start": 0, "end": 6, "skip": [[2, 3]]}, sents, 20, 60, 90)
    assert cut2 is not None and cut2["skip"] == [[2, 3]], log2


def test_jump_cut_allows_connective_after_seam() -> None:
    """뒤 문장이 "그런데/근데/그래서"로 시작해도 막지 않는다.

    예전엔 무조건 막았는데, 실제로 터진 쇼츠의 이음새 8곳(21%)이 바로 이 모양이고 전부 자연스러웠다
    (접속어가 가리키는 건 잘려나간 부분이 아니라 '남아 있는 앞 문장'이기 때문). 이 규칙을 되살리지 말 것."""
    sents = _story_sentences()
    sents[4] = _sent(4, 40.0, 49.5, "그런데 살아남는 사람이 있습니다.")
    cut, log = verify_and_fix({"core": 5, "start": 0, "end": 6, "skip": [[2, 3]]}, sents, 20, 60, 90)
    assert cut is not None and cut["skip"] == [[2, 3]], log


def test_jump_cut_rejects_unnatural_seams() -> None:
    """나머지 이음새 금지 조건 — 전부 실제 이음새 39곳에서 위반 0건이었던 것들."""
    # (1) 앞 문장이 말이 안 끝났는데 뒤를 들어냄
    sents = _story_sentences()
    sents[1] = _sent(1, 10.0, 19.5, "제가 그때 참 힘들었는데,")
    cut, log = verify_and_fix({"core": 5, "start": 0, "end": 6, "skip": [[2, 3]]}, sents, 20, 60, 90)
    assert cut is not None and cut["skip"] == [], log
    # (2) 뒤 문장이 순서 표지 — 앞 항목이 잘리면 말이 안 된다
    sents = _story_sentences()
    sents[4] = _sent(4, 40.0, 49.5, "둘째, 우리가 기억할 것이 있습니다.")
    cut2, log2 = verify_and_fix({"core": 5, "start": 0, "end": 6, "skip": [[2, 3]]}, sents, 20, 60, 90)
    assert cut2 is not None and cut2["skip"] == [], log2
    # (3) 핵심 문장은 절대 못 뺀다
    sents = _story_sentences()
    cut3, _ = verify_and_fix({"core": 5, "start": 0, "end": 6, "skip": [[5, 5]]}, sents, 20, 60, 90)
    assert cut3 is not None and cut3["skip"] == []


def test_sync_keep_ranges_follows_boundary_snap() -> None:
    """렌더의 끝 스냅이 clip.end를 늘리면 마지막 점프컷 구간도 따라 늘어나야 한다.
    (안 그러면 '말이 중간에 끊긴다'를 고치는 스냅이 점프컷 클립에서만 조용히 무효화된다.)"""
    clip = Clip(start=100.0, end=140.0, title="t", caption="", hashtags=[], reason="")
    clip.keep_ranges = [[100.0, 110.0], [120.0, 140.0]]
    clip.end = 143.0
    _sync_keep_ranges(clip, 100.0, 140.0)
    assert clip.keep_ranges == [[100.0, 110.0], [120.0, 143.0]]

    clip2 = Clip(start=100.0, end=140.0, title="t", caption="", hashtags=[], reason="")
    clip2.keep_ranges = [[100.0, 110.0], [120.0, 140.0]]
    clip2.start = 98.0
    _sync_keep_ranges(clip2, 100.0, 140.0)
    assert clip2.keep_ranges == [[98.0, 110.0], [120.0, 140.0]]

    # 스냅이 커서 첫 조각이 통째로 사라지면 점프컷을 포기하고 통짜로 돌아간다
    clip3 = Clip(start=100.0, end=140.0, title="t", caption="", hashtags=[], reason="")
    clip3.keep_ranges = [[100.0, 110.0], [120.0, 140.0]]
    clip3.start = 125.0
    _sync_keep_ranges(clip3, 100.0, 140.0)
    assert clip3.keep_ranges == []


def test_clip_duration_sec_uses_keep_ranges() -> None:
    """후보 목록의 'N초' 라벨은 원본 구간이 아니라 실제 재생 길이여야 한다."""
    clip = Clip(start=100.0, end=200.0, title="t", caption="", hashtags=[], reason="")
    assert clip.duration_sec == 100.0
    clip.keep_ranges = [[100.0, 130.0], [160.0, 200.0]]
    assert clip.duration_sec == 70.0


def test_filler_prefixed_sentence_can_be_a_hook() -> None:
    """말버릇 한 단어("아") 때문에 설정 문장을 통째로 버리면 안 된다.

    2026-09-22 실측(mjg0tcadzjY): "아 하나님께서 원하시면 나도 원하죠."가 버려져, 그걸 반박하는
    크레센도("여러분, 그렇지 않습니다 → 예배를 원하시는데 우리는 별로…")만 남아 반박 대상이 사라졌다.
    렌더 직전 trim_lead_words가 그 단어만 잘라내므로 선정에서 버릴 이유가 없다."""
    from src.sermon_cut import _is_dirty_start, trim_lead_words

    assert not _is_dirty_start("아 하나님께서 원하시면 나도 원하죠.")
    assert _is_dirty_start("아멘.")                      # 문장 전체가 추임새면 여전히 버린다
    assert _is_dirty_start("그래서 우리가 그렇게 해야 합니다.")   # 접속어 시작은 여전히 더럽다

    sent = Sentence(0, 10.0, 14.0, "아 하나님께서 원하시면 나도 원하죠.",
                    words=[(10.0, 10.3, "아"), (10.3, 11.0, "하나님께서"), (11.0, 12.0, "원하시면"),
                           (12.0, 13.0, "나도"), (13.0, 14.0, "원하죠.")])
    start, text = trim_lead_words(sent)
    assert start == 10.3 and text.startswith("하나님께서")


def test_verify_drops_clips_below_config_minimum() -> None:
    """하한은 config min_duration_sec 그대로 적용한다(예전 0.8배 여유가 30초대 반쪽 후보를 통과시켰다)."""
    sents = [Sentence(i, i * 10.0, i * 10.0 + 9.5, f"문장 {i}번입니다 여러분.") for i in range(4)]
    cut, log = verify_and_fix({"core": 1, "start": 0, "end": 2}, sents, 30, 60, 90)
    assert cut is None and any("하한" in m for m in log)
    cut2, _ = verify_and_fix({"core": 1, "start": 0, "end": 3}, sents, 30, 60, 90)
    assert cut2 is not None
