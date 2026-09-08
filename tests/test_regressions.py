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
    y_center = _y_position(res, "center", 0.2, 0.1)
    y_bottom = _y_position(res, "bottom", 0.2, 0.1)
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
