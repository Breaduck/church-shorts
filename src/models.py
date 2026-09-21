"""claude CLI에 넘길 모델 ID를 한 곳에 모은다.

왜 필요했나: 모델 ID가 main.py 3곳 + web_app.py 8곳 + config.yaml 2곳, 총 3파일 11군데에
문자열로 흩어져 있었고 허용목록(_ALLOWED_MODELS)은 같은 내용이 4벌 복사돼 있었다.
그 결과 세대 교체가 한 번 일어나면 일부만 갱신되고 나머지는 조용히 구형 모델로 남는다
(실제로 자막 교정·번역·가사 5곳이 두 세대 전인 sonnet-4-5로 돌고 있었다).

모델 정책(memory: project_model_policy_analysis_opus)은 그대로다 —
  분석/선정(무엇을 자를지 판단) = Opus,  자막 실행(교정·번역·가사) = Sonnet.
바뀐 건 그 Opus/Sonnet이 가리키는 세대뿐이다.
"""
from __future__ import annotations

# 현행 세대 (2026-09 기준)
ANALYSIS_MODEL = "claude-opus-5"      # 하이라이트/찬양 선정 — 판단 품질 우선
EXECUTION_MODEL = "claude-sonnet-5"   # 자막 교정·영어 번역·가사 정리 — 지시 수행, 속도/비용 우선
PREMIUM_MODEL = "claude-fable-5-1"    # 가장 비싸고 강한 모델. 사용자가 명시적으로 고를 때만.

# UI가 모델을 지정해 보낼 때 허용할 값. 여기 없는 값은 무시하고 config 기본값을 쓴다
# (임의 문자열이 그대로 CLI --model로 넘어가지 않게 하는 가드).
# 직전 세대도 남겨둔다: 예전 clips.json/북마크한 요청이 그대로 들어와도 거부되지 않게.
ALLOWED_MODELS = frozenset({
    ANALYSIS_MODEL, EXECUTION_MODEL, PREMIUM_MODEL,
    "claude-opus-4-8", "claude-sonnet-4-5", "claude-fable-5",  # 직전 세대(하위호환)
})


def sanitize(model: str) -> str:
    """UI에서 온 모델 문자열을 검증한다. 허용목록 밖이면 ""(=config 기본값)을 돌려준다."""
    m = (model or "").strip()
    return m if m in ALLOWED_MODELS else ""


# config.yaml이 세대 ID를 직접 적으면(예: "claude-opus-5") 세대 교체 때 이 파일만 고쳐선
# 안 되고 config도 같이 고쳐야 한다 — 이 파일이 막으려던 바로 그 상황이다. 그래서 config는
# 별칭("opus"/"sonnet"/"fable")만 쓰고, 실제 세대 ID는 여기서 한 번에 정한다.
_ALIASES = {
    "opus": ANALYSIS_MODEL,
    "sonnet": EXECUTION_MODEL,
    "fable": PREMIUM_MODEL,
}


def resolve(model: str) -> str:
    """config/CLI에서 온 모델 값을 실제 모델 ID로 바꾼다.

    별칭이면 현행 세대 ID로, 이미 허용된 ID면 그대로, 그 외(빈 값·오타)면 ""(=CLI 기본값).
    """
    m = (model or "").strip()
    if m in _ALIASES:
        return _ALIASES[m]
    return m if m in ALLOWED_MODELS else ""
