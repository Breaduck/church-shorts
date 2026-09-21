"""하이라이트 점수를 '진짜로' 계산하는 결정론적 로직.

배경(왜 이 모듈이 필요한가):
  예전엔 Claude에게 곧바로 0~100점(`score`)을 눈대중으로 매기게 했다. 그 결과
  (1) 같은 클립도 실행마다 점수가 흔들리고, (2) "곱셈적으로 생각하라"는 지시가
  실제 곱셈으로 강제되지 않았으며, (3) 쇼츠에서 치명적인 '첫 2초 훅'이 다른 요소와
  평균돼 묻혔다. 이 모듈은 Claude가 매기는 건 **세부 축(1~10) 뿐**으로 좁히고,
  최종 viral_score와 통합 score는 여기서 **일관된 공식**으로 계산한다.

핵심 성질:
  - 통합 score = core와 viral의 **가중 기하평균** → 한 축이라도 낮으면 확 떨어진다(곱셈적).
    지수 합이 1이라 "둘 다 8이면 80, 둘 다 9면 90"처럼 직관과도 맞는다.
  - viral은 세부 축의 가중합이되, **hook이 낮으면 관문(gate)으로 통째로 깎는다**
    (첫 2초가 죽은 클립은 나머지가 아무리 좋아도 쇼츠에선 사망하기 때문).
"""
from __future__ import annotations


# viral을 이루는 세부 축의 가중치 (합 = 1.0).
# hook을 가장 크게, 그다음 리텐션·감정·공감 순. payoff(마무리)도 무겁게.
VIRAL_WEIGHTS: dict[str, float] = {
    "hook": 0.30,          # 첫 1~2초 훅 (스크롤 멈춤)
    "retention": 0.22,     # 전진감/죽은 구간 없음
    "emotion": 0.18,       # 감정 스파이크(전율·위로·뜨끔)
    "relatability": 0.15,  # "이거 완전 내 얘기" 공감
    "payoff": 0.12,        # 깔끔한 마무리/펀치라인 착지
    "quotability": 0.03,   # 스샷 떠서 공유할 인용각
}

# 통합 score의 core/viral 가중치(기하평균 지수, 합 = 1.0).
# viral을 살짝 더 실어 실제 조회수 동인을 반영하되, core로 '낚시'를 방지.
CORE_EXPONENT = 0.45
VIRAL_EXPONENT = 0.55

# hook 관문: 이 값 미만이면 viral 본문을 (hook/기준) 비율로 통째 깎는다.
HOOK_GATE_THRESHOLD = 6.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_viral_score(subscores: dict) -> float:
    """세부 축(1~10)으로부터 viral_score(1~10)를 계산한다.

    subscores: hook/retention/emotion/relatability/payoff/quotability 키. 빠진 건 5로 간주.
    """
    body = 0.0
    for axis, weight in VIRAL_WEIGHTS.items():
        body += weight * _clamp(float(subscores.get(axis, 5) or 5), 1.0, 10.0)

    hook = _clamp(float(subscores.get("hook", 5) or 5), 1.0, 10.0)
    if hook < HOOK_GATE_THRESHOLD:
        # 죽은 훅 관문: hook=3이면 절반으로, hook=1이면 1/6로 깎인다.
        body *= hook / HOOK_GATE_THRESHOLD

    return round(_clamp(body, 1.0, 10.0), 2)


def compute_total_score(core_score: float, viral_score: float) -> int:
    """core(1~10)와 viral(1~10)의 가중 기하평균으로 0~100 통합 점수를 낸다."""
    core01 = _clamp(float(core_score), 1.0, 10.0) / 10.0
    viral01 = _clamp(float(viral_score), 1.0, 10.0) / 10.0
    total = 100.0 * (core01 ** CORE_EXPONENT) * (viral01 ** VIRAL_EXPONENT)
    return int(round(_clamp(total, 0.0, 100.0)))


def compute_scores(raw: dict) -> dict:
    """모델이 준 원시 dict에서 최종 점수 묶음을 계산해 돌려준다.

    입력 우선순위:
      1) 세부 축(hook/retention/...)이 있으면 그걸로 viral·score를 '계산'한다(권장 경로).
      2) 세부 축이 없고 예전 형식(viral_score/score 직접 제공)이면 그 값을 존중한다(하위호환).

    반환: {core_score, viral_score, score, subscores(dict)}
    """
    core_score = raw.get("core_score")
    core_score = float(core_score) if core_score is not None else 5.0

    axis_keys = list(VIRAL_WEIGHTS.keys())
    has_subscores = any(raw.get(k) is not None for k in axis_keys)

    if has_subscores:
        subscores = {k: (float(raw[k]) if raw.get(k) is not None else 5.0) for k in axis_keys}
        viral_score = compute_viral_score(subscores)
    else:
        # 하위호환: 세부 축이 없으면 예전에 모델이 직접 준 viral_score를 쓴다.
        v = raw.get("viral_score")
        viral_score = float(v) if v is not None else 5.0
        subscores = {}

    # score는 항상 우리 공식으로 재계산해 일관성을 보장한다(모델이 준 score는 무시).
    score = compute_total_score(core_score, viral_score)

    return {
        "core_score": round(core_score, 2),
        "viral_score": round(viral_score, 2),
        "score": score,
        "subscores": subscores,
    }
