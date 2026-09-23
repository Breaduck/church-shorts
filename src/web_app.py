"""웹 검토 흐름: 링크 입력 -> 후보 목록(바이럴 순위) 확인 -> 고른 것만 렌더링

사용법:
    python -m src.web_app
    -> http://127.0.0.1:5000 접속 (Cloudflare Tunnel 등으로 외부 노출 가능)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template_string, request, send_file

# .env를 읽는다(.env.example이 YOUTUBE_CLIENT_SECRETS_PATH를 안내하는데 여기서 안 읽으면
# 파일에 적어도 조용히 무시된다). 시스템 환경변수가 이미 있으면 그쪽이 우선.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from src.feedback import PerformanceRecord, upsert_feedback
from src.models import EXECUTION_MODEL, resolve as resolve_model, sanitize as sanitize_model
from src.highlights import CLIPS_LOCK, load_clips_json, save_clips_json
from src.main import analyze, reanalyze_clip_region, render_selected, render_signature
from src.upload.tracking import find_upload, load_uploads, record_upload, run_due_checks

app = Flask(__name__)
OUTPUT_ROOT = Path("output")

# 단일 사용자 로컬 도구이므로 메모리 내 딕셔너리로 작업 상태를 추적한다 (DB 불필요).
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# 클립 '구간 재분석' 작업 상태(영상별). 팝업이 폴링해서 완료되면 새 후보를 보여준다.
# 아래 _retrans_jobs(자막 정밀 재전사)와 함께 _jobs_lock으로 보호한다 — 검사와 표시가
# 원자적이지 않으면 같은 작업이 중복 실행된다(각 라우트 주석 참고).
_reanalyze_jobs: dict[str, dict] = {}


def _update_job(video_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(video_id, {}).update(fields)


def _clips_ready(video_id: str, job: dict | None) -> bool:
    """이 영상의 분석이 '지금 요청 기준으로' 끝났는지 판정한다.

    단순히 clips.json 존재만 보면, 이전에 분석했던 링크에 '새로 분석'을 걸었을 때
    옛 clips.json이 아직 디스크에 남아 있어(재선정 job이 .bak으로 옮기기 직전 찰나)
    즉시 '완료+옛 캐시'로 오인된다. 그래서 재분석 job이 도는 중(status=analyzing)이면
    clips.json이 '그 job 시작 시각(started) 이후'에 쓰였을 때만 완료로 인정한다.
    (분석 중이 아닌 경우엔 디스크를 진실의 원천으로 삼아 캐시 재사용을 그대로 허용.)"""
    p = OUTPUT_ROOT / video_id / "clips.json"
    if not p.exists():
        return False
    if job and job.get("status") == "analyzing":
        try:
            # 2초 여유: 파일시스템 mtime이 초 단위로 내림돼 started보다 살짝 작아지는 경계
            # 오차만 흡수한다. 옛 캐시는 수 분~수일 전이라 이 여유로도 절대 통과하지 못한다.
            return p.stat().st_mtime >= float(job.get("started", 0) or 0) - 2.0
        except OSError:
            return False
    return True


def _load_config() -> dict:
    return yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))


# ETA는 파이프라인(main.StageProgress / _run_with_progress_ticker)이 단계별 예상시간으로
# 직접 계산해 job dict의 eta_seconds/render_eta_seconds로 넘겨준다. (예전의 '경과/진행률'
# 선형 추정은 단계마다 속도가 크게 달라 값이 튀어서 폐기했다.)


# 토스/애플 느낌: 넉넉한 여백, 부드러운 그림자, 큰 라운드 코너, 프리텐다드 폰트, 절제된 포인트 컬러.
# ── 템플릿/정적 파일 ─────────────────────────────────────────────────────────
# 예전엔 이 5개가 web_app.py 안의 문자열 리터럴이었다(EDIT_TEMPLATE 500줄, STUDIO 2,900줄,
# preview-modal.js 1,150줄 — 파일 하나가 7,400줄/453KB). f-string 안의 HTML/JS라서
# Jinja 중괄호를 {{{{ }}}}로 4중 이스케이프해야 했고, JS 문법 검사도 CSS 하이라이팅도
# 없어서 오타가 런타임에야 드러났다. 이제 진짜 .html/.css/.js 파일이다.
_ASSET_DIR = Path(__file__).parent


def _load_asset(relpath: str) -> str:
    """src/ 밑의 템플릿·정적 파일을 읽어 __BASE_STYLE__ 자리표시자를 채운다.
    BASE_STYLE(전역 CSS)은 여러 화면이 공유하므로 파일마다 복사하지 않고 여기서 주입한다."""
    text = (_ASSET_DIR / relpath).read_text(encoding="utf-8")
    return text.replace("__BASE_STYLE__", BASE_STYLE)


BASE_STYLE = (_ASSET_DIR / "templates" / "base_style.css").read_text(encoding="utf-8")
INDEX_TEMPLATE = _load_asset("templates/index.html")
EDIT_TEMPLATE = _load_asset("templates/edit.html")
STUDIO_TEMPLATE = _load_asset("templates/studio.html")
PREVIEW_MODAL_JS = (_ASSET_DIR / "static" / "preview-modal.js").read_text(encoding="utf-8")


CANDIDATES_TEMPLATE = f"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>하이라이트 후보 - {{{{ video_id }}}}</title>
{BASE_STYLE}
<style>
  /* 애플 느낌: 한 카드 = 하나의 후보. 메타데이터는 조용하게, 제목이 주인공,
     기술적인 분석(core/viral/훅 …)은 기본으로 접어두고 눌렀을 때만 펼친다. */
  .subtitle {{ margin-bottom: 26px; }}
  .candidate {{ padding: 22px 24px; transition: box-shadow .2s ease; }}
  .candidate.picked {{ box-shadow: 0 0 0 1.5px var(--accent), 0 6px 22px rgba(49, 130, 246, 0.10); }}
  .cand-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 13px; }}
  .cand-meta {{ display: flex; align-items: center; gap: 9px; min-width: 0; flex-wrap: wrap; }}
  .rank {{ font-size: 12.5px; font-weight: 800; letter-spacing: .02em; color: var(--text-faint); font-variant-numeric: tabular-nums; }}
  .candidate.top .rank {{ color: var(--accent); }}
  .dot-sep {{ width: 3px; height: 3px; border-radius: 50%; background: var(--text-faint); opacity: .55; flex-shrink: 0; }}
  .score-badge {{
    font-size: 12px; font-weight: 700; letter-spacing: -0.01em;
    padding: 3px 9px; border-radius: 999px; white-space: nowrap; line-height: 1.35;
  }}
  /* 등급 기준 재보정(2026-09-22): 채점 프롬프트가 "8=잘 되는 채널 상위 클립 수준, 9~10=채널 1위감(드묾)"이라
     현실적인 상단이 70~80점인데, 기준이 90/85/80이라 실측 61개 중 54개(89%)가 '참고'로 표시됐다 —
     좋은 클립도 전부 나쁘게 읽혔다. 두 축이 8이면 80, 7이면 70이 되는 공식(scoring.py)에 맞춰 다시 잡는다. */
  /* 경계는 실측 분포의 분위수로 잡는다(2026-09-23 재보정, 54클립 기준 대략 상위 15/45/75%).
     예전 78/68/58은 "전 축 평균 7.8/6.8/5.8"을 요구했는데 프롬프트는 8점을 "드물게" 쓰라고
     지시해서, 최근 설교 17클립 중 최상(78+)이 0개·절반이 참고로 몰렸다. */
  .score-badge.tier-top {{ background: #fff4e5; color: #c2620c; }}       /* 73점 이상: 최상(상위 15%) */
  .score-badge.tier-high {{ background: #e7f7ec; color: #1a7f37; }}      /* 63점 이상: 추천(상위 45%) */
  .score-badge.tier-ok {{ background: #eaf1ff; color: #2563eb; }}        /* 53점 이상: 후보 */
  .score-badge.tier-low {{ background: #f1f3f5; color: #868e96; }}       /* 53점 미만: 참고용 */
  .dur {{ font-size: 12.5px; color: var(--text-faint); font-variant-numeric: tabular-nums; }}
  .pick {{
    display: inline-flex; align-items: center; gap: 7px; cursor: pointer; user-select: none; flex-shrink: 0;
    padding: 6px 12px 6px 10px; border-radius: 999px; border: 1.5px solid var(--border);
    transition: border-color .15s, background .15s;
  }}
  .pick:hover {{ border-color: #d4dae2; }}
  .candidate.picked .pick {{ border-color: var(--accent); background: #eef4ff; }}
  .pick input {{ width: 16px; height: 16px; accent-color: var(--accent); cursor: pointer; margin: 0; }}
  .pick-label {{ font-size: 13px; font-weight: 600; color: var(--text-muted); }}
  .candidate.picked .pick-label {{ color: var(--accent); }}
  .title {{ font-size: 19px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.3; margin: 0 0 8px; }}
  .caption {{ color: var(--text); font-size: 14.5px; line-height: 1.6; margin: 0 0 10px; }}
  .hashtags {{ color: var(--text-faint); font-size: 13px; letter-spacing: -0.01em; margin: 0; }}
  .cand-foot {{
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--border);
  }}
  .reason-toggle {{
    font-size: 13px; font-weight: 600; color: var(--text-muted); background: none; border: none;
    padding: 0; cursor: pointer; font-family: inherit; display: inline-flex; align-items: center; gap: 5px;
  }}
  .reason-toggle:hover {{ color: var(--text); }}
  .reason-toggle .chev {{ transition: transform .2s ease; font-size: 10px; }}
  .reason-toggle[aria-expanded="true"] .chev {{ transform: rotate(180deg); }}
  .edit-link {{ font-size: 13px; font-weight: 600; color: var(--accent); text-decoration: none; white-space: nowrap; }}
  .edit-link:hover {{ text-decoration: underline; }}
  .cand-foot-actions {{ display: flex; align-items: center; gap: 12px; }}
  .dup-btn {{ font-size: 13px; font-weight: 600; color: var(--text-muted); background: none;
    border: none; padding: 0; cursor: pointer; font-family: inherit; white-space: nowrap; }}
  .dup-btn:hover {{ color: var(--accent); }}
  .dup-btn:disabled {{ opacity: .5; cursor: default; }}
  .reason {{
    background: #f7f8fa; border-radius: 12px; padding: 13px 15px; margin: 12px 0 0;
  }}
  .reason-caption {{ color: var(--text); font-size: 13.5px; line-height: 1.6; margin: 0 0 8px; }}
  .reason-hashtags {{ color: var(--text-faint); font-size: 12.5px; margin: 0 0 10px; }}
  .reason-text {{ color: var(--text-muted); font-size: 13px; line-height: 1.65; white-space: pre-line; margin: 0; }}
  .capcopy {{ background: #fff; border: 1.5px solid var(--border); border-radius: 10px; padding: 10px 12px; margin: 0 0 10px; }}
  .capcopy-head {{ display: flex; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 700; color: var(--text-muted); margin-bottom: 7px; }}
  .capcopy-btn {{ margin-left: auto; padding: 5px 12px; font-size: 12px; font-weight: 700; font-family: inherit;
    border: 1.5px solid var(--border); background: #fafbfc; color: var(--accent); border-radius: 8px; cursor: pointer; }}
  .capcopy-btn:hover {{ border-color: var(--accent); background: #f0f6ff; }}
  .capcopy-done {{ font-size: 12px; color: #16a34a; font-weight: 700; }}
  .capcopy-text {{ width: 100%; border: none; background: none; resize: vertical; font-family: inherit;
    font-size: 13px; line-height: 1.55; color: var(--text); padding: 0; margin: 0; outline: none; }}
  video {{ width: 100%; max-width: 260px; border-radius: 14px; background: #000; margin-top: 14px; display: block; }}
  .actions {{ position: sticky; bottom: 20px; margin-top: 24px; }}
  .actions button {{ box-shadow: 0 8px 24px rgba(49, 130, 246, 0.35); }}
  .actions button:disabled {{ opacity: .5; cursor: not-allowed; box-shadow: none; }}
  .error-box {{ color: #e02424; }}
  /* 스크롤해도 항상 보이는 우측 하단 고정 진행 위젯 */
  .mini-prog {{
    position: fixed; right: 20px; bottom: 20px; z-index: 9999; width: 268px; max-width: calc(100vw - 40px);
    background: var(--card); border-radius: 16px; padding: 14px 16px;
    box-shadow: 0 10px 30px rgba(15,23,42,0.18), 0 2px 8px rgba(15,23,42,0.08);
    border: 1px solid var(--border); animation: miniIn .25s ease;
  }}
  @keyframes miniIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  .mini-prog.hidden {{ display: none; }}
  .mini-top {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 9px; }}
  .mini-msg {{ font-size: 13px; font-weight: 600; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .mini-x {{ cursor: pointer; color: var(--text-faint); font-size: 18px; line-height: 1; flex-shrink: 0; padding: 0 2px; }}
  .mini-x:hover {{ color: var(--text); }}
  .mini-track {{ width: 100%; height: 6px; border-radius: 999px; background: var(--border); overflow: hidden; }}
  .mini-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), #5aa2ff); width: 0%; transition: width .5s cubic-bezier(.22,.61,.36,1); }}
  .mini-bot {{ display: flex; align-items: center; justify-content: space-between; margin-top: 7px; }}
  .mini-eta {{ font-size: 11.5px; color: var(--text-faint); font-variant-numeric: tabular-nums; }}
  .mini-pct {{ font-size: 15px; font-weight: 800; color: var(--accent); font-variant-numeric: tabular-nums; }}
  .mini-actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }}
  .mini-go {{ padding: 8px 12px; font-size: 13px; font-weight: 700; font-family: inherit; color: #fff; background: var(--accent); border: none; border-radius: 10px; cursor: pointer; }}
  .mini-go:hover {{ background: var(--accent-hover); }}
  .mini-spinner {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; border: 2px solid #dbe4f0; border-top-color: var(--accent); animation: spin .8s linear infinite; margin-right: 6px; vertical-align: -1px; }}
  .cand-video {{ display: block; }}
  .cand-video video {{ width: 100%; max-width: 260px; border-radius: 14px; background: #000; margin-top: 14px; display: block; }}
  .dl-link {{ display: inline-block; margin-top: 8px; font-size: 13px; font-weight: 600; color: var(--accent); text-decoration: none; }}
  .dl-link:hover {{ text-decoration: underline; }}
  .candidate.flash {{ box-shadow: 0 0 0 2px var(--accent), 0 10px 30px rgba(49,130,246,0.22); transition: box-shadow .3s; }}
  .candidate.rendered-done {{ box-shadow: 0 0 0 1.5px #12b886, 0 6px 22px rgba(18,184,134,0.10); }}
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="/">&larr; 새 링크</a>
  {{% if status != 'ready' %}}
  <h1>쇼츠를 준비하고 있어요</h1>
  <p class="subtitle">링크를 받아 하이라이트를 뽑는 중이에요. 잠시만 기다려주세요…</p>
  {{% else %}}
  <div class="page-head">
    <div>
      <h1>하이라이트 후보</h1>
      <p class="subtitle">바이럴 예상 순위 순으로 정렬했어요. 만들고 싶은 걸 골라주세요.</p>
    </div>
    <div class="page-head-btns">
      <button type="button" class="pill-btn" id="studioBtn">스튜디오</button>
      <button type="button" class="pill-btn" id="ytUploadTop">YouTube 업로드</button>
      <button type="button" class="pill-btn" id="perfTop" title="손으로 올린 쇼츠의 실제 성과를 적어두면 다음 선정 때 채점 기준으로 씁니다">성과 기록</button>
    </div>
  </div>
  {{% endif %}}
  {{% if analyze_error %}}
  <div class="status-box error-box" style="margin-bottom:16px">
    새로 분석 실패 — {{{{ analyze_error }}}}<br>
    <span style="font-size:12.5px;color:var(--text-muted)">아래 목록은 <b>이전 분석 결과</b>입니다. 잠시 후 첫 화면에서 다시 시도하세요.</span>
  </div>
  {{% endif %}}

  {{% if status != 'ready' %}}
  <div class="prog-card" id="prog" data-kind="analyze" data-pct="{{{{ pct }}}}" data-msg="{{{{ status_message }}}}" data-url="/video/{{{{ video_id }}}}/status">
    <div class="prog-head">
      <div class="prog-headL">
        <span class="prog-msg">{{{{ status_message }}}}</span>
        <span class="prog-eta">예상 시간 계산 중…</span>
      </div>
      <span class="prog-pct">{{{{ "%.0f"|format(pct) }}}}%</span>
    </div>
    <div class="stepper">
      <div class="rail"><div class="rail-fill"></div></div>
      <div class="step" data-min="0" data-max="10"><span class="dot"></span><span class="lbl">다운로드</span></div>
      <div class="step" data-min="10" data-max="30"><span class="dot"></span><span class="lbl">전사</span></div>
      <div class="step" data-min="30" data-max="100"><span class="dot"></span><span class="lbl">하이라이트 선정</span></div>
    </div>
  </div>
  {{% else %}}
  <form id="renderForm">
  {{% for c in clips %}}
  <div class="card candidate{{% if loop.index == 1 %}} top{{% endif %}}" id="cand-{{{{ loop.index0 }}}}">
    <div class="cand-head">
      <div class="cand-meta">
        <span class="rank">{{% if loop.index == 1 %}}TOP{{% else %}}{{{{ loop.index }}}}위{{% endif %}}</span>
        {{% if c.score is not none %}}
        <span class="dot-sep"></span>
        <span class="score-badge {{% if c.score >= 73 %}}tier-top{{% elif c.score >= 63 %}}tier-high{{% elif c.score >= 53 %}}tier-ok{{% else %}}tier-low{{% endif %}}">{{{{ "%.0f"|format(c.score) }}}}점{{% if c.score >= 73 %}} · 최상{{% elif c.score >= 63 %}} · 추천{{% elif c.score < 53 %}} · 참고{{% endif %}}</span>
        {{% endif %}}
        <span class="dot-sep"></span>
        <span class="dur">{{{{ "%.0f"|format(c.duration_sec) }}}}초</span>{{% if c.keep_ranges and c.keep_ranges|length > 1 %}}<span class="dot-sep"></span><span class="dur" title="중간을 들어낸 점프컷 클립">점프컷 {{{{ c.keep_ranges|length }}}}조각</span>{{% endif %}}
      </div>
      <label class="pick">
        <input type="checkbox" name="idx" value="{{{{ loop.index0 }}}}">
        <span class="pick-label">만들기</span>
      </label>
    </div>
    {{% set choir = (c.clip_type == 'praise' and c.appeal == '성가대') %}}
    <h3 class="title"{{% if choir %}} data-prefix="[성가대] "{{% endif %}}>{{% if choir %}}[성가대] {{% endif %}}{{{{ c.title }}}}</h3>
    {{# 캡션·해시태그·추천 이유는 기본으로 접어 화면을 조용하게 유지한다(제목이 주인공). #}}
    <div class="cand-foot">
      <button type="button" class="reason-toggle" aria-expanded="false">상세 보기 <span class="chev">▾</span></button>
      <span class="cand-foot-actions">
        <button type="button" class="dup-btn" data-idx="{{{{ loop.index0 }}}}" title="이 후보를 통째로 복제합니다(자막·제목·배속 등 그대로) — 다른 설정으로 한 번 더 만들 때 유용">복제</button>
        <a class="edit-link" href="/video/{{{{ video_id }}}}/clip/{{{{ loop.index0 }}}}/edit">위치·자막 편집 &rarr;</a>
      </span>
    </div>
    <div class="reason" hidden>
      {{% if c.appeal or c.hook_line %}}
      <p class="reason-hashtags">{{% if c.appeal %}}{{{{ c.appeal }}}}{{% endif %}}{{% if c.hook_line %}} · 첫 문장: “{{{{ c.hook_line }}}}”{{% endif %}}{{% if c.payoff_line %}} · 끝 문장: “{{{{ c.payoff_line }}}}”{{% endif %}}</p>
      {{% endif %}}
      {{% if c.core_line %}}
      <p class="reason-hashtags">핵심 문장: “{{{{ c.core_line }}}}”</p>
      {{% endif %}}
      {{% if c.insight %}}
      <p class="reason-hashtags">{{{{ c.insight }}}}</p>
      {{% endif %}}
      {{# 업로드용 캡션: 제목+캡션+해시태그를 붙여넣기 좋은 형태로 조립, 한 번에 복사. #}}
      <div class="capcopy">
        <div class="capcopy-head">
          <span>업로드 캡션</span>
          <button type="button" class="capcopy-btn">복사</button>
          <span class="capcopy-done" hidden>복사됨 ✓</span>
        </div>
        <textarea class="capcopy-text" readonly rows="7">{{{{ c.title }}}}

{{{{ c.caption }}}}

풀 설교 보기: https://youtu.be/{{{{ video_id }}}}

{{{{ c.hashtags|join(' ') }}}} #shorts</textarea>
      </div>
      <p class="reason-text">{{{{ c.reason }}}}</p>
    </div>
    <div class="cand-video" id="candvid-{{{{ loop.index0 }}}}">
    {{% if c.rendered %}}
      <video controls src="/media/{{{{ video_id }}}}/{{{{ loop.index }}}}.mp4"></video>
      <a class="dl-link" href="/media/{{{{ video_id }}}}/{{{{ loop.index }}}}.mp4" download>⬇ 영상 저장</a>
    {{% endif %}}
    </div>
    {{# 업로드 버튼은 카드마다 두지 않고 우측 상단 'YouTube 업로드' 하나로 통합했다
       (사용자 요청 2026-09-06). 이미 업로드된 것의 링크·성과체크 상태만 카드에 남긴다. #}}
    {{% if c.rendered and c.youtube_id %}}
    <div class="yt-upload" data-idx="{{{{ loop.index0 }}}}">
      <a class="yt-link" href="https://youtu.be/{{{{ c.youtube_id }}}}" target="_blank" rel="noopener">▶ YouTube에서 보기</a>
      <span class="yt-track">
        {{% if c.upload_done %}}자동 성과 체크 완료 (4/4주)
        {{% elif c.upload_max_checks %}}자동 성과 체크 진행 중 ({{{{ c.upload_checks_done }}}}/{{{{ c.upload_max_checks }}}}주, 매주 자동 확인)
        {{% endif %}}
      </span>
    </div>
    {{% endif %}}
  </div>
  {{% endfor %}}
  <div class="actions">
    <button class="primary" type="submit" id="renderBtn" {{% if rendering %}}disabled{{% endif %}}>선택한 쇼츠 만들기</button>
  </div>
  </form>

  {{% if rendering %}}
  <div class="prog-card" id="prog" data-kind="render" style="margin-top:16px" data-pct="{{{{ render_pct }}}}" data-msg="{{{{ render_message }}}}" data-url="/video/{{{{ video_id }}}}/status">
    <div class="prog-head">
      <div class="prog-headL">
        <span class="prog-msg">{{{{ render_message }}}}</span>
        <span class="prog-eta">예상 시간 계산 중…</span>
      </div>
      <span class="prog-pct">{{{{ "%.0f"|format(render_pct) }}}}%</span>
    </div>
    <div class="stepper">
      <div class="rail"><div class="rail-fill"></div></div>
      <div class="step" data-min="0" data-max="50"><span class="dot"></span><span class="lbl">자막 인식</span></div>
      <div class="step" data-min="50" data-max="99"><span class="dot"></span><span class="lbl">쇼츠 렌더링</span></div>
      <div class="step" data-min="99" data-max="100"><span class="dot"></span><span class="lbl">완성</span></div>
    </div>
  </div>
  {{% elif render_error %}}
  <div class="status-box error-box" style="margin-top:16px">오류: {{{{ render_error }}}}</div>
  {{% endif %}}
  <div class="status-box" id="renderStatus" style="display:none; margin-top:16px"></div>
  <div class="yt-modal-back" id="ytModalBack">
    <div class="yt-modal">
      <h2>YouTube에 업로드</h2>
      <p class="sub">만든 쇼츠 중 하나를 골라 업로드하세요. 업로드 후 성과는 매주 자동으로 체크돼요.</p>
      <div id="ytPickList"></div>
      <button type="button" class="yt-modal-close" id="ytModalClose">닫기</button>
    </div>
  </div>
  {{# 성과 기록: 앱으로 업로드한 클립은 tracking.py가 매주 자동으로 채우지만, 손으로 올린
     클립은 채울 길이 없었다(선정 프롬프트에 되먹일 데이터가 계속 0건이던 이유). #}}
  <div class="yt-modal-back" id="perfModalBack">
    <div class="yt-modal">
      <h2>성과 기록</h2>
      <p class="sub">손으로 올린 쇼츠의 실제 성과를 적어두면, 다음 선정 때 "예측 → 실제" 사례로 되먹여 채점을 보정해요.</p>
      <select id="perfClip" class="perf-in"></select>
      <div class="perf-grid">
        <label>조회수<input type="number" id="perfViews" min="0" placeholder="예: 12000"></label>
        <label>평균 조회율 %<input type="number" id="perfRet" min="0" max="100" step="0.1" placeholder="예: 62"></label>
        <label>저장<input type="number" id="perfSaves" min="0"></label>
        <label>공유<input type="number" id="perfShares" min="0"></label>
        <label>좋아요<input type="number" id="perfLikes" min="0"></label>
        <label>댓글<input type="number" id="perfComments" min="0"></label>
      </div>
      <label class="perf-lbl">체감 등급
        <select id="perfRating" class="perf-in">
          <option value="hit">HIT — 잘 됨</option>
          <option value="ok" selected>보통</option>
          <option value="flop">FLOP — 망함</option>
        </select>
      </label>
      <label class="perf-lbl">메모<input type="text" id="perfNotes" class="perf-in" placeholder="예: 첫 3초 훅이 약했음"></label>
      <button type="button" class="pill-btn" id="perfSave" style="width:100%;margin-top:10px">저장</button>
      <p class="perf-msg" id="perfMsg"></p>
      <button type="button" class="yt-modal-close" id="perfModalClose">닫기</button>
    </div>
  </div>
  <div class="yt-modal-back" id="studioModalBack">
    <div class="yt-modal">
      <h2>스튜디오</h2>
      <p class="sub">만들 때 적용할 옵션이에요.</p>
      <label class="opt-chip" style="display:flex;width:100%;margin:0 0 8px"><input type="checkbox" id="outroChk" checked> 끝에 로고 2초 넣기</label>
      <label class="opt-chip" style="display:flex;width:100%;margin:0 0 8px"><input type="checkbox" id="sfxChk"> 효과음(전환 whoosh) 넣기</label>
      <label class="opt-chip" style="display:flex;width:100%;margin:0 0 8px"><input type="checkbox" id="motionChk"> 모션(제목 팝·자막 페이드)</label>
      <label class="opt-chip" style="display:flex;width:100%;margin:0 0 8px"><input type="checkbox" id="boldCapChk"> 레퍼런스 자막(볼드·형광펜)</label>
      <label class="opt-chip" style="display:flex;width:100%;margin:0 0 8px"><input type="checkbox" id="facetrackChk"> 얼굴 추적(화면 확대·화자 따라감)</label>
      <label class="opt-chip" style="display:flex;width:100%;margin:0 0 4px;justify-content:space-between">자막 언어
        <select id="capLangSel">
          <option value="bilingual" selected>한글+영어 2줄 (기본)</option>
          <option value="ko">한글만</option>
        </select>
      </label>
      <button type="button" class="yt-modal-close" id="studioModalClose">닫기</button>
    </div>
  </div>
  <script>
  const CLIPS_SUMMARY = {{{{ clips_summary_json | safe }}}};
  document.getElementById('renderForm').addEventListener('submit', async (e) => {{
    e.preventDefault();
    const idx = [...document.querySelectorAll('input[name=idx]:checked')].map(el => parseInt(el.value));
    if (idx.length === 0) {{ alert('클립을 하나 이상 선택하세요'); return; }}
    const outroChk = document.getElementById('outroChk');
    const sfxChk = document.getElementById('sfxChk');
    const motionChk = document.getElementById('motionChk');
    const boldCapChk = document.getElementById('boldCapChk');
    const capLangSel = document.getElementById('capLangSel');
    const facetrackChk = document.getElementById('facetrackChk');
    const doRender = async () => {{
      const res = await fetch('/video/{{{{ video_id }}}}/render', {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{indices: idx, outro: outroChk ? outroChk.checked : true, sfx: sfxChk ? sfxChk.checked : false, motion: motionChk ? motionChk.checked : false, bold_caption: boldCapChk ? boldCapChk.checked : false, caption_lang: capLangSel ? capLangSel.value : 'bilingual', facetrack: facetrackChk ? facetrackChk.checked : false, horizontal_indices: window.__pvHorizontal ? Array.from(window.__pvHorizontal) : []}})
      }});
      const data = await res.json().catch(function() {{ return {{}}; }});
      if (!res.ok) {{ alert('오류: ' + (data.error || '렌더 요청 실패')); return; }}
      // 리로드하지 않는다. 고정 미니위젯이 진행률을 보여주고, 완료되면 그 자리에 영상을 꽂는다.
      window.__startRenderWatch(idx);
    }};
    // 만들기 전 확인 팝업(첫 화면 미리보기 + 제목 후보 + 아이폰식 구간 다듬기).
    // 스크립트가 없으면(로드 실패 등) 예전처럼 바로 렌더로 폴백.
    if (window.__previewFlow) {{ window.__previewFlow(idx, doRender); }} else {{ await doRender(); }}
  }});

  // 선택한 카드에 파란 테두리(picked) 표시 — 무엇을 만들지 한눈에 보이게.
  document.querySelectorAll('.candidate input[name=idx]').forEach(function(cb) {{
    var card = cb.closest('.candidate');
    var sync = function() {{ card.classList.toggle('picked', cb.checked); }};
    cb.addEventListener('change', sync);
    sync();
  }});
  // 기술적인 분석은 기본으로 접어두고 눌렀을 때만 펼친다.
  document.querySelectorAll('.reason-toggle').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      var reason = btn.closest('.candidate').querySelector('.reason');
      var willOpen = reason.hidden;
      reason.hidden = !willOpen;
      btn.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
      btn.firstChild.textContent = willOpen ? '접기 ' : '상세 보기 ';
    }});
  }});

  // 후보 복제: 자막·배속 등을 그대로 복사한 새 후보를 목록 끝에 추가한다.
  document.querySelectorAll('.dup-btn').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      btn.disabled = true; var old = btn.textContent; btn.textContent = '복제 중…';
      fetch('/video/{{{{ video_id }}}}/clip/' + btn.dataset.idx + '/duplicate', {{ method: 'POST' }})
        .then(function(r) {{ return r.json().then(function(d) {{ return {{ok: r.ok, d: d}}; }}); }})
        .then(function(res) {{
          if (!res.ok) {{ alert('복제 실패: ' + (res.d.error || '')); btn.disabled = false; btn.textContent = old; return; }}
          sessionStorage.setItem('scrollToCand', res.d.new_idx);
          location.reload();
        }})
        .catch(function() {{ alert('복제 요청 실패'); btn.disabled = false; btn.textContent = old; }});
    }});
  }});
  (function() {{
    var target = sessionStorage.getItem('scrollToCand');
    if (target !== null) {{
      sessionStorage.removeItem('scrollToCand');
      var el = document.getElementById('cand-' + target);
      if (el) setTimeout(function() {{ el.scrollIntoView({{behavior: 'smooth', block: 'center'}}); }}, 100);
    }}
  }})();

  // 업로드 캡션 한 번에 복사 (클립보드 API 실패 시 select+execCommand 폴백)
  document.querySelectorAll('.capcopy').forEach(function(box) {{
    var btn = box.querySelector('.capcopy-btn');
    var done = box.querySelector('.capcopy-done');
    var ta = box.querySelector('.capcopy-text');
    btn.addEventListener('click', function() {{
      var text = ta.value;
      var ok = function() {{
        done.hidden = false;
        setTimeout(function() {{ done.hidden = true; }}, 1800);
      }};
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(text).then(ok, function() {{ ta.select(); document.execCommand('copy'); ok(); }});
      }} else {{
        ta.select(); document.execCommand('copy'); ok();
      }}
    }});
  }});

  // YouTube 업로드: 카드마다 버튼을 두지 않고 우측 상단 하나로 통합했다(사용자 요청
  // 2026-09-06 "가장 우측 상단에 하나만 있고 누르면 선택하게 해줘"). 눌러 업로드되면
  // 이후 성과는 주 1회 최대 4주 자동으로 체크된다.
  const ytModalBack = document.getElementById('ytModalBack');
  const ytPickList = document.getElementById('ytPickList');
  function renderYtPickList() {{
    // 제목은 사용자가 고치는 문자열이라 '<'가 들어가면 아래 innerHTML 조립이 깨진다.
    const esc = (t) => String(t == null ? '' : t).replace(/[&<>"]/g, (ch) => (
      {{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }}[ch]
    ));
    const renderedClips = CLIPS_SUMMARY.filter((c) => c.rendered);
    if (!renderedClips.length) {{
      ytPickList.innerHTML = '<p class="yt-empty">아직 만든 쇼츠가 없어요. 먼저 후보를 선택해 "만들기"를 눌러주세요.</p>';
      return;
    }}
    ytPickList.innerHTML = renderedClips.map((c) => {{
      if (c.youtube_id) {{
        return '<div class="yt-pick-row done"><span class="name">' + esc(c.title) + '</span>' +
          '<a class="yt-link" href="https://youtu.be/' + c.youtube_id + '" target="_blank" rel="noopener">▶ 이미 업로드됨</a></div>';
      }}
      return '<div class="yt-pick-row" data-idx="' + c.idx + '"><span class="name">' + esc(c.title) + '</span>' +
        '<button type="button" class="yt-pick-btn">업로드</button></div>';
    }}).join('');
    ytPickList.querySelectorAll('.yt-pick-btn').forEach((btn) => {{
      btn.addEventListener('click', async () => {{
        const row = btn.closest('.yt-pick-row');
        btn.disabled = true; btn.textContent = '업로드 중…';
        try {{
          const res = await fetch('/video/{{{{ video_id }}}}/clip/' + row.dataset.idx + '/upload', {{ method: 'POST' }});
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || '업로드 실패');
          const c = CLIPS_SUMMARY.find((x) => String(x.idx) === row.dataset.idx);
          if (c) c.youtube_id = data.youtube_id || data.url.split('/').pop();
          row.className = 'yt-pick-row done';
          row.innerHTML = '<span class="name">' + row.querySelector('.name').textContent + '</span>' +
            '<a class="yt-link" href="' + data.url + '" target="_blank" rel="noopener">▶ 업로드 완료</a>';
        }} catch (e) {{
          btn.disabled = false; btn.textContent = '실패: ' + e.message;
        }}
      }});
    }});
  }}
  document.getElementById('ytUploadTop').addEventListener('click', () => {{
    renderYtPickList();
    ytModalBack.classList.add('show');
  }});
  document.getElementById('ytModalClose').addEventListener('click', () => ytModalBack.classList.remove('show'));
  ytModalBack.addEventListener('click', (e) => {{ if (e.target === ytModalBack) ytModalBack.classList.remove('show'); }});

  // 성과 기록: 만든 클립 목록을 채우고, 입력한 수치를 /feedback에 올린다(빈 칸은 안 보냄).
  const perfModalBack = document.getElementById('perfModalBack');
  const perfMsg = document.getElementById('perfMsg');
  document.getElementById('perfTop').addEventListener('click', () => {{
    const sel = document.getElementById('perfClip');
    const made = CLIPS_SUMMARY.filter((c) => c.rendered);
    sel.innerHTML = '';
    made.forEach((c) => {{
      const o = document.createElement('option');
      o.value = c.idx; o.textContent = (c.idx + 1) + '. ' + c.title;
      sel.appendChild(o);
    }});
    if (!made.length) {{
      const o = document.createElement('option');
      o.value = ''; o.textContent = '아직 만든 쇼츠가 없어요';
      sel.appendChild(o);
    }}
    perfMsg.textContent = '';
    perfModalBack.classList.add('show');
  }});
  document.getElementById('perfModalClose').addEventListener('click', () => perfModalBack.classList.remove('show'));
  perfModalBack.addEventListener('click', (e) => {{ if (e.target === perfModalBack) perfModalBack.classList.remove('show'); }});
  document.getElementById('perfSave').addEventListener('click', async () => {{
    const idx = document.getElementById('perfClip').value;
    if (idx === '') {{ perfMsg.textContent = '먼저 쇼츠를 만들어 주세요.'; return; }}
    const num = (id) => {{
      const v = document.getElementById(id).value.trim();
      return v === '' ? null : Number(v);
    }};
    const btn = document.getElementById('perfSave');
    btn.disabled = true; perfMsg.textContent = '저장 중…';
    try {{
      const res = await fetch('/video/{{{{ video_id }}}}/feedback', {{
        method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          clip_index: Number(idx),
          views: num('perfViews'), retention_pct: num('perfRet'),
          saves: num('perfSaves'), shares: num('perfShares'),
          likes: num('perfLikes'), comments: num('perfComments'),
          rating: document.getElementById('perfRating').value,
          notes: document.getElementById('perfNotes').value,
        }}),
      }});
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || ('HTTP ' + res.status));
      perfMsg.textContent = '저장했어요 ✓ 다음 선정부터 반영돼요.';
    }} catch (e) {{
      perfMsg.textContent = '실패: ' + e.message;
    }} finally {{
      btn.disabled = false;
    }}
  }});

  // 스튜디오: 로고/효과음/모션/얼굴추적 등 렌더 옵션을 여기 모아둔다(사용자 요청 2026-09-06,
  // 메인 화면을 간결하게 유지).
  const studioModalBack = document.getElementById('studioModalBack');
  document.getElementById('studioBtn').addEventListener('click', () => studioModalBack.classList.add('show'));
  document.getElementById('studioModalClose').addEventListener('click', () => studioModalBack.classList.remove('show'));
  studioModalBack.addEventListener('click', (e) => {{ if (e.target === studioModalBack) studioModalBack.classList.remove('show'); }});
  </script>
  {{% endif %}}

  <script>
  (function() {{
    var el = document.getElementById('prog');
    if (!el) return;
    var kind = el.dataset.kind, url = el.dataset.url;
    var fill = el.querySelector('.rail-fill');
    var steps = Array.prototype.slice.call(el.querySelectorAll('.step'));
    var pctEl = el.querySelector('.prog-pct');
    var msgEl = el.querySelector('.prog-msg');
    var etaEl = el.querySelector('.prog-eta');
    function fmtEta(s) {{
      if (s == null || s < 0) return '';
      s = Math.round(s);
      if (s < 60) return '약 ' + Math.max(1, s) + '초 남음';
      var m = Math.round(s / 60);
      return '약 ' + m + '분 남음';
    }}
    function setEta(s) {{
      if (!etaEl) return;
      var t = fmtEta(s);
      etaEl.textContent = t || '예상 시간 계산 중…';
    }}
    function paint(pct, msg) {{
      pct = Math.max(0, Math.min(100, pct || 0));
      fill.style.width = pct + '%';
      pctEl.textContent = Math.round(pct) + '%';
      if (msg) msgEl.textContent = msg;
      steps.forEach(function(s) {{
        var mn = parseFloat(s.dataset.min), mx = parseFloat(s.dataset.max);
        s.classList.remove('active', 'done');
        if (pct >= mx) s.classList.add('done');
        else if (pct >= mn) s.classList.add('active');
      }});
    }}
    paint(parseFloat(el.dataset.pct), el.dataset.msg);
    function poll() {{
      fetch(url).then(function(r) {{ return r.json(); }}).then(function(j) {{
        if (kind === 'analyze') {{
          paint(j.pct, j.message);
          setEta(j.eta_seconds);
          if (j.ready) {{ paint(100, '완료'); if (etaEl) etaEl.textContent = '거의 완료…'; setTimeout(function() {{ location.reload(); }}, 500); return; }}
          if (j.status === 'error') {{ msgEl.textContent = j.message; if (etaEl) etaEl.textContent = ''; return; }}
        }} else {{
          paint(j.render_pct, j.render_message);
          setEta(j.render_eta_seconds);
          if (j.render_error) {{ location.reload(); return; }}
          if (!j.rendering) {{ paint(100, '완성'); if (etaEl) etaEl.textContent = '거의 완료…'; setTimeout(function() {{ location.reload(); }}, 500); return; }}
        }}
        setTimeout(poll, 650);
      }}).catch(function() {{ setTimeout(poll, 1200); }});
    }}
    setTimeout(poll, 650);
  }})();
  </script>

  <div class="mini-prog hidden" id="miniProg">
    <div class="mini-top">
      <span class="mini-msg" id="miniMsg"><span class="mini-spinner"></span>진행 중…</span>
      <span class="mini-x" id="miniClose" title="닫기" hidden>&times;</span>
    </div>
    <div class="mini-track" id="miniTrack"><div class="mini-fill" id="miniFill"></div></div>
    <div class="mini-bot">
      <span class="mini-eta" id="miniEta"></span>
      <span class="mini-pct" id="miniPct">0%</span>
    </div>
    <div class="mini-actions" id="miniActions" hidden></div>
  </div>
  <script>
  // 스크롤 위치와 무관하게 항상 보이는 고정 진행 위젯. 분석/렌더 상태를 폴링해 갱신하고,
  // 렌더가 끝나면 그 자리(카드)에 영상을 꽂고 위젯을 '완성' 상태로 남긴다(사라지지 않게).
  (function() {{
    var VID = "{{{{ video_id }}}}";
    var box = document.getElementById('miniProg');
    var msg = document.getElementById('miniMsg'), pct = document.getElementById('miniPct');
    var fill = document.getElementById('miniFill'), eta = document.getElementById('miniEta');
    var track = document.getElementById('miniTrack'), actions = document.getElementById('miniActions');
    var closeBtn = document.getElementById('miniClose');
    var statusUrl = "/video/" + VID + "/status";
    var renderIndices = null;   // 이번 세션에서 렌더 요청한 클립들(버튼 클릭 시 채워짐)
    var wasRendering = false;   // 렌더 진행 중이었는지(완료 전이 감지용)
    var polling = false;
    var idleTicks = 0;          // 연속으로 '아무것도 안 도는' 응답을 받은 횟수(폴링 백오프용)
    var reanalyzeUrl = "/video/" + VID + "/reanalyze_status";
    var wasReanalyzing = false;  // 팝업을 닫아도(만들기 전 확인 창) 재분석은 서버에서 계속 돌고,
    // 이 위젯이 이어서 진행률을 보여준다("팝업 나가면 취소되는 것처럼 보인다"는 신고 수정.
    // 실제로는 서버 스레드가 계속 도는데, 진행 상황을 보여줄 UI가 팝업 안에만 있었을 뿐).
    closeBtn.addEventListener('click', function() {{ box.classList.add('hidden'); }});

    function fmtEta(s) {{
      if (s == null || s < 0) return '';
      s = Math.round(s);
      return s < 60 ? ('약 ' + Math.max(1, s) + '초 남음') : ('약 ' + Math.round(s / 60) + '분 남음');
    }}
    function showProg(p, m, e) {{
      box.classList.remove('hidden'); track.hidden = false; actions.hidden = true; closeBtn.hidden = true;
      p = Math.max(0, Math.min(100, p || 0));
      fill.style.width = p + '%'; pct.style.display = '';
      pct.textContent = Math.round(p) + '%';
      msg.innerHTML = '<span class="mini-spinner"></span>' + (m || '진행 중…');
      eta.textContent = fmtEta(e) || (p >= 100 ? '거의 완료…' : '예상 시간 계산 중…');
    }}
    function injectVideo(i) {{
      var slot = document.getElementById('candvid-' + i);
      if (!slot) return;
      var src = '/media/' + VID + '/' + (i + 1) + '.mp4?t=' + Date.now();
      slot.innerHTML = '<video controls src="' + src + '"></video>' +
        '<a class="dl-link" href="/media/' + VID + '/' + (i + 1) + '.mp4" download>⬇ 영상 저장</a>';
      var card = document.getElementById('cand-' + i);
      if (card) card.classList.add('rendered-done');
    }}
    function showDone(indices) {{
      box.classList.remove('hidden'); track.hidden = true; closeBtn.hidden = false; pct.style.display = 'none';
      msg.innerHTML = '쇼츠 완성! 영상은 자동 저장됐어요';
      eta.textContent = '';
      actions.hidden = false;
      actions.innerHTML = '';
      indices.forEach(function(i) {{
        var b = document.createElement('button');
        b.className = 'mini-go';
        b.textContent = '#' + (i + 1) + ' 영상 보기';
        b.addEventListener('click', function() {{
          var card = document.getElementById('cand-' + i);
          if (card) {{ card.scrollIntoView({{behavior: 'smooth', block: 'center'}}); card.classList.add('flash'); setTimeout(function() {{ card.classList.remove('flash'); }}, 1500); }}
        }});
        actions.appendChild(b);
      }});
    }}
    function onRenderDone() {{
      if (renderIndices && renderIndices.length) {{
        renderIndices.forEach(injectVideo);
        showDone(renderIndices);
        renderIndices = null;
      }} else {{
        location.reload();  // 렌더 중 새로고침한 경우 등: 서버 렌더 상태로 복원
      }}
    }}
    function showReanalyzeDone(ok, errMsg) {{
      box.classList.remove('hidden'); track.hidden = true; closeBtn.hidden = false; pct.style.display = 'none';
      msg.innerHTML = ok ? '구간 재분석 완료 — 후보 목록 맨 아래 추가됨' : ('재분석 실패: ' + errMsg);
      eta.textContent = '';
      actions.hidden = false; actions.innerHTML = '';
      var b = document.createElement('button');
      b.className = 'mini-go'; b.textContent = '새로고침해서 보기';
      b.addEventListener('click', function() {{ location.reload(); }});
      actions.appendChild(b);
    }}
    function pollReanalyze() {{
      fetch(reanalyzeUrl).then(function(r) {{ return r.json(); }}).then(function(j) {{
        if (j.running) {{ wasReanalyzing = true; showProg((j.pct || 0) * 100, '구간 재분석 중…', null); }}
        else if (wasReanalyzing) {{ wasReanalyzing = false; showReanalyzeDone(!j.error, j.error); }}
      }}).catch(function() {{}});
    }}
    function poll() {{
      fetch(statusUrl).then(function(r) {{ return r.json(); }}).then(function(j) {{
        var idle = false;   // 이번 응답 기준으로 '아무것도 안 돌고 있음'이면 아래에서 백오프
        if (j.render_error) {{ box.classList.remove('hidden'); track.hidden = true; closeBtn.hidden = false; pct.style.display='none'; msg.textContent = '오류: ' + j.render_error; eta.textContent=''; actions.hidden=true; wasRendering=false; setTimeout(poll, 1500); return; }}
        if (j.rendering) {{ wasRendering = true; showProg(j.render_pct, j.render_message || '쇼츠 렌더링 중…', j.render_eta_seconds); }}
        else if (wasRendering) {{ wasRendering = false; onRenderDone(); }}
        else if (!j.ready && j.status !== 'error') {{ showProg(j.pct, j.message || '분석 중…', j.eta_seconds); }}
        else {{ pollReanalyze(); idle = true; }}   // 렌더/분석이 안 도는 동안만 재분석 상태를 확인(위젯 하나 공유)
        // 놀고 있을 때는 폴링 간격을 늘린다(백오프). 예전엔 조건 없이 800ms로 자기를 다시
        // 걸어서, 분석·렌더가 다 끝난 뒤에도 초당 2회 요청이 영원히 나갔다(실측: 페이지를
        // 열어둔 9시간 28분 동안 계속 — server_err.txt 1MB). 배터리/CPU를 먹고, 터널로
        // 열어두면 그대로 외부 트래픽이 된다. 뭔가 시작되면 즉시 800ms로 되돌아간다.
        if (idle) {{ idleTicks++; }} else {{ idleTicks = 0; }}
        setTimeout(poll, idle ? Math.min(10000, 800 * Math.pow(2, Math.min(idleTicks, 4))) : 800);
      }}).catch(function() {{ setTimeout(poll, 1500); }});
    }}
    // 버튼 클릭 시 호출: 이번에 렌더할 인덱스를 기억하고 즉시 위젯을 띄운다.
    window.__startRenderWatch = function(indices) {{
      renderIndices = indices; wasRendering = true;
      idleTicks = 0;   // 백오프 초기화 — 방금 시작한 렌더는 800ms 간격으로 즉시 따라붙는다
      showProg(0, '렌더링 시작…', null);
      if (!polling) {{ polling = true; poll(); }}
    }};
    poll(); polling = true;
  }})();
  </script>
  <script src="/js/preview-modal.js"></script>
</div>
</body>
</html>
"""


def _run_analyze_job(
    video_id_holder: dict, url: str, transcript_text: str = "", force: bool = False,
    model: str = "", mode: str = "sermon", song_titles: str = "",
) -> None:
    try:
        video_dir, clips = analyze(
            url,
            progress=lambda msg, pct, eta=None: _update_job(
                video_id_holder["id"], message=msg, pct=pct, eta_seconds=eta
            ),
            transcript_text=transcript_text,
            force=force,
            model=model,
            mode=mode,
            song_titles=song_titles,
        )
        video_id_holder["id"] = video_dir.name
        _update_job(video_dir.name, status="ready", clips=clips, message="완료", pct=100)
    except Exception as e:  # noqa: BLE001 - 사용자에게 실패 사유를 그대로 보여줘야 함
        vid = video_id_holder.get("id", "unknown")
        _update_job(vid, status="error", message=f"실패: {e}", pct=0)
        traceback.print_exc()


@app.route("/")
def index():
    return render_template_string(INDEX_TEMPLATE)


@app.route("/analyze", methods=["POST"])
def analyze_route():
    body = request.get_json()
    url = body.get("url", "").strip()
    transcript_text = (body.get("transcript_text") or "").strip()
    force = bool(body.get("force"))
    # 선정 모델은 UI 라디오(소넷/오푸스)에서 온다. 임의 문자열을 CLI에 넘기지 않도록
    # 허용 목록으로 제한하고, 벗어나면 빈 값("")으로 둬 analyze()가 config 기본을 쓴다.
    model = sanitize_model(body.get("model") or "")
    # 인덱스의 말씀/찬양 버튼. 허용 목록 밖 값은 기본(말씀)으로.
    mode = (body.get("mode") or "sermon").strip()
    if mode not in ("sermon", "praise"):
        mode = "sermon"
    song_titles = (body.get("song_titles") or "").strip() if mode == "praise" else ""
    if not url:
        return jsonify({"error": "URL이 비어있습니다"}), 400

    from src.download import extract_video_id

    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # 같은 영상이 이미 분석 중이면 재요청은 새 스레드를 또 띄우지 않는다. 인덱스 페이지를
    # 다시 열고 폼을 재제출하는 식으로 중복 POST가 오면(실측: 30초 새 3번), analyze()마다
    # 독립된 StageProgress 티커가 같은 job의 pct를 동시에 덮어써 진행률이 40%→80%→50%처럼
    # 널뛰었고, claude -p 하이라이트 선정도 그만큼 중복 실행돼 세션 한도만 낭비됐다.
    with _jobs_lock:
        existing = _jobs.get(video_id)
        if existing and existing.get("status") == "analyzing":
            return jsonify({"video_id": video_id})
        _jobs.setdefault(video_id, {}).update(
            status="analyzing", message="분석 시작...", pct=0, started=time.time()
        )

    holder = {"id": video_id}
    threading.Thread(
        target=_run_analyze_job,
        args=(holder, url, transcript_text, force, model, mode, song_titles),
        daemon=True,
    ).start()
    return jsonify({"video_id": video_id})


@app.route("/analyze_upload", methods=["POST"])
def analyze_upload_route():
    """직접 찍은 동영상 파일 업로드 → 저장 후 로컬 분석("local:<vid>" 경로).

    유튜브 링크 대신 파일이 소스다: output/upload_<ts>/source.mp4로 저장하고,
    analyze()가 다운로드/유튜브 자막을 건너뛰고 whisper 직접 전사부터 시작한다."""
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "업로드된 파일이 없습니다"}), 400
    mode = (request.form.get("mode") or "sermon").strip()
    if mode not in ("sermon", "praise"):
        mode = "sermon"
    model = sanitize_model(request.form.get("model") or "")
    # 찬양 곡 제목(한 줄에 한 곡, 부른 순서). 입력하면 whisper 전사를 건너뛰고 정식 가사를
    # 자막으로 쓴다(사용자 요청 2026-09-05: 노래 전사 정확도가 너무 낮음). sermon엔 무의미.
    song_titles = (request.form.get("song_titles") or "").strip() if mode == "praise" else ""

    video_id = "upload_" + time.strftime("%Y%m%d_%H%M%S")
    video_dir = OUTPUT_ROOT / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    # 확장자와 무관하게 source.mp4로 저장한다 — ffmpeg/whisper는 파일 내용으로 컨테이너를
    # 판별하므로(mov/mp4/mkv 모두 OK) 이름은 파이프라인 규약(source.mp4)만 따르면 된다.
    save_path = video_dir / "source.mp4"
    f.save(save_path)
    if not save_path.exists() or save_path.stat().st_size == 0:
        return jsonify({"error": "파일 저장에 실패했습니다"}), 500

    with _jobs_lock:
        _jobs.setdefault(video_id, {}).update(
            status="analyzing", message="업로드 완료, 분석 시작...", pct=0, started=time.time()
        )
    holder = {"id": video_id}
    threading.Thread(
        target=_run_analyze_job,
        args=(holder, f"local:{video_id}", "", False, model, mode, song_titles),
        daemon=True,
    ).start()
    return jsonify({"video_id": video_id})


@app.route("/video/<video_id>")
def video_detail(video_id: str):
    with _jobs_lock:
        job = _jobs.get(video_id)

    clips_path = OUTPUT_ROOT / video_id / "clips.json"

    # 재분석 중이면 옛 clips.json을 완료로 오인하지 않는다(_clips_ready). 그 외엔 디스크가
    # 진실의 원천 — 렌더링이 job status를 흔들어도 완료 상태가 유지된다.
    clips_ready = _clips_ready(video_id, job)
    job = job or {}

    if clips_ready:
        status = "ready"
        clips = load_clips_json(clips_path)
        pct = 100
    elif job:
        status = job.get("status", "analyzing")
        clips = job.get("clips", [])
        pct = job.get("pct", 0)
    else:
        return "해당 영상 작업을 찾을 수 없습니다. 처음부터 다시 시도하세요.", 404

    clips_dir = OUTPUT_ROOT / video_id / "clips"
    for i, c in enumerate(clips, start=1):
        # mp4가 있어도, 그게 '지금 이 클립'의 렌더인지 서명(start_end)으로 대조한다.
        # 재선정 등으로 clips.json이 바뀌면 옛 short_N.mp4가 다른 내용인데도 붙어 보이던
        # 문제(사용자: "안 만들었는데 멋대로 만들어짐")를 막는다. 서명 없거나 불일치면 미렌더 취급.
        mp4 = clips_dir / f"short_{i}.mp4"
        src = clips_dir / f"short_{i}.src"
        sig = render_signature(c.start, c.end)
        c.rendered = (
            mp4.exists() and src.exists()
            and src.read_text(encoding="utf-8").strip() == sig
        )
        up = find_upload(video_id, i - 1)
        # 재선정 등으로 같은 인덱스가 다른 클립이 됐으면(서명 불일치) 업로드 정보를 붙이지
        # 않는다. 서명 없는 옛 레코드는 기존처럼 인덱스만으로 매칭(하위호환).
        if up and up.clip_sig and up.clip_sig != sig:
            up = None
        c.youtube_id = up.youtube_video_id if up else ""
        c.upload_checks_done = up.checks_done if up else 0
        c.upload_max_checks = up.max_checks if up else 0
        c.upload_done = up.done if up else False

    # '새로 분석'이 실패한 경우(세션 한도 등): 옛 clips.json이 남아 있으면 페이지는 그걸
    # '완료'로 보여주는데, 에러 표시가 없으면 "새로 분석했는데 옛날 그대로"로 오인된다
    # (실신고). 실패 사유를 배너로 함께 보여준다.
    analyze_error = job.get("message") if job.get("status") == "error" else None

    # YouTube 업로드 선택 모달용 요약 데이터(카드마다 있던 업로드 버튼을 우측 상단
    # 하나로 통합하면서 필요해짐, 사용자 요청 2026-09-06).
    clips_summary = [
        {
            "idx": i,
            "title": ("[성가대] " if (c.clip_type == "praise" and c.appeal == "성가대") else "") + c.title,
            "rendered": bool(c.rendered),
            "youtube_id": c.youtube_id or "",
        }
        for i, c in enumerate(clips)
    ]

    return render_template_string(
        CANDIDATES_TEMPLATE,
        video_id=video_id,
        status=status,
        status_message=job.get("message", "처리 중..."),
        pct=pct,
        clips=clips,
        clips_summary_json=json.dumps(clips_summary, ensure_ascii=False),
        analyze_error=analyze_error,
        rendering=job.get("rendering", False),
        render_message=job.get("render_message", "렌더링 준비 중..."),
        render_pct=job.get("render_pct", 0),
        render_error=job.get("render_error"),
    )


@app.route("/video/<video_id>/status")
def video_status(video_id: str):
    """진행바 폴링용 경량 JSON. 전체 페이지를 새로고침하지 않고 이 상태만 받아
    스텝 진행바를 부드럽게 갱신한다 (분석/렌더 두 단계 모두 커버)."""
    with _jobs_lock:
        job = dict(_jobs.get(video_id) or {})
    clips_ready = _clips_ready(video_id, job)
    analyze_pct = 100 if clips_ready else job.get("pct", 0)
    return jsonify({
        "status": "ready" if clips_ready else job.get("status", "analyzing"),
        "ready": clips_ready,
        "pct": analyze_pct,
        "message": job.get("message", "처리 중..."),
        "eta_seconds": 0 if clips_ready else job.get("eta_seconds"),
        "rendering": job.get("rendering", False),
        "render_pct": job.get("render_pct", 0),
        "render_message": job.get("render_message", "렌더링 준비 중..."),
        "render_error": job.get("render_error"),
        "render_eta_seconds": job.get("render_eta_seconds"),
    })


@app.route("/video/<video_id>/render", methods=["POST"])
def render_route(video_id: str):
    body = request.get_json() or {}
    indices = body.get("indices", [])
    if not indices:
        return jsonify({"error": "선택된 항목이 없습니다"}), 400
    outro_enabled = bool(body.get("outro", True))
    sfx_enabled = bool(body.get("sfx", False))
    motion_enabled = bool(body.get("motion", False))
    caption_preset = "bold_yellow" if body.get("bold_caption") else ""
    # 자막 언어: 명시적 caption_lang("bilingual"/"ko"/"en")을 우선한다. 기본은 한글+영어
    # 2줄(사용자 요청 "디폴트로") — 영어 트랙이 있는 클립만 실제 2줄이 되므로 무해.
    # 예전 체크박스 2개(english/bilingual)는 '둘 다 체크하면 영어만'이 되는 혼란이 있었다
    # (실신고 2026-09-05) — 셀렉트 하나로 교체, 구 클라이언트 값은 bilingual 우선으로 해석.
    _cl = str(body.get("caption_lang") or "").strip()
    if _cl in ("en", "bilingual"):
        caption_lang = _cl
    elif _cl == "ko":
        caption_lang = ""
    else:  # 구 체크박스/스튜디오(미전송) 하위호환
        caption_lang = "bilingual" if body.get("bilingual", True) else ("en" if body.get("english") else "")
    facetrack_enabled = bool(body.get("facetrack", False))
    # 팝업 '가로 원본' 버튼으로 고른 클립들: 찬양을 쇼츠 레이아웃 없이 원본 가로 그대로,
    # 자막·제목 없이 잘라만 낸다(곡별 개별 업로드용).
    try:
        horizontal_indices = [int(x) for x in (body.get("horizontal_indices") or [])]
    except (TypeError, ValueError):
        horizontal_indices = []

    video_dir = OUTPUT_ROOT / video_id
    # 같은 영상 렌더가 이미 도는 중이면 새 스레드를 또 띄우지 않는다(분석과 동일한 가드).
    # 버튼 비활성화는 클라이언트에만 있어서, 다른 탭/새로고침 후 재클릭이면 서버로 중복
    # POST가 온다 — 렌더 2개가 같은 short_N.mp4와 clips.json을 동시에 쓰며 꼬인다.
    # 프론트는 ok 응답을 받고 진행 위젯 폴링을 시작하므로, 기존 렌더에 그냥 합류하게 된다.
    with _jobs_lock:
        job = _jobs.get(video_id) or {}
        if job.get("rendering"):
            return jsonify({"ok": True, "already": True})
        _jobs.setdefault(video_id, {}).update(
            rendering=True, render_pct=0, render_message="렌더링 시작...",
            render_error=None, render_started=time.time(),
        )

    def _job():
        try:
            render_selected(
                video_dir, indices,
                progress=lambda msg, pct, eta=None: _update_job(
                    video_id, render_message=msg, render_pct=pct, render_eta_seconds=eta
                ),
                outro_enabled=outro_enabled,
                sfx_enabled=sfx_enabled,
                motion_enabled=motion_enabled,
                caption_preset=caption_preset,
                caption_lang=caption_lang,
                facetrack_enabled=facetrack_enabled,
                horizontal_indices=horizontal_indices,
            )
        except Exception as e:  # noqa: BLE001 - 사용자에게 실패 사유를 그대로 보여줘야 함
            _update_job(video_id, render_error=str(e))
            traceback.print_exc()
        finally:
            _update_job(video_id, rendering=False)

    threading.Thread(target=_job, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/video/<video_id>/feedback", methods=["POST"])
def feedback_route(video_id: str):
    """올린 클립의 실제 성과를 기록한다. 다음 선정 프롬프트에 캘리브레이션 사례로 주입된다."""
    body = request.get_json() or {}
    try:
        clip_index = int(body.get("clip_index"))
    except (TypeError, ValueError):
        return jsonify({"error": "clip_index가 필요합니다"}), 400

    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    clip = None
    if clips_path.exists():
        clips = load_clips_json(clips_path)
        if 0 <= clip_index < len(clips):
            clip = clips[clip_index]

    def _num(key, cast=float):
        v = body.get(key)
        try:
            return cast(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # 어떤 훅이 실제로 먹혔는지가 캘리브레이션의 핵심 신호인데, 예전엔 사용자가 자막을 손으로
    # 고친 클립(caption_overrides)에서만 채워져 대부분 빈 값으로 기록됐다(2026-09-22). 선정이 확정한
    # 첫 문장(hook_line)을 기본값으로 쓴다.
    hook_text = ""
    if clip is not None:
        if clip.caption_overrides:
            hook_text = (clip.caption_overrides[0] or {}).get("text", "") or ""
        if not hook_text:
            hook_text = (getattr(clip, "hook_line", "") or "").strip()

    record = PerformanceRecord(
        video_id=video_id,
        clip_index=clip_index,
        title=(body.get("title") or (clip.title if clip else "")).strip(),
        hook_text=hook_text,
        start=clip.start if clip else 0.0,
        end=clip.end if clip else 0.0,
        predicted_score=clip.score if clip else None,
        predicted_core=clip.core_score if clip else None,
        predicted_viral=clip.viral_score if clip else None,
        predicted_subscores=(
            {
                "hook": clip.hook_score, "retention": clip.retention_score,
                "emotion": clip.emotion_score, "relatability": clip.relatability_score,
                "payoff": clip.payoff_score, "quotability": clip.quotability_score,
            }
            if clip else {}
        ),
        views=_num("views", int),
        retention_pct=_num("retention_pct", float),
        saves=_num("saves", int),
        shares=_num("shares", int),
        likes=_num("likes", int),
        comments=_num("comments", int),
        rating=(body.get("rating") or "ok").strip(),
        notes=(body.get("notes") or "").strip(),
    )
    upsert_feedback(record)
    return jsonify({"ok": True})


@app.route("/video/<video_id>/clip/<int:idx>/upload", methods=["POST"])
def upload_clip_route(video_id: str, idx: int):
    """렌더된 클립을 YouTube에 업로드하고, 이후 자동 성과 체크(주 1회, 최대 한달)를 예약한다."""
    from src.upload.youtube import upload_short

    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "clips.json이 없습니다"}), 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "invalid index"}), 400
    clip = clips[idx]

    video_path = OUTPUT_ROOT / video_id / "clips" / f"short_{idx + 1}.mp4"
    if not video_path.exists():
        return jsonify({"error": "먼저 이 클립을 렌더링하세요"}), 400

    cfg = _load_config()["upload"]["youtube"]
    if not cfg.get("enabled", True):
        return jsonify({"error": "config.yaml에서 upload.youtube가 비활성화돼 있습니다"}), 400

    try:
        yt_id = upload_short(
            video_path,
            title=clip.title or f"{video_id} 쇼츠 {idx + 1}",
            description=clip.caption,
            tags=[h.lstrip("#") for h in clip.hashtags],
            category_id=cfg.get("category_id", "22"),
            privacy_status=cfg.get("default_privacy", "unlisted"),
        )
    except Exception as e:  # noqa: BLE001 - 업로드 실패 사유를 그대로 사용자에게 보여줘야 함
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    record_upload(
        video_id, idx, yt_id, title=clip.title,
        clip_sig=render_signature(clip.start, clip.end),
    )
    return jsonify({"ok": True, "youtube_id": yt_id, "url": f"https://youtu.be/{yt_id}"})


@app.route("/media/<video_id>/<int:rank>.mp4")
def media(video_id: str, rank: int):
    # send_from_directory 대신 send_file + 직접 resolve()한 절대경로를 쓴다.
    # 이 프로젝트 경로에 한글("코딩")이 섞여 있는데, werkzeug의 send_from_directory
    # 내부 safe_join이 비-ASCII 경로에서 실패해 파일이 있어도 404를 반환하는 문제가 있었다.
    path = (OUTPUT_ROOT / video_id / "clips" / f"short_{rank}.mp4").resolve()
    if not path.exists():
        return jsonify({"error": "파일을 찾을 수 없습니다"}), 404
    return send_file(path)


PREVIEW_CANVAS_WIDTH = 360  # 실제 렌더 해상도(보통 1080px 폭)를 화면에 축소해서 보여줄 너비(px)

# 클립별 무음 지도 캐시(ffmpeg silencedetect 2~3초 — 팝업 열 때마다 다시 돌리지 않게)
# 서버를 며칠씩 켜두면 클립 경계를 조금씩 바꿀 때마다 키가 새로 생겨 무한정 쌓인다 → 상한.
_silences_cache: dict = {}
_SILENCES_CACHE_MAX = 200


def _clip_voice_silences(video_id: str, start: float, end: float) -> list:
    """렌더와 동일한 파라미터(-32dB/0.3s)로 클립 구간 무음 지도를 얻는다(클립 상대시간)."""
    key = (video_id, round(start, 1), round(end, 1))
    if key in _silences_cache:
        return _silences_cache[key]
    sil: list = []
    src = OUTPUT_ROOT / video_id / "source.mp4"
    if src.exists():
        try:
            from src.render import _detect_silences
            sil = _detect_silences(src, start, end, -32.0, 0.3)
        except Exception:  # noqa: BLE001 - 무음 감지 실패 시 보정 없이 진행
            sil = []
    if len(_silences_cache) >= _SILENCES_CACHE_MAX:
        _silences_cache.pop(next(iter(_silences_cache)), None)  # 가장 오래된 것부터 버린다
    _silences_cache[key] = sil
    return sil


def _voice_corrected_words(words_abs: list, video_id: str, clip_start: float, clip_end: float) -> list:
    """단어 시각에 렌더와 동일한 무음 보정(_snap_word_starts_to_voice)을 적용한다.

    처음 렌더(자동 경로)는 이 보정을 거친 시각으로 자막을 굽는다 — 편집기 초안도 같은
    보정을 거쳐야 "처음 영상은 싱크가 맞는데 편집만 하면 무너진다"가 안 생긴다(쉼 직후
    단어는 whisper가 침묵을 흡수해 1~2초 이르게 찍히는데, 초안이 그 생시각을 보여주면
    저장하는 순간 그 이른 시각이 그대로 구워져 싱크가 무너졌다)."""
    from src.captions import _snap_word_starts_to_voice
    from src.transcribe import Word

    sil = _clip_voice_silences(video_id, clip_start, clip_end)
    if not sil:
        return words_abs
    rel = [Word(start=w.start - clip_start, end=w.end - clip_start, text=w.text) for w in words_abs]
    rel = _snap_word_starts_to_voice(rel, sil)
    return [Word(start=w.start + clip_start, end=w.end + clip_start, text=w.text) for w in rel]


def _is_upload_praise(video_id: str, clip) -> bool:
    """직접 업로드한 찬양 클립인가 — 렌더가 원본 16:9 풀프레임+제목 없음으로 가는 경로.
    미리보기(_compute_layout full_frame)도 이 판정과 반드시 같이 움직여야 한다."""
    return video_id.startswith("upload_") and getattr(clip, "clip_type", "") == "praise"


def _compute_layout(
    cfg: dict, clip, source_resolution: tuple[int, int], full_frame: bool = False,
) -> dict:
    """위치 편집 화면에 필요한 좌표들을 render.py/captions.py와 동일한 공식으로 계산한다.
    이 값이 실제 렌더링(render_clip)과 어긋나면 편집 화면에서 본 위치와 실제 결과물의
    위치가 달라지므로, 반드시 같은 헬퍼 함수(_compute_card_video_box_*, compute_card_margins)
    를 재사용한다.

    full_frame=True(업로드 찬양): 실제 렌더가 원본 16:9 그대로 + 제목 없음 + 자막만
    오버레이하므로, 미리보기도 세로 카드가 아니라 소스 비율 캔버스 전체 = 영상으로 그린다
    (사용자 신고: "미리보기는 왜 설교영상처럼 나오냐"). 해상도 축소 규칙(1920 상한·짝수)은
    render_selected의 업로드 찬양 분기와 동일하게 맞춘다."""
    from src.captions import _fit_title_font_size, compute_card_margins
    from src.fonts import ass_size_coeff
    from src.render import _compute_card_video_box_height, _compute_card_video_box_y

    render_cfg = cfg["render"]
    captions_cfg = cfg["captions"]

    if full_frame:
        src_w, src_h = source_resolution
        if src_w > 1920:
            src_h = int(src_h * (1920 / src_w))
            src_w = 1920
        res_w, res_h = src_w - src_w % 2, src_h - src_h % 2
        caption_size = int(captions_cfg.get("font_size", 72) * 1.28)  # 렌더의 업로드 찬양 확대와 동일(1.28)
        scale = PREVIEW_CANVAS_WIDTH / res_w
        caption_font_name = getattr(clip, "caption_font", "") or captions_cfg.get("font_family", "")
        # 자막은 하단 기준(ASS Alignment 2)이다 — 크기를 키우면 글자가 '위로' 자란다.
        # 미리보기도 같은 기준(bottom)으로 그려야 크기를 바꿔도 위치가 안 어긋난다.
        # cap_y는 captions._y_position과 같은 식(-40 포함)이어야 렌더와 일치한다.
        cap_y = int(res_h * (1 - captions_cfg.get("safe_area_bottom_pct", 0.2)) - 40)
        return {
            "resolution": (res_w, res_h),
            "scale": scale,
            "canvas_w": PREVIEW_CANVAS_WIDTH,
            "canvas_h": round(res_h * scale),
            "video_box": {"x": 0, "y": 0, "w": res_w, "h": res_h, "r": 0},
            "no_title": True,   # 실제 렌더에 제목 오버레이가 없다 — 팝업도 제목을 숨긴다
            "title_base_margin_v": 0,
            "caption_base_margin_v": max(0, cap_y - int(caption_size * 1.3)),
            # 렌더 기준: 자막 블록의 '아래 끝'이 화면 바닥에서 이만큼 떨어진다(offset_y만큼 내려감).
            "caption_anchor": "bottom",
            "caption_bottom_px": max(0, res_h - cap_y),
            "title_size": 0,
            "caption_font_size": caption_size,
            "caption_en_font_size": max(16, int(caption_size * 0.60)),
            "title_ass_coeff": 1,
            "caption_ass_coeff": ass_size_coeff(caption_font_name),
        }

    resolution = tuple(render_cfg.get("resolution", [1080, 1920]))
    card = render_cfg["card_layout"]

    vbw = card["video_box_width"]
    vbh = _compute_card_video_box_height(card, source_resolution)
    vby = _compute_card_video_box_y(card, resolution, vbh)
    vbx = (resolution[0] - vbw) // 2
    card_layout = {**card, "video_box_height": vbh, "video_box_y": vby}

    # build_ass()와 정확히 같은 공식(같은 폰트·같은 상한)을 써야 편집 화면 크기가 실제
    # 렌더 크기와 일치한다. clip.title_size/title_font 오버라이드를 여기서도 반영한다
    # (예전엔 config 기본값만 써서, 사용자가 크기를 바꾸면 미리보기가 그걸 무시했다).
    title_font_name = getattr(clip, "title_font", "") or captions_cfg.get("title_font_family") or captions_cfg["font_family"]
    max_title_size = (
        getattr(clip, "title_size", 0) or captions_cfg.get("title_font_size") or int(captions_cfg["font_size"] * 1.3)
    )
    _title_top = max(24, int(resolution[1] * 0.06))
    title_avail_h = max(80, int(vby) - _title_top - 20)
    title_size = _fit_title_font_size(
        clip.title or "", max_title_size, min_size=captions_cfg["font_size"],
        available_width_px=resolution[0] - 80, font_family=title_font_name,
        available_height_px=title_avail_h,
    )
    base_title_margin_v, base_caption_margin_v = compute_card_margins(card_layout, resolution)

    caption_font_name = getattr(clip, "caption_font", "") or captions_cfg.get("font_family", "")
    scale = PREVIEW_CANVAS_WIDTH / resolution[0]
    return {
        "resolution": resolution,
        "scale": scale,
        "canvas_w": PREVIEW_CANVAS_WIDTH,
        "canvas_h": round(resolution[1] * scale),
        "video_box": {"x": vbx, "y": vby, "w": vbw, "h": vbh, "r": card.get("corner_radius", 36)},
        "title_base_margin_v": base_title_margin_v,
        "caption_base_margin_v": base_caption_margin_v,
        "title_size": title_size,
        "caption_font_size": captions_cfg["font_size"],
        "caption_en_font_size": max(16, int(captions_cfg["font_size"] * 0.60)),
        # 카드형 자막은 영상 박스 아래(위 기준, ASS Alignment 8)에 붙는다.
        "caption_anchor": "top",
        "caption_bottom_px": 0,
        # libass는 Fontsize를 셀 높이로 해석해 같은 숫자라도 브라우저보다 작게 그린다
        # (fonts.ass_size_coeff 주석 참고). 미리보기 CSS px = ASS 크기 × 이 계수 × scale.
        "title_ass_coeff": ass_size_coeff(title_font_name),
        "caption_ass_coeff": ass_size_coeff(caption_font_name),
    }


def _split_long_caption_lines(video_id: str, clip, cfg: dict, lines: list[dict]) -> list[dict]:
    """자막 목록에서 '남들보다 유난히 긴 줄'만 앞뒤 조각으로 쪼갠다(데이터 단계).

    왜 렌더가 아니라 여기서 하나: 쪼갠 조각이 타임라인에 별도 자막 칸으로 보여야 사용자가
    각각 고칠 수 있다(실신고 2026-09-06 "쪼갰으면 스튜디오에서도 2칸으로 나와야 수정하지").
    렌더는 저장된 자막을 그대로 굽는다(WYSIWYG).
    """
    from src.captions import caption_usable_width_px, split_outlier_lines
    from src.render import _probe_display_resolution

    if not lines:
        return lines
    try:
        res = _probe_display_resolution(OUTPUT_ROOT / video_id / "source.mp4")
        full_frame = _is_upload_praise(video_id, clip)
        layout = _compute_layout(cfg, clip, res, full_frame=full_frame)
        width = int((layout.get("resolution") or [1080, 1920])[0])
        size = int(
            getattr(clip, "caption_size", 0)
            or layout.get("caption_font_size")
            or cfg["captions"]["font_size"]
        )
        font = clip.caption_font or cfg["captions"]["font_family"]
        usable = caption_usable_width_px(width, card_layout=not full_frame)
        return split_outlier_lines(lines, font, size, usable)
    except Exception as e:  # noqa: BLE001
        print(f"[caption-split] 건너뜀({e})")
        return lines


def _tighten_caption_lines(lines: list[dict], clip, hold_max: float = 6.0):
    """'싱크 맞추기' 결과를 빈 칸 없이 딱딱 붙게 다듬는다.

    (2026-09-21 신고: "싱크 맞추기 하면 빈 칸도 자동으로 없애버리고 시간이 딱딱 정확하게
    박혀야 할 거 아니야" — 실제로 저장된 자막을 재보니 한 클립에 0.25초 넘는 빈 칸이 16개,
    최대 2.8초까지 있었다. 싱크는 단어 시각에 줄을 맞출 뿐 그 사이 '말 쉬는 틈'은 그대로
    비워 뒀다.)

    - 글자가 빈 줄은 삭제(화면엔 안 보이면서 타임라인에 빈 칸만 만든다).
    - 겹치는 줄은 앞 줄을 잘라 겹침 제거.
    - 줄 사이 빈틈은 '앞 줄 끝'을 다음 줄 시작까지 늘려 메운다(최대 hold_max초).
      뒤 줄을 앞당기지 않는 이유: 말하기 전에 자막이 먼저 뜨는 '선행 싱크'는 이미 고친
      과거 신고라 절대 다시 만들지 않는다. 앞 줄을 늘리는 쪽은 싱크를 해치지 않는다.
    - 클립 맨 앞 빈틈은 1.5초 이내일 때만 첫 줄을 당겨 메우고(긴 무음에 글자를 미리
      띄우지 않으려고), 맨 뒤는 마지막 줄을 클립 끝까지(최대 hold_max초) 늘린다.

    반환: (다듬은 줄, 삭제한 빈 줄 수, 메운 빈 칸 수)
    """
    cs, ce = float(clip.start), float(clip.end)
    out: list[dict] = []
    dropped = 0
    for ln in lines:
        text = str(ln.get("text", "")).strip()
        if not text:
            dropped += 1
            continue
        s = max(cs, min(ce, float(ln.get("start", cs))))
        e = max(cs, min(ce, float(ln.get("end", s))))
        if e <= s + 0.05:
            e = min(ce, s + 0.3)
        item = dict(ln)
        item["text"], item["start"], item["end"] = text, s, e
        out.append(item)
    if not out:
        return [], dropped, 0
    out.sort(key=lambda x: (x["start"], x["end"]))
    filled = 0
    for k in range(len(out) - 1):
        gap = out[k + 1]["start"] - out[k]["end"]
        if gap < 0:
            # 겹침 제거: 앞 줄을 다음 줄 시작에 딱 붙여 끊는다(두 줄이 같이 뜨는 것 방지).
            nxt = out[k + 1]["start"]
            out[k]["end"] = nxt if nxt > out[k]["start"] + 0.05 else out[k]["start"] + 0.05
        elif gap > 0.05:
            out[k]["end"] = min(out[k + 1]["start"], out[k]["end"] + hold_max)
            if gap > 0.25:
                filled += 1
    head = out[0]["start"] - cs
    if 0.05 < head <= 1.5:
        out[0]["start"] = cs
        if head > 0.25:
            filled += 1
    tail = ce - out[-1]["end"]
    if tail > 0.05:
        out[-1]["end"] = min(ce, out[-1]["end"] + hold_max)
        if tail > 0.25:
            filled += 1
    for ln in out:
        ln["start"] = round(ln["start"], 2)
        ln["end"] = round(ln["end"], 2)
    return out, dropped, filled


def _segments_for_clip(video_id: str, clip, cfg: dict, transcript_path: Path) -> list:
    """이 클립의 자막을 만들 전사 세그먼트 — 실제 렌더가 쓰는 것과 같은 소스를 고른다.
    정밀 재전사 캐시가 있고 '구멍'이 없으면 그것, 아니면 유튜브 자동자막(transcript.json).
    편집기 초안(_caption_lines_for_clip)과 실제결과 미리보기(clip_truth_frame)가 공유한다."""
    from src.main import (
        _apply_corrections, _build_clip_hotwords, _precise_cache_find, _precise_worst_hole,
        json_load_transcript,
    )
    segs = None
    try:
        w = cfg["whisper"]
        tdata = json_load_transcript(transcript_path)
        base_text_all = " ".join((s.text or "") for s in tdata["segments"])
        hotwords = _build_clip_hotwords(clip.keywords, w.get("bible_hotwords", ""), base_text_all)
        sig = hashlib.md5(
            f"{w.get('initial_prompt', '')}|{hotwords or ''}".encode("utf-8")
        ).hexdigest()[:8]
        model = w.get("precise_model_size", w["model_size"])
        segs = _precise_cache_find(
            OUTPUT_ROOT / video_id / "precise_cache", model, sig, clip.start, clip.end + 4.0
        )
        # 렌더와 같은 '구멍' 검사: 앞/중간이 통째로 빈 불량 캐시(VAD/배치 사고)를 신뢰하면
        # 초안 자체가 십수 초 어긋난다(실측: 19초 구멍 캐시 → 초안 전체 밀림).
        if segs is not None and _precise_worst_hole(
            tdata["segments"], segs, clip.start, clip.end
        ) >= 5.0:
            segs = None
    except Exception:  # noqa: BLE001 - 캐시 조회 실패는 조용히 자동자막 폴백
        segs = None
    if segs is None:
        segs = json_load_transcript(transcript_path)["segments"]
    # 오탈자 교정도 렌더와 동일하게 적용(예전엔 편집기 초안에만 미적용 → 저장 시 오탈자 굳음).
    _apply_corrections(segs, cfg.get("captions", {}).get("corrections") or {})
    return segs


def _caption_lines_for_clip(video_id: str, clip, cfg: dict) -> list[dict]:
    """자막 편집기에 채워 넣을 자막 라인 목록을 만든다.
    이미 편집·저장된 caption_overrides가 있으면 그걸 쓰고, 없으면 원본 전사에서 클립
    구간 단어를 뽑아 max_words_per_line 단위로 잘라 라인({start,end,text})으로 만든다."""
    from src.captions import (
        _clean_word_text, _collect_words_in_range, _display_text, chunk_words_into_lines,
    )

    # 유튜브 실황의 찬양 클립은 가사 자막을 넣지 않는다(화면에 교회 가사 슬라이드가 이미
    # 있음) — 편집기에도 초안을 채우지 않는다. 단, 직접 찍어 업로드한 영상(upload_*)은
    # 가사 표시가 없어 whisper 자막을 넣으므로 초안도 같은 소스로 채운다(아래 일반 경로).
    if getattr(clip, "clip_type", "") == "praise" and not video_id.startswith("upload_"):
        return []

    def _strip_trailing_dots(text: str) -> str:
        # 실제 렌더(_karaoke_text)는 단어별로 끝 마침표를 뗀다. 편집기 미리보기도 같은
        # 규칙을 적용해야 "화면엔 있는데 실제 영상엔 없는" 불일치가 안 생긴다.
        # 비언어 표기([한숨]·[웃음]…)도 렌더가 걷어내므로 여기서도 같이 걷어낸다 —
        # 이미 저장된 caption_overrides에 박혀 있는 경우까지 덮는다(2026-09-23).
        cleaned = _clean_word_text(text)
        return " ".join(_display_text(w) for w in cleaned.split())

    if getattr(clip, "caption_overrides", None):
        # 시간순 정렬해서 보여준다 — 저장 순서가 어긋나 있으면(과거 자동자막 초안 오염 등)
        # 편집기에 자막이 뒤죽박죽 순서로 떠 "순서가 뒤바뀐다"는 혼란을 준다.
        rows = [
            {
                "start": float(o["start"]), "end": float(o["end"]),
                "text": _strip_trailing_dots(str(o.get("text", ""))),
            }
            for o in clip.caption_overrides
        ]
        # 통째로 비언어 표기였던 줄은 빈 줄이 되므로 목록에서 뺀다(빈 자막 칸이 남지 않게).
        rows = [r for r in rows if r["text"].strip()]
        return _split_long_caption_lines(
            video_id, clip, cfg, sorted(rows, key=lambda r: r["start"]),
        )

    transcript_path = OUTPUT_ROOT / video_id / "transcript.json"
    if not transcript_path.exists():
        return []
    from src.main import (
        _apply_corrections, _build_clip_hotwords, _precise_cache_find, _precise_worst_hole,
        json_load_transcript,
    )

    # ── 구조적 핵심(2026-09-03 "편집할수록 자막이 무너진다" 근본 수정) ──
    # 편집기 초안은 "실제 렌더가 구울 것과 동일한 소스"에서 만들어야 한다. 예전엔 무조건
    # 유튜브 자동자막(부정확)으로 초안을 만들어서, 사용자가 2~3줄만 고쳐 저장해도 나머지
    # 수십 줄이 전부 '정확한 정밀 자막 → 부정확한 자동자막 초안'으로 통째로 바뀌어 굳었다.
    # 이제 정밀 재전사 캐시가 있으면 그걸 초안 소스로 쓴다(렌더와 같은 결과) — 한 줄만
    # 고치면 정말 그 한 줄만 달라진다. 캐시가 없으면 예전처럼 자동자막 폴백.
    segs = _segments_for_clip(video_id, clip, cfg, transcript_path)
    # 필러 제거 설정도 렌더와 동일하게(예전엔 기본값이라 렌더가 지우는 '그/막/뭐'가 초안에 남았다).
    words = _collect_words_in_range(
        segs, clip.start, clip.end,
        strip_filler=cfg["captions"].get("strip_filler", True),
        aggressive_filler=cfg["captions"].get("aggressive_filler", False),
    )
    # 처음 렌더(자동 경로)와 '정확히 같은 시각'을 보여준다: 무음 보정 + 전역 오프셋까지.
    # 이 두 보정이 초안에 빠져 있으면, 텍스트만 고쳐 저장해도 시간축 전체가 (보정 안 된)
    # 다른 기준으로 바뀌어 "편집하는 순간 싱크가 무너지는" 사후 회귀가 났다(실신고).
    words = _voice_corrected_words(words, video_id, clip.start, clip.end)
    sync_off = float(cfg["captions"].get("sync_offset_sec", 0.0) or 0.0)
    max_wpl = cfg["captions"].get("max_words_per_line", 4)
    # 렌더(build_ass)와 같은 '화면 1줄 폭' 규칙으로 잘라, 편집기에서 본 줄이 실제 자막과 일치하게.
    res_w = (cfg.get("render", {}).get("resolution") or [1080, 1920])[0]
    max_units = max(4.0, (res_w - 104) / max(1, cfg["captions"].get("font_size", 72)))
    lines = chunk_words_into_lines(words, max_wpl, max_units=max_units)
    out = [
        {
            "start": ln.start + sync_off, "end": ln.end + sync_off,
            "text": " ".join(_display_text(w.text) for w in ln.words),
        }
        for ln in lines
    ]
    # 각 줄 끝이 다음 줄 시작을 넘지 않게 잘라 자막이 겹쳐 뜨는 걸 막는다(싱크 안정).
    for i in range(len(out) - 1):
        if out[i]["end"] > out[i + 1]["start"]:
            out[i]["end"] = max(out[i]["start"] + 0.3, out[i + 1]["start"] - 0.02)
    return out


def _preview_caption_text(video_id: str, clip) -> str:
    """편집 화면에 보여줄 자막 미리보기 텍스트. 실제 카라오케 자막 로직을 그대로 쓰지 않고,
    클립 시작 지점과 겹치는 전사 세그먼트의 앞부분만 대충 가져온다 (위치 감을 잡는 용도)."""
    transcript_path = OUTPUT_ROOT / video_id / "transcript.json"
    if not transcript_path.exists():
        return "여기에 자막이 표시됩니다"
    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        for seg in data.get("segments", []):
            if seg["end"] > clip.start:
                words = seg["text"].strip().split()
                text = " ".join(words[:4])
                return text or "여기에 자막이 표시됩니다"
    except (OSError, ValueError, KeyError):
        pass
    return "여기에 자막이 표시됩니다"


@app.route("/video/<video_id>/clip/<int:idx>/studio")
def clip_studio(video_id: str, idx: int):
    """프리미어식 스튜디오: 미리보기 + 속성 패널(자막 크기/위치) + 타임라인 트랙(자막 블록
    드래그·리사이즈·더블클릭 편집). 데이터/저장/AI 도구는 기존 엔드포인트를 재사용한다."""
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return "해당 영상 작업을 찾을 수 없습니다.", 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return "잘못된 클립 번호입니다.", 404
    # no-store: 스튜디오를 고쳐도 브라우저가 옛 페이지를 캐시하면 "고쳤다는데 그대로"가
    # 된다(preview-modal.js와 같은 이유 — 실제로 겪은 유형의 사고).
    resp = Response(render_template_string(STUDIO_TEMPLATE, video_id=video_id, idx=idx))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/video/<video_id>/clip/<int:idx>/edit")
def clip_edit(video_id: str, idx: int):
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return "해당 영상 작업을 찾을 수 없습니다.", 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return "잘못된 클립 번호입니다.", 404
    clip = clips[idx]

    from src.render import _probe_resolution

    cfg = _load_config()
    # 회전 메타데이터 반영(세로 촬영 업로드가 가로로 계산되던 실사고 — 렌더와 동일 함수).
    from src.render import _probe_display_resolution

    source_res = _probe_display_resolution(OUTPUT_ROOT / video_id / "source.mp4")
    layout = _compute_layout(
        cfg, clip, source_res,
        full_frame=_is_upload_praise(video_id, clip),
    )
    caption_preview = _preview_caption_text(video_id, clip)
    caption_lines = _caption_lines_for_clip(video_id, clip, cfg)

    from src.fonts import get_font_registry

    fonts = get_font_registry()
    defaults = {
        "title_font": cfg["captions"].get("title_font_family", ""),
        "title_size": cfg["captions"].get("title_font_size", 160),
        "caption_font": cfg["captions"].get("font_family", ""),
        "caption_size": cfg["captions"].get("font_size", 72),
        "fill_mode": cfg["render"]["card_layout"].get("fill_mode", "fit"),
    }

    return render_template_string(
        EDIT_TEMPLATE, video_id=video_id, idx=idx, clip=clip, layout=layout,
        caption_preview=caption_preview, caption_lines=caption_lines,
        fonts=fonts, defaults=defaults,
    )


@app.route("/video/<video_id>/clip/<int:idx>/duplicate", methods=["POST"])
def duplicate_clip(video_id: str, idx: int):
    """후보 하나를 통째로 복제해 목록 끝에 새 후보로 추가한다(원본은 그대로).

    자막·제목·배속·위치 등 모든 필드를 그대로 복사해, 같은 장면을 다른 설정(배속·자막
    스타일 등)으로 한 번 더 만들어보고 싶을 때 처음부터 다시 잡을 필요가 없게 한다
    (사용자 요청 2026-09-05 "하이라이트 후보 섹션 복제 가능하게"). 렌더 결과물(mp4)은
    복제 안 됨 — 서명(render_signature)이 원본과 같은 시간대라도 새 인덱스라 '미렌더'로
    시작해, 복제본에서 설정을 바꾸고 다시 만들면 그 설정으로 새로 렌더된다."""
    import copy

    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    with CLIPS_LOCK:
        clips = load_clips_json(clips_path)
        if idx < 0 or idx >= len(clips):
            return jsonify({"error": "잘못된 클립 번호"}), 400
        dup = copy.deepcopy(clips[idx])
        dup.title = (dup.title or "클립") + " (복제)"
        # 편집창을 처음 열 때 잘라낸 구간이 아니라 원본 영상 전체가 보이게 하는 일회성
        # 힌트(사용자 요청: "복제될 때는 원본 영상 전체가 편집창에서 보이게").
        dup.show_full_source_once = True
        clips.append(dup)
        new_idx = len(clips) - 1
        save_clips_json(clips, clips_path)
    return jsonify({"ok": True, "new_idx": new_idx})


@app.route("/video/<video_id>/clip/<int:idx>/position", methods=["POST"])
def save_clip_position(video_id: str, idx: int):
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    # 읽기→수정→저장 전 구간을 락으로 보호한다. 렌더 종료(main.render_selected의 병합
    # 저장)나 다른 편집 저장과 겹치면 마지막 저장이 상대 수정을 덮어쓰기 때문.
    with CLIPS_LOCK:
        return _save_clip_position_locked(clips_path, idx)


def _save_clip_position_locked(clips_path: Path, idx: int):
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "invalid index"}), 400

    body = request.get_json() or {}
    clip = clips[idx]
    if "title" in body and str(body["title"]).strip():
        clip.title = str(body["title"]).strip()
    # 영상 구간(길이 자르기). 사용자가 명시하면 그대로 존중한다(렌더 시 자동 확장/스냅 안 함).
    if body.get("clip_start") is not None and body.get("clip_end") is not None:
        try:
            s = max(0.0, float(body["clip_start"]))  # 시작은 0(영상 맨 앞) 밑으로 못 내림
            e = float(body["clip_end"])
        except (TypeError, ValueError):
            s = e = 0.0
        if e - s >= 1.0:  # 최소 1초
            clip.start, clip.end = s, e
            clip.trimmed = True
    # 확인 팝업의 분할·삭제 결과(남길 구간 절대초). 렌더가 이 구간들만 이어붙인다.
    if "keep_ranges" in body:
        kr = []
        for r in (body.get("keep_ranges") or []):
            try:
                a, b = float(r[0]), float(r[1])
            except (TypeError, ValueError, IndexError):
                continue
            if b - a > 0.1:
                kr.append([a, b])
        # 한 조각이 start~end 전체와 같으면 굳이 저장하지 않는다(전체 사용 = 기본).
        if len(kr) == 1 and abs(kr[0][0] - clip.start) < 0.05 and abs(kr[0][1] - clip.end) < 0.05:
            kr = []
        clip.keep_ranges = kr
    clip.title_offset_x = float(body.get("title_offset_x", clip.title_offset_x))
    clip.title_offset_y = float(body.get("title_offset_y", clip.title_offset_y))
    # 편집기의 숫자 입력칸을 비우면 JSON에 null이 실려 온다. 키는 있고 값이 null이라
    # 기본값 인자가 안 먹으므로(float(None) → TypeError → 저장 전체가 500) 따로 막는다.
    for _k in ("caption_offset_x", "caption_offset_y"):
        if body.get(_k) is not None:
            try:
                setattr(clip, _k, float(body[_k]))
            except (TypeError, ValueError):
                pass
    # 자막 편집기에서 확정한 라인들(텍스트가 남아있는 것만). 저장되면 다음 렌더는 재전사 없이
    # 이 자막을 그대로 쓴다. 넘어오지 않으면(위치만 저장) 기존 caption_overrides를 유지한다.
    if "captions" in body:
        # 시간순으로 정렬해 저장한다. 편집기에서 시간을 고치거나 줄을 끼워 넣어 행 순서가
        # 시간순과 어긋나도, 렌더/편집기 어디서든 항상 시간순으로 일관되게 다뤄지도록.
        _new_caps = sorted(
            (
                {"start": float(c["start"]), "end": float(c["end"]), "text": str(c.get("text", "")).strip()}
                for c in body["captions"]
                if str(c.get("text", "")).strip()
            ),
            key=lambda o: o["start"],
        )
        # 데이터 소실 방지(찬양): 찬양 클립은 재전사 폴백이 없어, 빈 captions로 덮어쓰면
        # 자막이 영구 소실되고 렌더에 아무 자막도 안 나간다(실사고 2026-09-05 — 팝업의
        # '구간 변경 시 자막 비우기'가 찬양에도 적용돼 저장 자막이 증발). 찬양에서 비어있는
        # captions는 무시하고 기존 자막을 지킨다(정말 지우려면 편집기에서 줄을 지우는 게
        # 아니라 텍스트를 남겨야 하는 구조라, 전량 삭제 의도는 사실상 없다).
        if not _new_caps and getattr(clip, "clip_type", "") == "praise" and clip.caption_overrides:
            print(f"[save] praise 클립 {idx}: 빈 captions 저장 무시(기존 {len(clip.caption_overrides)}줄 유지)")
        else:
            clip.caption_overrides = _new_caps
    # 재생 배속(1.0~2.0, 팝업에서 선택). 렌더가 완성본에 후처리로 적용한다.
    if "playback_speed" in body:
        try:
            _spd = float(body.get("playback_speed") or 1.0)
        except (TypeError, ValueError):
            _spd = 1.0
        clip.playback_speed = min(2.0, max(1.0, _spd))
    # 업로드 찬양 클립의 카라오케 자막 토글(사용자가 "싱크 맞추기"를 눌러 확인한 경우만 켬).
    if "caption_karaoke" in body:
        clip.caption_karaoke = bool(body.get("caption_karaoke"))
    # AI 자막 교정이 뽑은 형광 강조어(caption_highlights). 렌더가 이 단어를 자막에서 강조한다.
    if "caption_highlights" in body:
        clip.caption_highlights = [
            str(h).strip() for h in (body.get("caption_highlights") or []) if str(h).strip()
        ]
    # 자유 텍스트 트랙(스튜디오 '+' 로 만든 텍스트들). 자막과 달리 겹쳐도 되고
    # 요소마다 화면 위치·크기를 갖는다. 렌더는 ASS에 \pos로 따로 얹는다.
    if "free_texts" in body:
        _ft = []
        for t in (body.get("free_texts") or []):
            try:
                a, b = float(t["start"]), float(t["end"])
            except (TypeError, ValueError, KeyError):
                continue
            txt = str(t.get("text", "")).strip()
            if not txt or b - a <= 0.05:
                continue
            _ft.append({
                "start": a, "end": b, "text": txt,
                "x": float(t.get("x", 0) or 0), "y": float(t.get("y", 0) or 0),
                "size": int(float(t.get("size", 0) or 0)),
                "track": max(0, int(float(t.get("track", 0) or 0))),
            })
        clip.free_texts = sorted(_ft, key=lambda o: (o["track"], o["start"]))
    # 영어 자막 트랙(번역). 한국어는 그대로 두고 별도 저장 → 렌더 옵션으로 전환.
    if "caption_overrides_en" in body:
        clip.caption_overrides_en = sorted(
            (
                {"start": float(c["start"]), "end": float(c["end"]), "text": str(c.get("text", "")).strip()}
                for c in (body.get("caption_overrides_en") or [])
                if str(c.get("text", "")).strip()
            ),
            key=lambda o: o["start"],
        )
    # 화면모드 + 제목/자막 글꼴 스타일(편집기에서 선택). 빈 값이면 config 기본값 사용.
    if "fill_mode" in body:
        clip.fill_mode = str(body.get("fill_mode", "") or "")
    for k in ("title_font", "title_align", "caption_font", "caption_align"):
        if k in body:
            setattr(clip, k, str(body.get(k, "") or ""))
    for k in ("title_size", "caption_size", "caption_size_en"):
        if k in body:
            setattr(clip, k, int(float(body.get(k) or 0)))
    for k in ("title_spacing", "caption_spacing"):
        if k in body:
            setattr(clip, k, float(body.get(k) or 0))
    # 자막 스타일 프리셋(캡컷식 박스·색상, 2026-09-08 요청). caption_text_color/box_color는
    # CSS #RRGGBB 문자열 그대로 저장하고, ASS 변환(BGR·알파 반전)은 렌더 시점에 한다.
    for k in ("caption_text_color", "caption_box_color"):
        if k in body:
            setattr(clip, k, str(body.get(k, "") or ""))
    if "caption_box" in body:
        clip.caption_box = bool(body.get("caption_box"))
    if "caption_box_opacity" in body:
        try:
            clip.caption_box_opacity = max(0.0, min(1.0, float(body.get("caption_box_opacity"))))
        except (TypeError, ValueError):
            pass
    if "caption_box_radius" in body:
        try:
            clip.caption_box_radius = max(0.0, min(100.0, float(body.get("caption_box_radius"))))
        except (TypeError, ValueError):
            pass
    for k in ("caption_box_width_pct", "caption_box_height_pct"):
        if k in body:
            try:
                setattr(clip, k, max(0.0, min(100.0, float(body.get(k)))))
            except (TypeError, ValueError):
                pass
    for k in ("caption_box_offset_x", "caption_box_offset_y"):
        if k in body:
            try:
                setattr(clip, k, float(body.get(k)))
            except (TypeError, ValueError):
                pass
    # 캡컷 '텍스트' 패널의 패턴(B/U/I)·불투명도·획·글로우(2026-09-08: 스샷 4장 전부 구현 요청).
    for k in ("caption_bold", "caption_italic", "caption_underline", "caption_outline_enabled", "caption_glow"):
        if k in body:
            setattr(clip, k, bool(body.get(k)))
    if "caption_text_opacity" in body:
        try:
            clip.caption_text_opacity = max(0.0, min(1.0, float(body.get("caption_text_opacity"))))
        except (TypeError, ValueError):
            pass
    for k in ("caption_outline_color", "caption_glow_color"):
        if k in body:
            setattr(clip, k, str(body.get(k, "") or ""))
    if "caption_outline_width" in body:
        try:
            clip.caption_outline_width = float(body.get("caption_outline_width"))
        except (TypeError, ValueError):
            pass
    # 배경 음악 볼륨/음소거/시작 오프셋(파일은 별도 업로드 라우트에서 받는다).
    # 오프셋은 타임라인에서 BGM 막대를 끌어 정한, 음원 파일 안에서 재생을 시작할 지점(초)
    # — 무한 반복 위에서 자르는 창(atrim)만 옮기므로 클립 길이·다른 트랙과는 무관하다.
    if clip.bgm and ("bgm_volume" in body or "bgm_muted" in body or "bgm_offset" in body):
        if "bgm_volume" in body:
            try:
                clip.bgm["volume"] = max(0.0, min(1.0, float(body.get("bgm_volume"))))
            except (TypeError, ValueError):
                pass
        if "bgm_muted" in body:
            clip.bgm["muted"] = bool(body.get("bgm_muted"))
        if "bgm_offset" in body:
            try:
                clip.bgm["offset"] = max(0.0, float(body.get("bgm_offset")))
            except (TypeError, ValueError):
                pass
    save_clips_json(clips, clips_path)
    return jsonify({"ok": True})


@app.route("/video/<video_id>/clip/<int:idx>/bgm", methods=["POST"])
def upload_clip_bgm(video_id: str, idx: int):
    """스튜디오 A2 트랙 '배경 음악 추가'. 파일을 output/<video_id>/bgm/에 저장하고
    clip.bgm에 참조를 남긴다 — 렌더가 클립 길이에 맞춰 반복/트림해 원본 오디오와 믹싱한다
    (_add_sfx와 같은 ffmpeg amix 패턴, render._mix_bgm 참고)."""
    from werkzeug.utils import secure_filename

    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "업로드된 파일이 없습니다"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"):
        return jsonify({"error": "mp3/wav/m4a/aac/ogg/flac 파일만 지원합니다"}), 400
    with CLIPS_LOCK:
        clips = load_clips_json(clips_path)
        if idx < 0 or idx >= len(clips):
            return jsonify({"error": "invalid index"}), 400
        old = clips[idx].bgm
        bgm_dir = OUTPUT_ROOT / video_id / "bgm"
        bgm_dir.mkdir(parents=True, exist_ok=True)
        fname = f"clip{idx}_{int(time.time())}_{secure_filename(f.filename)}"
        f.save(bgm_dir / fname)
        clips[idx].bgm = {"filename": fname, "volume": 0.25, "muted": False}
        save_clips_json(clips, clips_path)
        result = clips[idx].bgm
    if old and old.get("filename"):
        try:
            (OUTPUT_ROOT / video_id / "bgm" / old["filename"]).unlink(missing_ok=True)
        except OSError:
            pass
    return jsonify({"ok": True, "bgm": result})


@app.route("/video/<video_id>/clip/<int:idx>/bgm", methods=["DELETE"])
def remove_clip_bgm(video_id: str, idx: int):
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    with CLIPS_LOCK:
        clips = load_clips_json(clips_path)
        if idx < 0 or idx >= len(clips):
            return jsonify({"error": "invalid index"}), 400
        old = clips[idx].bgm
        clips[idx].bgm = None
        save_clips_json(clips, clips_path)
    if old and old.get("filename"):
        try:
            (OUTPUT_ROOT / video_id / "bgm" / old["filename"]).unlink(missing_ok=True)
        except OSError:
            pass
    return jsonify({"ok": True})


@app.route("/font/<path:filename>")
def serve_font(filename: str):
    """편집기 드롭다운/미리보기에서 실제 글꼴로 보여주기 위해 통합 폴더의 폰트 파일을 서빙."""
    from src.fonts import CONSOLIDATED_DIR

    base = CONSOLIDATED_DIR.resolve()
    p = (base / filename).resolve()
    if base != p.parent or not p.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(p)


@app.route("/media/<video_id>/preview/<int:idx>.jpg")
def clip_preview_frame(video_id: str, idx: int):
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "invalid index"}), 404
    clip = clips[idx]

    video_dir = OUTPUT_ROOT / video_id
    # fill_mode별로 캐시를 분리한다: 편집기에서 화면모드를 바꾸면 미리보기 프레임도 다시
    # 만들어져야 하는데, 파일명이 같으면 옛 모드의 캐시가 계속 나간다.
    fill_tag = (getattr(clip, "fill_mode", "") or "cfg")
    out_path = (video_dir / "clips" / f"_preview_{idx}_{fill_tag}.jpg").resolve()
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = _load_config()
        card = cfg["render"]["card_layout"]
        source_path = video_dir / "source.mp4"

        from src.render import (
            _compute_card_video_box_height,
            _probe_resolution,
            card_source_video_filter,
        )

        # 클립별 fill_mode 오버라이드를 렌더(render_clip)와 똑같이 반영한다.
        clip_fill = getattr(clip, "fill_mode", "") or ""
        if clip_fill:
            card = {**card, "fill_mode": clip_fill}
        src_res = _probe_resolution(source_path)
        vbh = _compute_card_video_box_height(card, src_res)
        vbw = card["video_box_width"]
        ts = clip.start + min(1.0, max(0.0, (clip.end - clip.start) / 2))
        # crop/scale 공식은 실제 렌더와 같은 함수(card_source_video_filter)를 쓴다 —
        # 문자열을 복제하면 기본값 하나만 어긋나도 미리보기 ≠ 결과물이 된다(실제 잠복 사례).
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(ts), "-i", str(source_path),
            "-frames:v", "1",
            "-vf", card_source_video_filter(card, vbw, vbh),
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out_path.exists():
            return jsonify({"error": "미리보기 프레임 생성 실패"}), 500

    return send_file(out_path)


@app.route("/media/<video_id>/truth/<int:idx>.jpg")
def clip_truth_frame(video_id: str, idx: int):
    """'실제 결과' 미리보기: 절대시각 t의 화면을 실제 렌더와 **같은 ASS·같은 필터**로 그린다.
    CSS 미리보기(_compute_layout)는 근사치라 결과물과 어긋날 수 있다 — 이건 render_clip이
    쓰는 함수(render.build_clip_ass 등)를 그대로 호출하므로 정의상 결과물과 같다.
    캐시하지 않는다(자막·스타일을 고칠 때마다 달라져야 하므로). 카드/블러 레이아웃만 지원."""
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "invalid index"}), 404
    clip = clips[idx]
    if _is_upload_praise(video_id, clip):
        return jsonify({"error": "업로드 찬양(원본 비율) 클립은 아직 지원하지 않습니다"}), 400
    try:
        t = float(request.args.get("t", clip.start))
    except ValueError:
        return jsonify({"error": "bad t"}), 400
    t = max(clip.start, min(clip.end - 0.05, t))
    video_dir = OUTPUT_ROOT / video_id
    source = video_dir / "source.mp4"
    if not source.exists():
        return jsonify({"error": "원본 영상이 없습니다"}), 404
    cfg = _load_config()
    segs = _segments_for_clip(video_id, clip, cfg, video_dir / "transcript.json")
    out = (video_dir / "clips" / f"_truth_{idx}.jpg").resolve()
    from src.render import render_truth_frame
    try:
        render_truth_frame(source, segs, clip, cfg["render"], cfg["captions"], t, out)
    except Exception as e:  # noqa: BLE001 - 실패 사유를 팝업에 그대로 보여준다
        traceback.print_exc()
        return jsonify({"error": str(e)[:600]}), 500
    resp = send_file(out, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/media/<video_id>/source.mp4")
def serve_source_video(video_id: str):
    """'만들기 전 확인' 팝업의 <video>용 원본 서빙. conditional=True로 HTTP Range를 지원해
    아이폰식 트림 핸들을 끌 때 브라우저가 필요한 구간만 받아 즉시 탐색된다."""
    p = (OUTPUT_ROOT / video_id / "source.mp4").resolve()
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(p, conditional=True)


@app.route("/media/<video_id>/bgm/<path:filename>")
def serve_bgm(video_id: str, filename: str):
    """스튜디오 미리듣기용 배경 음악 서빙(업로드한 파일 그대로)."""
    base = (OUTPUT_ROOT / video_id / "bgm").resolve()
    p = (base / filename).resolve()
    if base != p.parent or not p.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(p, conditional=True)


@app.route("/media/<video_id>/thumb/<int:sec>.jpg")
def clip_trim_thumb(video_id: str, sec: int):
    """트림 필름스트립용 소형 프레임(가로 160px, 초 단위). 한 번 만들면 캐시로 재사용."""
    video_dir = OUTPUT_ROOT / video_id
    source = video_dir / "source.mp4"
    if not source.exists():
        return jsonify({"error": "not found"}), 404
    # send_file은 상대경로를 앱 루트(src/) 기준으로 해석하므로 반드시 절대경로로 만든다.
    out = (video_dir / "clips" / "_thumbs" / f"{sec}.jpg").resolve()
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(sec), "-i", str(source),
            "-frames:v", "1", "-vf", "scale=160:-2", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            return jsonify({"error": "thumb 실패"}), 500
    return send_file(out)


@app.route("/media/<video_id>/peaks.json")
def audio_peaks(video_id: str):
    """스튜디오 타임라인 A1 트랙 파형용 피크 배열. 구간 [a,b]를 n칸으로 나눠 각 칸의
    최대 진폭(0~1)을 돌려준다. ffmpeg로 8kHz 모노 PCM만 뽑으므로 1시간 원본도 수 초.
    (a,b,n) 조합별로 파일 캐시 — 같은 클립을 다시 열면 즉시."""
    source = OUTPUT_ROOT / video_id / "source.mp4"
    if not source.exists():
        return jsonify({"error": "not found"}), 404
    try:
        a = max(0.0, float(request.args.get("a", 0)))
        b = float(request.args.get("b", 0))
        n = int(request.args.get("n", 400))
    except ValueError:
        return jsonify({"error": "bad args"}), 400
    n = max(50, min(8000, n))
    if b <= a:
        return jsonify({"a": a, "b": b, "peaks": []})
    cache_dir = (OUTPUT_ROOT / video_id / "clips" / "_peaks").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{a:.2f}_{b:.2f}_{n}.json"
    if out.exists():
        return send_file(out, mimetype="application/json")
    rate = 8000
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return jsonify({"error": "ffmpeg 실패"}), 500
    import array as _array
    samples = _array.array("h")
    raw = proc.stdout
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    total = len(samples)
    peaks = []
    if total:
        per = total / n
        for i in range(n):
            s0, s1 = int(i * per), max(int(i * per) + 1, int((i + 1) * per))
            seg = samples[s0:s1]
            pk = max((abs(x) for x in seg), default=0)
            peaks.append(round(pk / 32768.0, 3))
    data = {"a": a, "b": b, "peaks": peaks}
    out.write_text(json.dumps(data), encoding="utf-8")
    return jsonify(data)


@app.route("/video/<video_id>/clip/<int:idx>/preview_info")
def clip_preview_info(video_id: str, idx: int):
    """'만들기 전 확인' 팝업이 쓰는 클립·레이아웃 정보 JSON (편집 페이지와 같은 계산을 재사용)."""
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "not found"}), 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "invalid index"}), 400
    clip = clips[idx]
    # 복제 직후 힌트(원본 전체 보기)는 이 조회 한 번으로 소비한다 — 다음에 같은 클립을
    # 열면 보통 클립처럼 트림된 구간 위주로 보이게(계속 전체로 열리면 오히려 불편함).
    show_full_once = bool(getattr(clip, "show_full_source_once", False))
    if show_full_once:
        with CLIPS_LOCK:
            fresh = load_clips_json(clips_path)
            if idx < len(fresh):
                fresh[idx].show_full_source_once = False
                save_clips_json(fresh, clips_path)

    from src.fonts import get_font_registry
    from src.render import _probe_resolution

    cfg = _load_config()
    from src.render import _probe_display_resolution

    layout = _compute_layout(
        cfg, clip, _probe_display_resolution(OUTPUT_ROOT / video_id / "source.mp4"),
        full_frame=_is_upload_praise(video_id, clip),
    )

    # 소스 전체 길이(트림 확장 한계). transcript.json의 duration_sec가 가장 싸게 정확하다.
    duration = 0.0
    tp = OUTPUT_ROOT / video_id / "transcript.json"
    if tp.exists():
        try:
            duration = float(json.loads(tp.read_text(encoding="utf-8")).get("duration_sec") or 0)
        except (OSError, ValueError):
            pass
    if duration <= 0:
        duration = clip.end + 10.0

    fonts = get_font_registry()

    def font_entry(family: str):
        for f in fonts:
            if f["family"] == family:
                return {"family": f["family"], "file": f["file"]}
        return None

    # 제목 후보: 현재 제목을 맨 앞에 두고 중복 제거해 최대 5개(팝업에서 한눈에 고르게).
    cands = [clip.title] + [t for t in (clip.title_candidates or []) if t and t != clip.title]
    return jsonify({
        "layout": layout,
        "clip": {
            "title": clip.title,
            "title_candidates": cands[:5],
            "start": clip.start,
            "end": clip.end,
            "keep_ranges": (getattr(clip, "keep_ranges", None) or []),
            "title_offset_x": clip.title_offset_x,
            "title_offset_y": clip.title_offset_y,
            "title_size": getattr(clip, "title_size", 0) or 0,
            "caption_offset_x": clip.caption_offset_x,
            "caption_offset_y": clip.caption_offset_y,
            "fill_mode": (getattr(clip, "fill_mode", "") or cfg["render"]["card_layout"].get("fill_mode", "fit")),
            "caption_karaoke": bool(getattr(clip, "caption_karaoke", False)),
            "caption_highlights": (getattr(clip, "caption_highlights", None) or []),
            "free_texts": (getattr(clip, "free_texts", None) or []),
            "caption_font": getattr(clip, "caption_font", "") or "",
            "caption_align": getattr(clip, "caption_align", "") or "",
            "caption_spacing": float(getattr(clip, "caption_spacing", 0) or 0),
            "caption_text_color": getattr(clip, "caption_text_color", "") or "",
            "caption_box": bool(getattr(clip, "caption_box", False)),
            "caption_box_color": getattr(clip, "caption_box_color", "") or "#000000",
            "caption_box_opacity": float(getattr(clip, "caption_box_opacity", 0.55) or 0.55),
            "caption_box_radius": float(getattr(clip, "caption_box_radius", 40.0) or 40.0),
            "caption_box_width_pct": float(getattr(clip, "caption_box_width_pct", 28.0) or 28.0),
            "caption_box_height_pct": float(getattr(clip, "caption_box_height_pct", 28.0) or 28.0),
            "caption_box_offset_x": float(getattr(clip, "caption_box_offset_x", 0.0) or 0.0),
            "caption_box_offset_y": float(getattr(clip, "caption_box_offset_y", 0.0) or 0.0),
            "caption_bold": bool(getattr(clip, "caption_bold", True)),
            "caption_italic": bool(getattr(clip, "caption_italic", False)),
            "caption_underline": bool(getattr(clip, "caption_underline", False)),
            "caption_text_opacity": float(getattr(clip, "caption_text_opacity", 1.0) or 1.0),
            "caption_outline_enabled": bool(getattr(clip, "caption_outline_enabled", True)),
            "caption_outline_color": getattr(clip, "caption_outline_color", "") or "",
            "caption_outline_width": getattr(clip, "caption_outline_width", -1.0),
            "caption_glow": bool(getattr(clip, "caption_glow", False)),
            "caption_glow_color": getattr(clip, "caption_glow_color", "") or "",
            "caption_overrides_en": (getattr(clip, "caption_overrides_en", None) or []),
            "clip_type": getattr(clip, "clip_type", "") or "",
            "caption_size": int(getattr(clip, "caption_size", 0) or 0),
            "caption_size_en": int(getattr(clip, "caption_size_en", 0) or 0),
            "playback_speed": float(getattr(clip, "playback_speed", 1.0) or 1.0),
            "show_full_source_once": bool(getattr(clip, "show_full_source_once", False)),
            "bgm": (getattr(clip, "bgm", None) or None),
        },
        "caption_preview": _preview_caption_text(video_id, clip),
        "caption_lines": _caption_lines_for_clip(video_id, clip, cfg),
        "source_duration": duration,
        "title_font": font_entry(clip.title_font or cfg["captions"].get("title_font_family", "")),
        "caption_font": font_entry(clip.caption_font or cfg["captions"].get("font_family", "")),
        "fonts": [{"name": f["name"], "family": f["family"], "file": f["file"]} for f in fonts],
    })


# '만들기 전 확인' 팝업 스크립트. CANDIDATES_TEMPLATE는 f-string이라 중괄호를 전부 이스케이프
# 해야 해서, JS는 별도 상수로 두고 라우트로 서빙한다(브레이스 지옥 방지 + 캐시 가능).


@app.route("/js/preview-modal.js")
def preview_modal_js():
    # no-store: 캐시 헤더가 없으면 브라우저가 알아서(휴리스틱) 캐시해서, 팝업을 고쳐도
    # 사용자에겐 옛 스크립트가 계속 돌아 "고쳤다는데 그대로"가 된다(실측, 2026-09-03).
    resp = Response(PREVIEW_MODAL_JS, mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _run_reanalyze_job(video_id: str, idx: int) -> None:
    """백그라운드: 한 클립의 구간만 재분석해 새 후보를 목록에 '추가'한다(원본 유지)."""
    video_dir = OUTPUT_ROOT / video_id
    clips_path = video_dir / "clips.json"
    try:
        cfg = _load_config()
        clips = load_clips_json(clips_path)
        if idx < 0 or idx >= len(clips):
            with _jobs_lock:
                _reanalyze_jobs[video_id] = {"running": False, "error": "잘못된 클립 번호"}
            return
        orig = clips[idx]
        with _jobs_lock:
            _reanalyze_jobs[video_id] = {"running": True, "error": None, "new_idx": None, "pct": 0.0}

        def _prog(frac, msg):
            j = _reanalyze_jobs.get(video_id)
            if j is not None:
                j["pct"] = min(0.99, max(0.0, float(frac)))

        new_clip = reanalyze_clip_region(
            video_dir, orig, cfg, model=resolve_model(cfg["highlights"].get("model", "")), on_progress=_prog
        )
        # 하이라이트 후보 목록 '맨 아래'에 새 후보로 추가한다(원본은 그대로 유지).
        with CLIPS_LOCK:
            clips = load_clips_json(clips_path)  # 그 사이 바뀌었을 수 있어 다시 읽는다
            clips.append(new_clip)
            new_idx = len(clips) - 1
            save_clips_json(clips, clips_path)
        with _jobs_lock:
            _reanalyze_jobs[video_id] = {"running": False, "error": None, "new_idx": new_idx, "pct": 1.0}
    except Exception as e:  # noqa: BLE001 - 실패해도 서버는 살아야 하고 팝업에 사유를 알린다
        traceback.print_exc()
        with _jobs_lock:
            _reanalyze_jobs[video_id] = {"running": False, "error": str(e)[:300], "new_idx": None}


@app.route("/video/<video_id>/clip/<int:idx>/reanalyze", methods=["POST"])
def reanalyze_clip_route(video_id: str, idx: int):
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    # 검사(running?)와 표시(running=True)를 한 락 안에서 한다. 예전엔 락이 없어서 그 사이에
    # 두 번째 요청이 끼어들면(다른 탭·더블클릭) 같은 클립 재분석 스레드가 2개 떠서 claude
    # 호출을 두 배로 태우고 서로의 결과를 덮어썼다(TOCTOU).
    with _jobs_lock:
        cur = _reanalyze_jobs.get(video_id)
        if cur and cur.get("running"):
            return jsonify({"error": "이미 재분석이 진행 중입니다"}), 409
        _reanalyze_jobs[video_id] = {"running": True, "error": None, "new_idx": None, "pct": 0.0}
    threading.Thread(target=_run_reanalyze_job, args=(video_id, idx), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/video/<video_id>/reanalyze_status")
def reanalyze_status(video_id: str):
    j = _reanalyze_jobs.get(video_id) or {"running": False, "error": None, "new_idx": None, "pct": 0.0}
    return jsonify(j)


# ── 팝업 "자막 꼼꼼 재분석": 이 클립 구간만 정밀 재전사(large-v3)해 자막 초안을 새로 뽑는다 ──
_retrans_jobs: dict[str, dict] = {}


def _run_retranscribe_job(video_id: str, idx: int) -> None:
    """백그라운드: 클립 구간을 정밀 재전사해 자막 라인을 만들고, 결과를 정밀 캐시에도
    저장한다(다음 렌더가 같은 결과를 재사용 → 결정적). 토큰 비용 없음(로컬 whisper)."""
    key = f"{video_id}:{idx}"
    try:
        video_dir = OUTPUT_ROOT / video_id
        cfg = _load_config()
        w = cfg["whisper"]
        clips = load_clips_json(video_dir / "clips.json")
        clip = clips[idx]

        from src.captions import _collect_words_in_range, _display_text, chunk_words_into_lines
        from src.main import (
            _apply_corrections, _build_clip_hotwords, _precise_cache_save, _precise_worst_hole,
            json_load_transcript,
        )
        from src.transcribe import transcribe_clip_precise

        tdata = json_load_transcript(video_dir / "transcript.json")
        base_segs = tdata["segments"]
        base_text_all = " ".join((s.text or "") for s in base_segs)
        hotwords = _build_clip_hotwords(clip.keywords, w.get("bible_hotwords", ""), base_text_all)
        sig = hashlib.md5(
            f"{w.get('initial_prompt', '')}|{hotwords or ''}".encode("utf-8")
        ).hexdigest()[:8]
        model = w.get("precise_model_size", w["model_size"])
        tr_a = max(0.0, clip.start - 4.0)
        tr_b = clip.end + 16.0  # 렌더 경로와 같은 버퍼(문장 끝 탐색 여유)

        def _precise(vad: bool, batched: bool = True):
            return transcribe_clip_precise(
                video_dir / "source.mp4", tr_a, tr_b,
                model_size=model, device=w["device"], compute_type=w["compute_type"],
                language=w["language"], vad_filter=vad,
                initial_prompt=w.get("initial_prompt"), hotwords=hotwords,
                cpu_threads=int(w.get("cpu_threads", 0)), batch_size=int(w.get("batch_size", 8)),
                batched=batched,
            )

        def _hole(s):
            return _precise_worst_hole(base_segs, s, clip.start, clip.end)

        # 렌더 경로와 같은 '구멍' 방어: 배치 인식이 앞/중간을 통째로 놓치면(실측: 19초 구멍)
        # VAD 끔 → 순차 모드 순으로 재시도한다. 구멍 난 결과는 캐시에 저장하지 않는다.
        segs = _precise(w.get("vad_filter", True))
        if _hole(segs) >= 5.0:
            try:
                s2 = _precise(False)
                if _hole(s2) < _hole(segs):
                    segs = s2
            except Exception:  # noqa: BLE001
                pass
        if _hole(segs) >= 5.0:
            try:
                s3 = _precise(False, batched=False)
                if _hole(s3) < _hole(segs):
                    segs = s3
            except Exception:  # noqa: BLE001
                pass
        corrections = cfg.get("captions", {}).get("corrections") or {}
        if _hole(segs) < 5.0:
            _apply_corrections(segs, corrections)
            _precise_cache_save(video_dir / "precise_cache", model, sig, tr_a, tr_b, segs)
        else:
            # 재시도까지 해도 5초+ 구멍이 남는 난구간: 렌더와 같은 규칙으로 base(참조 전사)
            # 폴백 — 초안과 실제 렌더가 항상 같은 소스를 쓰게 유지한다(자막 실종 방지 우선).
            segs = base_segs
            _apply_corrections(segs, corrections)

        captions_cfg = cfg["captions"]
        words = _collect_words_in_range(
            segs, clip.start, clip.end,
            strip_filler=captions_cfg.get("strip_filler", True),
            aggressive_filler=captions_cfg.get("aggressive_filler", False),
        )
        # 처음 렌더와 같은 시간축(무음 보정 + 전역 오프셋)으로 초안을 만든다(_caption_lines_for_clip 주석).
        words = _voice_corrected_words(words, video_id, clip.start, clip.end)
        sync_off = float(captions_cfg.get("sync_offset_sec", 0.0) or 0.0)
        max_wpl = captions_cfg.get("max_words_per_line", 4)
        res_w = (cfg.get("render", {}).get("resolution") or [1080, 1920])[0]
        max_units = max(4.0, (res_w - 104) / max(1, captions_cfg.get("font_size", 72)))
        lines = chunk_words_into_lines(words, max_wpl, max_units=max_units)
        out = [
            {
                "start": ln.start + sync_off, "end": ln.end + sync_off,
                "text": " ".join(_display_text(x.text) for x in ln.words),
            }
            for ln in lines
        ]
        for i in range(len(out) - 1):
            if out[i]["end"] > out[i + 1]["start"]:
                out[i]["end"] = max(out[i]["start"] + 0.3, out[i + 1]["start"] - 0.02)
        if not out:
            raise RuntimeError("정밀 재전사 결과가 비었습니다 (무음 구간이거나 인식 실패)")
        _retrans_jobs[key] = {"running": False, "error": None, "lines": out}
    except Exception as e:  # noqa: BLE001 - 실패 사유를 팝업에 그대로 알린다
        traceback.print_exc()
        _retrans_jobs[key] = {"running": False, "error": str(e)[:300], "lines": None}


@app.route("/video/<video_id>/clip/<int:idx>/retranscribe", methods=["POST"])
def retranscribe_route(video_id: str, idx: int):
    if not (OUTPUT_ROOT / video_id / "clips.json").exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    key = f"{video_id}:{idx}"
    # 재분석과 같은 이유로 락 안에서 검사+표시(TOCTOU). 정밀 재전사는 large-v3라 중복 실행이
    # 뜨면 CPU를 두 배로 먹고 whisper 모델 사본까지 늘어난다.
    with _jobs_lock:
        cur = _retrans_jobs.get(key)
        if cur and cur.get("running"):
            return jsonify({"ok": True, "already": True})
        _retrans_jobs[key] = {"running": True, "error": None, "lines": None}
    threading.Thread(target=_run_retranscribe_job, args=(video_id, idx), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/video/<video_id>/clip/<int:idx>/retranscribe_status")
def retranscribe_status(video_id: str, idx: int):
    j = _retrans_jobs.get(f"{video_id}:{idx}") or {"running": False, "error": None, "lines": None}
    return jsonify(j)


def _sync_captions_by_voice(video_dir: Path, clip, in_lines: list, fresh: bool = False) -> "Response":
    """노래(찬양) 자막 싱크: 클립을 온디맨드 전사해 '가창 단어 시각'을 얻고, 가사 자막을
    퍼지 앵커링으로 그 위치에 매핑한다(내용이 틀려도 타이밍은 맞음). 결과는 캐시.

    fresh=True(같은 팝업에서 두 번째 이상 누름): 캐시 재사용이 아니라 최고 정밀 모델
    (whisper.precise_model_size, 기본 large-v3)로 다시 전사해 다시 맞춘다 — "계속 눌러도
    결과가 똑같다" 신고 대응. 같은 모델 결과는 결정적이라 세 번째부터는 large-v3 캐시를
    재사용한다(더 좋아질 여지가 없음)."""
    cfg = _load_config()
    from src.main import map_lines_to_voice_times

    w = cfg["whisper"]
    model_override = (w.get("precise_model_size") or "large-v3") if fresh else ""
    texts = [str(l.get("text", "")).strip() for l in in_lines]
    mapped = map_lines_to_voice_times(
        video_dir / "source.mp4", clip.start, clip.end, texts,
        w, cache_dir=video_dir / "precise_cache",
        model_override=model_override,
    )
    if not mapped:
        # 가창 단어가 거의 안 잡혔다 → 매핑 불가, 원래 시각 유지(빈 칸만 정리).
        lines = [
            {"start": float(l.get("start", 0)), "end": float(l.get("end", 0)),
             "text": str(l.get("text", "")).strip()}
            for l in in_lines
        ]
        lines, dropped, filled = _tighten_caption_lines(lines, clip)
        return jsonify({"lines": lines, "matched": 0, "total": len(lines),
                        "dropped": dropped, "filled": filled, "retranscribe_helps": False,
                        "mode": "voice", "source": "전사 단어 부족 — 원래 시각 유지"})
    src = "최고 정밀(large-v3) 재분석" if fresh else "가창 단어 시각 기준(노래 싱크)"
    mapped = _split_long_caption_lines(video_dir.name, clip, cfg, mapped)
    mapped, dropped, filled = _tighten_caption_lines(mapped, clip)
    return jsonify({"lines": mapped, "matched": len(mapped), "total": len(mapped), "source": src,
                    "dropped": dropped, "filled": filled, "retranscribe_helps": False,
                    "mode": "voice"})


@app.route("/video/<video_id>/clip/<int:idx>/sync_captions", methods=["POST"])
def sync_captions_route(video_id: str, idx: int):
    """'싱크 맞추기': 현재 자막 줄 구조(텍스트·분할)는 그대로 두고, 각 줄의 시작·끝만
    정밀 인식 단어 시각에 다시 정렬한다. 줄 텍스트를 참조 단어열과 퍼지 매칭(순차)해서
    맞는 구간을 찾는다 — 몇 초면 끝나고 토큰 비용 없음."""
    import difflib
    import re as _re

    video_dir = OUTPUT_ROOT / video_id
    clips_path = video_dir / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "잘못된 클립 번호"}), 400
    clip = clips[idx]
    body = request.get_json() or {}
    in_lines = body.get("captions") or []
    if not in_lines:
        return jsonify({"error": "정렬할 자막이 없습니다"}), 400

    # 찬양(제목 기반 업로드 등)은 전사본이 없거나 노래 오인식이 심해 '텍스트 매칭' 싱크가
    # 무의미하다(사용자 신고: "업로드 영상은 싱크 맞추기해도 안 맞아"). 대신 실제 '가창 단어
    # 시각'에 가사 진행을 비례로 매핑한다 — 단어 내용이 틀려도 '언제 노래하는지'는 맞으므로,
    # 인트로/간주를 건너뛰고 가사가 노래에 붙는다.
    if getattr(clip, "clip_type", "") == "praise" or not (video_dir / "transcript.json").exists():
        try:
            return _sync_captions_by_voice(
                video_dir, clip, in_lines, fresh=bool(body.get("fresh")),
            )
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"싱크 맞추기 실패(전사): {e}"}), 500

    from src.captions import _collect_words_in_range
    from src.main import (
        _apply_corrections, _build_clip_hotwords, _precise_cache_find, _precise_worst_hole,
        json_load_transcript,
    )

    cfg = _load_config()
    w = cfg["whisper"]
    tdata = json_load_transcript(video_dir / "transcript.json")
    base_text_all = " ".join((s.text or "") for s in tdata["segments"])
    hotwords = _build_clip_hotwords(clip.keywords, w.get("bible_hotwords", ""), base_text_all)
    sig = hashlib.md5(
        f"{w.get('initial_prompt', '')}|{hotwords or ''}".encode("utf-8")
    ).hexdigest()[:8]
    model = w.get("precise_model_size", w["model_size"])
    segs = _precise_cache_find(video_dir / "precise_cache", model, sig, clip.start, clip.end + 4.0)
    used = "정밀 캐시"
    if segs is not None and _precise_worst_hole(tdata["segments"], segs, clip.start, clip.end) >= 5.0:
        segs = None  # 구멍 난 불량 캐시는 신뢰하지 않는다
    precise = segs is not None
    if segs is None:
        segs = tdata["segments"]
        used = "참조 전사(캐시 없음 — 정밀 인식 후 다시 맞추면 더 정확)"
    _apply_corrections(segs, cfg.get("captions", {}).get("corrections") or {})
    words = _collect_words_in_range(segs, clip.start - 2.0, clip.end + 4.0)
    if not words:
        return jsonify({"error": "참조할 단어 시각이 없습니다"}), 400
    # 정렬 목표 시각도 처음 렌더와 같은 시간축(무음 보정 + 전역 오프셋)으로.
    words = _voice_corrected_words(words, video_id, clip.start, clip.end)
    sync_off = float(cfg["captions"].get("sync_offset_sec", 0.0) or 0.0)

    def norm(s: str) -> str:
        return _re.sub(r"[^0-9가-힣a-zA-Z]", "", s or "")

    out = []
    matched_idx: list[int] = []  # out 안에서 확신 매칭(>=0.6)된 인덱스들
    wi = 0  # 순차 정렬: 다음 줄은 이전 줄 매칭 지점 이후에서 찾는다
    n_words = len(words)
    matched_n = 0
    for ln in in_lines:
        text = str(ln.get("text", "")).strip()
        target = norm(text)
        cur = {"start": float(ln.get("start", 0)), "end": float(ln.get("end", 0)), "text": text}
        out.append(cur)
        if not target or wi >= n_words:
            continue
        best = None  # (score, i, j)
        for i in range(wi, min(wi + 30, n_words)):
            acc = ""
            for j in range(i, min(i + 12, n_words)):
                acc += norm(words[j].text)
                if len(acc) > len(target) * 2 + 8:
                    break
                score = difflib.SequenceMatcher(None, acc, target).ratio()
                if best is None or score > best[0]:
                    best = (score, i, j)
        if best and best[0] >= 0.6:
            _, i, j = best
            cur["start"] = round(words[i].start + sync_off, 2)
            cur["end"] = round(max(words[j].end, words[i].start + 0.3) + sync_off, 2)
            wi = j + 1
            matched_n += 1
            matched_idx.append(len(out) - 1)
    # 매칭 실패(신뢰도<0.6, 또는 단어 소진)한 줄은 예전 절대 시각이 그대로 남는다 — 매칭된
    # 줄들은 '새' 시간대로 옮겨졌는데 얘들만 '옛' 시간대에 남아 뒤섞이는 게 실제 신고
    # ("빈 칸 삭제는 되는데 싱크가 안 맞다", 2026-09-20)의 원인이다. 매칭된 앵커들 사이를
    # 원래 줄 길이 비율로 새 시간대 위에 재배치하고, 첫/마지막 앵커 앞뒤도 같은 방식으로
    # 이어 붙인다(매칭이 하나도 없으면 옛 시각 그대로 — 폴백).
    if matched_idx:
        def _orig_dur(k: int) -> float:
            return max(0.05, float(in_lines[k].get("end", 0)) - float(in_lines[k].get("start", 0)))

        for ai in range(len(matched_idx) - 1):
            a, b = matched_idx[ai], matched_idx[ai + 1]
            mids = list(range(a + 1, b))
            if not mids:
                continue
            span = out[b]["start"] - out[a]["end"]
            gap = max(0.0, float(in_lines[b].get("start", 0)) - float(in_lines[a].get("end", 0)))
            lens = [_orig_dur(k) for k in mids]
            total = sum(lens) + gap
            if span <= 0 or total <= 0:
                step = max(0.3, span / (len(mids) + 1)) if span > 0 else 0.3
                t = out[a]["end"]
                for k in mids:
                    t += step
                    out[k]["start"] = round(t - step, 2)
                    out[k]["end"] = round(t, 2)
                continue
            scale = span / total
            t = out[a]["end"] + gap * scale
            for k, dur in zip(mids, lens):
                out[k]["start"] = round(t, 2)
                t += dur * scale
                out[k]["end"] = round(t, 2)
        first = matched_idx[0]
        t = out[first]["start"]
        for k in range(first - 1, -1, -1):
            out[k]["end"] = round(t, 2)
            t = max(0.0, t - _orig_dur(k))
            out[k]["start"] = round(t, 2)
        last = matched_idx[-1]
        t = out[last]["end"]
        for k in range(last + 1, len(out)):
            dur = _orig_dur(k)
            out[k]["start"] = round(t, 2)
            t += dur
            out[k]["end"] = round(t, 2)
    # 줄끼리 겹치지 않게(다음 줄 시작 - 0.02까지만) 정리해 화면에 두 줄이 겹쳐 뜨는 것 방지.
    for k in range(len(out) - 1):
        if out[k]["end"] > out[k + 1]["start"]:
            out[k]["end"] = round(max(out[k]["start"] + 0.2, out[k + 1]["start"] - 0.02), 2)
    out = _split_long_caption_lines(video_id, clip, cfg, out)
    # 빈 줄 삭제 + 줄 사이 빈 칸 메우기(딱딱 붙는 자막) — 신고 2026-09-21.
    out, dropped, filled = _tighten_caption_lines(out, clip)
    # retranscribe_helps: 정밀 인식 캐시가 없어 '참조 전사'(대충 시각)로 맞췄다는 뜻 —
    # 편집기가 이걸 보고 정밀 재전사를 돌린 뒤 자동으로 한 번 더 맞춘다.
    return jsonify({"lines": out, "matched": matched_n, "total": len(out), "source": used,
                    "dropped": dropped, "filled": filled, "retranscribe_helps": not precise,
                    "mode": "sermon"})


@app.route("/video/<video_id>/clip/<int:idx>/correct_captions", methods=["POST"])
def correct_captions_route(video_id: str, idx: int):
    """'AI 자막 교정': 현재 자막 줄의 텍스트를 Claude가 문맥·성경지식으로 교정하고
    핵심 강조어(caption_highlights)를 뽑는다. 줄 수·시간은 그대로 두고 '내용(단어)'만
    고친다(1:1 매핑). 결과는 편집기에 반영되고, 저장 시 확정된다."""
    video_dir = OUTPUT_ROOT / video_id
    clips_path = video_dir / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "잘못된 클립 번호"}), 400
    clip = clips[idx]
    body = request.get_json() or {}
    in_lines = body.get("captions") or []
    if not in_lines:
        return jsonify({"error": "교정할 자막이 없습니다"}), 400
    model = sanitize_model(body.get("model") or "")
    # 자막 교정은 '자막 실행' 단계 — 분석용 모델(opus)과 무관하게 기본 Sonnet(2026-09-06).
    model = model or EXECUTION_MODEL

    from src.highlights import correct_sermon_captions

    texts = [str(ln.get("text", "")) for ln in in_lines]
    cfg = _load_config()
    p = cfg.get("praise", {}) or {}
    try:
        corrected, highlights = correct_sermon_captions(
            texts,
            context=clip.title or "",
            model=model,
            thinking_tokens=int(p.get("lyrics_thinking_tokens", 2048)),
        )
    except Exception as e:  # noqa: BLE001 - 실패해도 원문 유지로 안내
        return jsonify({"error": f"교정 실패: {e}"}), 500

    out = []
    for i, ln in enumerate(in_lines):
        txt = corrected[i] if i < len(corrected) else str(ln.get("text", ""))
        out.append({
            "start": float(ln.get("start", 0)),
            "end": float(ln.get("end", 0)),
            "text": txt,
        })
    changed = sum(1 for i, ln in enumerate(in_lines) if out[i]["text"].strip() != str(ln.get("text", "")).strip())
    return jsonify({"lines": out, "highlights": highlights, "changed": changed, "total": len(out)})


@app.route("/video/<video_id>/clip/<int:idx>/fetch_lyrics", methods=["POST"])
def fetch_lyrics_route(video_id: str, idx: int):
    """'가사 자동 가져오기'(찬양): 곡 제목으로 정식 가사를 인터넷 검색(WebSearch)해 가져오고,
    이 클립의 실제 가창 시각에 맞춰(싱크까지) 자막 라인으로 반환한다. 저장은 편집기에서."""
    video_dir = OUTPUT_ROOT / video_id
    clips_path = video_dir / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "잘못된 클립 번호"}), 400
    clip = clips[idx]
    body = request.get_json() or {}
    title = (body.get("title") or clip.title or "").strip()
    if not title:
        return jsonify({"error": "곡 제목이 없습니다"}), 400

    from src.highlights import fetch_praise_lyrics_by_titles
    from src.main import map_lines_to_voice_times

    cfg = _load_config()
    p = cfg.get("praise", {}) or {}
    try:
        # 가사 검색도 '자막 실행' 단계 — 분석용 모델(opus)과 무관하게 기본 Sonnet(2026-09-06).
        lyrics_by_idx = fetch_praise_lyrics_by_titles(
            [title], model=EXECUTION_MODEL,
            thinking_tokens=int(p.get("lyrics_thinking_tokens", 2048)),
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"가사 검색 실패: {e}"}), 500
    lines = lyrics_by_idx.get(0) or []
    if not lines:
        return jsonify({"error": f"'{title}' 가사를 찾지 못했습니다. 곡 제목을 확인해 주세요."}), 404
    # 이 클립의 실제 가창 시각에 매핑(분석 단계와 동일: 정밀 모델+캐시). 실패 시 글자수 비례.
    mapped = None
    try:
        mapped = map_lines_to_voice_times(
            video_dir / "source.mp4", clip.start, clip.end, lines,
            cfg["whisper"], cache_dir=video_dir / "precise_cache",
            # 싱크는 단어 '시각'만 쓰므로 medium이 맞다 — large-v3는 3배 느리고 단어도 덜
            # 잡는다(실측 2026-09-07: 223초/63단어 vs 68초/73단어). main._sync_praise_clips_bg 참고.
            model_override=(
                cfg["whisper"].get("praise_model_size")
                or cfg["whisper"].get("model_size", "medium")
            ),
        )
    except Exception:  # noqa: BLE001 - 매핑 실패해도 가사는 반환(균등 분배)
        traceback.print_exc()
    if not mapped:
        from src.highlights import _distribute_lines_by_chars
        mapped = _distribute_lines_by_chars(lines, clip.start, clip.end)
    return jsonify({"lines": mapped, "total": len(mapped), "title": title})


@app.route("/video/<video_id>/clip/<int:idx>/translate_captions", methods=["POST"])
def translate_captions_route(video_id: str, idx: int):
    """'영어 자막': 현재 자막 줄을 영어로 번역한다(줄 수·시간 유지). 한국어는 그대로 두고
    영어 트랙(caption_overrides_en)으로 저장 → 렌더 시 '영어 자막' 옵션으로 전환 가능."""
    video_dir = OUTPUT_ROOT / video_id
    clips_path = video_dir / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "잘못된 클립 번호"}), 400
    clip = clips[idx]
    body = request.get_json() or {}
    in_lines = body.get("captions") or []
    if not in_lines:
        return jsonify({"error": "번역할 자막이 없습니다"}), 400
    model = sanitize_model(body.get("model") or "")
    # 번역도 '자막 실행' 단계 — 분석용 모델(opus)과 무관하게 기본 Sonnet(2026-09-06).
    model = model or EXECUTION_MODEL

    from src.highlights import translate_captions_to_english

    texts = [str(ln.get("text", "")) for ln in in_lines]
    try:
        en = translate_captions_to_english(texts, context=clip.title or "", model=model)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"번역 실패: {e}"}), 500
    out = []
    for i, ln in enumerate(in_lines):
        out.append({
            "start": float(ln.get("start", 0)),
            "end": float(ln.get("end", 0)),
            "text": en[i] if i < len(en) else str(ln.get("text", "")),
        })
    return jsonify({"lines": out, "total": len(out)})


def _tracking_scheduler_loop() -> None:
    """앱이 켜져 있는 동안 주기적으로 성과 자동 체크를 돌린다(예정일 지난 업로드만 실제로 호출됨).
    앱이 꺼져 있는 동안 예정일이 지난 건 `python -m src.upload.tracking`(작업 스케줄러)이 대신 처리한다."""
    while True:
        try:
            done = run_due_checks()
            if done:
                print(f"[tracking] 자동 성과 체크 {len(done)}건 완료")
        except Exception:  # noqa: BLE001 - 스케줄러는 죽으면 안 됨, 다음 주기에 재시도
            traceback.print_exc()
        time.sleep(6 * 3600)


if __name__ == "__main__":
    threading.Thread(target=_tracking_scheduler_loop, daemon=True).start()
    # use_reloader=False: 리로더(파일 변경 감지 자동재시작)를 끈다. 분석/렌더가 백그라운드
    # 스레드+서브프로세스로 몇 분씩 걸리는데, 리로더가 프로젝트 폴더 아무 .py 파일 변경에나
    # 반응해 서버를 재시작하면 진행 중이던 작업(및 메모리 상 _jobs 상태)이 통째로 날아간다
    # (실제로 겪은 문제: 관련 없는 스크립트 파일이 바뀌었는데도 분석 작업이 끊김).
    # 코드를 고친 뒤에는 터미널에서 수동으로 재시작해야 한다.
    # debug=True는 절대 쓰지 않는다. Werkzeug 디버거가 켜지면 예외가 나는 순간 브라우저에서
    # 임의의 파이썬 코드를 실행할 수 있는 콘솔이 열리는데(use_reloader=False로도 안 꺼진다),
    # README가 안내하는 cloudflared 터널로 이 서버를 공개하면 그 콘솔이 인터넷에 그대로
    # 노출된다 = 이 PC 원격 장악. 라우트에 인증도 없으므로 기본 바인드도 루프백으로 둔다.
    # 외부(터널/휴대폰)에서 써야 할 때만 SHORTS_HOST=0.0.0.0 로 명시적으로 연다.
    host = os.environ.get("SHORTS_HOST", "127.0.0.1")
    app.run(debug=False, use_reloader=False, threaded=True, host=host, port=5000)
