"""웹 검토 흐름: 링크 입력 -> 후보 목록(바이럴 순위) 확인 -> 고른 것만 렌더링

사용법:
    python -m src.web_app
    -> http://127.0.0.1:5000 접속 (Cloudflare Tunnel 등으로 외부 노출 가능)
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template_string, request, send_file

from src.feedback import PerformanceRecord, upsert_feedback
from src.highlights import CLIPS_LOCK, load_clips_json, save_clips_json
from src.main import analyze, reanalyze_clip_region, render_selected, render_signature
from src.upload.tracking import find_upload, load_uploads, record_upload, run_due_checks

app = Flask(__name__)
OUTPUT_ROOT = Path("output")

# 단일 사용자 로컬 도구이므로 메모리 내 딕셔너리로 작업 상태를 추적한다 (DB 불필요).
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# 클립 '구간 재분석' 작업 상태(영상별). 팝업이 폴링해서 완료되면 새 후보를 보여준다.
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
BASE_STYLE = """
<link rel="stylesheet" as="style" crossorigin
  href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css" />
<style>
  :root {
    --bg: #f7f8fa;
    --card: #ffffff;
    --text: #191f28;
    --text-muted: #6b7684;
    --text-faint: #8b95a1;
    --accent: #3182f6;
    --accent-hover: #1b64da;
    --border: #f0f1f3;
    --shadow: 0 2px 8px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.03);
  }
  * { box-sizing: border-box; }
  body {
    font-family: "Pretendard", -apple-system, BlinkMacSystemFont, "Malgun Gothic", sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 56px 20px 100px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 720px; margin: 0 auto; }
  a.back {
    color: var(--text-muted); text-decoration: none; font-size: 14px; font-weight: 500;
    display: inline-block; margin-bottom: 20px;
  }
  a.back:hover { color: var(--text); }
  h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 6px; }
  .subtitle { color: var(--text-muted); font-size: 15px; margin: 0 0 32px; }
  .card {
    background: var(--card); border-radius: 20px; box-shadow: var(--shadow);
    padding: 28px; margin-bottom: 16px;
  }
  input[type=text] {
    width: 100%; padding: 16px 18px; font-size: 15px; font-family: inherit;
    border: 1.5px solid var(--border); border-radius: 14px; background: #fafbfc;
    transition: border-color .15s, background .15s; margin-top: 14px;
  }
  input[type=text]:focus {
    outline: none; border-color: var(--accent); background: #fff;
  }
  textarea {
    width: 100%; padding: 14px 16px; font-size: 14px; font-family: inherit; line-height: 1.5;
    border: 1.5px solid var(--border); border-radius: 14px; background: #fafbfc;
    transition: border-color .15s, background .15s; margin-top: 12px; resize: vertical;
  }
  textarea:focus { outline: none; border-color: var(--accent); background: #fff; }
  .hint { color: var(--text-faint); font-size: 12.5px; margin: 10px 2px 0; }
  input[type=number] {
    padding: 9px 11px; font-size: 13px; font-family: inherit; width: 100%;
    border: 1.5px solid var(--border); border-radius: 10px; background: #fafbfc;
  }
  input[type=number]:focus { outline: none; border-color: var(--accent); background: #fff; }
  /* 점수 세부축 막대 */
  .subscores { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 10px 0 2px; }
  .subscore { font-size: 12px; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }
  .subscore b { color: var(--text); font-variant-numeric: tabular-nums; font-weight: 700; }
  .sbar { width: 46px; height: 5px; border-radius: 999px; background: var(--border); overflow: hidden; }
  .sbar > i { display: block; height: 100%; background: var(--accent); border-radius: 999px; }
  /* 성과 피드백 폼 */
  .fb { margin-top: 14px; border-top: 1px dashed var(--border); padding-top: 14px; }
  .fb summary { cursor: pointer; font-size: 13px; font-weight: 600; color: var(--text-muted); }
  .fb-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }
  .fb-grid label { font-size: 11.5px; color: var(--text-faint); font-weight: 600; display: block; margin-bottom: 4px; }
  .fb-rate { display: flex; gap: 8px; margin-top: 10px; }
  .fb-rate button {
    flex: 1; padding: 9px; font-size: 13px; font-weight: 600; font-family: inherit;
    border: 1.5px solid var(--border); border-radius: 10px; background: #fff; cursor: pointer; color: var(--text-muted);
  }
  .fb-rate button.sel { border-color: var(--accent); color: var(--accent); background: #f0f6ff; }
  .fb-save {
    margin-top: 12px; padding: 10px 14px; font-size: 13px; font-weight: 600; font-family: inherit;
    color: #fff; background: var(--accent); border: none; border-radius: 10px; cursor: pointer;
  }
  .fb-saved { font-size: 12.5px; color: #12b886; font-weight: 600; margin-left: 10px; }
  /* YouTube 업로드 */
  .yt-upload { margin-top: 12px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .yt-upload-btn {
    padding: 9px 14px; font-size: 13px; font-weight: 600; font-family: inherit;
    border: 1.5px solid var(--accent); border-radius: 10px; background: #fff; color: var(--accent); cursor: pointer;
  }
  .yt-upload-btn:disabled { opacity: 0.6; cursor: default; }
  .yt-status { font-size: 12.5px; color: var(--text-muted); }
  .yt-link { font-size: 13px; font-weight: 600; color: var(--accent); text-decoration: none; }
  .yt-link:hover { text-decoration: underline; }
  .yt-track { font-size: 12px; color: var(--text-faint); }
  button.primary {
    width: 100%; margin-top: 16px; padding: 16px; font-size: 15px; font-weight: 600;
    font-family: inherit; color: #fff; background: var(--accent); border: none;
    border-radius: 14px; cursor: pointer; transition: background .15s;
  }
  button.primary:hover { background: var(--accent-hover); }
  button.primary:active { transform: scale(0.99); }
  .status-box {
    padding: 20px 24px; background: var(--card); border-radius: 16px; box-shadow: var(--shadow);
    white-space: pre-line; color: var(--text-muted); font-size: 14px;
  }
  .spinner {
    display: inline-block; width: 14px; height: 14px; border-radius: 50%;
    border: 2px solid #dbe4f0; border-top-color: var(--accent);
    animation: spin .8s linear infinite; margin-right: 8px; vertical-align: -2px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .progress-row {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px; font-size: 14px; color: var(--text);
  }
  .progress-pct { font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
  .progress-track {
    width: 100%; height: 8px; border-radius: 999px; background: var(--border); overflow: hidden;
  }
  .progress-fill {
    height: 100%; border-radius: 999px; background: var(--accent);
    transition: width .4s ease;
  }
  /* 애플식 단계형 진행바 */
  .prog-card { padding: 26px 28px; background: var(--card); border-radius: 20px; box-shadow: var(--shadow); }
  .prog-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 26px; gap: 12px; }
  .prog-headL { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
  .prog-msg { font-size: 15px; color: var(--text); font-weight: 600; min-width: 0; }
  .prog-eta { font-size: 12.5px; font-weight: 500; color: var(--text-faint); font-variant-numeric: tabular-nums; }
  .prog-pct { font-size: 27px; font-weight: 800; color: var(--accent); font-variant-numeric: tabular-nums; letter-spacing: -0.02em; white-space: nowrap; }
  .stepper { position: relative; display: flex; justify-content: space-between; padding: 0; }
  /* 양 끝 스텝의 원이 레일의 정확한 시작·끝에 걸리도록 정렬(가운데 스텝만 중앙). */
  .stepper .step:first-child { align-items: flex-start; }
  .stepper .step:last-child { align-items: flex-end; }
  .rail { position: absolute; top: 11px; left: 11px; right: 11px; height: 4px; background: var(--border); border-radius: 999px; }
  .rail-fill { position: absolute; top: 0; bottom: 0; left: 0; width: 0%; background: linear-gradient(90deg, var(--accent), #5aa2ff); border-radius: 999px; transition: width .6s cubic-bezier(.22,.61,.36,1); }
  .step { position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; gap: 11px; }
  .dot { width: 22px; height: 22px; border-radius: 50%; background: var(--card); border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; transition: border-color .35s ease, background .35s ease, box-shadow .35s ease; }
  .dot::after { content: ''; width: 7px; height: 7px; border-radius: 50%; background: transparent; transition: background .3s ease; }
  .step.active .dot { border-color: var(--accent); box-shadow: 0 0 0 5px rgba(49, 130, 246, .15); }
  .step.active .dot::after { background: var(--accent); animation: dotpulse 1.2s ease-in-out infinite; }
  .step.done .dot { border-color: var(--accent); background: var(--accent); }
  .step.done .dot::after { content: '✓'; color: #fff; font-size: 12px; font-weight: 800; width: auto; height: auto; background: transparent; }
  .step .lbl { font-size: 12.5px; color: var(--text-faint); font-weight: 600; white-space: nowrap; transition: color .3s ease; }
  .step.active .lbl { color: var(--accent); font-weight: 700; }
  .step.done .lbl { color: var(--text-muted); }
  @keyframes dotpulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(.55); opacity: .5; } }
</style>
"""

INDEX_TEMPLATE = f"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>교회 쇼츠 생성기</title>
{BASE_STYLE}
<style>
  .adv {{ margin-top: 12px; }}
  .adv summary {{
    cursor: pointer; font-size: 13px; font-weight: 600; color: var(--text-muted);
    padding: 4px 2px; user-select: none; list-style-position: inside;
  }}
  .adv summary:hover {{ color: var(--text); }}
  .adv-sub {{ font-weight: 500; color: var(--text-faint); margin-left: 4px; }}
  /* 모델 선택(소넷/오푸스) 세그먼트 버튼 */
  .model-pick {{ margin: 14px 0 4px; }}
  .model-pick-lbl {{ font-size: 12.5px; font-weight: 600; color: var(--text-muted); margin-bottom: 7px; }}
  .seg {{ display: flex; gap: 6px; background: var(--border); padding: 4px; border-radius: 12px; }}
  .seg label {{
    flex: 1; text-align: center; cursor: pointer; border-radius: 9px; padding: 9px 8px;
    font-size: 13.5px; font-weight: 700; color: var(--text-muted); transition: background .15s, color .15s, box-shadow .15s;
  }}
  .seg label .seg-sub {{ display: block; font-size: 11px; font-weight: 500; color: var(--text-faint); margin-top: 2px; }}
  .seg input {{ position: absolute; opacity: 0; pointer-events: none; }}
  .seg label:has(input:checked) {{ background: var(--card); color: var(--accent); box-shadow: 0 1px 4px rgba(15,23,42,.12); }}
  .seg label:has(input:checked) .seg-sub {{ color: var(--text-muted); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>교회 쇼츠 생성기</h1>
  <p class="subtitle">유튜브 설교 링크를 넣으면 하이라이트 후보를 뽑아드려요.</p>
  <div class="card">
    <form id="f">
      <input type="text" id="url" placeholder="https://www.youtube.com/watch?v=..." required autofocus>
      <details class="adv">
        <summary>고급 옵션 <span class="adv-sub">자막 붙여넣기 · 새로 분석</span></summary>
        <textarea id="transcript" rows="5" placeholder="(선택) 자막 붙여넣기 — 붙여넣으면 자동 전사를 건너뛰고 이걸로 하이라이트를 찾습니다. 유튜브 '스크립트 표시' 복사 또는 SRT/VTT 권장."></textarea>
        <label style="display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text-muted);margin-top:10px;cursor:pointer">
          <input type="checkbox" id="force"> 새로 분석 (저장된 후보 무시하고 다시 뽑기)
        </label>
      </details>
      <div class="model-pick">
        <div class="model-pick-lbl">하이라이트 선정 AI 모델</div>
        <div class="seg">
          <label><input type="radio" name="model" value="claude-sonnet-4-5" checked>소넷<span class="seg-sub">빠름 · 한도 절약 (기본)</span></label>
          <label><input type="radio" name="model" value="claude-opus-4-8">오푸스<span class="seg-sub">품질 우선 · 한도 더 씀</span></label>
          <label><input type="radio" name="model" value="claude-fable-5">페이블<span class="seg-sub">최고 품질 · 한도 많이 씀</span></label>
        </div>
      </div>
      <button class="primary" type="submit">분석 시작</button>
    </form>
    <p class="hint">같은 영상은 저장된 후보를 재사용해 사용량을 아껴요.</p>
  </div>
  <div class="status-box" id="status" style="display:none"></div>
</div>
<script>
const f = document.getElementById('f');
const statusEl = document.getElementById('status');
const submitBtn = f.querySelector('button[type="submit"]');
f.addEventListener('submit', async (e) => {{
  e.preventDefault();
  if (submitBtn.disabled) return;  // 중복 클릭 방지: 하이라이트 선정은 AI가 실제로 읽고 고르는
                                    // 단계라 보통 2~5분 걸린다. 재클릭하면 같은 영상 분석이
                                    // 중복 실행돼 진행률이 널뛰고 세션 한도만 낭비된다.
  submitBtn.disabled = true;
  const url = document.getElementById('url').value;
  const transcript_text = document.getElementById('transcript').value;
  const force = document.getElementById('force').checked;  // 기본은 캐시 재사용, 체크 시에만 새로 분석
  const model = (f.querySelector('input[name="model"]:checked') || {{}}).value || '';
  statusEl.style.display = 'block';
  statusEl.innerHTML = '<span class="spinner"></span>진행률 화면으로 이동 중… (곧 %와 남은 예상시간이 표시돼요)';
  try {{
    const res = await fetch('/analyze', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{url, transcript_text, force, model}})
    }});
    const data = await res.json();
    if (!res.ok) {{ statusEl.innerText = '오류: ' + data.error; submitBtn.disabled = false; return; }}
    // 진행률(%)과 ETA는 영상 진행바 페이지에서 폴링으로 실시간 표시된다. 링크만 넣어도
    // 이 페이지로 즉시 이동해 %가 바로 보이게 한다(예전엔 이 인덱스 문구에 %가 없어
    // "몇 %인지 안 나온다"는 오해가 있었다).
    window.location.replace('/video/' + data.video_id);
  }} catch (err) {{
    statusEl.innerText = '오류: ' + err;
    submitBtn.disabled = false;
  }}
}});
</script>
</body>
</html>
"""

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
  .score-badge.tier-top {{ background: #fff4e5; color: #c2620c; }}       /* 90점 이상: 최상 */
  .score-badge.tier-high {{ background: #e7f7ec; color: #1a7f37; }}      /* 85점 이상: 추천 */
  .score-badge.tier-ok {{ background: #eaf1ff; color: #2563eb; }}        /* 80점 이상: 후보 */
  .score-badge.tier-low {{ background: #f1f3f5; color: #868e96; }}       /* 80점 미만: 참고용 */
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
  .reason {{
    background: #f7f8fa; border-radius: 12px; padding: 13px 15px; margin: 12px 0 0;
  }}
  .reason-caption {{ color: var(--text); font-size: 13.5px; line-height: 1.6; margin: 0 0 8px; }}
  .reason-hashtags {{ color: var(--text-faint); font-size: 12.5px; margin: 0 0 10px; }}
  .reason-text {{ color: var(--text-muted); font-size: 13px; line-height: 1.65; white-space: pre-line; margin: 0; }}
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
  <h1>하이라이트 후보</h1>
  <p class="subtitle">바이럴 예상 순위 순으로 정렬했어요. 만들고 싶은 걸 골라주세요.</p>
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
        <span class="score-badge {{% if c.score >= 90 %}}tier-top{{% elif c.score >= 85 %}}tier-high{{% elif c.score >= 80 %}}tier-ok{{% else %}}tier-low{{% endif %}}">{{{{ "%.0f"|format(c.score) }}}}점{{% if c.score >= 90 %}} · 최상{{% elif c.score >= 85 %}} · 추천{{% elif c.score < 80 %}} · 참고{{% endif %}}</span>
        {{% endif %}}
        <span class="dot-sep"></span>
        <span class="dur">{{{{ "%.0f"|format(c.end - c.start) }}}}초</span>
      </div>
      <label class="pick">
        <input type="checkbox" name="idx" value="{{{{ loop.index0 }}}}">
        <span class="pick-label">만들기</span>
      </label>
    </div>
    <h3 class="title">{{{{ c.title }}}}</h3>
    {{# 캡션·해시태그·추천 이유는 기본으로 접어 화면을 조용하게 유지한다(제목이 주인공). #}}
    <div class="cand-foot">
      <button type="button" class="reason-toggle" aria-expanded="false">상세 보기 <span class="chev">▾</span></button>
      <a class="edit-link" href="/video/{{{{ video_id }}}}/clip/{{{{ loop.index0 }}}}/edit">위치·자막 편집 &rarr;</a>
    </div>
    <div class="reason" hidden>
      {{% if c.appeal or c.hook_line %}}
      <p class="reason-hashtags">{{% if c.appeal %}}🎯 {{{{ c.appeal }}}}{{% endif %}}{{% if c.hook_line %}} · 첫 문장: “{{{{ c.hook_line }}}}”{{% endif %}}{{% if c.payoff_line %}} · 끝 문장: “{{{{ c.payoff_line }}}}”{{% endif %}}</p>
      {{% endif %}}
      {{% if c.insight %}}
      <p class="reason-hashtags">💡 {{{{ c.insight }}}}</p>
      {{% endif %}}
      <p class="reason-caption">{{{{ c.caption }}}}</p>
      <p class="reason-hashtags">{{{{ c.hashtags|join(' ') }}}}</p>
      <p class="reason-text">{{{{ c.reason }}}}</p>
    </div>
    <div class="cand-video" id="candvid-{{{{ loop.index0 }}}}">
    {{% if c.rendered %}}
      <video controls src="/media/{{{{ video_id }}}}/{{{{ loop.index }}}}.mp4"></video>
      <a class="dl-link" href="/media/{{{{ video_id }}}}/{{{{ loop.index }}}}.mp4" download>⬇ 영상 저장</a>
    {{% endif %}}
    </div>
    {{% if c.rendered %}}
    <div class="yt-upload" data-idx="{{{{ loop.index0 }}}}">
      {{% if c.youtube_id %}}
      <a class="yt-link" href="https://youtu.be/{{{{ c.youtube_id }}}}" target="_blank" rel="noopener">▶ YouTube에서 보기</a>
      <span class="yt-track">
        {{% if c.upload_done %}}자동 성과 체크 완료 (4/4주)
        {{% elif c.upload_max_checks %}}자동 성과 체크 진행 중 ({{{{ c.upload_checks_done }}}}/{{{{ c.upload_max_checks }}}}주, 매주 자동 확인)
        {{% endif %}}
      </span>
      {{% else %}}
      <button type="button" class="yt-upload-btn">YouTube에 업로드</button>
      <span class="yt-status"></span>
      {{% endif %}}
    </div>
    {{% endif %}}
    {{% if c.rendered %}}
    {{# 성과 입력은 실제로 만든(렌더된) 클립에서만 의미가 있다 — 안 만든 후보 카드는 조용하게. #}}
    <details class="fb" data-idx="{{{{ loop.index0 }}}}" data-title="{{{{ c.title|e }}}}">
      <summary>📊 실제 성과 입력 (업로드 후 조회수는 매주 자동 수집돼요 — 느낀 점만 적어도 충분)</summary>
      <div class="fb-grid">
        <div><label>조회수</label><input type="number" class="fb-views" min="0" placeholder="예: 12000"></div>
        <div><label>평균 조회율(%)</label><input type="number" class="fb-ret" min="0" max="100" placeholder="예: 45"></div>
        <div><label>저장</label><input type="number" class="fb-saves" min="0" placeholder="예: 320"></div>
        <div><label>공유</label><input type="number" class="fb-shares" min="0" placeholder="예: 80"></div>
        <div><label>좋아요</label><input type="number" class="fb-likes" min="0" placeholder="예: 540"></div>
        <div><label>댓글</label><input type="number" class="fb-comments" min="0" placeholder="예: 25"></div>
      </div>
      <div class="fb-rate">
        <button type="button" data-rate="hit">잘 됨 ✅</button>
        <button type="button" data-rate="ok">보통</button>
        <button type="button" data-rate="flop">망함 ❌</button>
      </div>
      <div style="margin-top:12px">
        <label style="font-size:11.5px;color:var(--text-faint);font-weight:600;display:block;margin-bottom:4px">메모(선택)</label>
        <input type="text" class="fb-notes" placeholder="예: 훅이 강했다 / 초반 이탈 많음">
      </div>
      <button type="button" class="fb-save">성과 저장</button>
      <span class="fb-saved" hidden>저장됨 ✓</span>
    </details>
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
  <script>
  document.getElementById('renderForm').addEventListener('submit', async (e) => {{
    e.preventDefault();
    const idx = [...document.querySelectorAll('input[name=idx]:checked')].map(el => parseInt(el.value));
    if (idx.length === 0) {{ alert('클립을 하나 이상 선택하세요'); return; }}
    const doRender = async () => {{
      const res = await fetch('/video/{{{{ video_id }}}}/render', {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{indices: idx}})
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

  // 성과 피드백: 등급 선택 + 저장 → 서버에 기록(다음 선정 프롬프트에 캘리브레이션으로 주입됨).
  document.querySelectorAll('details.fb').forEach(function(fb) {{
    var rate = 'ok';
    fb.querySelectorAll('.fb-rate button').forEach(function(b) {{
      b.addEventListener('click', function() {{
        rate = b.dataset.rate;
        fb.querySelectorAll('.fb-rate button').forEach(x => x.classList.remove('sel'));
        b.classList.add('sel');
      }});
    }});
    var num = function(sel) {{ var v = fb.querySelector(sel).value.trim(); return v === '' ? null : parseFloat(v); }};
    fb.querySelector('.fb-save').addEventListener('click', async function() {{
      var payload = {{
        clip_index: parseInt(fb.dataset.idx), title: fb.dataset.title, rating: rate,
        views: num('.fb-views'), retention_pct: num('.fb-ret'), saves: num('.fb-saves'),
        shares: num('.fb-shares'), likes: num('.fb-likes'), comments: num('.fb-comments'),
        notes: fb.querySelector('.fb-notes').value.trim(),
      }};
      var res = await fetch('/video/{{{{ video_id }}}}/feedback', {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(payload)
      }});
      var saved = fb.querySelector('.fb-saved');
      if (res.ok) {{ saved.hidden = false; setTimeout(() => {{ saved.hidden = true; }}, 2500); }}
      else {{ alert('저장 실패'); }}
    }});
  }});

  // YouTube 업로드 버튼: 눌러 업로드되면 이후 성과는 주 1회 최대 4주 자동으로 체크된다.
  document.querySelectorAll('.yt-upload').forEach(function(box) {{
    var btn = box.querySelector('.yt-upload-btn');
    if (!btn) return;
    var status = box.querySelector('.yt-status');
    btn.addEventListener('click', async function() {{
      btn.disabled = true;
      status.textContent = '업로드 중... (영상 크기에 따라 시간이 걸릴 수 있어요)';
      try {{
        var res = await fetch('/video/{{{{ video_id }}}}/clip/' + box.dataset.idx + '/upload', {{ method: 'POST' }});
        var data = await res.json();
        if (!res.ok) throw new Error(data.error || '업로드 실패');
        status.textContent = '';
        box.innerHTML = '<a class="yt-link" href="' + data.url + '" target="_blank" rel="noopener">▶ YouTube에서 보기</a>' +
          '<span class="yt-track">자동 성과 체크 예약됨 (매주, 최대 4주)</span>';
      }} catch (e) {{
        btn.disabled = false;
        status.textContent = '실패: ' + e.message;
      }}
    }});
  }});
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
      msg.innerHTML = '✅ 쇼츠 완성! 영상은 자동 저장됐어요';
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
    function poll() {{
      fetch(statusUrl).then(function(r) {{ return r.json(); }}).then(function(j) {{
        if (j.render_error) {{ box.classList.remove('hidden'); track.hidden = true; closeBtn.hidden = false; pct.style.display='none'; msg.innerHTML = '⚠️ 오류: ' + j.render_error; eta.textContent=''; actions.hidden=true; wasRendering=false; setTimeout(poll, 1500); return; }}
        if (j.rendering) {{ wasRendering = true; showProg(j.render_pct, j.render_message || '쇼츠 렌더링 중…', j.render_eta_seconds); }}
        else if (wasRendering) {{ wasRendering = false; onRenderDone(); }}
        else if (!j.ready && j.status !== 'error') {{ showProg(j.pct, j.message || '분석 중…', j.eta_seconds); }}
        setTimeout(poll, 800);
      }}).catch(function() {{ setTimeout(poll, 1500); }});
    }}
    // 버튼 클릭 시 호출: 이번에 렌더할 인덱스를 기억하고 즉시 위젯을 띄운다.
    window.__startRenderWatch = function(indices) {{
      renderIndices = indices; wasRendering = true;
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
    model: str = "",
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
    _ALLOWED_MODELS = {"claude-sonnet-4-5", "claude-opus-4-8", "claude-fable-5"}
    model = (body.get("model") or "").strip()
    if model and model not in _ALLOWED_MODELS:
        model = ""
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
        target=_run_analyze_job, args=(holder, url, transcript_text, force, model), daemon=True
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

    return render_template_string(
        CANDIDATES_TEMPLATE,
        video_id=video_id,
        status=status,
        status_message=job.get("message", "처리 중..."),
        pct=pct,
        clips=clips,
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
    indices = request.get_json().get("indices", [])
    if not indices:
        return jsonify({"error": "선택된 항목이 없습니다"}), 400

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

    hook_text = ""
    if clip is not None and clip.caption_overrides:
        hook_text = (clip.caption_overrides[0] or {}).get("text", "")

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


def _compute_layout(cfg: dict, clip, source_resolution: tuple[int, int]) -> dict:
    """위치 편집 화면에 필요한 좌표들을 render.py/captions.py와 동일한 공식으로 계산한다.
    이 값이 실제 렌더링(render_clip)과 어긋나면 편집 화면에서 본 위치와 실제 결과물의
    위치가 달라지므로, 반드시 같은 헬퍼 함수(_compute_card_video_box_*, compute_card_margins)
    를 재사용한다."""
    from src.captions import _fit_title_font_size, compute_card_margins
    from src.render import _compute_card_video_box_height, _compute_card_video_box_y

    render_cfg = cfg["render"]
    captions_cfg = cfg["captions"]
    resolution = tuple(render_cfg.get("resolution", [1080, 1920]))
    card = render_cfg["card_layout"]

    vbw = card["video_box_width"]
    vbh = _compute_card_video_box_height(card, source_resolution)
    vby = _compute_card_video_box_y(card, resolution, vbh)
    vbx = (resolution[0] - vbw) // 2
    card_layout = {**card, "video_box_height": vbh, "video_box_y": vby}

    max_title_size = captions_cfg.get("title_font_size") or int(captions_cfg["font_size"] * 1.3)
    title_size = _fit_title_font_size(
        clip.title or "", max_title_size, min_size=captions_cfg["font_size"],
        available_width_px=resolution[0] - 80,
    )
    base_title_margin_v, base_caption_margin_v = compute_card_margins(card_layout, resolution, title_size)

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
    }


def _caption_lines_for_clip(video_id: str, clip, cfg: dict) -> list[dict]:
    """자막 편집기에 채워 넣을 자막 라인 목록을 만든다.
    이미 편집·저장된 caption_overrides가 있으면 그걸 쓰고, 없으면 원본 전사에서 클립
    구간 단어를 뽑아 max_words_per_line 단위로 잘라 라인({start,end,text})으로 만든다."""
    if getattr(clip, "caption_overrides", None):
        return [
            {"start": float(o["start"]), "end": float(o["end"]), "text": str(o.get("text", ""))}
            for o in clip.caption_overrides
        ]
    from src.captions import _collect_words_in_range, chunk_words_into_lines

    transcript_path = OUTPUT_ROOT / video_id / "transcript.json"
    if not transcript_path.exists():
        return []
    from src.main import json_load_transcript

    segs = json_load_transcript(transcript_path)["segments"]
    words = _collect_words_in_range(segs, clip.start, clip.end)  # 롤링 중복 제거됨
    max_wpl = cfg["captions"].get("max_words_per_line", 4)
    # 렌더(build_ass)와 같은 '화면 1줄 폭' 규칙으로 잘라, 편집기에서 본 줄이 실제 자막과 일치하게.
    res_w = (cfg.get("render", {}).get("resolution") or [1080, 1920])[0]
    max_units = max(4.0, (res_w - 104) / max(1, cfg["captions"].get("font_size", 72)))
    lines = chunk_words_into_lines(words, max_wpl, max_units=max_units)
    out = [
        {"start": ln.start, "end": ln.end, "text": " ".join(w.text for w in ln.words)}
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


EDIT_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>위치 편집 - {{ video_id }}</title>
__BASE_STYLE__
<style>
  .canvas-wrap { display: flex; justify-content: center; margin: 24px 0; }
  .canvas {
    position: relative; width: {{ layout.canvas_w }}px; height: {{ layout.canvas_h }}px;
    background: #f7f8fa; border-radius: 20px; box-shadow: var(--shadow); overflow: hidden;
  }
  .video-box { position: absolute; background: #000; overflow: hidden; }
  .video-box img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .drag-box {
    position: absolute; cursor: grab; touch-action: none; user-select: none;
    transform: translate(-50%, 0); text-align: center; white-space: nowrap;
    padding: 4px 10px; border-radius: 8px; border: 1.5px dashed transparent;
  }
  .drag-box:hover, .drag-box.dragging { border-color: var(--accent); background: rgba(49,130,246,0.08); }
  .drag-box.title {
    font-weight: 800; color: #191f28;
    /* 렌더(libass)처럼 긴 제목은 2줄로 자연 줄바꿈(1.4배 부스트와 일치). nowrap이면
       fitToWidth가 한 줄로 다시 쪼그라뜨려 미리보기가 실제보다 작아 보인다. */
    white-space: normal; word-break: keep-all;
    max-width: {{ ((layout.resolution[0] - 80) * layout.scale)|round|int }}px;
  }
  .drag-box.caption { font-weight: 700; color: #191f28; }
  .hint { color: var(--text-muted); font-size: 13px; text-align: center; margin-top: 4px; }
  .btn-row { display: flex; gap: 10px; margin-top: 20px; }
  .btn-row button, .btn-row a {
    flex: 1; text-align: center; padding: 14px; border-radius: 14px; font-size: 14px; font-weight: 600;
    font-family: inherit; border: none; cursor: pointer; text-decoration: none;
  }
  #resetBtn { background: var(--border); color: var(--text); }
  #saveBtn { background: var(--accent); color: #fff; }
  #saveStatus { text-align: center; color: var(--text-muted); font-size: 13px; margin-top: 10px; min-height: 16px; }
  .cap-editor { margin-top: 28px; }
  .cap-editor h2 { font-size: 17px; font-weight: 700; margin: 0 0 4px; letter-spacing: -0.01em; }
  .cap-sub { color: var(--text-muted); font-size: 13px; margin: 0 0 14px; }
  .cap-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .cap-time {
    flex-shrink: 0; width: 52px; text-align: right; font-size: 12px; font-weight: 600;
    color: var(--text-faint); font-variant-numeric: tabular-nums;
  }
  .cap-input {
    flex: 1; min-width: 0; padding: 10px 12px; font-size: 14px; font-family: inherit;
    border: 1.5px solid var(--border); border-radius: 10px; background: #fafbfc;
    transition: border-color .15s, background .15s;
  }
  .cap-input:focus { outline: none; border-color: var(--accent); background: #fff; }
  .cap-empty { color: var(--text-faint); font-size: 13px; }
  .title-edit { margin: 0 0 22px; }
  .title-edit label { display: block; font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
  .title-edit input {
    width: 100%; padding: 13px 15px; font-size: 16px; font-weight: 700; font-family: inherit;
    border: 1.5px solid var(--border); border-radius: 12px; background: #fafbfc;
  }
  .title-edit input:focus { outline: none; border-color: var(--accent); background: #fff; }
  .cap-num {
    width: 54px; flex-shrink: 0; padding: 8px 4px; font-size: 12px; text-align: center; font-family: inherit;
    border: 1.5px solid var(--border); border-radius: 8px; background: #fafbfc;
    font-variant-numeric: tabular-nums;
  }
  .cap-num:focus { outline: none; border-color: var(--accent); background: #fff; }
  .cap-del {
    flex-shrink: 0; width: 32px; height: 32px; border: none; border-radius: 8px;
    background: var(--border); color: var(--text-muted); font-size: 18px; cursor: pointer; line-height: 1;
  }
  .cap-del:hover { background: #ffe3e3; color: #e02424; }
  .cap-add {
    margin-top: 8px; padding: 10px 16px; font-size: 13px; font-weight: 600; font-family: inherit;
    color: var(--accent); background: #eef4ff; border: none; border-radius: 10px; cursor: pointer;
  }
  .cap-add:hover { background: #dfeafe; }
  #renderBtn {
    width: 100%; margin-top: 12px; padding: 15px; font-size: 15px; font-weight: 700; font-family: inherit;
    color: #fff; background: var(--accent); border: none; border-radius: 14px; cursor: pointer;
    box-shadow: 0 8px 24px rgba(49,130,246,0.28);
  }
  #renderBtn:hover { background: var(--accent-hover); }
  #renderBtn:disabled { opacity: .5; cursor: not-allowed; box-shadow: none; }
  .hidden { display: none; }
  #renderResult video { width: 100%; max-width: 300px; display: block; margin: 14px auto 0; border-radius: 14px; background: #000; }
  .ed-card { background: var(--card); border-radius: 16px; box-shadow: var(--shadow); padding: 20px 22px; margin-bottom: 14px; }
  .ed-card h2 { font-size: 16px; font-weight: 700; margin: 0 0 10px; letter-spacing: -0.01em; }
  .ed-card .cap-sub { margin-top: 0; }
  .sty-h { font-size: 13px; font-weight: 700; color: var(--text-muted); margin: 16px 0 6px; }
  .sty-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .sty-row > label { font-size: 13px; font-weight: 600; color: var(--text-muted); width: 44px; flex-shrink: 0; }
  .sty-row select { flex: 1; }
  .sty-grid { display: grid; grid-template-columns: 1fr 70px 86px 70px; gap: 8px; }
  .trim-num { width: 88px; }
  .unit { font-size: 13px; color: var(--text-muted); }
  .title-input {
    width: 100%; padding: 13px 15px; font-size: 16px; font-weight: 700; font-family: inherit;
    border: 1.5px solid var(--border); border-radius: 12px; background: #fafbfc;
  }
  .title-input:focus { outline: none; border-color: var(--accent); background: #fff; }
  .sty-grid select, .sty-grid input, .sty-row select, .sty-row input {
    padding: 9px 10px; font-size: 13px; font-family: inherit; border: 1.5px solid var(--border);
    border-radius: 9px; background: #fafbfc; min-width: 0;
  }
  .sty-grid select:focus, .sty-grid input:focus, .sty-row select:focus, .sty-row input:focus { outline: none; border-color: var(--accent); background: #fff; }
  /* 추천 제목 후보 칩 */
  .title-cands { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
  .title-cand {
    text-align: left; padding: 12px 14px; font-size: 14.5px; font-weight: 600; font-family: inherit;
    color: var(--text); background: #fafbfc; border: 1.5px solid var(--border); border-radius: 12px;
    cursor: pointer; transition: border-color .15s, background .15s; line-height: 1.4;
  }
  .title-cand:hover { border-color: var(--accent); background: #f0f6ff; }
  .title-cand.sel { border-color: var(--accent); background: #eef4ff; color: var(--accent); }
  /* 세부 편집 접기 */
  details.advanced { background: transparent; box-shadow: none; padding: 0; margin-bottom: 14px; }
  details.advanced > summary {
    cursor: pointer; list-style: none; padding: 15px 20px; background: var(--card);
    border-radius: 14px; box-shadow: var(--shadow); font-size: 15px; font-weight: 700;
    color: var(--text); display: flex; align-items: center; justify-content: space-between;
  }
  details.advanced > summary::-webkit-details-marker { display: none; }
  details.advanced > summary::after { content: '▾'; color: var(--text-faint); font-size: 13px; transition: transform .2s; }
  details.advanced[open] > summary::after { transform: rotate(180deg); }
  details.advanced > summary .sum-sub { font-size: 12.5px; font-weight: 500; color: var(--text-faint); }
  details.advanced .adv-body { margin-top: 12px; }
</style>
<style>
/* 각 글꼴을 실제 모양으로 미리보기(드롭다운 옵션 + 캔버스). font-display:swap로 늦게 떠도 UI는 안 막힘 */
{% for f in fonts %}@font-face{ font-family:'{{ f.family }}'; src:url('/font/{{ f.file }}'); font-display:swap; }
{% endfor %}
.fontopt option { font-size: 15px; }
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="/video/{{ video_id }}">&larr; 후보 목록</a>
  <h1>제목·자막 위치 편집</h1>
  <p class="subtitle">글자를 드래그해서 원하는 위치로 옮기세요. #{{ idx + 1 }} 클립 ({{ clip.title }})</p>

  <div class="btn-row" style="margin:14px 0 4px">
    <button id="resetBtn" type="button">위치 초기화</button>
    <button id="saveBtn" type="button">저장만</button>
  </div>
  <button id="renderBtn" type="button">저장하고 영상 만들기</button>
  <p id="saveStatus"></p>

  <div class="canvas-wrap">
    <div class="canvas" id="canvas">
      <div class="video-box" id="videoBox" style="
        left: {{ (layout.video_box.x * layout.scale)|round|int }}px;
        top: {{ (layout.video_box.y * layout.scale)|round|int }}px;
        width: {{ (layout.video_box.w * layout.scale)|round|int }}px;
        height: {{ (layout.video_box.h * layout.scale)|round|int }}px;
        border-radius: {{ (layout.video_box.r * layout.scale)|round|int }}px;">
        <img src="/media/{{ video_id }}/preview/{{ idx }}.jpg" alt="미리보기 프레임">
      </div>
      <div class="drag-box title" id="titleBox" style="font-size: {{ (layout.title_size * layout.scale)|round|int }}px;">
        {{ clip.title }}
      </div>
      <div class="drag-box caption" id="captionBox" style="font-size: {{ (layout.caption_font_size * layout.scale)|round|int }}px;">
        {{ caption_preview }}
      </div>
    </div>
  </div>
  <p class="hint">글자를 드래그해 위치를 옮기고, 아래에서 화면·글꼴·자막을 편집하세요. 최종 결과는 렌더링해야 반영됩니다.</p>

  <section class="ed-card">
    <h2>제목</h2>
    <input type="text" id="titleInput" class="title-input" value="{{ clip.title }}">
    {% if clip.title_candidates %}
    <p class="cap-sub" style="margin:12px 0 0">추천 제목 — 누르면 위에 바로 적용돼요</p>
    <div class="title-cands">
      {% for t in clip.title_candidates %}
      <button type="button" class="title-cand">{{ t }}</button>
      {% endfor %}
    </div>
    {% endif %}
  </section>

  <details class="advanced">
    <summary>세부 편집 <span class="sum-sub">화면·글꼴·길이·자막 손보기</span></summary>
    <div class="adv-body">
      <section class="ed-card">
        <h2>영상 길이</h2>
        <p class="cap-sub">클립 시작·끝을 초 단위로 조절합니다. (전체 {{ '%.1f'|format(clip.end - clip.start) }}초) · 시작을 음수로, 끝을 전체보다 크게 하면 원본에서 앞뒤로 <b>최대 10초까지 늘릴</b> 수 있어요.</p>
        <div class="sty-row">
          <label>시작</label>
          <input type="number" id="trimStart" class="trim-num" min="-10" step="0.5"> <span class="unit">초</span>
          <label style="width:auto">끝</label>
          <input type="number" id="trimEnd" class="trim-num" min="0" max="{{ '%.1f'|format(clip.end - clip.start + 10) }}" step="0.5"> <span class="unit">초</span>
        </div>
      </section>

      <section class="ed-card">
        <h2>스타일</h2>
        <div class="sty-row">
          <label>화면</label>
          <select id="fillMode">
            <option value="fit">풀 화면 (유튜브 원본 그대로, 안 잘림)</option>
            <option value="cover">화면 확대 (세로 꽉 채움, 좌우·하단 잘림)</option>
          </select>
        </div>
        <div class="sty-h">제목 — 글꼴 · 크기 · 정렬 · 자간</div>
        <div class="sty-grid">
          <select id="titleFont" class="fontopt">{% for f in fonts %}<option value="{{ f.family }}" style="font-family:'{{ f.family }}',sans-serif">{{ f.name }}</option>{% endfor %}</select>
          <input type="number" id="titleSize" min="20" max="400" step="2" title="크기(px)">
          <select id="titleAlign"><option value="left">왼쪽</option><option value="center">가운데</option><option value="right">오른쪽</option></select>
          <input type="number" id="titleSpacing" step="0.5" title="자간(px)">
        </div>
        <div class="sty-h">자막 — 글꼴 · 크기 · 정렬 · 자간</div>
        <div class="sty-grid">
          <select id="captionFont" class="fontopt">{% for f in fonts %}<option value="{{ f.family }}" style="font-family:'{{ f.family }}',sans-serif">{{ f.name }}</option>{% endfor %}</select>
          <input type="number" id="captionSize" min="20" max="300" step="2" title="크기(px)">
          <select id="captionAlign"><option value="left">왼쪽</option><option value="center">가운데</option><option value="right">오른쪽</option></select>
          <input type="number" id="captionSpacing" step="0.5" title="자간(px)">
        </div>
      </section>

      <section class="ed-card">
        <h2>자막</h2>
        <p class="cap-sub">왼쪽 두 칸은 시작·끝 시간(클립 시작 기준 초). 내용을 고치고 칸을 추가/삭제하세요.</p>
        <div id="capList">
          {% for c in caption_lines %}
          <div class="cap-row">
            <input type="number" class="cap-num cap-start" step="0.1" value="{{ '%.1f'|format(c.start - clip.start) }}">
            <input type="number" class="cap-num cap-end" step="0.1" value="{{ '%.1f'|format(c.end - clip.start) }}">
            <input type="text" class="cap-input" value="{{ c.text }}">
            <button type="button" class="cap-del" title="삭제">&times;</button>
          </div>
          {% endfor %}
        </div>
        <button type="button" class="cap-add" id="capAdd">+ 자막 칸 추가</button>
      </section>
    </div>
  </details>

  <div class="prog-card hidden" id="renderProg" style="margin-top:16px">
    <div class="prog-head">
      <div class="prog-headL"><span class="prog-msg" id="rpMsg">렌더링 준비...</span><span class="prog-eta" id="rpEta">예상 시간 계산 중…</span></div>
      <span class="prog-pct" id="rpPct">0%</span>
    </div>
    <div class="stepper">
      <div class="rail"><div class="rail-fill" id="rpFill"></div></div>
      <div class="step" data-min="0" data-max="50"><span class="dot"></span><span class="lbl">자막 인식</span></div>
      <div class="step" data-min="50" data-max="99"><span class="dot"></span><span class="lbl">렌더링</span></div>
      <div class="step" data-min="99" data-max="100"><span class="dot"></span><span class="lbl">완성</span></div>
    </div>
  </div>
  <div id="renderResult"></div>
</div>

<script>
const SCALE = {{ layout.scale }};
const state = {
  title: { x: {{ clip.title_offset_x }}, y: {{ clip.title_offset_y }} },
  caption: { x: {{ clip.caption_offset_x }}, y: {{ clip.caption_offset_y }} },
};
const bases = {
  title: { left: {{ layout.resolution[0] / 2 * layout.scale }}, top: {{ layout.title_base_margin_v * layout.scale }} },
  caption: { left: {{ layout.resolution[0] / 2 * layout.scale }}, top: {{ layout.caption_base_margin_v * layout.scale }} },
};

function render(el, key) {
  el.style.left = (bases[key].left + state[key].x * SCALE) + 'px';
  el.style.top = (bases[key].top + state[key].y * SCALE) + 'px';
}

function makeDraggable(el, key) {
  let dragging = false;
  let startX = 0, startY = 0, origX = 0, origY = 0;

  el.addEventListener('pointerdown', (e) => {
    dragging = true;
    el.classList.add('dragging');
    el.setPointerCapture(e.pointerId);
    startX = e.clientX; startY = e.clientY;
    origX = state[key].x; origY = state[key].y;
  });
  el.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    state[key].x = origX + (e.clientX - startX) / SCALE;
    state[key].y = origY + (e.clientY - startY) / SCALE;
    render(el, key);
  });
  const stop = () => { dragging = false; el.classList.remove('dragging'); };
  el.addEventListener('pointerup', stop);
  el.addEventListener('pointercancel', stop);
}

const titleEl = document.getElementById('titleBox');
const captionEl = document.getElementById('captionBox');
const titleInput = document.getElementById('titleInput');
const CLIP_START = {{ clip.start }};
const CLIP_DUR = {{ '%.2f'|format(clip.end - clip.start) }};
const TITLE_BASE_FS = {{ (layout.title_size * layout.scale)|round|int }};

// 실제 렌더(libass)는 제목/자막을 프레임 폭(좌우 여백 40px 제외) 안에 맞춘다. 하지만
// 브라우저는 한글 글리프를 libass보다 넓게 그려서, 같은 폰트 크기라도 미리보기에서만
// 글자가 캔버스를 넘쳐 "과도하게 커" 보였다. 그래서 미리보기도 실제처럼 사용 가능한
// 프레임 폭 안에 들어가도록 폰트를 축소해, 결과물과 시각적으로 일치시킨다.
const USABLE_W = ({{ layout.resolution[0] }} - 80) * SCALE;
function fitToWidth(el, maxPx) {
  let fs = parseFloat(getComputedStyle(el).fontSize);
  let guard = 0;
  while (el.scrollWidth > maxPx && fs > 5 && guard < 300) { fs -= 0.5; el.style.fontSize = fs + 'px'; guard++; }
}
fitToWidth(titleEl, USABLE_W);
fitToWidth(captionEl, USABLE_W);

makeDraggable(titleEl, 'title');
makeDraggable(captionEl, 'caption');
render(titleEl, 'title');
render(captionEl, 'caption');

// 제목 편집 → 미리보기 즉시 반영(폰트도 다시 맞춤)
titleInput.addEventListener('input', () => {
  titleEl.textContent = titleInput.value || ' ';
  titleEl.style.fontSize = TITLE_BASE_FS + 'px';
  fitToWidth(titleEl, USABLE_W);
});

// 추천 제목 후보: 누르면 제목 칸에 적용 + 미리보기 갱신.
document.querySelectorAll('.title-cand').forEach(function(chip) {
  chip.addEventListener('click', function() {
    titleInput.value = chip.textContent.trim();
    titleInput.dispatchEvent(new Event('input'));
    document.querySelectorAll('.title-cand').forEach(x => x.classList.remove('sel'));
    chip.classList.add('sel');
  });
});

// 자막 칸 추가/삭제
const capList = document.getElementById('capList');
function addRow(startRel, endRel, text) {
  const row = document.createElement('div');
  row.className = 'cap-row';
  row.innerHTML =
    '<input type="number" class="cap-num cap-start" step="0.1">' +
    '<input type="number" class="cap-num cap-end" step="0.1">' +
    '<input type="text" class="cap-input">' +
    '<button type="button" class="cap-del" title="삭제">&times;</button>';
  row.querySelector('.cap-start').value = startRel.toFixed(1);
  row.querySelector('.cap-end').value = endRel.toFixed(1);
  row.querySelector('.cap-input').value = text || '';
  capList.appendChild(row);
}
document.getElementById('capAdd').addEventListener('click', () => {
  const rows = capList.querySelectorAll('.cap-row');
  let s = 0;
  if (rows.length) s = parseFloat(rows[rows.length - 1].querySelector('.cap-end').value) || 0;
  addRow(s, s + 2, '');
});
capList.addEventListener('click', (e) => {
  if (e.target.classList.contains('cap-del')) e.target.closest('.cap-row').remove();
});

function collectCaptions() {
  return [...capList.querySelectorAll('.cap-row')].map(function(r) {
    const s = parseFloat(r.querySelector('.cap-start').value);
    const en = parseFloat(r.querySelector('.cap-end').value);
    return { start: CLIP_START + (isNaN(s) ? 0 : s), end: CLIP_START + (isNaN(en) ? 0 : en),
             text: r.querySelector('.cap-input').value };
  });
}
// 스타일 컨트롤 초기값 세팅(클립에 저장된 값 우선, 없으면 config 기본값)
const CUR = {
  fill_mode: '{{ clip.fill_mode or defaults.fill_mode }}',
  title_font: '{{ clip.title_font or defaults.title_font }}',
  title_size: {{ clip.title_size or defaults.title_size }},
  title_align: '{{ clip.title_align or "center" }}',
  title_spacing: {{ clip.title_spacing or 0 }},
  caption_font: '{{ clip.caption_font or defaults.caption_font }}',
  caption_size: {{ clip.caption_size or defaults.caption_size }},
  caption_align: '{{ clip.caption_align or "center" }}',
  caption_spacing: {{ clip.caption_spacing or 0 }},
};
function setVal(id, v) { const el = document.getElementById(id); if (el != null) el.value = v; }
setVal('fillMode', CUR.fill_mode);
setVal('titleFont', CUR.title_font); setVal('titleSize', CUR.title_size);
setVal('titleAlign', CUR.title_align); setVal('titleSpacing', CUR.title_spacing);
setVal('captionFont', CUR.caption_font); setVal('captionSize', CUR.caption_size);
setVal('captionAlign', CUR.caption_align); setVal('captionSpacing', CUR.caption_spacing);
setVal('trimStart', '0'); setVal('trimEnd', CLIP_DUR.toFixed(1));
function val(id) { const el = document.getElementById(id); return el ? el.value : ''; }

// 선택한 글꼴을 캔버스 미리보기에 즉시 반영(@font-face로 로드된 실제 글꼴)
function applyFontPreview() {
  if (titleEl && val('titleFont')) titleEl.style.fontFamily = "'" + val('titleFont') + "', sans-serif";
  if (captionEl && val('captionFont')) captionEl.style.fontFamily = "'" + val('captionFont') + "', sans-serif";
  titleEl.style.fontSize = TITLE_BASE_FS + 'px';
  fitToWidth(titleEl, USABLE_W);
}
document.getElementById('titleFont').addEventListener('change', applyFontPreview);
document.getElementById('captionFont').addEventListener('change', applyFontPreview);
document.fonts && document.fonts.ready.then(applyFontPreview);
applyFontPreview();

function payload() {
  const p = {
    title: titleInput.value,
    title_offset_x: state.title.x, title_offset_y: state.title.y,
    caption_offset_x: state.caption.x, caption_offset_y: state.caption.y,
    captions: collectCaptions(),
    fill_mode: val('fillMode'),
    title_font: val('titleFont'), title_size: val('titleSize'),
    title_align: val('titleAlign'), title_spacing: val('titleSpacing'),
    caption_font: val('captionFont'), caption_size: val('captionSize'),
    caption_align: val('captionAlign'), caption_spacing: val('captionSpacing'),
  };
  // 영상 길이를 실제로 조절했을 때만 전송(안 건드리면 문장 끝 자동 확장 유지).
  const ts = parseFloat(val('trimStart')) || 0;
  const te = parseFloat(val('trimEnd'));
  if (Math.abs(ts - 0) > 0.05 || Math.abs((isNaN(te) ? CLIP_DUR : te) - CLIP_DUR) > 0.05) {
    p.clip_start = CLIP_START + ts;
    p.clip_end = CLIP_START + (isNaN(te) ? CLIP_DUR : te);
  }
  return p;
}
async function save() {
  const res = await fetch('/video/{{ video_id }}/clip/{{ idx }}/position', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload()),
  });
  return res.ok;
}

const statusEl = document.getElementById('saveStatus');
document.getElementById('resetBtn').addEventListener('click', () => {
  state.title = { x: 0, y: 0 };
  state.caption = { x: 0, y: 0 };
  render(titleEl, 'title');
  render(captionEl, 'caption');
});
document.getElementById('saveBtn').addEventListener('click', async () => {
  statusEl.innerText = '저장 중...';
  statusEl.innerText = (await save()) ? '저장됨. "저장하고 영상 만들기"로 렌더링하세요.' : '저장 실패';
});

// 저장하고 영상 만들기 (+ 진행바)
const prog = document.getElementById('renderProg');
const rpFill = document.getElementById('rpFill');
const rpMsg = document.getElementById('rpMsg');
const rpPct = document.getElementById('rpPct');
const steps = [...prog.querySelectorAll('.step')];
function paint(pct, msg) {
  pct = Math.max(0, Math.min(100, pct || 0));
  rpFill.style.width = pct + '%';
  rpPct.textContent = Math.round(pct) + '%';
  if (msg) rpMsg.textContent = msg;
  steps.forEach(function(s) {
    var mn = parseFloat(s.dataset.min), mx = parseFloat(s.dataset.max);
    s.classList.remove('active', 'done');
    if (pct >= mx) s.classList.add('done'); else if (pct >= mn) s.classList.add('active');
  });
}
document.getElementById('renderBtn').addEventListener('click', async () => {
  const btn = document.getElementById('renderBtn');
  statusEl.innerText = '저장 중...';
  if (!(await save())) { statusEl.innerText = '저장 실패'; return; }
  statusEl.innerText = '';
  btn.disabled = true;
  prog.classList.remove('hidden');
  document.getElementById('renderResult').innerHTML = '';
  paint(0, '렌더링 시작...');
  const r = await fetch('/video/{{ video_id }}/render', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ indices: [{{ idx }}] }),
  });
  if (!r.ok) { paint(0, '렌더 요청 실패'); btn.disabled = false; return; }
  var rpEta = document.getElementById('rpEta');
  function fmtEta(s) {
    if (s == null || s < 0) return '';
    s = Math.round(s);
    if (s < 60) return '약 ' + Math.max(1, s) + '초 남음';
    return '약 ' + Math.round(s / 60) + '분 남음';
  }
  function poll() {
    fetch('/video/{{ video_id }}/status').then(x => x.json()).then(function(j) {
      paint(j.render_pct, j.render_message);
      if (rpEta) rpEta.textContent = fmtEta(j.render_eta_seconds) || '예상 시간 계산 중…';
      if (j.render_error) { rpMsg.textContent = '오류: ' + j.render_error; btn.disabled = false; return; }
      if (!j.rendering) {
        paint(100, '완성');
        if (rpEta) rpEta.textContent = '거의 완료…';
        document.getElementById('renderResult').innerHTML =
          '<video controls src="/media/{{ video_id }}/{{ idx + 1 }}.mp4?t=' + Date.now() + '"></video>';
        btn.disabled = false; return;
      }
      setTimeout(poll, 650);
    }).catch(function() { setTimeout(poll, 1200); });
  }
  setTimeout(poll, 800);
});
</script>
</body>
</html>
""".replace("__BASE_STYLE__", BASE_STYLE)


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
    source_res = _probe_resolution(OUTPUT_ROOT / video_id / "source.mp4")
    layout = _compute_layout(cfg, clip, source_res)
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
        s = max(0.0, float(body["clip_start"]))  # 시작은 0(영상 맨 앞) 밑으로 못 내림
        e = float(body["clip_end"])
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
    clip.caption_offset_x = float(body.get("caption_offset_x", clip.caption_offset_x))
    clip.caption_offset_y = float(body.get("caption_offset_y", clip.caption_offset_y))
    # 자막 편집기에서 확정한 라인들(텍스트가 남아있는 것만). 저장되면 다음 렌더는 재전사 없이
    # 이 자막을 그대로 쓴다. 넘어오지 않으면(위치만 저장) 기존 caption_overrides를 유지한다.
    if "captions" in body:
        clip.caption_overrides = [
            {"start": float(c["start"]), "end": float(c["end"]), "text": str(c.get("text", "")).strip()}
            for c in body["captions"]
            if str(c.get("text", "")).strip()
        ]
    # 화면모드 + 제목/자막 글꼴 스타일(편집기에서 선택). 빈 값이면 config 기본값 사용.
    if "fill_mode" in body:
        clip.fill_mode = str(body.get("fill_mode", "") or "")
    for k in ("title_font", "title_align", "caption_font", "caption_align"):
        if k in body:
            setattr(clip, k, str(body.get(k, "") or ""))
    for k in ("title_size", "caption_size"):
        if k in body:
            setattr(clip, k, int(float(body.get(k) or 0)))
    for k in ("title_spacing", "caption_spacing"):
        if k in body:
            setattr(clip, k, float(body.get(k) or 0))
    save_clips_json(clips, clips_path)
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


@app.route("/media/<video_id>/source.mp4")
def serve_source_video(video_id: str):
    """'만들기 전 확인' 팝업의 <video>용 원본 서빙. conditional=True로 HTTP Range를 지원해
    아이폰식 트림 핸들을 끌 때 브라우저가 필요한 구간만 받아 즉시 탐색된다."""
    p = (OUTPUT_ROOT / video_id / "source.mp4").resolve()
    if not p.exists():
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

    from src.fonts import get_font_registry
    from src.render import _probe_resolution

    cfg = _load_config()
    layout = _compute_layout(cfg, clip, _probe_resolution(OUTPUT_ROOT / video_id / "source.mp4"))

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
            "caption_offset_x": clip.caption_offset_x,
            "caption_offset_y": clip.caption_offset_y,
            "fill_mode": (getattr(clip, "fill_mode", "") or cfg["render"]["card_layout"].get("fill_mode", "fit")),
        },
        "caption_preview": _preview_caption_text(video_id, clip),
        "source_duration": duration,
        "title_font": font_entry(clip.title_font or cfg["captions"].get("title_font_family", "")),
        "caption_font": font_entry(clip.caption_font or cfg["captions"].get("font_family", "")),
    })


# '만들기 전 확인' 팝업 스크립트. CANDIDATES_TEMPLATE는 f-string이라 중괄호를 전부 이스케이프
# 해야 해서, JS는 별도 상수로 두고 라우트로 서빙한다(브레이스 지옥 방지 + 캐시 가능).
PREVIEW_MODAL_JS = r"""
(function () {
  'use strict';
  const VIDEO_ID = location.pathname.split('/')[2];

  // 만들기 흐름: 선택한 클립들을 순서대로 팝업 확인 → 모두 확인되면 onAllConfirmed() 실행.
  window.__previewFlow = function (indices, onAllConfirmed) {
    let i = 0;
    const next = () => {
      if (i >= indices.length) { onAllConfirmed(); return; }
      const idx = indices[i];
      i += 1;
      openModal(idx, i, indices.length, next);
    };
    next();
  };

  const CSS = `
  .pv-backdrop { position: fixed; inset: 0; background: rgba(15,23,42,.55); z-index: 1000;
    display: flex; align-items: center; justify-content: center; padding: 16px;
    animation: pvFade .18s ease; }
  @keyframes pvFade { from { opacity: 0; } to { opacity: 1; } }
  .pv-card { background: var(--card, #fff); border-radius: 22px; box-shadow: 0 24px 80px rgba(15,23,42,.35);
    width: min(420px, 96vw); max-height: 94vh; overflow-y: auto; padding: 18px 18px 16px;
    animation: pvUp .22s cubic-bezier(.22,.61,.36,1); }
  @keyframes pvUp { from { opacity: 0; transform: translateY(14px) scale(.98); } to { opacity: 1; transform: none; } }
  .pv-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
  .pv-head b { font-size: 16px; letter-spacing: -0.01em; }
  .pv-x { cursor: pointer; border: none; background: none; font-size: 22px; color: #8b95a1; line-height: 1; padding: 2px 6px; }
  .pv-head-r { display: flex; align-items: center; gap: 6px; }
  .pv-reanalyze { cursor: pointer; border: 1.5px solid #f0f1f3; background: #fafbfc; color: #3182f6;
    font-size: 12.5px; font-weight: 700; font-family: inherit; border-radius: 9px; padding: 6px 10px; white-space: nowrap; }
  .pv-reanalyze:hover { border-color: #3182f6; background: #f0f6ff; }
  .pv-reanalyze:disabled { opacity: .6; cursor: default; }
  .pv-sub { font-size: 12.5px; color: #8b95a1; margin: 0 0 10px; }
  .pv-canvas-wrap { display: flex; justify-content: center; }
  .pv-canvas { position: relative; background: #f1f3f5; border-radius: 14px; overflow: hidden; flex-shrink: 0; }
  .pv-vbox { position: absolute; background: #000; overflow: hidden; }
  .pv-vbox video { width: 100%; height: 100%; display: block; }
  .pv-drag { position: absolute; cursor: grab; touch-action: none; user-select: none;
    transform: translate(-50%, 0); text-align: center; padding: 3px 8px; border-radius: 7px;
    border: 1.5px dashed transparent; color: #191f28; text-shadow: 0 0 6px rgba(255,255,255,.85);
    font-weight: 800; white-space: normal; word-break: keep-all; width: max-content; }
  .pv-drag:hover, .pv-drag.dragging { border-color: #3182f6; background: rgba(49,130,246,.18); }
  .pv-play { position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%);
    width: 54px; height: 54px; border-radius: 50%; border: none; cursor: pointer;
    background: rgba(15,23,42,.55); color: #fff; font-size: 22px; display: flex;
    align-items: center; justify-content: center; backdrop-filter: blur(2px); }
  .pv-play.hidden { display: none; }
  /* NLE식 타임라인(휠 확대 · 분할 · 구간 삭제) */
  .pv-trimwrap { margin: 14px 2px 2px; }
  .pv-trim { position: relative; height: 60px; touch-action: none; user-select: none; }
  .pv-strip { position: absolute; inset: 0; display: flex; border-radius: 10px; overflow: hidden; background: #dee2e6; }
  .pv-strip img { flex: 1; min-width: 0; object-fit: cover; height: 100%; display: block; pointer-events: none; }
  .pv-segs { position: absolute; inset: 0; pointer-events: none; }
  .pv-seg { position: absolute; top: 0; bottom: 0; box-sizing: border-box; pointer-events: auto;
    border: 3px solid #f7c325; border-radius: 8px; background: rgba(247,195,37,.14); cursor: grab; }
  .pv-seg.active { border-color: #f5a623; box-shadow: 0 0 0 2px rgba(245,166,35,.30); background: rgba(247,195,37,.22); }
  .pv-seg .h { position: absolute; top: -4px; bottom: -4px; width: 22px; cursor: ew-resize; }
  .pv-seg .hL { left: -12px; } .pv-seg .hR { right: -12px; }
  .pv-seg .h::after { content: ''; position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    width: 4px; height: 22px; border-radius: 2px; background: #b8860b; }
  .pv-seg .del { position: absolute; top: -10px; right: -10px; width: 21px; height: 21px; border-radius: 50%;
    border: none; background: #e02424; color: #fff; font-size: 13px; line-height: 1; cursor: pointer;
    display: none; align-items: center; justify-content: center; box-shadow: 0 1px 4px rgba(0,0,0,.25); }
  .pv-trim.pv-multi .pv-seg .del { display: flex; }
  .pv-play-head { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #3182f6; pointer-events: none; }
  .pv-times { display: flex; justify-content: space-between; font-size: 12px; color: #6b7684;
    font-variant-numeric: tabular-nums; margin-top: 8px; }
  .pv-times b { color: #191f28; }
  .pv-tools { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
  .pv-tool { padding: 7px 11px; font-size: 12.5px; font-weight: 700; font-family: inherit;
    border: 1.5px solid #f0f1f3; background: #fafbfc; color: #191f28; border-radius: 9px; cursor: pointer; }
  .pv-tool:hover { border-color: #3182f6; color: #3182f6; }
  .pv-cands { display: flex; flex-direction: column; gap: 7px; margin-top: 12px; }
  .pv-cand { text-align: left; padding: 10px 12px; font-size: 13.5px; font-weight: 600; font-family: inherit;
    color: #191f28; background: #fafbfc; border: 1.5px solid #f0f1f3; border-radius: 11px;
    cursor: pointer; line-height: 1.35; }
  .pv-cand:hover { border-color: #3182f6; background: #f0f6ff; }
  .pv-cand.sel { border-color: #3182f6; background: #eef4ff; color: #1b64da; }
  .pv-foot { display: flex; gap: 9px; margin-top: 14px; align-items: center; }
  .pv-cancel { flex: 0 0 auto; padding: 13px 16px; border: none; border-radius: 12px; background: #f0f1f3;
    color: #191f28; font-weight: 600; font-size: 14px; font-family: inherit; cursor: pointer; }
  .pv-ok { flex: 1; padding: 13px; border: none; border-radius: 12px; background: #3182f6; color: #fff;
    font-weight: 700; font-size: 14.5px; font-family: inherit; cursor: pointer;
    box-shadow: 0 8px 24px rgba(49,130,246,.28); }
  .pv-ok:hover { background: #1b64da; }
  .pv-edit-link { display: block; text-align: center; margin-top: 10px; font-size: 12.5px; color: #8b95a1; text-decoration: none; }
  .pv-edit-link:hover { color: #3182f6; }
  `;

  function injectOnce(id, cssText) {
    if (document.getElementById(id)) return;
    const s = document.createElement('style');
    s.id = id; s.textContent = cssText;
    document.head.appendChild(s);
  }
  async function openModal(idx, seq, total, onConfirm) {
    let info;
    try {
      const r = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/preview_info');
      if (!r.ok) throw new Error('bad');
      info = await r.json();
    } catch (e) { alert('미리보기 정보를 불러오지 못했어요'); return; }

    injectOnce('pv-style', CSS);
    // 실제 렌더 글꼴을 팝업에서도 그대로 보여준다.
    let faceCss = '';
    for (const f of [info.title_font, info.caption_font]) {
      if (f) faceCss += "@font-face{font-family:'" + f.family + "';src:url('/font/" + f.file + "');font-display:swap}\n";
    }
    if (faceCss) injectOnce('pv-fonts-' + idx, faceCss);

    const L = info.layout, C = info.clip;
    // 팝업엔 트림바·후보·버튼까지 들어가므로 편집 캔버스(360px)를 화면 높이에 맞춰 줄인다.
    const MS = Math.max(0.5, Math.min(0.72, (window.innerHeight - 430) / L.canvas_h));
    const SC = L.scale * MS;               // 렌더 px -> 팝업 px
    const W = Math.round(L.canvas_w * MS), H = Math.round(L.canvas_h * MS);

    const back = document.createElement('div');
    back.className = 'pv-backdrop';
    back.innerHTML =
      '<div class="pv-card">' +
      '  <div class="pv-head"><b>만들기 전 확인' + (total > 1 ? ' (' + seq + '/' + total + ')' : '') + '</b>' +
      '    <div class="pv-head-r">' +
      '      <button class="pv-reanalyze" title="주제는 그대로 두고 이 장면의 시작·끝만 다시 잡아 새 후보로 추가합니다(원본 유지)">↻ 구간 재분석</button>' +
      '      <button class="pv-x" title="취소">&times;</button>' +
      '    </div></div>' +
      '  <p class="pv-sub">첫 화면 미리보기예요. 제목·자막을 드래그해 옮기고, 노란 핸들로 구간을 다듬으세요.</p>' +
      '  <div class="pv-canvas-wrap"><div class="pv-canvas" style="width:' + W + 'px;height:' + H + 'px">' +
      '    <div class="pv-vbox"><video playsinline preload="metadata"></video></div>' +
      '    <div class="pv-drag pv-title"></div>' +
      '    <div class="pv-drag pv-caption"></div>' +
      '    <button class="pv-play">▶</button>' +
      '  </div></div>' +
      '  <div class="pv-trimwrap">' +
      '    <div class="pv-trim">' +
      '      <div class="pv-strip"></div>' +
      '      <div class="pv-segs"></div>' +
      '      <div class="pv-play-head" style="display:none"></div>' +
      '    </div>' +
      '    <div class="pv-times"><span>시작 <b class="pv-t0"></b></span><span class="pv-dur"></span><span>끝 <b class="pv-t1"></b></span></div>' +
      '    <div class="pv-tools">' +
      '      <button type="button" class="pv-tool pv-split">✂ 재생 위치서 분할</button>' +
      '      <button type="button" class="pv-tool pv-zoomout">− 축소</button>' +
      '      <button type="button" class="pv-tool pv-zoomin">+ 확대</button>' +
      '    </div>' +
      '  </div>' +
      '  <div class="pv-cands"></div>' +
      '  <div class="pv-foot"><button class="pv-cancel">취소</button><button class="pv-ok">이 설정으로 만들기</button></div>' +
      '  <a class="pv-edit-link" href="/video/' + VIDEO_ID + '/clip/' + idx + '/edit">자막 내용·글꼴까지 바꾸려면 상세 편집 →</a>' +
      '</div>';
    document.body.appendChild(back);
    const $ = (sel) => back.querySelector(sel);

    // ── 비디오 박스(레이아웃 좌표 그대로 축소) ──
    const vbox = $('.pv-vbox');
    // video_box는 렌더 해상도(px) 좌표 → 팝업 px로는 SC(=layout.scale × 팝업 축소율)를 곱한다.
    vbox.style.left = Math.round(L.video_box.x * SC) + 'px';
    vbox.style.top = Math.round(L.video_box.y * SC) + 'px';
    vbox.style.width = Math.round(L.video_box.w * SC) + 'px';
    vbox.style.height = Math.round(L.video_box.h * SC) + 'px';
    vbox.style.borderRadius = Math.round(L.video_box.r * SC) + 'px';
    const video = $('video');
    video.src = '/media/' + VIDEO_ID + '/source.mp4';
    video.style.objectFit = (C.fill_mode === 'cover') ? 'cover' : 'contain';

    // ── NLE식 타임라인: 휠 확대 · 분할 · 조각 삭제 ──
    const dur = info.source_duration;
    const trimEl = $('.pv-trim'), strip = $('.pv-strip'), segsLayer = $('.pv-segs'), headEl = $('.pv-play-head');
    const playBtn = $('.pv-play');
    // 남길 구간(절대초). 이전에 분할·저장했으면 복원, 아니면 클립 전체 한 조각.
    let segs = (C.keep_ranges && C.keep_ranges.length)
      ? C.keep_ranges.map((r) => ({ s: r[0], e: r[1] }))
      : [{ s: C.start, e: C.end }];
    let view = { a: 0, b: dur };   // 보이는 시간창(확대/축소). 처음엔 설교 전체.
    let activeSeg = 0;

    const fmt = (t) => Math.floor(t / 60) + ':' + String(Math.floor(t % 60)).padStart(2, '0');
    const t2x = (t) => (t - view.a) / (view.b - view.a) * trimEl.clientWidth;
    const x2t = (px) => view.a + (px / trimEl.clientWidth) * (view.b - view.a);
    function clampView() {
      const minSpan = Math.min(dur, 3);  // 최대 확대는 3초 폭까지
      if (view.b - view.a < minSpan) { const c = (view.a + view.b) / 2; view.a = c - minSpan / 2; view.b = c + minSpan / 2; }
      if (view.a < 0) { view.b -= view.a; view.a = 0; }
      if (view.b > dur) { view.a -= (view.b - dur); view.b = dur; if (view.a < 0) view.a = 0; }
    }

    function renderStrip() {
      strip.innerHTML = '';
      const N = 12;
      for (let k = 0; k < N; k++) {
        const t = view.a + (view.b - view.a) * (k + 0.5) / N;
        const img = document.createElement('img');
        img.src = '/media/' + VIDEO_ID + '/thumb/' + Math.round(t) + '.jpg';
        img.draggable = false; strip.appendChild(img);
      }
    }
    function renderHead() {
      const t = video.currentTime;
      if (t >= view.a && t <= view.b) { headEl.style.display = 'block'; headEl.style.left = t2x(t) + 'px'; }
      else headEl.style.display = 'none';
    }
    function renderSegs() {
      segsLayer.innerHTML = '';
      const W = trimEl.clientWidth;
      segs.forEach((sg, i) => {
        const x0 = Math.max(0, t2x(sg.s)), x1 = Math.min(W, t2x(sg.e));
        if (x1 <= 0 || x0 >= W) return;  // 확대 시 화면 밖 조각은 안 그림
        const el = document.createElement('div');
        el.className = 'pv-seg' + (i === activeSeg ? ' active' : '');
        el.style.left = x0 + 'px'; el.style.width = Math.max(6, x1 - x0) + 'px';
        el.innerHTML = '<div class="h hL"></div><div class="h hR"></div><button type="button" class="del" title="이 조각 삭제">×</button>';
        segsLayer.appendChild(el);
        bindSeg(el, i);
      });
      trimEl.classList.toggle('pv-multi', segs.length > 1);
      const total = segs.reduce((a, s) => a + (s.e - s.s), 0);
      $('.pv-t0').textContent = fmt(segs[0].s);
      $('.pv-t1').textContent = fmt(segs[segs.length - 1].e);
      $('.pv-dur').textContent = '남는 길이 ' + total.toFixed(1) + '초' + (segs.length > 1 ? (' · ' + segs.length + '조각') : '');
    }
    function redraw() { renderStrip(); renderSegs(); renderHead(); }

    function bindSeg(el, i) {
      const Wpx = () => trimEl.clientWidth;
      el.addEventListener('pointerdown', (e) => {
        if (e.target.classList.contains('del')) return;
        e.preventDefault();
        activeSeg = i;
        const isL = e.target.classList.contains('hL'), isR = e.target.classList.contains('hR');
        el.setPointerCapture(e.pointerId);
        const startX = e.clientX, o = { s: segs[i].s, e: segs[i].e };
        const prev = segs[i - 1], next = segs[i + 1];
        const move = (ev) => {
          const dt = (ev.clientX - startX) / Wpx() * (view.b - view.a);
          if (isL) {
            segs[i].s = Math.min(Math.max(prev ? prev.e + 0.1 : 0, o.s + dt), segs[i].e - 0.5);
          } else if (isR) {
            segs[i].e = Math.max(Math.min(next ? next.s - 0.1 : dur, o.e + dt), segs[i].s + 0.5);
          } else {
            const len = o.e - o.s;
            let ns = Math.max(prev ? prev.e + 0.1 : 0, Math.min(o.s + dt, (next ? next.s - 0.1 : dur) - len));
            segs[i].s = ns; segs[i].e = ns + len;
          }
          video.pause(); playBtn.classList.remove('hidden');
          video.currentTime = isR ? segs[i].e : segs[i].s;
          renderSegs(); renderHead();
        };
        const up = () => { el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up); };
        el.addEventListener('pointermove', move); el.addEventListener('pointerup', up);
      });
      el.querySelector('.del').addEventListener('click', (ev) => {
        ev.stopPropagation();
        if (segs.length <= 1) return;   // 최소 한 조각은 남긴다
        segs.splice(i, 1);
        activeSeg = Math.max(0, Math.min(activeSeg, segs.length - 1));
        redraw();
      });
    }

    // 휠로 확대/축소 (커서 위치 중심)
    trimEl.addEventListener('wheel', (e) => {
      e.preventDefault();
      const pivot = x2t(e.clientX - trimEl.getBoundingClientRect().left);
      const f = e.deltaY < 0 ? 0.8 : 1.25;  // 위로 굴리면 확대
      view.a = pivot - (pivot - view.a) * f;
      view.b = pivot + (view.b - pivot) * f;
      clampView(); redraw();
    }, { passive: false });
    // 확대/축소는 '지금 보는 조각'을 중심으로 한다(영상 한가운데로 확대돼 클립이 화면 밖으로
    // 사라지던 문제 수정). 재생 위치가 보이면 그 위치를, 아니면 활성 조각 중앙을 기준으로.
    const focusT = () => {
      const t = video.currentTime;
      if (t >= view.a && t <= view.b) return t;
      const s = segs[activeSeg] || segs[0];
      return (s.s + s.e) / 2;
    };
    $('.pv-zoomin').addEventListener('click', () => { const c = focusT(), h = (view.b - view.a) * 0.4; view.a = c - h; view.b = c + h; clampView(); redraw(); });
    $('.pv-zoomout').addEventListener('click', () => { const c = focusT(), h = (view.b - view.a) * 0.625; view.a = c - h; view.b = c + h; clampView(); redraw(); });

    // 분할: 재생 위치(playhead)가 든 조각을 그 지점에서 둘로 나눔
    $('.pv-split').addEventListener('click', () => {
      const t = video.currentTime;
      for (let i = 0; i < segs.length; i++) {
        if (t > segs[i].s + 0.3 && t < segs[i].e - 0.3) {
          segs.splice(i + 1, 0, { s: t, e: segs[i].e });
          segs[i].e = t; activeSeg = i + 1; redraw(); return;
        }
      }
    });

    // 타임라인 빈 곳 클릭 = 재생 위치 이동
    trimEl.addEventListener('click', (e) => {
      if (e.target.closest('.pv-seg')) return;
      const t = Math.max(0, Math.min(dur, x2t(e.clientX - trimEl.getBoundingClientRect().left)));
      video.currentTime = t; video.pause(); playBtn.classList.remove('hidden'); renderHead();
    });

    video.addEventListener('loadedmetadata', () => { video.currentTime = segs[0].s; });
    redraw();

    // ── 재생: 남긴 조각들만 순서대로 미리듣기(잘린 gap은 건너뜀) ──
    playBtn.addEventListener('click', () => {
      if (video.paused) {
        const inSeg = segs.some((sg) => video.currentTime >= sg.s - 0.05 && video.currentTime < sg.e - 0.05);
        if (!inSeg) video.currentTime = segs[0].s;
        video.play(); playBtn.classList.add('hidden');
      }
    });
    video.addEventListener('click', () => { if (!video.paused) { video.pause(); playBtn.classList.remove('hidden'); } });
    video.addEventListener('timeupdate', () => {
      renderHead();
      const t = video.currentTime;
      for (let i = 0; i < segs.length; i++) { if (t >= segs[i].s - 0.05 && t < segs[i].e) return; }  // 아직 조각 안
      const nxt = segs.find((sg) => sg.s > t);  // gap이면 다음 조각으로 점프
      if (nxt && !video.paused) { video.currentTime = nxt.s; return; }
      video.pause(); video.currentTime = segs[0].s; playBtn.classList.remove('hidden'); renderHead();
    });

    // ── 제목/자막 드래그 (편집 페이지와 같은 좌표계: 렌더 px 오프셋 저장) ──
    const bases = {
      title: { left: L.resolution[0] / 2 * SC, top: L.title_base_margin_v * SC },
      caption: { left: L.resolution[0] / 2 * SC, top: L.caption_base_margin_v * SC },
    };
    const state = {
      title: { x: C.title_offset_x, y: C.title_offset_y },
      caption: { x: C.caption_offset_x, y: C.caption_offset_y },
    };
    const titleEl = $('.pv-title'), capEl = $('.pv-caption');
    titleEl.textContent = C.title;
    capEl.textContent = info.caption_preview;
    titleEl.style.fontSize = Math.round(L.title_size * SC) + 'px';
    capEl.style.fontSize = Math.round(L.caption_font_size * SC) + 'px';
    titleEl.style.maxWidth = Math.round((L.resolution[0] - 80) * SC) + 'px';
    if (info.title_font) titleEl.style.fontFamily = "'" + info.title_font.family + "', sans-serif";
    if (info.caption_font) capEl.style.fontFamily = "'" + info.caption_font.family + "', sans-serif";
    const USABLE_W = (L.resolution[0] - 80) * SC;
    function fitToWidth(el) {
      let fs = parseFloat(getComputedStyle(el).fontSize), guard = 0;
      while (el.scrollWidth > USABLE_W && fs > 5 && guard < 300) { fs -= 0.5; el.style.fontSize = fs + 'px'; guard++; }
    }
    function paintBox(el, key) {
      el.style.left = (bases[key].left + state[key].x * SC) + 'px';
      el.style.top = (bases[key].top + state[key].y * SC) + 'px';
    }
    function makeDraggable(el, key) {
      el.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        el.classList.add('dragging');
        el.setPointerCapture(e.pointerId);
        const sx = e.clientX, sy = e.clientY, ox = state[key].x, oy = state[key].y;
        const move = (ev) => {
          state[key].x = ox + (ev.clientX - sx) / SC;
          state[key].y = oy + (ev.clientY - sy) / SC;
          paintBox(el, key);
        };
        const up = () => {
          el.classList.remove('dragging');
          el.removeEventListener('pointermove', move);
          el.removeEventListener('pointerup', up);
        };
        el.addEventListener('pointermove', move);
        el.addEventListener('pointerup', up);
      });
    }
    fitToWidth(titleEl); fitToWidth(capEl);
    paintBox(titleEl, 'title'); paintBox(capEl, 'caption');
    makeDraggable(titleEl, 'title'); makeDraggable(capEl, 'caption');
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => {
      titleEl.style.fontSize = Math.round(L.title_size * SC) + 'px';
      fitToWidth(titleEl);
    });

    // ── 제목 후보 5개 ──
    const candsBox = $('.pv-cands');
    let chosenTitle = C.title;
    (C.title_candidates || []).forEach((t, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'pv-cand' + (t === chosenTitle ? ' sel' : '');
      b.textContent = t;
      b.addEventListener('click', () => {
        chosenTitle = t;
        candsBox.querySelectorAll('.pv-cand').forEach((x) => x.classList.remove('sel'));
        b.classList.add('sel');
        titleEl.textContent = t;
        titleEl.style.fontSize = Math.round(L.title_size * SC) + 'px';
        fitToWidth(titleEl);
      });
      candsBox.appendChild(b);
    });

    // ── 구간 재분석: 주제는 그대로, 이 장면 시작·끝만 다시 잡아 새 후보로 추가(원본 유지) ──
    const reBtn = $('.pv-reanalyze');
    reBtn.addEventListener('click', async () => {
      reBtn.disabled = true; reBtn.textContent = '재분석 중…';
      let started = null;
      try { started = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/reanalyze', { method: 'POST' }); }
      catch (e) { started = null; }
      if (!started || !started.ok) {
        const d = started ? await started.json().catch(() => ({})) : {};
        alert('재분석 시작 실패: ' + (d.error || '네트워크 오류'));
        reBtn.disabled = false; reBtn.textContent = '↻ 구간 재분석'; return;
      }
      const poll = () => {
        fetch('/video/' + VIDEO_ID + '/reanalyze_status').then((r) => r.json()).then((j) => {
          if (j.running) { reBtn.textContent = '재분석 중… ' + Math.round((j.pct || 0) * 100) + '%'; setTimeout(poll, 1000); return; }
          if (j.error) { alert('재분석 실패: ' + j.error); reBtn.disabled = false; reBtn.textContent = '↻ 구간 재분석'; return; }
          alert('구간 재분석 완료 — 새 후보를 원본 바로 아래에 추가했어요(원본은 그대로).');
          location.reload();
        }).catch(() => setTimeout(poll, 1500));
      };
      poll();
    });

    // ── 닫기/확정 ──
    function close() { video.pause(); back.remove(); }
    $('.pv-x').addEventListener('click', close);
    $('.pv-cancel').addEventListener('click', close);
    $('.pv-ok').addEventListener('click', async () => {
      const payload = {
        title: chosenTitle,
        title_offset_x: state.title.x, title_offset_y: state.title.y,
        caption_offset_x: state.caption.x, caption_offset_y: state.caption.y,
      };
      const outS = segs[0].s, outE = segs[segs.length - 1].e;
      const changed = segs.length > 1 || Math.abs(outS - C.start) > 0.05 || Math.abs(outE - C.end) > 0.05;
      if (changed) {
        payload.clip_start = outS; payload.clip_end = outE;
        payload.keep_ranges = segs.map((sg) => [sg.s, sg.e]);
        // 구간을 바꾸면 이전에 저장된 자막 타임스탬프는 무효 → 비워서 렌더 때 재전사 유도.
        payload.captions = [];
      }
      const r = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/position', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      if (!r.ok) { alert('저장에 실패했어요'); return; }
      // 목록 카드의 제목도 갱신해 팝업에서 고른 제목이 바로 보이게.
      const card = document.getElementById('cand-' + idx);
      if (card) { const h = card.querySelector('h3.title'); if (h) h.textContent = chosenTitle; }
      close();
      onConfirm();
    });
  }
})();
"""


@app.route("/js/preview-modal.js")
def preview_modal_js():
    return Response(PREVIEW_MODAL_JS, mimetype="application/javascript")


def _run_reanalyze_job(video_id: str, idx: int) -> None:
    """백그라운드: 한 클립의 구간만 재분석해 새 후보를 목록에 '추가'한다(원본 유지)."""
    video_dir = OUTPUT_ROOT / video_id
    clips_path = video_dir / "clips.json"
    try:
        cfg = _load_config()
        clips = load_clips_json(clips_path)
        if idx < 0 or idx >= len(clips):
            _reanalyze_jobs[video_id] = {"running": False, "error": "잘못된 클립 번호"}
            return
        orig = clips[idx]
        _reanalyze_jobs[video_id] = {"running": True, "error": None, "new_idx": None, "pct": 0.0}

        def _prog(frac, msg):
            j = _reanalyze_jobs.get(video_id)
            if j is not None:
                j["pct"] = min(0.99, max(0.0, float(frac)))

        new_clip = reanalyze_clip_region(
            video_dir, orig, cfg, model=cfg["highlights"].get("model", ""), on_progress=_prog
        )
        # 원본 바로 뒤에 삽입해 '이전 버전 유지 + 새 버전 추가'가 목록에서 나란히 보이게 한다.
        with CLIPS_LOCK:
            clips = load_clips_json(clips_path)  # 그 사이 바뀌었을 수 있어 다시 읽는다
            insert_at = min(idx + 1, len(clips))
            clips.insert(insert_at, new_clip)
            save_clips_json(clips, clips_path)
        _reanalyze_jobs[video_id] = {"running": False, "error": None, "new_idx": insert_at, "pct": 1.0}
    except Exception as e:  # noqa: BLE001 - 실패해도 서버는 살아야 하고 팝업에 사유를 알린다
        traceback.print_exc()
        _reanalyze_jobs[video_id] = {"running": False, "error": str(e)[:300], "new_idx": None}


@app.route("/video/<video_id>/clip/<int:idx>/reanalyze", methods=["POST"])
def reanalyze_clip_route(video_id: str, idx: int):
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if not clips_path.exists():
        return jsonify({"error": "해당 영상 작업을 찾을 수 없습니다"}), 404
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
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
