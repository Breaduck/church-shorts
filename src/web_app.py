"""웹 검토 흐름: 링크 입력 -> 후보 목록(바이럴 순위) 확인 -> 고른 것만 렌더링

사용법:
    python -m src.web_app
    -> http://127.0.0.1:5000 접속 (Cloudflare Tunnel 등으로 외부 노출 가능)
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template_string, request, send_file

from src.highlights import load_clips_json, save_clips_json
from src.main import analyze, render_selected

app = Flask(__name__)
OUTPUT_ROOT = Path("output")

# 단일 사용자 로컬 도구이므로 메모리 내 딕셔너리로 작업 상태를 추적한다 (DB 불필요).
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _update_job(video_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(video_id, {}).update(fields)


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
</head>
<body>
<div class="wrap">
  <h1>교회 쇼츠 생성기</h1>
  <p class="subtitle">유튜브 설교 링크를 넣으면 하이라이트 후보를 뽑아드려요.</p>
  <div class="card">
    <form id="f">
      <input type="text" id="url" placeholder="https://www.youtube.com/watch?v=..." required autofocus>
      <button class="primary" type="submit">분석 시작</button>
    </form>
  </div>
  <div class="status-box" id="status" style="display:none"></div>
</div>
<script>
const f = document.getElementById('f');
const statusEl = document.getElementById('status');
f.addEventListener('submit', async (e) => {{
  e.preventDefault();
  const url = document.getElementById('url').value;
  statusEl.style.display = 'block';
  statusEl.innerHTML = '<span class="spinner"></span>분석 요청 중...';
  const res = await fetch('/analyze', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{url}})
  }});
  const data = await res.json();
  if (!res.ok) {{ statusEl.innerText = '오류: ' + data.error; return; }}
  window.location = '/video/' + data.video_id;
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
    color: var(--text-muted); font-size: 13px; line-height: 1.65; white-space: pre-line;
    background: #f7f8fa; border-radius: 12px; padding: 13px 15px; margin: 12px 0 0;
  }}
  video {{ width: 100%; max-width: 260px; border-radius: 14px; background: #000; margin-top: 14px; display: block; }}
  .actions {{ position: sticky; bottom: 20px; margin-top: 24px; }}
  .actions button {{ box-shadow: 0 8px 24px rgba(49, 130, 246, 0.35); }}
  .actions button:disabled {{ opacity: .5; cursor: not-allowed; box-shadow: none; }}
  .error-box {{ color: #e02424; }}
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
      <div class="step" data-min="0" data-max="30"><span class="dot"></span><span class="lbl">다운로드</span></div>
      <div class="step" data-min="30" data-max="75"><span class="dot"></span><span class="lbl">전사</span></div>
      <div class="step" data-min="75" data-max="100"><span class="dot"></span><span class="lbl">하이라이트 선정</span></div>
    </div>
  </div>
  {{% else %}}
  <form id="renderForm">
  {{% for c in clips %}}
  {{% if c.score is none or c.score >= 80 %}}
  <div class="card candidate{{% if loop.index == 1 %}} top{{% endif %}}">
    <div class="cand-head">
      <div class="cand-meta">
        <span class="rank">{{% if loop.index == 1 %}}TOP{{% else %}}{{{{ loop.index }}}}위{{% endif %}}</span>
        {{% if c.score is not none %}}
        <span class="dot-sep"></span>
        <span class="score-badge {{% if c.score >= 90 %}}tier-top{{% elif c.score >= 85 %}}tier-high{{% else %}}tier-ok{{% endif %}}">{{{{ "%.0f"|format(c.score) }}}}점{{% if c.score >= 90 %}} · 최상{{% elif c.score >= 85 %}} · 추천{{% endif %}}</span>
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
    <p class="caption">{{{{ c.caption }}}}</p>
    <p class="hashtags">{{{{ c.hashtags|join(' ') }}}}</p>
    <div class="cand-foot">
      <button type="button" class="reason-toggle" aria-expanded="false">왜 추천하나요? <span class="chev">▾</span></button>
      <a class="edit-link" href="/video/{{{{ video_id }}}}/clip/{{{{ loop.index0 }}}}/edit">위치·자막 편집 &rarr;</a>
    </div>
    <p class="reason" hidden>{{{{ c.reason }}}}</p>
    {{% if c.rendered %}}
      <video controls src="/media/{{{{ video_id }}}}/{{{{ loop.index }}}}.mp4"></video>
    {{% endif %}}
  </div>
  {{% endif %}}
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
    const box = document.getElementById('renderStatus');
    box.style.display = 'block';
    box.innerHTML = '<span class="spinner"></span>렌더링 요청 중...';
    const res = await fetch('/video/{{{{ video_id }}}}/render', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{indices: idx}})
    }});
    const data = await res.json();
    if (!res.ok) {{ box.innerText = '오류: ' + data.error; return; }}
    location.reload();
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
      btn.firstChild.textContent = willOpen ? '분석 접기 ' : '왜 추천하나요? ';
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
</div>
</body>
</html>
"""


def _run_analyze_job(video_id_holder: dict, url: str) -> None:
    try:
        video_dir, clips = analyze(
            url,
            progress=lambda msg, pct, eta=None: _update_job(
                video_id_holder["id"], message=msg, pct=pct, eta_seconds=eta
            ),
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
    url = request.get_json().get("url", "").strip()
    if not url:
        return jsonify({"error": "URL이 비어있습니다"}), 400

    from src.download import extract_video_id

    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    _update_job(video_id, status="analyzing", message="분석 시작...", pct=0, started=time.time())
    holder = {"id": video_id}
    threading.Thread(target=_run_analyze_job, args=(holder, url), daemon=True).start()
    return jsonify({"video_id": video_id})


@app.route("/video/<video_id>")
def video_detail(video_id: str):
    with _jobs_lock:
        job = _jobs.get(video_id)

    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    job = job or {}

    # clips.json이 디스크에 있으면 분석은 이미 끝난 것 -> 항상 'ready'로 취급한다.
    # (render_route가 _update_job으로 job dict에 rendering/render_pct만 채워 넣으면
    # status/clips 키가 없는 채로 남는데, 그걸 그대로 쓰면 렌더링 시작과 동시에 상태가
    # "분석 중"으로 되돌아가 버리는 버그가 있었다. 디스크를 진실의 원천으로 삼아 방지.)
    if clips_path.exists():
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
        c.rendered = (clips_dir / f"short_{i}.mp4").exists()

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
    clips_ready = (OUTPUT_ROOT / video_id / "clips.json").exists()
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
    _update_job(
        video_id, rendering=True, render_pct=0, render_message="렌더링 시작...",
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
    lines = chunk_words_into_lines(words, max_wpl)
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
  .drag-box.title { font-weight: 800; color: #191f28; }
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
    flex: 1; padding: 10px 12px; font-size: 14px; font-family: inherit;
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
    <h2>영상 길이</h2>
    <p class="cap-sub">클립 시작·끝을 초 단위로 잘라 길이를 조절합니다. (전체 {{ '%.1f'|format(clip.end - clip.start) }}초)</p>
    <div class="sty-row">
      <label>시작</label>
      <input type="number" id="trimStart" class="trim-num" min="0" step="0.5"> <span class="unit">초</span>
      <label style="width:auto">끝</label>
      <input type="number" id="trimEnd" class="trim-num" min="0" step="0.5"> <span class="unit">초</span>
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
    <h2>제목</h2>
    <input type="text" id="titleInput" class="title-input" value="{{ clip.title }}">
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

  <div class="btn-row">
    <button id="resetBtn" type="button">위치 초기화</button>
    <button id="saveBtn" type="button">저장만</button>
  </div>
  <button id="renderBtn" type="button">저장하고 영상 만들기</button>
  <p id="saveStatus"></p>

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
    clips = load_clips_json(clips_path)
    if idx < 0 or idx >= len(clips):
        return jsonify({"error": "invalid index"}), 400

    body = request.get_json() or {}
    clip = clips[idx]
    if "title" in body and str(body["title"]).strip():
        clip.title = str(body["title"]).strip()
    # 영상 구간(길이 자르기). 사용자가 명시하면 그대로 존중한다(렌더 시 자동 확장/스냅 안 함).
    if body.get("clip_start") is not None and body.get("clip_end") is not None:
        s = float(body["clip_start"])
        e = float(body["clip_end"])
        if e - s >= 1.0:  # 최소 1초
            clip.start, clip.end = s, e
            clip.trimmed = True
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
    out_path = (video_dir / "clips" / f"_preview_{idx}.jpg").resolve()
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = _load_config()
        card = cfg["render"]["card_layout"]
        source_path = video_dir / "source.mp4"

        from src.render import _compute_card_video_box_height, _probe_resolution

        src_res = _probe_resolution(source_path)
        vbh = _compute_card_video_box_height(card, src_res)
        vbw = card["video_box_width"]
        crop_pct = card.get("source_crop_bottom_pct", 0.12)
        ts = clip.start + min(1.0, max(0.0, (clip.end - clip.start) / 2))
        # 실제 렌더(render.py)의 fill_mode와 동일하게 프레임을 만들어야 편집 미리보기가
        # 결과물과 일치한다. cover는 박스를 채우도록 확대 후 중앙 크롭, fit은 비율 유지.
        if card.get("fill_mode", "cover") == "cover":
            scale_vf = f"scale={vbw}:{vbh}:force_original_aspect_ratio=increase:flags=lanczos,crop={vbw}:{vbh}"
        else:
            scale_vf = f"scale={vbw}:{vbh}:flags=lanczos"
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(ts), "-i", str(source_path),
            "-frames:v", "1",
            "-vf", f"crop=iw:ih*{1 - crop_pct}:0:0,{scale_vf}",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out_path.exists():
            return jsonify({"error": "미리보기 프레임 생성 실패"}), 500

    return send_file(out_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
