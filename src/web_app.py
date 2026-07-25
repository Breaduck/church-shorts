"""웹 검토 흐름: 링크 입력 -> 후보 목록(바이럴 순위) 확인 -> 고른 것만 렌더링

사용법:
    python -m src.web_app
    -> http://127.0.0.1:5000 접속 (Cloudflare Tunnel 등으로 외부 노출 가능)
"""
from __future__ import annotations

import json
import subprocess
import threading
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
  .prog-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 26px; gap: 12px; }
  .prog-msg { font-size: 15px; color: var(--text); font-weight: 600; min-width: 0; }
  .prog-pct { font-size: 27px; font-weight: 800; color: var(--accent); font-variant-numeric: tabular-nums; letter-spacing: -0.02em; white-space: nowrap; }
  .stepper { position: relative; display: flex; justify-content: space-between; padding: 0 6px; }
  .rail { position: absolute; top: 11px; left: 17px; right: 17px; height: 4px; background: var(--border); border-radius: 999px; }
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
  .candidate {{ display: flex; gap: 20px; }}
  .rank {{
    flex-shrink: 0; width: 36px; height: 36px; border-radius: 50%;
    background: #eef2ff; color: var(--accent); font-weight: 700; font-size: 15px;
    display: flex; align-items: center; justify-content: center;
  }}
  .candidate:nth-child(1) .rank {{ background: var(--accent); color: #fff; }}
  .body {{ flex: 1; min-width: 0; }}
  .top-row {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
  .title {{ font-size: 17px; font-weight: 700; letter-spacing: -0.01em; margin: 0; }}
  .duration {{ color: var(--text-faint); font-size: 13px; white-space: nowrap; }}
  .title-wrap {{ display: flex; align-items: center; gap: 10px; min-width: 0; }}
  .score-badge {{
    flex-shrink: 0; font-size: 12px; font-weight: 800; letter-spacing: -0.01em;
    padding: 4px 9px; border-radius: 999px; white-space: nowrap; line-height: 1;
    display: inline-flex; align-items: center; gap: 4px;
  }}
  .score-badge .num {{ font-size: 13px; }}
  .score-badge.tier-top {{ background: #fef3c7; color: #b45309; }}       /* 90점 이상: 최상 */
  .score-badge.tier-high {{ background: #dcfce7; color: #15803d; }}      /* 85점 이상: 추천 */
  .score-badge.tier-ok {{ background: #e0edff; color: #1d4ed8; }}        /* 80점 이상: 후보 */
  .caption {{ color: var(--text); font-size: 14.5px; margin: 10px 0; }}
  .hashtags {{ color: var(--accent); font-size: 13px; margin: 0 0 10px; }}
  .reason {{
    color: var(--text-muted); font-size: 13px; background: #f7f8fa; border-radius: 10px;
    padding: 10px 14px; margin: 0;
  }}
  label.select {{
    display: inline-flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600;
    color: var(--text-muted); cursor: pointer; user-select: none;
  }}
  input[type=checkbox] {{ width: 18px; height: 18px; accent-color: var(--accent); cursor: pointer; }}
  video {{
    width: 100%; max-width: 260px; border-radius: 14px; background: #000; margin-top: 14px;
    display: block;
  }}
  .actions {{ position: sticky; bottom: 20px; margin-top: 24px; }}
  .actions button {{ box-shadow: 0 8px 24px rgba(49, 130, 246, 0.35); }}
  .actions button:disabled {{ opacity: .5; cursor: not-allowed; box-shadow: none; }}
  .edit-link {{ font-size: 13px; font-weight: 600; color: var(--accent); text-decoration: none; white-space: nowrap; }}
  .edit-link:hover {{ text-decoration: underline; }}
  .error-box {{ color: #e02424; }}
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="/">&larr; 새 링크</a>
  <h1>하이라이트 후보</h1>
  <p class="subtitle">바이럴 예상 순위 순으로 정렬했어요. 만들고 싶은 걸 골라주세요.</p>

  {{% if status != 'ready' %}}
  <div class="prog-card" id="prog" data-kind="analyze" data-pct="{{{{ pct }}}}" data-msg="{{{{ status_message }}}}" data-url="/video/{{{{ video_id }}}}/status">
    <div class="prog-head">
      <span class="prog-msg">{{{{ status_message }}}}</span>
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
  <div class="card candidate">
    <div class="rank">{{{{ loop.index }}}}</div>
    <div class="body">
      <div class="top-row">
        <div class="title-wrap">
          {{% if c.score is not none %}}
          <span class="score-badge {{% if c.score >= 90 %}}tier-top{{% elif c.score >= 85 %}}tier-high{{% else %}}tier-ok{{% endif %}}">
            <span class="num">{{{{ "%.0f"|format(c.score) }}}}</span>점{{% if c.score >= 90 %}} · 최상{{% elif c.score >= 85 %}} · 추천{{% endif %}}
          </span>
          {{% endif %}}
          <h3 class="title">{{{{ c.title }}}}</h3>
        </div>
        <span class="duration">{{{{ "%.0f"|format(c.end - c.start) }}}}초</span>
      </div>
      <p class="caption">{{{{ c.caption }}}}</p>
      <p class="hashtags">{{{{ c.hashtags|join(' ') }}}}</p>
      <p class="reason">{{{{ c.reason }}}}</p>
      <div class="top-row" style="margin-top:14px">
        <label class="select">
          <input type="checkbox" name="idx" value="{{{{ loop.index0 }}}}" {{% if loop.index0 < 3 %}}checked{{% endif %}}>
          이 클립 만들기
        </label>
        <a class="edit-link" href="/video/{{{{ video_id }}}}/clip/{{{{ loop.index0 }}}}/edit">위치 편집 &rarr;</a>
      </div>
      {{% if c.rendered %}}
        <video controls src="/media/{{{{ video_id }}}}/{{{{ loop.index }}}}.mp4"></video>
      {{% endif %}}
    </div>
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
      <span class="prog-msg">{{{{ render_message }}}}</span>
      <span class="prog-pct">{{{{ "%.0f"|format(render_pct) }}}}%</span>
    </div>
    <div class="stepper">
      <div class="rail"><div class="rail-fill"></div></div>
      <div class="step" data-min="0" data-max="100"><span class="dot"></span><span class="lbl">쇼츠 렌더링</span></div>
      <div class="step" data-min="100" data-max="100"><span class="dot"></span><span class="lbl">완성</span></div>
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
          if (j.ready) {{ paint(100, '완료'); setTimeout(function() {{ location.reload(); }}, 500); return; }}
          if (j.status === 'error') {{ msgEl.textContent = j.message; return; }}
        }} else {{
          paint(j.render_pct, j.render_message);
          if (j.render_error) {{ location.reload(); return; }}
          if (!j.rendering) {{ paint(100, '완성'); setTimeout(function() {{ location.reload(); }}, 500); return; }}
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
            progress=lambda msg, pct: _update_job(video_id_holder["id"], message=msg, pct=pct),
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

    _update_job(video_id, status="analyzing", message="분석 시작...", pct=0)
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
    return jsonify({
        "status": "ready" if clips_ready else job.get("status", "analyzing"),
        "ready": clips_ready,
        "pct": 100 if clips_ready else job.get("pct", 0),
        "message": job.get("message", "처리 중..."),
        "rendering": job.get("rendering", False),
        "render_pct": job.get("render_pct", 0),
        "render_message": job.get("render_message", "렌더링 준비 중..."),
        "render_error": job.get("render_error"),
    })


@app.route("/video/<video_id>/render", methods=["POST"])
def render_route(video_id: str):
    indices = request.get_json().get("indices", [])
    if not indices:
        return jsonify({"error": "선택된 항목이 없습니다"}), 400

    video_dir = OUTPUT_ROOT / video_id
    _update_job(
        video_id, rendering=True, render_pct=0, render_message="렌더링 시작...", render_error=None
    )

    def _job():
        try:
            render_selected(
                video_dir, indices,
                progress=lambda msg, pct: _update_job(video_id, render_message=msg, render_pct=pct),
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
  <p class="hint">실제 영상 프레임 위에서의 대략적인 위치입니다. 최종 결과는 다시 렌더링해야 반영됩니다.</p>

  <div class="btn-row">
    <button id="resetBtn" type="button">초기화</button>
    <button id="saveBtn" type="button">저장</button>
  </div>
  <p id="saveStatus"></p>
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
makeDraggable(titleEl, 'title');
makeDraggable(captionEl, 'caption');
render(titleEl, 'title');
render(captionEl, 'caption');

document.getElementById('resetBtn').addEventListener('click', () => {
  state.title = { x: 0, y: 0 };
  state.caption = { x: 0, y: 0 };
  render(titleEl, 'title');
  render(captionEl, 'caption');
});

document.getElementById('saveBtn').addEventListener('click', async () => {
  const statusEl = document.getElementById('saveStatus');
  statusEl.innerText = '저장 중...';
  const res = await fetch('/video/{{ video_id }}/clip/{{ idx }}/position', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      title_offset_x: state.title.x, title_offset_y: state.title.y,
      caption_offset_x: state.caption.x, caption_offset_y: state.caption.y,
    }),
  });
  statusEl.innerText = res.ok ? '저장됨. "선택한 쇼츠 만들기"로 다시 렌더링하면 반영됩니다.' : '저장 실패';
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

    source_res = _probe_resolution(OUTPUT_ROOT / video_id / "source.mp4")
    layout = _compute_layout(_load_config(), clip, source_res)
    caption_preview = _preview_caption_text(video_id, clip)

    return render_template_string(
        EDIT_TEMPLATE, video_id=video_id, idx=idx, clip=clip, layout=layout,
        caption_preview=caption_preview,
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
    clip.title_offset_x = float(body.get("title_offset_x", clip.title_offset_x))
    clip.title_offset_y = float(body.get("title_offset_y", clip.title_offset_y))
    clip.caption_offset_x = float(body.get("caption_offset_x", clip.caption_offset_x))
    clip.caption_offset_y = float(body.get("caption_offset_y", clip.caption_offset_y))
    save_clips_json(clips, clips_path)
    return jsonify({"ok": True})


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
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(ts), "-i", str(source_path),
            "-frames:v", "1",
            "-vf", f"crop=iw:ih*{1 - crop_pct}:0:0,scale={vbw}:{vbh}:flags=lanczos",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out_path.exists():
            return jsonify({"error": "미리보기 프레임 생성 실패"}), 500

    return send_file(out_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
