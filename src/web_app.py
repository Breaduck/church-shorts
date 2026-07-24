"""웹 검토 흐름: 링크 입력 -> 후보 목록(바이럴 순위) 확인 -> 고른 것만 렌더링

사용법:
    python -m src.web_app
    -> http://127.0.0.1:5000 접속 (Cloudflare Tunnel 등으로 외부 노출 가능)
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

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
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="/">&larr; 새 링크</a>
  <h1>하이라이트 후보</h1>
  <p class="subtitle">바이럴 예상 순위 순으로 정렬했어요. 만들고 싶은 걸 골라주세요.</p>

  {{% if status != 'ready' %}}
  <div class="status-box"><span class="spinner"></span>{{{{ status_message }}}}</div>
  <script>setTimeout(() => location.reload(), 3000);</script>
  {{% else %}}
  <form id="renderForm">
  {{% for c in clips %}}
  <div class="card candidate">
    <div class="rank">{{{{ loop.index }}}}</div>
    <div class="body">
      <div class="top-row">
        <h3 class="title">{{{{ c.title }}}}</h3>
        <span class="duration">{{{{ "%.0f"|format(c.end - c.start) }}}}초</span>
      </div>
      <p class="caption">{{{{ c.caption }}}}</p>
      <p class="hashtags">{{{{ c.hashtags|join(' ') }}}}</p>
      <p class="reason">{{{{ c.reason }}}}</p>
      <label class="select" style="margin-top:14px">
        <input type="checkbox" name="idx" value="{{{{ loop.index0 }}}}" {{% if loop.index0 < 3 %}}checked{{% endif %}}>
        이 클립 만들기
      </label>
      {{% if c.rendered %}}
        <video controls src="/media/{{{{ video_id }}}}/{{{{ loop.index }}}}.mp4"></video>
      {{% endif %}}
    </div>
  </div>
  {{% endfor %}}
  <div class="actions">
    <button class="primary" type="submit">선택한 쇼츠 만들기</button>
  </div>
  </form>
  <div class="status-box" id="renderStatus" style="display:none; margin-top:16px"></div>
  <script>
  document.getElementById('renderForm').addEventListener('submit', async (e) => {{
    e.preventDefault();
    const idx = [...document.querySelectorAll('input[name=idx]:checked')].map(el => parseInt(el.value));
    const box = document.getElementById('renderStatus');
    box.style.display = 'block';
    box.innerHTML = '<span class="spinner"></span>렌더링 요청 중...';
    const res = await fetch('/video/{{{{ video_id }}}}/render', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{indices: idx}})
    }});
    const data = await res.json();
    box.innerText = res.ok ? '렌더링 시작됨. 잠시 후 새로고침하면 결과가 보입니다.' : ('오류: ' + data.error);
  }});
  </script>
  {{% endif %}}
</div>
</body>
</html>
"""


def _run_analyze_job(video_id_holder: dict, url: str) -> None:
    try:
        video_dir, clips = analyze(
            url, progress=lambda msg: _update_job(video_id_holder["id"], message=msg)
        )
        video_id_holder["id"] = video_dir.name
        _update_job(video_dir.name, status="ready", clips=clips, message="완료")
    except Exception as e:  # noqa: BLE001 - 사용자에게 실패 사유를 그대로 보여줘야 함
        vid = video_id_holder.get("id", "unknown")
        _update_job(vid, status="error", message=f"실패: {e}")
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

    _update_job(video_id, status="analyzing", message="분석 시작...")
    holder = {"id": video_id}
    threading.Thread(target=_run_analyze_job, args=(holder, url), daemon=True).start()
    return jsonify({"video_id": video_id})


@app.route("/video/<video_id>")
def video_detail(video_id: str):
    with _jobs_lock:
        job = _jobs.get(video_id)

    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    if job is None and clips_path.exists():
        job = {"status": "ready", "clips": load_clips_json(clips_path), "message": "완료"}

    if job is None:
        return "해당 영상 작업을 찾을 수 없습니다. 처음부터 다시 시도하세요.", 404

    clips = job.get("clips", [])
    clips_dir = OUTPUT_ROOT / video_id / "clips"
    for i, c in enumerate(clips, start=1):
        c.rendered = (clips_dir / f"short_{i}.mp4").exists()

    return render_template_string(
        CANDIDATES_TEMPLATE,
        video_id=video_id,
        status=job.get("status", "analyzing"),
        status_message=job.get("message", "처리 중..."),
        clips=clips,
    )


@app.route("/video/<video_id>/render", methods=["POST"])
def render_route(video_id: str):
    indices = request.get_json().get("indices", [])
    if not indices:
        return jsonify({"error": "선택된 항목이 없습니다"}), 400

    video_dir = OUTPUT_ROOT / video_id

    def _job():
        try:
            render_selected(video_dir, indices, progress=lambda msg: _update_job(video_id, render_message=msg))
        except Exception:
            traceback.print_exc()

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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
