"""로컬 검토 웹 UI: 생성된 쇼츠를 미리보고, 문구를 수정하고, 승인 후 YouTube에 업로드.

사용법:
    python -m src.review_app
    -> http://127.0.0.1:5000 접속
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template_string, request, send_from_directory

from src.highlights import Clip, load_clips_json, save_clips_json

app = Flask(__name__)
OUTPUT_ROOT = Path("output")


def _load_config() -> dict:
    return yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))


INDEX_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>쇼츠 검토</title>
<style>
  body { font-family: -apple-system, "Malgun Gothic", sans-serif; max-width: 900px; margin: 40px auto; }
  a { text-decoration: none; color: #2563eb; }
  li { margin-bottom: 10px; }
</style>
</head>
<body>
<h1>처리된 영상 목록</h1>
<ul>
{% for v in videos %}
  <li><a href="/video/{{ v }}">{{ v }}</a></li>
{% endfor %}
</ul>
</body>
</html>
"""

VIDEO_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{{ video_id }} 검토</title>
<style>
  body { font-family: -apple-system, "Malgun Gothic", sans-serif; max-width: 1100px; margin: 30px auto; }
  .clip { display: flex; gap: 20px; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
  video { width: 260px; border-radius: 8px; background: #000; }
  .fields { flex: 1; }
  label { display: block; font-weight: bold; margin-top: 10px; }
  input[type=text], textarea { width: 100%; box-sizing: border-box; padding: 6px; margin-top: 4px; }
  .actions { margin-top: 12px; }
  button { padding: 8px 14px; margin-right: 8px; cursor: pointer; }
  .reason { color: #666; font-size: 0.9em; }
  .status { font-size: 0.85em; color: green; margin-left: 8px; }
</style>
</head>
<body>
<p><a href="/">&larr; 목록으로</a></p>
<h1>{{ video_id }}</h1>
{% for clip in clips %}
<div class="clip" data-index="{{ loop.index0 }}">
  <video controls src="/media/{{ video_id }}/clips/short_{{ loop.index }}.mp4"></video>
  <div class="fields">
    <label>훅 제목 (title)</label>
    <input type="text" class="title" value="{{ clip.title }}">
    <label>캡션 (caption)</label>
    <textarea class="caption" rows="3">{{ clip.caption }}</textarea>
    <label>해시태그 (쉼표 구분)</label>
    <input type="text" class="hashtags" value="{{ clip.hashtags|join(', ') }}">
    <p class="reason">선정 이유: {{ clip.reason }}</p>
    <div class="actions">
      <button onclick="saveClip({{ loop.index0 }})">수정 저장</button>
      <button onclick="uploadClip({{ loop.index0 }})">YouTube 업로드 (unlisted)</button>
      <span class="status" id="status-{{ loop.index0 }}"></span>
    </div>
  </div>
</div>
{% endfor %}

<script>
async function saveClip(i) {
  const el = document.querySelectorAll('.clip')[i];
  const body = {
    title: el.querySelector('.title').value,
    caption: el.querySelector('.caption').value,
    hashtags: el.querySelector('.hashtags').value.split(',').map(s => s.trim()).filter(Boolean),
  };
  const res = await fetch('/video/{{ video_id }}/clip/' + i, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  document.getElementById('status-' + i).innerText = res.ok ? '저장됨' : '저장 실패';
}

async function uploadClip(i) {
  document.getElementById('status-' + i).innerText = '업로드 중...';
  const res = await fetch('/video/{{ video_id }}/clip/' + i + '/upload', { method: 'POST' });
  const data = await res.json();
  document.getElementById('status-' + i).innerText = res.ok ? ('업로드 완료: ' + data.video_id) : ('실패: ' + data.error);
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    videos = sorted(
        p.name for p in OUTPUT_ROOT.iterdir() if p.is_dir() and (p / "clips.json").exists()
    ) if OUTPUT_ROOT.exists() else []
    return render_template_string(INDEX_TEMPLATE, videos=videos)


@app.route("/video/<video_id>")
def video_detail(video_id: str):
    clips = load_clips_json(OUTPUT_ROOT / video_id / "clips.json")
    return render_template_string(VIDEO_TEMPLATE, video_id=video_id, clips=clips)


@app.route("/media/<video_id>/clips/<filename>")
def media(video_id: str, filename: str):
    return send_from_directory(OUTPUT_ROOT / video_id / "clips", filename)


@app.route("/video/<video_id>/clip/<int:index>", methods=["POST"])
def update_clip(video_id: str, index: int):
    clips_path = OUTPUT_ROOT / video_id / "clips.json"
    clips = load_clips_json(clips_path)
    if index < 0 or index >= len(clips):
        return jsonify({"error": "invalid index"}), 400

    body = request.get_json()
    clips[index].title = body.get("title", clips[index].title)
    clips[index].caption = body.get("caption", clips[index].caption)
    clips[index].hashtags = body.get("hashtags", clips[index].hashtags)
    save_clips_json(clips, clips_path)
    return jsonify({"ok": True})


@app.route("/video/<video_id>/clip/<int:index>/upload", methods=["POST"])
def upload_clip(video_id: str, index: int):
    from src.upload.youtube import upload_short

    clips = load_clips_json(OUTPUT_ROOT / video_id / "clips.json")
    if index < 0 or index >= len(clips):
        return jsonify({"error": "invalid index"}), 400
    clip = clips[index]
    video_path = OUTPUT_ROOT / video_id / "clips" / f"short_{index + 1}.mp4"

    cfg = _load_config()["upload"]["youtube"]
    try:
        yt_id = upload_short(
            video_path,
            title=clip.title or f"{video_id} 쇼츠 {index + 1}",
            description=clip.caption,
            tags=[h.lstrip("#") for h in clip.hashtags],
            category_id=cfg.get("category_id", "22"),
            privacy_status=cfg.get("default_privacy", "unlisted"),
        )
    except Exception as e:  # noqa: BLE001 - 업로드 실패 사유를 그대로 사용자에게 보여줘야 함
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "video_id": yt_id})


if __name__ == "__main__":
    app.run(debug=True)
