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

from src.feedback import PerformanceRecord, upsert_feedback
from src.models import EXECUTION_MODEL, sanitize as sanitize_model
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
  /* 애플 리퀴드 글래스: 반투명 유리 표면(blur+saturate) + 얇은 하이라이트 테두리 +
     넓고 부드러운 다층 그림자. 배경에 은은한 그라디언트를 깔아야 블러가 실제로
     보인다(평면 단색 위에서는 유리 느낌이 안 남) — CLAUDE.md 지침(애플 스타일) 반영. */
  :root {
    --bg: #eef1f6;
    --card: #ffffff;
    --glass-bg: rgba(255, 255, 255, 0.68);
    --glass-border: rgba(255, 255, 255, 0.55);
    --text: #1d1d1f;
    --text-muted: #6e7175;
    --text-faint: #98999d;
    --accent: #0a84ff;
    --accent-hover: #0071e3;
    --border: rgba(60, 60, 67, 0.1);
    --shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px rgba(15, 23, 42, 0.07);
    --radius: 20px;
  }
  * { box-sizing: border-box; }
  body {
    font-family: "Pretendard", -apple-system, BlinkMacSystemFont, "Malgun Gothic", sans-serif;
    background:
      radial-gradient(1100px 700px at 12% -10%, rgba(10, 132, 255, 0.10), transparent 60%),
      radial-gradient(900px 600px at 100% 0%, rgba(191, 90, 242, 0.08), transparent 55%),
      var(--bg);
    background-attachment: fixed;
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
    background: var(--glass-bg); -webkit-backdrop-filter: blur(24px) saturate(180%);
    backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid var(--glass-border); border-radius: var(--radius); box-shadow: var(--shadow);
    padding: 28px; margin-bottom: 16px;
  }
  input[type=text] {
    width: 100%; padding: 16px 18px; font-size: 15px; font-family: inherit;
    border: 1px solid rgba(255,255,255,0.8); border-radius: 14px; background: #fff;
    box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 6px 18px rgba(15,23,42,.08);
    transition: border-color .15s, box-shadow .15s; margin-top: 14px;
  }
  input[type=text]:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 4px rgba(10,132,255,.14), 0 6px 18px rgba(15,23,42,.08);
  }
  textarea {
    width: 100%; padding: 14px 16px; font-size: 14px; font-family: inherit; line-height: 1.5;
    border: 1px solid rgba(255,255,255,0.8); border-radius: 14px; background: #fff;
    box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 6px 18px rgba(15,23,42,.08);
    transition: border-color .15s, box-shadow .15s; margin-top: 12px; resize: vertical;
  }
  textarea:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 4px rgba(10,132,255,.14), 0 6px 18px rgba(15,23,42,.08);
  }
  .hint { color: var(--text-faint); font-size: 12.5px; margin: 10px 2px 0; }
  input[type=number] {
    padding: 9px 11px; font-size: 13px; font-family: inherit; width: 100%;
    border: 1px solid var(--border); border-radius: 10px; background: rgba(120,120,128,0.08);
  }
  input[type=number]:focus { outline: none; border-color: var(--accent); background: #fff; }
  /* 점수 세부축 막대 */
  .subscores { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 10px 0 2px; }
  .subscore { font-size: 12px; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }
  .subscore b { color: var(--text); font-variant-numeric: tabular-nums; font-weight: 700; }
  .sbar { width: 46px; height: 5px; border-radius: 999px; background: var(--border); overflow: hidden; }
  .sbar > i { display: block; height: 100%; background: var(--accent); border-radius: 999px; }
  /* 렌더 옵션 유리 칩(체크박스/셀렉트 라벨). 예전엔 카드마다 같은 인라인 스타일을
     복붙해 유지보수가 어려웠다 — 클래스 하나로 통일(애플 리퀴드 글래스 톤). */
  .opt-chip {
    display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-muted);
    margin: 0 8px 10px 0; user-select: none; cursor: pointer;
    background: var(--glass-bg); -webkit-backdrop-filter: blur(16px) saturate(180%);
    backdrop-filter: blur(16px) saturate(180%); border: 1px solid var(--glass-border);
    border-radius: 12px; padding: 8px 12px; box-shadow: var(--shadow); transition: box-shadow .15s;
  }
  .opt-chip:hover { box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 10px 26px rgba(15,23,42,.09); }
  .opt-chip select {
    font-size: 13px; font-family: inherit; border: 1px solid var(--border); border-radius: 8px;
    padding: 4px 6px; background: rgba(255,255,255,.7); color: var(--text);
  }
  /* 우측 상단 둥근 버튼(YouTube 업로드 · 스튜디오) — 애플식 흰 바탕 + 그림자, 완전한 알약 모양 */
  .page-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  .page-head-btns { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .pill-btn { flex: 0 0 auto; padding: 10px 18px; font-size: 13.5px; font-weight: 700; font-family: inherit;
    border: none; border-radius: 999px; background: #fff; color: var(--accent); cursor: pointer; white-space: nowrap;
    box-shadow: 0 1px 3px rgba(15,23,42,.06), 0 4px 14px rgba(15,23,42,.08); transition: background .15s, box-shadow .15s, transform .1s; }
  .pill-btn:hover { background: #f5f8ff; box-shadow: 0 2px 6px rgba(15,23,42,.08), 0 6px 18px rgba(15,23,42,.10); }
  .pill-btn:active { transform: scale(0.97); }
  .yt-modal-back { position: fixed; inset: 0; background: rgba(15,23,42,.32);
    -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px); display: none;
    align-items: center; justify-content: center; z-index: 60; }
  .yt-modal-back.show { display: flex; }
  .yt-modal { background: rgba(255,255,255,.78); -webkit-backdrop-filter: blur(30px) saturate(180%);
    backdrop-filter: blur(30px) saturate(180%); border: 1px solid var(--glass-border);
    border-radius: 22px; padding: 22px; width: 420px; max-width: 92vw;
    max-height: 80vh; overflow-y: auto; box-shadow: 0 20px 60px rgba(15,23,42,.25); }
  .yt-modal h2 { font-size: 16px; margin-bottom: 4px; }
  .yt-modal .sub { font-size: 12.5px; color: var(--text-muted); margin-bottom: 14px; }
  .yt-pick-row { display: flex; align-items: center; justify-content: space-between; gap: 10px;
    padding: 10px 12px; border: 1.5px solid var(--border, #f0f1f3); border-radius: 10px; margin-bottom: 8px; }
  .yt-pick-row .name { font-size: 13.5px; font-weight: 600; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; flex: 1; }
  .yt-pick-row .st { font-size: 11.5px; color: var(--text-faint); flex: 0 0 auto; }
  .yt-pick-row button { flex: 0 0 auto; padding: 6px 12px; font-size: 12.5px; font-weight: 700;
    font-family: inherit; border: none; border-radius: 8px; background: #3182f6; color: #fff; cursor: pointer; }
  .yt-pick-row button:disabled { opacity: .55; cursor: default; }
  .yt-pick-row.done button { background: #eee; color: #888; }
  .yt-empty { font-size: 13px; color: var(--text-muted); text-align: center; padding: 20px 0; }
  .yt-modal-close { display: block; width: 100%; margin-top: 8px; padding: 10px; border: none;
    border-radius: 10px; background: var(--border, #f0f1f3); color: var(--text-muted); font-weight: 700;
    font-family: inherit; cursor: pointer; }
  .yt-upload { margin-top: 12px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .yt-link { font-size: 13px; font-weight: 600; color: var(--accent); text-decoration: none; }
  .yt-link:hover { text-decoration: underline; }
  .yt-track { font-size: 12px; color: var(--text-faint); }
  button.primary {
    width: 100%; margin-top: 16px; padding: 16px; font-size: 15px; font-weight: 600;
    font-family: inherit; color: #fff;
    background: linear-gradient(180deg, #2a9bff, var(--accent));
    border: none; border-radius: 980px; cursor: pointer;
    box-shadow: 0 1px 1px rgba(255,255,255,.35) inset, 0 6px 16px rgba(10,132,255,.32);
    transition: background .15s, box-shadow .15s, transform .1s;
  }
  button.primary:hover { background: linear-gradient(180deg, #1f92ff, var(--accent-hover)); }
  button.primary:active { transform: scale(0.98); box-shadow: 0 1px 1px rgba(255,255,255,.3) inset, 0 3px 8px rgba(10,132,255,.28); }
  .status-box {
    padding: 20px 24px; background: var(--glass-bg); -webkit-backdrop-filter: blur(24px) saturate(180%);
    backdrop-filter: blur(24px) saturate(180%); border: 1px solid var(--glass-border);
    border-radius: var(--radius); box-shadow: var(--shadow);
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
<title>교회 쇼츠 시스템</title>
{BASE_STYLE}
<style>
  .adv {{ margin-top: 12px; }}
  .adv summary {{
    cursor: pointer; font-size: 13px; font-weight: 600; color: var(--text-muted);
    padding: 4px 2px; user-select: none; list-style-position: inside;
  }}
  .adv summary:hover {{ color: var(--text); }}
  .seg {{ display: flex; gap: 2px; background: rgba(120,120,128,0.12); -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px); padding: 3px; border-radius: 12px; }}
  .seg label {{
    flex: 1; text-align: center; cursor: pointer; border-radius: 9px; padding: 18px 8px;
    font-size: 14.5px; font-weight: 700; color: var(--text-muted); transition: background .2s, color .2s, box-shadow .2s;
  }}
  .seg input {{ position: absolute; opacity: 0; pointer-events: none; }}
  .seg label:has(input:checked) {{ background: #fff; color: var(--accent);
    box-shadow: 0 1px 1px rgba(0,0,0,.04), 0 3px 8px rgba(15,23,42,.10); }}
  /* 우측 상단 버튼 행 — 둥근 흰색 배경 + 그림자(애플 느낌) */
  .head-btns {{ display: flex; align-items: center; gap: 10px; flex: 0 0 auto; }}
  .force-toggle {{
    flex: 0 0 auto; padding: 16px 26px; font-size: 15px; font-weight: 700; font-family: inherit;
    border: none; border-radius: 999px; background: #fff; color: var(--text-muted); cursor: pointer;
    box-shadow: 0 2px 6px rgba(15,23,42,.08), 0 10px 26px rgba(15,23,42,.10);
    transition: color .15s, box-shadow .15s, transform .1s; white-space: nowrap;
  }}
  .force-toggle:hover {{ box-shadow: 0 3px 10px rgba(15,23,42,.10), 0 12px 30px rgba(15,23,42,.14); }}
  .force-toggle:active {{ transform: scale(0.97); }}
  .force-toggle[aria-pressed="true"] {{
    color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent) inset, 0 2px 6px rgba(15,23,42,.08), 0 10px 26px rgba(15,23,42,.10);
  }}
  /* 분석 시작 — 파란색 유지, 우측 상단으로 이동(사용자 요청 2026-09-06) */
  .analyze-btn {{
    flex: 0 0 auto; padding: 16px 30px; font-size: 15px; font-weight: 700; font-family: inherit;
    color: #fff; background: linear-gradient(180deg, #2a9bff, var(--accent));
    border: none; border-radius: 999px; cursor: pointer; white-space: nowrap;
    box-shadow: 0 1px 1px rgba(255,255,255,.35) inset, 0 6px 16px rgba(10,132,255,.32);
    transition: background .15s, box-shadow .15s, transform .1s;
  }}
  .analyze-btn:hover {{ background: linear-gradient(180deg, #1f92ff, var(--accent-hover)); }}
  .analyze-btn:active {{ transform: scale(0.97); }}
  /* 파일 드래그앤드롭 박스 — 애플 점선 업로드 카드 */
  .dropzone {{
    margin-top: 14px; padding: 36px 20px; text-align: center; cursor: pointer;
    border: 1.5px dashed rgba(60,60,67,0.28); border-radius: 18px;
    background: rgba(120,120,128,0.05); transition: border-color .15s, background .15s, box-shadow .15s;
  }}
  .dropzone:hover {{ border-color: var(--accent); background: rgba(10,132,255,0.05); }}
  .dropzone.drag {{ border-color: var(--accent); background: rgba(10,132,255,0.09);
    box-shadow: 0 0 0 5px rgba(10,132,255,0.08); }}
  .dropzone-icon {{
    width: 46px; height: 46px; margin: 0 auto 12px; border-radius: 50%;
    background: rgba(10,132,255,0.10); color: var(--accent); font-size: 20px; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
  }}
  .dropzone-text {{ font-size: 14.5px; font-weight: 600; color: var(--text); }}
  .dropzone-text b {{ color: var(--accent); }}
  .dropzone-file {{ font-size: 13.5px; color: var(--accent); font-weight: 700; margin-top: 4px; }}
  /* 찬양 제목 — 곡마다 개별 입력칸 + 추가 버튼(애플 스타일) */
  .song-title-row {{ display: flex; align-items: center; gap: 8px; margin-top: 8px; }}
  .song-title-row:first-child {{ margin-top: 0; }}
  .song-title-input {{
    flex: 1; min-width: 0; padding: 12px 16px; font-size: 14.5px; font-family: inherit;
    border: 1px solid rgba(255,255,255,0.8); border-radius: 12px; background: #fff;
    box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 6px 18px rgba(15,23,42,.08);
    transition: border-color .15s, box-shadow .15s;
  }}
  .song-title-input:focus {{
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 4px rgba(10,132,255,.14), 0 6px 18px rgba(15,23,42,.08);
  }}
  .song-title-remove {{
    flex: 0 0 auto; width: 32px; height: 32px; border-radius: 50%; border: none;
    background: rgba(120,120,128,0.10); color: var(--text-faint); font-size: 16px; line-height: 1;
    cursor: pointer; transition: background .15s, color .15s;
  }}
  .song-title-remove:hover {{ background: rgba(255,59,48,0.12); color: #ff3b30; }}
  .add-song-btn {{
    margin-top: 10px; padding: 10px 18px; font-size: 13.5px; font-weight: 700; font-family: inherit;
    border: none; border-radius: 999px; background: #fff; color: var(--accent); cursor: pointer;
    box-shadow: 0 1px 3px rgba(15,23,42,.06), 0 4px 14px rgba(15,23,42,.08);
    transition: background .15s, box-shadow .15s, transform .1s;
  }}
  .add-song-btn:hover {{ background: #f5f8ff; box-shadow: 0 2px 6px rgba(15,23,42,.08), 0 6px 18px rgba(15,23,42,.10); }}
  .add-song-btn:active {{ transform: scale(0.97); }}
  /* 사용자 요청(2026-09-06): 제목을 더 굵고 크게 + 말씀/찬양·유튜브 링크 입력·업로드
     영역을 전체적으로 약 2배 확대. h1/.seg/#url/.dropzone은 BASE_STYLE(전역)에도 있지만
     아래는 인덱스 페이지 전용 <style> 블록 안이라 다른 페이지(후보 목록 등)에는 영향 없다.
     2차 피드백(2026-09-06): "글씨 크기가 아니라 칸(박스) 전체를 키우라는 것" — 넓은
     화면에 비해 카드 자체가 작아 보였던 게 진짜 문제. 그래서 폭·패딩·버튼까지 전부 확대.
     3차 피드백(2026-09-06): "100% 비율로 보면 너무 크고 67% 비율이 적당" — 2차 확대판을
     브라우저 67% 축소로 봤을 때가 딱 맞다는 뜻이므로, 아래 모든 값에 0.67을 곱해 축소. */
  .wrap {{ max-width: 750px; }}
  .card {{ padding: 32px 35px; border-radius: 19px; }}
  .page-head {{ margin-bottom: 6px; }}
  #mainTitle {{ font-size: 26px; font-weight: 650; letter-spacing: -0.025em; }}
  .force-toggle {{ padding: 15px 25px; font-size: 13px; }}
  .analyze-btn {{ padding: 15px 28px; font-size: 13px; }}
  .seg {{ padding: 4px; border-radius: 12px; margin-bottom: 16px !important; }}
  .seg label {{ font-size: 17px; padding: 24px 9px; border-radius: 9px; }}
  #url {{ font-size: 17px; padding: 23px 20px; border-radius: 15px; margin-top: 13px; }}
  .dropzone {{ padding: 60px 27px; margin-top: 13px; border-radius: 16px; }}
  .dropzone-icon {{ width: 67px; height: 67px; font-size: 28px; margin-bottom: 15px; }}
  .dropzone-text {{ font-size: 17px; }}
  .dropzone-file {{ font-size: 16px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="page-head">
    <div>
      <h1 id="mainTitle">교회 쇼츠 시스템</h1>
    </div>
    <div class="head-btns">
      <button type="button" class="force-toggle" id="forceToggle" aria-pressed="false">새로 분석</button>
      <button class="analyze-btn" type="submit" form="f">분석 시작</button>
    </div>
  </div>
  <div class="card">
    <form id="f">
      <div class="seg" style="margin-bottom:14px">
        <label><input type="radio" name="mode" value="sermon" checked>말씀</label>
        <label><input type="radio" name="mode" value="praise">찬양</label>
      </div>
      <input type="text" id="url" placeholder="유튜브 링크" autofocus>
      <div class="dropzone" id="dropzone">
        <input type="file" id="vfile" accept="video/*,.mp4,.mov,.mkv,.avi" style="display:none">
        <div class="dropzone-icon">&#8593;</div>
        <div class="dropzone-text" id="dropzoneText">동영상 파일을 여기로 끌어다 놓거나 <b>클릭해서 선택</b></div>
      </div>
      <div id="songTitlesWrap" style="display:none;margin-top:10px">
        <div id="songTitleRows"></div>
        <button type="button" id="addSongTitle" class="add-song-btn">+ 찬양 추가</button>
      </div>
      <details class="adv">
        <summary>고급 옵션</summary>
        <textarea id="transcript" rows="5" placeholder="(선택) 자막 붙여넣기"></textarea>
      </details>
    </form>
  </div>
  <div class="status-box" id="status" style="display:none"></div>
</div>
<script>
const f = document.getElementById('f');
const statusEl = document.getElementById('status');
// '분석 시작' 버튼은 우측 상단으로 옮겨 폼 바깥에 있다 — form="f" 속성으로 제출과
// 연결되므로 querySelector는 문서 전체에서 찾아야 한다(f.querySelector는 못 찾음).
const submitBtn = document.querySelector('.analyze-btn');
// 찬양 모드일 때만 '곡 제목' 입력란을 보여준다(제목 → 정식 가사 자막).
const songTitlesWrap = document.getElementById('songTitlesWrap');
const songTitleRows = document.getElementById('songTitleRows');
const addSongTitleBtn = document.getElementById('addSongTitle');
function addSongTitleRow(value) {{
  const row = document.createElement('div');
  row.className = 'song-title-row';
  const input = document.createElement('input');
  input.type = 'text'; input.className = 'song-title-input'; input.placeholder = '찬양 제목 (또는 벅스 가사 페이지 URL)';
  input.value = value || '';
  const remove = document.createElement('button');
  remove.type = 'button'; remove.className = 'song-title-remove'; remove.setAttribute('aria-label', '삭제');
  remove.innerHTML = '&times;';
  remove.addEventListener('click', () => {{ row.remove(); ensureAtLeastOneSongTitleRow(); }});
  row.appendChild(input); row.appendChild(remove);
  songTitleRows.appendChild(row);
}}
function ensureAtLeastOneSongTitleRow() {{
  if (!songTitleRows.children.length) addSongTitleRow();
}}
function getSongTitlesValue() {{
  return Array.from(songTitleRows.querySelectorAll('.song-title-input'))
    .map((i) => i.value.trim()).filter(Boolean).join('\\n');
}}
addSongTitleBtn.addEventListener('click', () => addSongTitleRow());
ensureAtLeastOneSongTitleRow();
function syncSongTitlesVisibility() {{
  const mode = (f.querySelector('input[name="mode"]:checked') || {{}}).value || 'sermon';
  songTitlesWrap.style.display = (mode === 'praise') ? 'block' : 'none';
}}
f.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener('change', syncSongTitlesVisibility));
syncSongTitlesVisibility();
// '새로 분석' 토글 — 저장된 후보를 무시하고 다시 뽑을지 여부(우측 상단 버튼으로 이동).
const forceToggle = document.getElementById('forceToggle');
forceToggle.addEventListener('click', () => {{
  const on = forceToggle.getAttribute('aria-pressed') !== 'true';
  forceToggle.setAttribute('aria-pressed', String(on));
}});
// 파일 드래그앤드롭
const dropzone = document.getElementById('dropzone');
const vfile = document.getElementById('vfile');
const dropzoneText = document.getElementById('dropzoneText');
function showPickedFile(file) {{
  dropzoneText.innerHTML = file
    ? '<span class="dropzone-file">' + file.name + '</span>'
    : '동영상 파일을 여기로 끌어다 놓거나 <b>클릭해서 선택</b>';
}}
dropzone.addEventListener('click', () => vfile.click());
vfile.addEventListener('change', () => showPickedFile(vfile.files[0] || null));
['dragenter', 'dragover'].forEach((ev) => dropzone.addEventListener(ev, (e) => {{
  e.preventDefault(); e.stopPropagation(); dropzone.classList.add('drag');
}}));
['dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, (e) => {{
  e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('drag');
}}));
dropzone.addEventListener('drop', (e) => {{
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) {{ vfile.files = e.dataTransfer.files; showPickedFile(file); }}
}});
f.addEventListener('submit', async (e) => {{
  e.preventDefault();
  if (submitBtn.disabled) return;  // 중복 클릭 방지: 하이라이트 선정은 AI가 실제로 읽고 고르는
                                    // 단계라 보통 2~5분 걸린다. 재클릭하면 같은 영상 분석이
                                    // 중복 실행돼 진행률이 널뛰고 세션 한도만 낭비된다.
  submitBtn.disabled = true;
  const url = document.getElementById('url').value;
  const vf = vfile.files[0] || null;
  const transcript_text = document.getElementById('transcript').value;
  const force = forceToggle.getAttribute('aria-pressed') === 'true';  // 기본은 캐시 재사용
  const mode = (f.querySelector('input[name="mode"]:checked') || {{}}).value || 'sermon';
  if (!url.trim() && !vf) {{
    statusEl.style.display = 'block';
    statusEl.innerText = '유튜브 링크를 넣거나 동영상 파일을 선택해 주세요.';
    submitBtn.disabled = false; return;
  }}
  statusEl.style.display = 'block';
  statusEl.innerHTML = vf
    ? '<span class="spinner"></span>동영상 업로드 중… (파일이 크면 시간이 걸려요)'
    : '<span class="spinner"></span>진행률 화면으로 이동 중… (곧 %와 남은 예상시간이 표시돼요)';
  try {{
    let res;
    if (vf) {{
      // 파일 업로드 경로: multipart로 보내고, 서버가 저장 후 whisper 직접 전사부터 시작한다.
      const fd = new FormData();
      fd.append('file', vf); fd.append('mode', mode); fd.append('model', '');
      if (mode === 'praise') fd.append('song_titles', getSongTitlesValue());
      // fetch는 업로드 진행률을 알려주지 않아, 수백 MB 영상을 올리는 내내 화면이 멈춘 것처럼
      // 보였다("초반에 왜 이렇게 오래 분석하냐" — 실은 아직 업로드 중, 2026-09-07).
      // XHR로 바꿔 실제 업로드 %와 MB를 보여준다.
      res = await new Promise((resolve, reject) => {{
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/analyze_upload');
        xhr.upload.onprogress = (ev) => {{
          if (!ev.lengthComputable) return;
          const pct = Math.round(ev.loaded / ev.total * 100);
          const mb = (ev.total / 1048576).toFixed(0);
          statusEl.innerHTML = '<span class="spinner"></span>동영상 업로드 중… ' + pct + '% (' + mb + 'MB)'
            + (pct >= 100 ? ' — 업로드 완료, 분석 시작 중…' : '');
        }};
        xhr.onload = () => resolve({{ ok: xhr.status >= 200 && xhr.status < 300, json: async () => JSON.parse(xhr.responseText) }});
        xhr.onerror = () => reject(new Error('업로드 실패 (네트워크)'));
        xhr.send(fd);
      }});
    }} else {{
      const song_titles = mode === 'praise' ? getSongTitlesValue() : '';
      res = await fetch('/analyze', {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{url, transcript_text, force, model: '', mode, song_titles}})
      }});
    }}
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
    const renderedClips = CLIPS_SUMMARY.filter((c) => c.rendered);
    if (!renderedClips.length) {{
      ytPickList.innerHTML = '<p class="yt-empty">아직 만든 쇼츠가 없어요. 먼저 후보를 선택해 "만들기"를 눌러주세요.</p>';
      return;
    }}
    ytPickList.innerHTML = renderedClips.map((c) => {{
      if (c.youtube_id) {{
        return '<div class="yt-pick-row done"><span class="name">' + c.title + '</span>' +
          '<a class="yt-link" href="https://youtu.be/' + c.youtube_id + '" target="_blank" rel="noopener">▶ 이미 업로드됨</a></div>';
      }}
      return '<div class="yt-pick-row" data-idx="' + c.idx + '"><span class="name">' + c.title + '</span>' +
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
        if (j.render_error) {{ box.classList.remove('hidden'); track.hidden = true; closeBtn.hidden = false; pct.style.display='none'; msg.innerHTML = '오류: ' + j.render_error; eta.textContent=''; actions.hidden=true; wasRendering=false; setTimeout(poll, 1500); return; }}
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
            "title": c.title,
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

# 클립별 무음 지도 캐시(ffmpeg silencedetect 2~3초 — 팝업 열 때마다 다시 돌리지 않게)
_silences_cache: dict = {}


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
    base_title_margin_v, base_caption_margin_v = compute_card_margins(card_layout, resolution, title_size)

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


def _caption_lines_for_clip(video_id: str, clip, cfg: dict) -> list[dict]:
    """자막 편집기에 채워 넣을 자막 라인 목록을 만든다.
    이미 편집·저장된 caption_overrides가 있으면 그걸 쓰고, 없으면 원본 전사에서 클립
    구간 단어를 뽑아 max_words_per_line 단위로 잘라 라인({start,end,text})으로 만든다."""
    from src.captions import _collect_words_in_range, _display_text, chunk_words_into_lines

    # 유튜브 실황의 찬양 클립은 가사 자막을 넣지 않는다(화면에 교회 가사 슬라이드가 이미
    # 있음) — 편집기에도 초안을 채우지 않는다. 단, 직접 찍어 업로드한 영상(upload_*)은
    # 가사 표시가 없어 whisper 자막을 넣으므로 초안도 같은 소스로 채운다(아래 일반 경로).
    if getattr(clip, "clip_type", "") == "praise" and not video_id.startswith("upload_"):
        return []

    def _strip_trailing_dots(text: str) -> str:
        # 실제 렌더(_karaoke_text)는 단어별로 끝 마침표를 뗀다. 편집기 미리보기도 같은
        # 규칙을 적용해야 "화면엔 있는데 실제 영상엔 없는" 불일치가 안 생긴다.
        return " ".join(_display_text(w) for w in text.split())

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
  /* input[type=text]의 전역 width:100%/margin-top(BASE_STYLE)보다 상위 명시도가 필요해
     .cap-row input.cap-input로 부모+태그+클래스 조합을 쓴다(단일 클래스는 짐). */
  .cap-row input.cap-input {
    flex: 1; min-width: 0; padding: 10px 12px; font-size: 14px; font-family: inherit;
    border: 1.5px solid var(--border); border-radius: 10px; background: #fafbfc;
    transition: border-color .15s, background .15s; width: auto; margin-top: 0;
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
  /* input[type=number]의 전역 width:100%(BASE_STYLE)보다 상위 명시도가 필요해
     .cap-row input.cap-num로 부모+태그+클래스 조합을 쓴다(단일 클래스는 짐). */
  .cap-row input.cap-num {
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
      <div class="drag-box title" id="titleBox" style="font-size: {{ (layout.title_size * layout.title_ass_coeff * layout.scale)|round|int }}px; white-space: nowrap;">
        {{ clip.title }}
      </div>
      <div class="drag-box caption" id="captionBox" style="font-size: {{ (layout.caption_font_size * layout.caption_ass_coeff * layout.scale)|round|int }}px;">
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
const TITLE_BASE_FS = {{ (layout.title_size * layout.title_ass_coeff * layout.scale)|round|int }};

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


STUDIO_TEMPLATE = r"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>스튜디오 - 클립 {{ idx + 1 }}</title>
<style>
  /* ═══════════════════════════════════════════════════════════════════════
     프리미어 프로 2024 '편집(Editing)' 작업 영역을 그대로 재현한다.
       메뉴 막대 → 헤더(홈·가져오기/편집/내보내기·프로젝트명) →
       [좌상: 소스 | 효과 컨트롤]  [우상: 프로그램 모니터]
       [좌하: 프로젝트 | 효과]     [도구 막대 | 타임라인 | 오디오 미터]
     패널 사이의 어두운 홈(splitter)을 끌어 크기를 바꿀 수 있다(프리미어와 동일).
     ═══════════════════════════════════════════════════════════════════════ */
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body { background: #0f0f0f; color: #d6d6d6; font-family: 'Pretendard', -apple-system, 'Malgun Gothic', sans-serif;
    font-size: 12px; display: flex; flex-direction: column; overflow: hidden; user-select: none; }
  button { font-family: inherit; }
  svg { display: block; }
  :root { --blue: #3e8ef7; --hot: #4b9bff; --panel: #232323; --panel2: #1d1d1d; --bar: #1b1b1b; --line: #0c0c0c;
    --text: #d6d6d6; --dim: #8e8e8e; }

  /* ── 메뉴 막대(파일 편집 클립 시퀀스 마커 …) ── */
  .menubar { flex: 0 0 22px; display: flex; align-items: stretch; background: #1e1e1e; border-bottom: 1px solid #000;
    font-size: 11.5px; color: #cfcfcf; position: relative; z-index: 50; }
  .menubar .mi { padding: 0 9px; display: flex; align-items: center; cursor: default; position: relative; }
  .menubar .mi:hover, .menubar .mi.open { background: #3a3a3a; color: #fff; }
  .menu { display: none; position: absolute; top: 22px; left: 0; min-width: 210px; background: #2b2b2b;
    border: 1px solid #111; box-shadow: 0 6px 18px rgba(0,0,0,.6); padding: 4px 0; z-index: 60; }
  .mi.open .menu { display: block; }
  .menu .it, .ctxmenu .it { display: flex; align-items: center; justify-content: space-between; gap: 22px; padding: 4px 22px 4px 26px;
    color: #e0e0e0; white-space: nowrap; cursor: default; position: relative; }
  .menu .it:hover, .ctxmenu .it:hover { background: var(--blue); color: #fff; }
  .menu .it.dis, .ctxmenu .it.dis { color: #6a6a6a; pointer-events: none; }
  .menu .it .k, .ctxmenu .it .k { color: #9a9a9a; font-size: 11px; }
  .menu .it:hover .k, .ctxmenu .it:hover .k { color: #e8f0ff; }
  .menu .it.chk::before, .ctxmenu .it.chk::before { content: '✓'; position: absolute; left: 10px; font-size: 10px; }
  /* 타임라인 우클릭 컨텍스트 메뉴(프리미어처럼 클립/빈 영역에서 우클릭 시 뜬다) */
  .ctxmenu { display: none; position: fixed; min-width: 200px; background: #2b2b2b; border: 1px solid #111;
    box-shadow: 0 6px 18px rgba(0,0,0,.6); padding: 4px 0; z-index: 200; }
  .ctxmenu.on { display: block; }
  .menu .sep, .ctxmenu .sep { height: 1px; background: #444; margin: 4px 8px; }

  /* ── 헤더(홈 · 가져오기/편집/내보내기 · 프로젝트명 · 우측 아이콘) ── */
  .hdr { flex: 0 0 36px; display: flex; align-items: center; background: #121212; border-bottom: 1px solid #000;
    padding: 0 10px; gap: 4px; }
  .hdr .home { width: 26px; height: 26px; display: flex; align-items: center; justify-content: center; color: #bdbdbd;
    border-radius: 4px; text-decoration: none; }
  .hdr .home:hover { background: #2a2a2a; color: #fff; }
  .hdr .mode { padding: 0 12px; height: 36px; display: flex; align-items: center; color: #9a9a9a; font-size: 12.5px;
    border-bottom: 2px solid transparent; cursor: pointer; }
  .hdr .mode:hover { color: #e6e6e6; }
  .hdr .mode.on { color: #fff; border-bottom-color: var(--blue); }
  .hdr .proj { flex: 1 1 auto; text-align: center; font-size: 12.5px; color: #e6e6e6; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; padding: 0 12px; }
  .hdr .proj .edited { color: #9a9a9a; }
  .hdr .hic { width: 28px; height: 26px; display: flex; align-items: center; justify-content: center; color: #bdbdbd;
    background: transparent; border: none; border-radius: 4px; cursor: pointer; }
  .hdr .hic:hover { background: #2a2a2a; color: #fff; }
  .hdr .hic.export { color: var(--hot); }
  /* 저장하기: 어두운 프리미어 헤더에서 유일한 애플 스타일 흰 pill(프로젝트 규칙) —
     Ctrl+S를 모르면 편집한 값(크기·가로·세로)이 저장 안 된 채 내보내기로 가던 문제 해결. */
  .hdr .savebtn { margin: 0 4px 0 6px; padding: 4px 14px; border: 0; border-radius: 999px;
    background: #fff; color: #1d1d1f; font-size: 11.5px; font-weight: 700; letter-spacing: -0.01em;
    cursor: pointer; white-space: nowrap;
    box-shadow: 0 1px 2px rgba(0,0,0,.45), 0 4px 12px rgba(0,0,0,.35);
    transition: transform .12s ease, box-shadow .12s ease, background .12s ease; }
  .hdr .savebtn:hover { transform: translateY(-1px); box-shadow: 0 2px 5px rgba(0,0,0,.5), 0 8px 20px rgba(0,0,0,.45); }
  .hdr .savebtn:active { transform: translateY(0); }
  .hdr .savebtn:disabled { opacity: .65; cursor: default; transform: none; }
  .hdr .savebtn.saved { background: #34c759; color: #fff; }
  .hdr .savebtn.dirty { background: #ffd60a; }

  /* ── 작업 영역 그리드 ── */
  .ws { flex: 1 1 auto; min-height: 0; display: grid; gap: 0; padding: 3px;
    grid-template-columns: var(--colL, 34%) 5px 1fr;
    grid-template-rows: var(--rowT, 58%) 5px 1fr; }
  .split-v { grid-column: 2; grid-row: 1; cursor: col-resize; }
  .split-h { grid-column: 1 / 4; grid-row: 2; cursor: row-resize; }
  .split-v:hover, .split-h:hover, .split-v.on, .split-h.on { background: rgba(62,142,247,.35); }

  /* ── 패널 공통: 탭 막대 + 본문. 클릭한 패널은 파란 테두리(프리미어의 활성 패널 표시) ── */
  .panel { display: flex; flex-direction: column; min-width: 0; min-height: 0; background: var(--panel);
    border: 1px solid #0a0a0a; position: relative; }
  .panel.focus { border-color: var(--blue); }
  .tabs { flex: 0 0 26px; display: flex; align-items: stretch; background: var(--panel2); border-bottom: 1px solid #000; }
  .tab { display: flex; align-items: center; gap: 6px; padding: 0 12px; font-size: 11.5px; color: #8f8f8f;
    border-right: 1px solid #0d0d0d; cursor: default; white-space: nowrap; }
  .tab.on { background: var(--panel); color: #e6e6e6; }
  .tab .tname { max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
  .tab .mm { color: #6a6a6a; margin-left: 2px; font-size: 13px; line-height: 1; }
  .tabs .fill { flex: 1 1 auto; }
  .tabs .tabmenu { width: 24px; display: flex; align-items: center; justify-content: center; color: #6a6a6a; }
  .pbody { flex: 1 1 auto; min-height: 0; position: relative; display: none; }
  .pbody.on { display: flex; flex-direction: column; }

  /* ── 소스 모니터 ── */
  .srcstage { flex: 1 1 auto; min-height: 0; position: relative; background: #101010; display: flex; align-items: center; justify-content: center; }
  .srcstage video { max-width: 96%; max-height: 96%; background: #000; }
  .srcstage .empty { color: #6a6a6a; font-size: 12px; }

  /* ── 효과 컨트롤 ── */
  .ec { flex: 1 1 auto; min-height: 0; display: flex; }
  .ec-left { flex: 1 1 300px; min-width: 260px; display: flex; flex-direction: column; border-right: 1px solid #000; }
  .ec-right { flex: 0 1 220px; min-width: 0; display: flex; flex-direction: column; background: #1a1a1a; }
  .ec-head { flex: 0 0 26px; display: flex; align-items: center; gap: 8px; padding: 0 10px; font-size: 11.5px;
    background: #202020; border-bottom: 1px solid #000; color: #cfcfcf; }
  .ec-head .mst { color: #e6e6e6; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ec-head .seq { color: #9a9a9a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ec-list { flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden; padding: 4px 0 12px; }
  .ec-sec { padding: 6px 10px 2px; font-size: 11.5px; color: #cfcfcf; font-weight: 700; }
  .fx { }
  .fx-h { display: flex; align-items: center; gap: 6px; padding: 3px 10px; cursor: default; }
  .fx-h:hover { background: #2a2a2a; }
  .fx-h .tri { width: 10px; color: #9a9a9a; font-size: 9px; }
  .fx-h .fxi { width: 16px; height: 13px; border-radius: 2px; background: #3a3a3a; color: #bdbdbd; font-size: 9px;
    display: inline-flex; align-items: center; justify-content: center; font-style: italic; font-weight: 700; }
  .fx-h .fxn { flex: 1 1 auto; color: #e6e6e6; }
  .fx-h .rst { color: #8e8e8e; font-size: 10.5px; }
  .fx-h .rst:hover { color: #fff; }
  .fx.closed .fx-b { display: none; }
  .fx.closed .fx-h .tri { transform: rotate(-90deg); }
  .prm { display: flex; align-items: center; gap: 8px; padding: 3px 10px 3px 30px; }
  .prm:hover { background: #262626; }
  .prm .stop { width: 12px; height: 12px; border-radius: 50%; border: 1px solid #6a6a6a; flex: 0 0 12px; }
  .prm .pn { flex: 0 0 64px; color: #bdbdbd; white-space: nowrap; }
  .prm .pv { display: flex; align-items: center; gap: 4px; flex: 1 1 auto; min-width: 0; }
  .hot { background: transparent; border: none; border-bottom: 1px dotted transparent; color: var(--hot); font-family: inherit;
    font-size: 12px; width: 74px; padding: 1px 2px; text-align: right; cursor: ew-resize; font-variant-numeric: tabular-nums; }
  .hot:hover { border-bottom-color: var(--hot); }
  .hot:focus { outline: none; background: #111; border: 1px solid var(--blue); border-radius: 2px; cursor: text; }
  .hot:disabled { color: #5a5a5a; cursor: default; }
  .hot.wide { width: 96px; }
  .prm .unit { color: #7a7a7a; font-size: 11px; }
  .prm .txt { flex: 1 1 auto; min-width: 0; background: #151515; border: 1px solid #333; color: #e6e6e6;
    border-radius: 2px; padding: 3px 6px; font-family: inherit; font-size: 12px; }
  .prm .txt:focus { outline: none; border-color: var(--blue); }
  /* 값 조절은 막대(슬라이더) 대신 애플식 둥근 −/+ 버튼 + 숫자(좌우 드래그도 됨).
     사용자 요청 2026-09-07: "크기·위치 저 움직이는 바 형태 별로다". */
  .prm .stp { width: 20px; height: 20px; flex: 0 0 20px; border: 0; border-radius: 50%;
    background: #f5f5f7; color: #1d1d1f; font-size: 13px; font-weight: 700; line-height: 1;
    display: inline-flex; align-items: center; justify-content: center; cursor: pointer;
    box-shadow: 0 1px 2px rgba(0,0,0,.45), 0 3px 8px rgba(0,0,0,.30);
    transition: transform .12s ease, background .12s ease; }
  .prm .stp:hover { background: #fff; transform: translateY(-1px); }
  .prm .stp:active { background: #e5e5ea; transform: translateY(0); }
  /* 자막 스타일(글꼴·정렬·자간) — 캡컷처럼 스튜디오 안에서 바로 고를 수 있게(2026-09-08 요청) */
  #pFont { flex: 1 1 auto; min-width: 0; background: #151515; border: 1px solid #333; color: #e6e6e6;
    border-radius: 4px; padding: 3px 6px; font-family: inherit; font-size: 12px; }
  #pFont:focus { outline: none; border-color: var(--blue); }
  .pv.align3 { gap: 6px; }
  .align-btn { border-radius: 50%; font-size: 12px; }
  .align-btn.on { background: var(--blue); color: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.45), 0 0 0 2px rgba(75,155,255,.4); }
  .pat-btn { border-radius: 50%; font-size: 11px; }
  .pat-btn.on { background: var(--blue); color: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.45), 0 0 0 2px rgba(75,155,255,.4); }
  input.swatch[type="color"] { width: 26px; height: 20px; padding: 0; border: 1px solid #444; border-radius: 5px; background: none; cursor: pointer; }
  input.swatch.big[type="color"] { width: 40px; height: 30px; border-radius: 7px; border: 1px solid #555; }
  #capPresets { flex-wrap: wrap; gap: 6px; }
  .cap-preset { width: 22px; height: 22px; border-radius: 50%; border: 2px solid #444; cursor: pointer; padding: 0;
    display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 800; flex: 0 0 auto; }
  .cap-preset.on { border-color: #fff; box-shadow: 0 0 0 2px var(--blue); }
  .prm .chk { accent-color: var(--hot); }
  .prm .dimtxt { color: #7a7a7a; }
  .ec-none { padding: 10px 30px; color: #6a6a6a; }
  /* 우측 키프레임 미니 타임라인 */
  .kf-ruler { flex: 0 0 26px; position: relative; border-bottom: 1px solid #000; background: #202020; }
  .kf-ruler .tk { position: absolute; top: 0; height: 100%; border-left: 1px solid #3a3a3a; padding-left: 3px;
    font-size: 9.5px; color: #8e8e8e; line-height: 24px; white-space: nowrap; }
  .kf-body { flex: 1 1 auto; position: relative; overflow: hidden; }
  .kf-body .span { position: absolute; top: 34px; height: 18px; background: rgba(75,155,255,.25); border: 1px solid rgba(75,155,255,.7); }
  .kf-body .lbl { position: absolute; left: 8px; color: #6a6a6a; font-size: 10.5px; }
  .kf-body .ph { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--blue); }
  .kf-ruler .ph { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--blue); }
  .kf-ruler .ph::before { content: ''; position: absolute; left: -5px; top: 0; border: 5.5px solid transparent; border-top: 8px solid var(--blue); }

  /* ── 프로그램 모니터 ── */
  /* 프로그램 모니터는 '원본 영상'이 아니라 **완성될 세로 영상 한 프레임 그대로**를 그린다.
     (예전엔 원본 16:9 위에 자막을 bottom 24%로 얹어서, 스튜디오에서 맞춘 자막 위치가
      실제 추출물과 전혀 달랐다 — 실신고 2026-09-07.) 좌표는 확인 팝업과 같은 공식:
     프레임 = layout.resolution, 영상 = layout.video_box, 자막 top = caption_base_margin_v + y. */
  .stage { flex: 1 1 auto; min-height: 0; position: relative; background: #101010; overflow: hidden; }
  /* 영상만 전체화면(우측 하단) — 상단 메뉴의 '전체 화면'(hFull, 앱 전체 F11)과는 별개다. */
  .stageFullBtn { position: absolute; right: 10px; bottom: 10px; z-index: 6; width: 28px; height: 28px;
    border-radius: 8px; border: none; background: rgba(30,30,30,.72); color: #ddd; cursor: pointer;
    display: flex; align-items: center; justify-content: center; backdrop-filter: blur(4px); }
  .stageFullBtn:hover { background: rgba(60,60,60,.9); color: #fff; }
  .stage:fullscreen { background: #000; }
  .stage:fullscreen .stageFullBtn { background: rgba(30,30,30,.55); }
  .frame { position: absolute; background: #fff; overflow: hidden; box-shadow: 0 0 0 1px #303030; }
  .vbox { position: absolute; background: #000; overflow: hidden; }
  .vbox video { width: 100%; height: 100%; display: block; background: #000; }
  .ttl-ov { position: absolute; left: 50%; transform: translateX(-50%); text-align: center;
    white-space: nowrap; width: max-content; pointer-events: none; font-weight: 800;
    color: #191f28; line-height: 1.25; z-index: 3; }
  .cap-ov { position: absolute; left: 50%; transform: translateX(-50%); text-align: center;
    width: max-content; max-width: 96%; font-weight: 800; color: #191f28; line-height: 1.25;
    z-index: 4; cursor: grab; touch-action: none; padding: 1px 5px; border-radius: 7px;
    border: 1.5px dashed transparent; }
  .cap-ov:hover, .cap-ov.dragging { border-color: var(--hot); background: rgba(75,155,255,.16); }
  .cap-ov.dragging { cursor: grabbing; }
  .cap-ov .en { display: block; font-weight: 600; color: #8b95a1; }
  /* 영상 위 오버레이(업로드 찬양)는 실제 렌더가 흰 글씨 + 검은 외곽선이다. */
  .cap-ov.onvideo { color: #fff;
    text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000, 0 0 6px rgba(0,0,0,.7); }
  .cap-ov.onvideo .en { color: #f2f2f2; text-shadow: 0 1px 4px rgba(0,0,0,.8); }
  .safe { position: absolute; pointer-events: none; border: 1px dashed rgba(255,80,80,.5); display: none; z-index: 2; }
  .safe.right { top: 0; bottom: 0; right: 0; width: 13%; }
  .safe.bottom { left: 0; right: 0; bottom: 0; height: 12%; }
  .frame.showsafe .safe { display: block; }
  .mon-bar { flex: 0 0 26px; display: flex; align-items: center; gap: 10px; padding: 0 10px; background: #1e1e1e;
    border-top: 1px solid #000; font-size: 11.5px; }
  .mon-bar .tc { color: var(--hot); font-variant-numeric: tabular-nums; font-size: 12px; }
  .mon-bar .tc.dur { color: #9a9a9a; margin-left: auto; }
  .mon-bar .dd { color: #bdbdbd; background: transparent; border: none; font-family: inherit; font-size: 11.5px; cursor: pointer; }
  .mon-bar .dd:hover { color: #fff; }
  .transport { flex: 0 0 34px; display: flex; align-items: center; justify-content: center; gap: 2px; background: #1e1e1e;
    border-top: 1px solid #000; }
  .tb { width: 30px; height: 26px; display: inline-flex; align-items: center; justify-content: center; background: transparent;
    border: none; border-radius: 3px; color: #cfcfcf; cursor: pointer; }
  .tb:hover { background: #333; color: #fff; }
  .tb.play { width: 34px; }
  .tb.on { color: var(--hot); }
  .tsep { width: 1px; height: 16px; background: #3a3a3a; margin: 0 6px; }

  /* ── 프로젝트 패널 ── */
  .pj-tools { flex: 0 0 30px; display: flex; align-items: center; gap: 6px; padding: 0 8px; border-bottom: 1px solid #000; background: #202020; }
  .pj-tools .search { flex: 1 1 auto; min-width: 0; display: flex; align-items: center; gap: 6px; background: #151515;
    border: 1px solid #333; border-radius: 3px; padding: 3px 7px; color: #8e8e8e; }
  .pj-tools .search input { flex: 1 1 auto; min-width: 0; background: transparent; border: none; color: #e6e6e6;
    font-family: inherit; font-size: 11.5px; }
  .pj-tools .search input:focus { outline: none; }
  .pj-cols { flex: 0 0 22px; display: flex; align-items: center; padding: 0 8px; font-size: 10.5px; color: #8e8e8e;
    border-bottom: 1px solid #000; background: #1f1f1f; }
  .pj-cols span:first-child { flex: 1 1 auto; padding-left: 26px; }
  .pj-cols span { flex: 0 0 90px; }
  .pj-list { flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 2px 0; }
  .pj-row { display: flex; align-items: center; gap: 6px; padding: 3px 8px; cursor: default; }
  .pj-row:hover { background: #2a2a2a; }
  .pj-row.sel { background: #35476a; color: #fff; }
  .pj-row .ico { width: 18px; color: #9a9a9a; display: flex; justify-content: center; }
  .pj-row .nm { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .pj-row .meta { flex: 0 0 90px; color: #8e8e8e; font-size: 11px; font-variant-numeric: tabular-nums; }
  .pj-foot { flex: 0 0 26px; display: flex; align-items: center; gap: 2px; padding: 0 6px; border-top: 1px solid #000; background: #1e1e1e; }
  .pj-foot .fi { width: 24px; height: 22px; display: inline-flex; align-items: center; justify-content: center; color: #9a9a9a;
    background: transparent; border: none; border-radius: 3px; cursor: pointer; }
  .pj-foot .fi:hover { background: #333; color: #fff; }
  .pj-foot .fi.on { color: var(--hot); }
  .pj-foot .zoom { width: 70px; accent-color: #9a9a9a; height: 12px; }
  .pj-foot .fill { flex: 1 1 auto; }
  /* 효과 탭(트리) */
  .fx-tree { flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 4px 0; }
  .fx-fold { }
  .fx-fold > .fh { display: flex; align-items: center; gap: 6px; padding: 3px 8px; cursor: default; }
  .fx-fold > .fh:hover { background: #2a2a2a; }
  .fx-fold > .fh .tri { width: 10px; font-size: 9px; color: #9a9a9a; }
  .fx-fold.closed > .fh .tri { transform: rotate(-90deg); }
  .fx-fold.closed > .fl { display: none; }
  .fx-fold .fh .fico { color: #c9a24a; }
  .fx-item { display: flex; align-items: center; gap: 6px; padding: 3px 8px 3px 30px; cursor: default; }
  .fx-item:hover { background: #2a2a2a; }
  .fx-item .fxi { width: 16px; height: 13px; border-radius: 2px; background: #3a3a3a; color: #bdbdbd; font-size: 9px;
    display: inline-flex; align-items: center; justify-content: center; font-style: italic; font-weight: 700; }
  .fx-item .nm { flex: 1 1 auto; }
  .fx-item .apply { display: none; font-size: 10.5px; color: var(--hot); background: transparent; border: 1px solid var(--hot);
    border-radius: 3px; padding: 1px 7px; cursor: pointer; }
  .fx-item:hover .apply { display: inline-block; }
  .fx-item.busy .nm::after { content: ' · 처리 중…'; color: var(--hot); }

  /* ── 하단 우측: 도구 막대 + 타임라인 + 오디오 미터 ── */
  .bottom-right { display: flex; min-width: 0; min-height: 0; gap: 3px; }
  .tools { flex: 0 0 28px; display: flex; flex-direction: column; align-items: center; gap: 2px; padding-top: 6px;
    background: var(--panel); border: 1px solid #0a0a0a; }
  .tools.focus { border-color: var(--blue); }
  .tool { width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; background: transparent;
    border: none; border-radius: 3px; color: #cfcfcf; cursor: pointer; position: relative; }
  .tool:hover { background: #333; color: #fff; }
  .tool.on { background: #2f2f2f; color: var(--hot); }
  .tool .sub { position: absolute; right: 2px; bottom: 2px; width: 0; height: 0; border-left: 3px solid transparent; border-bottom: 3px solid #8e8e8e; }
  .tl-panel { flex: 1 1 auto; min-width: 0; }
  .tl-head { flex: 0 0 30px; display: flex; align-items: center; gap: 4px; padding: 0 8px; background: #1e1e1e; border-bottom: 1px solid #000; }
  .tl-head .tc { color: var(--hot); font-size: 12.5px; font-variant-numeric: tabular-nums; margin-right: 8px; }
  .tl-head .status { margin-left: auto; color: var(--hot); font-size: 11.5px; }
  .hb { width: 24px; height: 22px; display: inline-flex; align-items: center; justify-content: center; background: transparent;
    border: none; border-radius: 3px; color: #bdbdbd; cursor: pointer; }
  .hb:hover { background: #333; color: #fff; }
  .hb.on { color: var(--hot); }
  .tl-body { flex: 1 1 auto; min-height: 0; display: flex; }
  .tl-heads { flex: 0 0 168px; display: flex; flex-direction: column; background: #232323; border-right: 1px solid #000; overflow: hidden; }
  .tl-heads .ruler-pad { flex: 0 0 24px; border-bottom: 1px solid #000; background: #1e1e1e; }
  .tl-tracks { flex: 1 1 auto; min-width: 0; position: relative; display: flex; flex-direction: column; overflow: hidden; background: #1a1a1a; }
  .tl-tracks-scroll, .tl-heads-scroll { flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden; }
  .tl-heads-scroll { scrollbar-width: none; }
  .tl-heads-scroll::-webkit-scrollbar { display: none; }
  .trk-h { display: flex; align-items: center; gap: 3px; padding: 0 4px; border-bottom: 1px solid #111; }
  .trk-h .patch { width: 22px; height: 16px; font-size: 9.5px; display: inline-flex; align-items: center; justify-content: center;
    color: #bdbdbd; background: #3a3a3a; border-radius: 2px; }
  .trk-h .th { width: 18px; height: 18px; display: inline-flex; align-items: center; justify-content: center; background: transparent;
    border: none; border-radius: 2px; color: #7a7a7a; cursor: pointer; }
  .trk-h .th:hover { background: #333; color: #ddd; }
  .trk-h .th.on { color: #e6e6e6; }
  .trk-h .th.lock.on { color: #ffb454; }
  .trk-h .th.mute.on { color: #ffb454; }
  .trk-h .th.solo.on { color: #ffb454; }
  .trk-h .tn { flex: 1 1 auto; font-size: 10.5px; color: #bdbdbd; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; margin-left: 2px; }
  .trk { position: relative; border-bottom: 1px solid #111; background: #1f1f1f; }
  .trk.locked { background: repeating-linear-gradient(135deg, #1f1f1f 0 6px, #242424 6px 12px); }
  .trk.h-c { height: 34px; } .trk.h-v { height: 56px; } .trk.h-a { height: 52px; }
  .trk.a { background: #1c1f1f; }
  .ruler { flex: 0 0 24px; position: relative; background: #1e1e1e; border-bottom: 1px solid #000; cursor: pointer; overflow: hidden; }
  .ruler .tk { position: absolute; top: 0; height: 100%; border-left: 1px solid #4a4a4a; padding-left: 3px; font-size: 10px;
    color: #9a9a9a; line-height: 22px; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .ruler .tk.minor { border-left-color: #333; height: 6px; top: auto; bottom: 0; }
  .ruler .inout { position: absolute; top: 0; height: 100%; background: rgba(255,255,255,.08); border-left: 1px solid #bdbdbd; border-right: 1px solid #bdbdbd; pointer-events: none; }
  .ruler .mk { position: absolute; top: 12px; width: 9px; height: 9px; transform: translateX(-50%) rotate(45deg); background: #3fbf6f; border: 1px solid #1a1a1a; pointer-events: none; }
  .ph-line { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--blue); z-index: 6; pointer-events: none; }
  .ph-head { position: absolute; top: 0; width: 13px; height: 24px; transform: translateX(-50%); z-index: 7; pointer-events: none; }
  .ph-head::before { content: ''; position: absolute; left: 0; top: 0; width: 13px; height: 14px; background: var(--blue); border-radius: 2px 2px 0 0; }
  .ph-head::after { content: ''; position: absolute; left: 0; top: 14px; border-left: 6.5px solid transparent; border-right: 6.5px solid transparent; border-top: 8px solid var(--blue); }
  .thumbs { position: absolute; left: 0; right: 0; top: 4px; bottom: 4px; display: flex; overflow: hidden; }
  .thumbs img { flex: 1 1 0; min-width: 0; height: 100%; object-fit: cover; opacity: .85; pointer-events: none; }
  .vclip, .aclip { position: absolute; top: 3px; bottom: 3px; border-radius: 2px; overflow: hidden; pointer-events: none; }
  .vclip { background: #3e4c6e; border: 1px solid #56689a; }
  .aclip { background: #2f5f5b; border: 1px solid #3f8a84; }
  /* A1(원본 오디오)은 영상과 같은 소스라 계속 잠겨 있지만(pointer-events:none 상속),
     A2(배경 음악)는 자유롭게 끌 수 있다 — 재생 시작 지점을 옮기는 것이라 싱크가 안 깨진다. */
  #bgmClip { pointer-events: auto; cursor: grab; }
  #bgmClip:active { cursor: grabbing; }
  /* V1은 이제 실제로 잘라 조각낼 수 있다 — 조각마다 선택·리사이즈 손잡이가 있다. */
  .vclip { pointer-events: auto; cursor: default; }
  .vclip .h { position: absolute; top: 0; bottom: 0; width: 7px; cursor: ew-resize; }
  .vclip .h.l { left: 0; } .vclip .h.r { right: 0; }
  .vclip .h:hover { background: rgba(255,255,255,.3); }
  .vclip.sel { box-shadow: inset 0 0 0 1px #fff; }
  .vclip .cn, .aclip .cn { position: absolute; left: 6px; top: 2px; font-size: 10.5px; color: #e8ecf5; text-shadow: 0 1px 2px #000; z-index: 2; }
  .aclip canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  .blk { position: absolute; top: 3px; bottom: 3px; background: #5a4f8f; border: 1px solid #8072c9; border-radius: 2px; overflow: hidden;
    cursor: grab; display: flex; align-items: center; }
  .blk.sel { background: #7a6dc0; border-color: #fff; box-shadow: inset 0 0 0 1px #fff; z-index: 2; }
  .blk .txt { padding: 0 8px; font-size: 11px; color: #f0edff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; pointer-events: none; width: 100%; }
  .blk .h { position: absolute; top: 0; bottom: 0; width: 7px; cursor: ew-resize; }
  .blk .h.l { left: 0; } .blk .h.r { right: 0; }
  .blk .h:hover { background: rgba(255,255,255,.3); }
  .trk.en .blk { background: #3f5f3a; border-color: #6b9a63; cursor: default; }
  .trk.en .blk .txt { color: #dff0d8; }
  .tl-panel.tool-razor .trk.ko .blk { cursor: crosshair; }
  .tl-panel.tool-ripple .trk.ko .blk .h { cursor: e-resize; background: rgba(255,200,80,.15); }
  .tl-panel.tool-roll .trk.ko .blk .h { cursor: col-resize; background: rgba(255,80,80,.15); }
  .tl-panel.tool-hand .tl-tracks { cursor: grab; }
  .tl-panel.tool-hand .trk .blk { cursor: grab; }
  .tl-panel.tool-type .trk.ko .blk { cursor: text; }
  .tl-panel.tool-trackfwd .trk.ko .blk { cursor: e-resize; }
  .trk.locked .blk { cursor: not-allowed; opacity: .7; }
  /* 자막 사이 빈 구간: 점선 + '+' 를 두어 그 자리에 바로 자막을 만든다(사용자 요청). */
  .blk-add { position: absolute; top: 5px; bottom: 5px; border: 1px dashed #6b6b6b; border-radius: 3px;
    display: flex; align-items: center; justify-content: center; color: #8e8e8e; font-size: 13px;
    font-weight: 700; cursor: pointer; background: rgba(255,255,255,.02); }
  .blk-add:hover { border-color: var(--hot); color: var(--hot); background: rgba(75,155,255,.12); }
  .blk-edit { position: absolute; z-index: 9; background: #2b2b2b; border: 1px solid var(--blue); border-radius: 4px; padding: 5px; display: flex; gap: 5px; align-items: center; }
  .blk-edit input { width: 340px; background: #151515; color: #e6e6e6; border: 1px solid #333; border-radius: 3px; padding: 5px 8px; font-family: inherit; font-size: 12px; }
  .blk-edit button { padding: 4px 9px; font-size: 11.5px; border-radius: 3px; border: 1px solid #444; background: #3a3a3a; color: #e6e6e6; cursor: pointer; }
  /* 자유 텍스트 트랙(캡컷식): 자막과 달리 겹쳐도 되고, 요소마다 화면 위치·크기를 갖는다. */
  .trk.tx { height: 30px; background: #23202c; }
  .txblk { position: absolute; top: 3px; bottom: 3px; background: #2f6f5e; border: 1px solid #58b39a;
    border-radius: 2px; overflow: hidden; cursor: grab; display: flex; align-items: center; }
  .txblk.sel { background: #3d8f78; border-color: #fff; box-shadow: inset 0 0 0 1px #fff; z-index: 3; }
  .txblk .txt { padding: 0 8px; font-size: 11px; color: #e6fff7; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; pointer-events: none; width: 100%; }
  .txblk .h { position: absolute; top: 0; bottom: 0; width: 7px; cursor: ew-resize; }
  .txblk .h.l { left: 0; } .txblk .h.r { right: 0; }
  .txblk .h:hover { background: rgba(255,255,255,.3); }
  .trk-h.addtrk { height: 26px; cursor: pointer; }
  .trk-h.addtrk:hover .tn { color: var(--hot); }
  .trk-h .addbtn { width: 18px; height: 18px; display: inline-flex; align-items: center; justify-content: center;
    background: #333; color: #ddd; border: none; border-radius: 4px; font-size: 13px; line-height: 1; cursor: pointer; }
  .trk-h .addbtn:hover { background: var(--blue); color: #fff; }
  .trk.drop-hint { box-shadow: inset 0 0 0 2px var(--hot); }
  /* 미리보기(프로그램 모니터) 위의 자유 텍스트 — 끌어서 위치를 옮기고 더블클릭으로 고친다. */
  .tx-ov { position: absolute; text-align: center; width: max-content; max-width: 96%; font-weight: 800;
    color: #191f28; line-height: 1.25; z-index: 5; cursor: grab; touch-action: none; transform: translateX(-50%);
    padding: 1px 5px; border-radius: 7px; border: 1.5px dashed transparent; white-space: pre-wrap; }
  .tx-ov:hover, .tx-ov.sel, .tx-ov.dragging { border-color: var(--hot); background: rgba(75,155,255,.16); }
  .tx-ov.dragging { cursor: grabbing; }
  .tx-ov.editing { cursor: text; background: rgba(75,155,255,.22); outline: none; }
  .tl-scroll { flex: 0 0 16px; display: flex; align-items: center; background: #1e1e1e; border-top: 1px solid #000; }
  .tl-scroll .pad { flex: 0 0 168px; }
  .navi { flex: 1 1 auto; position: relative; height: 10px; margin: 0 8px; background: #2a2a2a; border-radius: 5px; }
  .navi .thumb { position: absolute; top: 0; height: 100%; background: #6a6a6a; border-radius: 5px; cursor: grab; min-width: 14px; }
  .navi .thumb::before, .navi .thumb::after { content: ''; position: absolute; top: 0; width: 6px; height: 100%; background: #9a9a9a; cursor: ew-resize; }
  .navi .thumb::before { left: 0; border-radius: 5px 0 0 5px; } .navi .thumb::after { right: 0; border-radius: 0 5px 5px 0; }
  /* 오디오 미터 */
  .meters { flex: 0 0 44px; display: flex; flex-direction: column; background: var(--panel); border: 1px solid #0a0a0a; }
  .meters .tabs .tab { padding: 0 6px; }
  .meters .mb { flex: 1 1 auto; min-height: 0; display: flex; padding: 8px 4px 6px; gap: 3px; }
  .meters .scale { flex: 0 0 16px; display: flex; flex-direction: column; justify-content: space-between; font-size: 8px; color: #8e8e8e; text-align: right; font-variant-numeric: tabular-nums; }
  .meters .bar { flex: 1 1 0; position: relative; background: #111; border-radius: 1px; overflow: hidden; }
  .meters .bar .lv { position: absolute; left: 0; right: 0; bottom: 0; height: 0; background: linear-gradient(180deg, #ff4b4b 0%, #ffcc33 12%, #3fbf6f 30%, #2c8f52 100%); background-size: 100% var(--mh, 100px); background-position: bottom; }
  .meters .bar .pk { position: absolute; left: 0; right: 0; height: 2px; background: #fff; bottom: 0; }

  /* 단축키 도움말 팝업 */
  .modal-bg { position: fixed; inset: 0; background: rgba(0,0,0,.55); display: none; align-items: center; justify-content: center; z-index: 100; }
  .modal-bg.on { display: flex; }
  .modal { background: #2b2b2b; border: 1px solid #111; border-radius: 6px; padding: 16px 20px; width: 520px; max-height: 80vh; overflow: auto; box-shadow: 0 10px 40px rgba(0,0,0,.7); }
  .modal h2 { font-size: 13px; margin-bottom: 10px; color: #fff; }
  .modal table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  .modal td { padding: 3px 4px; border-bottom: 1px solid #383838; }
  .modal td:first-child { color: var(--hot); width: 130px; font-variant-numeric: tabular-nums; }
  .modal .close { margin-top: 12px; padding: 5px 14px; border-radius: 3px; border: 1px solid #444; background: #3a3a3a; color: #fff; cursor: pointer; }
</style>
</head>
<body>

<!-- ─── 메뉴 막대 ─── -->
<div class="menubar" id="menubar">
  <div class="mi" data-menu="file">파일
    <div class="menu">
      <div class="it" data-act="save">저장 <span class="k">Ctrl+S</span></div>
      <div class="it" data-act="render">내보내기(만들기)… <span class="k">Ctrl+M</span></div>
      <div class="sep"></div>
      <div class="it" data-act="back">후보 목록으로 돌아가기</div>
    </div>
  </div>
  <div class="mi" data-menu="edit">편집
    <div class="menu">
      <div class="it" data-act="undo">실행 취소 <span class="k">Ctrl+Z</span></div>
      <div class="it" data-act="redo">다시 실행 <span class="k">Ctrl+Shift+Z</span></div>
      <div class="sep"></div>
      <div class="it" data-act="delete">지우기 <span class="k">Delete</span></div>
      <div class="it" data-act="ripple-delete">잔물결 삭제 <span class="k">Shift+Delete</span></div>
      <div class="sep"></div>
      <div class="it" data-act="select-all">모두 선택 <span class="k">Ctrl+A</span></div>
      <div class="it" data-act="deselect">모두 선택 해제 <span class="k">Ctrl+Shift+A</span></div>
    </div>
  </div>
  <div class="mi" data-menu="clip">클립
    <div class="menu">
      <div class="it" data-act="edit-text">텍스트 수정… <span class="k">Enter</span></div>
      <div class="it" data-act="split">재생 헤드에서 분할 <span class="k">Ctrl+K</span></div>
      <div class="it" data-act="add">재생 헤드에 자막 추가 <span class="k">Ctrl+Shift+N</span></div>
    </div>
  </div>
  <div class="mi" data-menu="seq">시퀀스
    <div class="menu">
      <div class="it" data-act="mark-in">시작 표시(클립 시작) <span class="k">I</span></div>
      <div class="it" data-act="mark-out">종료 표시(클립 끝) <span class="k">O</span></div>
      <div class="sep"></div>
      <div class="it" data-act="zoom-in">확대 <span class="k">=</span></div>
      <div class="it" data-act="zoom-out">축소 <span class="k">-</span></div>
      <div class="it" data-act="zoom-fit">시퀀스에 맞게 확대/축소 <span class="k">\</span></div>
      <div class="sep"></div>
      <div class="it" data-act="snap" id="miSnap">타임라인에서 스냅 <span class="k">S</span></div>
    </div>
  </div>
  <div class="mi" data-menu="mark">마커
    <div class="menu">
      <div class="it" data-act="marker">마커 추가 <span class="k">M</span></div>
      <div class="it" data-act="marker-clear">모든 마커 지우기</div>
    </div>
  </div>
  <div class="mi" data-menu="gfx">그래픽 및 제목
    <div class="menu">
      <div class="it" data-act="fx-correct">AI 교정 적용</div>
      <div class="it" data-act="fx-translate">영어 번역 적용</div>
      <div class="it" data-act="fx-sync">싱크 맞추기 적용</div>
    </div>
  </div>
  <div class="mi" data-menu="view">보기
    <div class="menu">
      <div class="it" data-act="safe" id="miSafe">안전 여백 표시</div>
      <div class="it" data-act="en-vis" id="miEn">영어 자막 표시</div>
      <div class="sep"></div>
      <div class="it" data-act="fit">재생 해상도 맞춤</div>
    </div>
  </div>
  <div class="mi" data-menu="win">창
    <div class="menu">
      <div class="it" data-act="ws-reset">작업 영역 › 편집(재설정)</div>
      <div class="it" data-act="fullscreen">전체 화면 <span class="k">F11</span></div>
    </div>
  </div>
  <div class="mi" data-menu="help">도움말
    <div class="menu">
      <div class="it" data-act="shortcuts">키보드 단축키… <span class="k">Ctrl+Alt+K</span></div>
    </div>
  </div>
</div>

<!-- ─── 헤더 ─── -->
<div class="hdr">
  <a class="home" href="/video/{{ video_id }}" title="홈(후보 목록)">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M3 11l9-8 9 8v9a1 1 0 01-1 1h-5v-6H9v6H4a1 1 0 01-1-1z"/></svg>
  </a>
  <div class="mode" id="modeImport">가져오기</div>
  <div class="mode on">편집</div>
  <div class="mode" id="modeExport">내보내기</div>
  <div class="proj"><span id="projName">클립 {{ idx + 1 }}</span><span class="edited" id="projEdited"></span></div>
  <button class="savebtn" id="hSave" title="편집한 내용(자막·글씨 크기·가로/세로 위치 등)을 저장합니다 (Ctrl+S)">저장하기</button>
  <button class="hic" id="hWorkspace" title="작업 영역">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 12h18M12 12v9"/></svg>
  </button>
  <button class="hic export" id="hExport" title="빠른 내보내기(만들기)">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M6 10l6-6 6 6"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/></svg>
  </button>
  <button class="hic" id="hFull" title="전체 화면">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>
  </button>
</div>

<!-- ─── 작업 영역: 상단 2분할(소스·효과·프로젝트 병합 / 프로그램) + 하단 전체 너비 타임라인 ─── -->
<div class="ws" id="ws">

  <!-- ① 소스 / 효과 컨트롤 / 프로젝트(병합) -->
  <div class="panel" id="pSrcEc" style="grid-column:1;grid-row:1">
    <div class="tabs">
      <div class="tab" data-tab="src"><span class="tname">소스: 원본 영상</span><span class="mm">≡</span></div>
      <div class="tab on" data-tab="ec"><span class="tname">효과 컨트롤</span><span class="mm">≡</span></div>
      <div class="tab" data-tab="proj"><span class="tname">프로젝트: <span id="projName2">클립 {{ idx + 1 }}</span></span><span class="mm">≡</span></div>
      <div class="tab" data-tab="media"><span class="tname">미디어 브라우저</span></div>
      <div class="tab" data-tab="fx"><span class="tname">효과</span></div>
      <div class="fill"></div><div class="tabmenu">»</div>
    </div>
    <div class="pbody" data-body="src">
      <div class="srcstage" id="srcStage"><span class="empty">(탭을 열면 원본 영상을 불러옵니다)</span></div>
      <div class="mon-bar"><span class="tc" id="srcTc">00:00:00.000</span><span class="tc dur" id="srcDur">00:00:00.000</span></div>
      <div class="transport">
        <button class="tb" id="srcBack" title="한 프레임 뒤로"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h2v14H6zM20 5v14L9 12z"/></svg></button>
        <button class="tb play" id="srcPlay" title="재생/정지"><svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4l13 8-13 8z"/></svg></button>
        <button class="tb" id="srcFwd" title="한 프레임 앞으로"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M16 5h2v14h-2zM4 5v14l11-7z"/></svg></button>
      </div>
    </div>
    <div class="pbody on" data-body="ec">
      <div class="ec">
        <div class="ec-left">
          <div class="ec-head"><span class="mst" id="ecMaster">마스터 * 클립</span><span class="seq">∨</span><span class="seq" id="ecSeq">시퀀스 * 클립</span></div>
          <div class="ec-list" id="ecList">
            <div class="ec-sec">자막(한글) — 트랙 전체</div>
            <div class="fx" id="fxStyle">
              <div class="fx-h"><span class="tri">▼</span><span class="fxi">fx</span><span class="fxn">텍스트 스타일</span><span class="rst" data-reset="style">재설정</span></div>
              <div class="fx-b">
                <div class="prm"><span class="stop"></span><span class="pn">스타일</span><div class="pv" id="capPresets"></div></div>
                <div class="prm"><span class="stop"></span><span class="pn">글꼴</span><div class="pv"><select class="hot wide" id="pFont"></select></div></div>
                <div class="prm"><span class="stop"></span><span class="pn">정렬</span><div class="pv align3">
                  <button type="button" class="stp align-btn" data-align="left" title="왼쪽 정렬">◧</button>
                  <button type="button" class="stp align-btn" data-align="center" title="가운데 정렬">▣</button>
                  <button type="button" class="stp align-btn" data-align="right" title="오른쪽 정렬">◨</button>
                </div></div>
                <div class="prm"><span class="stop"></span><span class="pn">글자색</span><div class="pv"><input type="color" id="pTextColor" class="swatch big"><span class="dimtxt">탭하면 원하는 색 아무거나</span><button type="button" class="stp txt-reset" id="pTextColorReset" title="기본색으로">↺</button></div></div>
                <div class="prm"><span class="stop" style="visibility:hidden"></span><span class="pn">박스</span><div class="pv"><input class="chk" type="checkbox" id="pBoxOn"><span class="dimtxt">글자 뒤에 색 박스 깔기</span></div></div>
                <div class="prm" id="pBoxRow1"><span class="stop"></span><span class="pn">박스 색</span><div class="pv"><input type="color" id="pBoxColor" class="swatch big"><input class="hot" type="number" id="pBoxOpacity" min="0" max="100" step="5"><span class="unit">% 진하기</span></div></div>
                <div class="prm" id="pBoxRow2"><span class="stop"></span><span class="pn">둥근 정도</span><div class="pv"><button class="stp" data-nudge="pBoxRadius:-5" title="각지게">&minus;</button><input class="hot" type="number" id="pBoxRadius" min="0" max="100" step="1"><button class="stp" data-nudge="pBoxRadius:5" title="둥글게">+</button><span class="unit">%</span></div></div>
                <div class="prm" id="pBoxRow3"><span class="stop"></span><span class="pn">박스 높이</span><div class="pv"><button class="stp" data-nudge="pBoxH:-2" title="좁게">&minus;</button><input class="hot" type="number" id="pBoxH" min="0" max="100" step="1"><button class="stp" data-nudge="pBoxH:2" title="넓게">+</button><span class="unit">%</span></div></div>
                <div class="prm" id="pBoxRow4"><span class="stop"></span><span class="pn">박스 너비</span><div class="pv"><button class="stp" data-nudge="pBoxW:-2" title="좁게">&minus;</button><input class="hot" type="number" id="pBoxW" min="0" max="100" step="1"><button class="stp" data-nudge="pBoxW:2" title="넓게">+</button><span class="unit">%</span></div></div>
                <div class="prm" id="pBoxRow5"><span class="stop"></span><span class="pn">박스 X</span><div class="pv"><button class="stp" data-nudge="pBoxOX:-4" title="왼쪽으로">&minus;</button><input class="hot" type="number" id="pBoxOX" min="-300" max="300" step="1"><button class="stp" data-nudge="pBoxOX:4" title="오른쪽으로">+</button><span class="unit">px</span></div></div>
                <div class="prm" id="pBoxRow6"><span class="stop"></span><span class="pn">박스 Y</span><div class="pv"><button class="stp" data-nudge="pBoxOY:-4" title="위로">&minus;</button><input class="hot" type="number" id="pBoxOY" min="-300" max="300" step="1"><button class="stp" data-nudge="pBoxOY:4" title="아래로">+</button><span class="unit">px</span></div></div>
                <div class="prm"><span class="stop"></span><span class="pn">패턴</span><div class="pv">
                  <button type="button" class="stp pat-btn" id="pBold" title="볼드"><b>B</b></button>
                  <button type="button" class="stp pat-btn" id="pUnderline" title="밑줄"><u>U</u></button>
                  <button type="button" class="stp pat-btn" id="pItalic" title="이탤릭"><i>I</i></button>
                </div></div>
                <div class="prm"><span class="stop"></span><span class="pn">불투명도</span><div class="pv"><button class="stp" data-nudge="pTextOpacity:-5" title="투명하게">&minus;</button><input class="hot" type="number" id="pTextOpacity" min="0" max="100" step="5"><button class="stp" data-nudge="pTextOpacity:5" title="진하게">+</button><span class="unit">%</span></div></div>
                <div class="prm"><span class="stop" style="visibility:hidden"></span><span class="pn">획</span><div class="pv"><input class="chk" type="checkbox" id="pOutlineOn" checked><span class="dimtxt">글자 외곽선</span></div></div>
                <div class="prm" id="pOutlineRow1"><span class="stop"></span><span class="pn">획 색</span><div class="pv"><input type="color" id="pOutlineColor" class="swatch big" value="#000000"><button type="button" class="stp txt-reset" id="pOutlineColorReset" title="기본색으로">↺</button></div></div>
                <div class="prm" id="pOutlineRow2"><span class="stop"></span><span class="pn">획 두께</span><div class="pv"><button class="stp" data-nudge="pOutlineWidth:-1" title="얇게">&minus;</button><input class="hot" type="number" id="pOutlineWidth" min="0" max="20" step="1"><button class="stp" data-nudge="pOutlineWidth:1" title="굵게">+</button><span class="unit">px</span></div></div>
                <div class="prm"><span class="stop" style="visibility:hidden"></span><span class="pn">글로우</span><div class="pv"><input class="chk" type="checkbox" id="pGlowOn"><span class="dimtxt">은은하게 번지는 효과</span></div></div>
                <div class="prm" id="pGlowRow1"><span class="stop"></span><span class="pn">글로우 색</span><div class="pv"><input type="color" id="pGlowColor" class="swatch big" value="#00ffff"></div></div>
                <div class="prm"><span class="stop"></span><span class="pn">자간</span><div class="pv"><button class="stp" data-nudge="pSpacing:-0.5" title="좁게">&minus;</button><input class="hot" type="number" id="pSpacing" min="-4" max="20" step="0.5"><button class="stp" data-nudge="pSpacing:0.5" title="넓게">+</button><span class="unit">px</span></div></div>
                <div class="prm"><span class="stop"></span><span class="pn">크기</span><div class="pv"><button class="stp" data-nudge="pSize:-2" title="작게">&minus;</button><input class="hot" type="number" id="pSize" min="28" max="240" step="1"><button class="stp" data-nudge="pSize:2" title="크게">+</button><span class="unit">px</span></div></div>
                <div class="prm"><span class="stop"></span><span class="pn">가로 위치</span><div class="pv"><button class="stp" data-nudge="pX:-6" title="왼쪽으로">&minus;</button><input class="hot" type="number" id="pX" min="-400" max="400" step="1"><button class="stp" data-nudge="pX:6" title="오른쪽으로">+</button><span class="unit">px</span></div></div>
                <div class="prm"><span class="stop"></span><span class="pn">세로 위치</span><div class="pv"><button class="stp" data-nudge="pY:-6" title="위로">&minus;</button><input class="hot" type="number" id="pY" min="-700" max="700" step="1"><button class="stp" data-nudge="pY:6" title="아래로">+</button><span class="unit">px</span></div></div>
                <div class="prm"><span class="stop" style="visibility:hidden"></span><span class="pn">안전 여백</span><div class="pv"><input class="chk" type="checkbox" id="pSafe" checked><span class="dimtxt">휴대폰 UI 가이드(우측·하단)</span></div></div>
                <div class="prm"><span class="stop" style="visibility:hidden"></span><span class="pn"></span><div class="pv"><span class="dimtxt">미리보기의 자막을 끌어서 옮길 수도 있어요</span></div></div>
              </div>
            </div>
            <div class="ec-sec">자막(영어)</div>
            <div class="fx" id="fxStyleEn">
              <div class="fx-h"><span class="tri">▼</span><span class="fxi">fx</span><span class="fxn">텍스트 스타일</span><span class="rst" data-reset="styleEn">재설정</span></div>
              <div class="fx-b">
                <div class="prm"><span class="stop"></span><span class="pn">크기</span><div class="pv"><button class="stp" data-nudge="pSizeEn:-2" title="작게">&minus;</button><input class="hot" type="number" id="pSizeEn" min="0" max="200" step="1"><button class="stp" data-nudge="pSizeEn:2" title="크게">+</button><span class="unit">px</span></div></div>
                <div class="prm"><span class="stop" style="visibility:hidden"></span><span class="pn"></span><div class="pv"><span class="dimtxt">0이면 한글 크기의 45%로 자동 · 한글 자막 바로 아래에 붙습니다</span></div></div>
              </div>
            </div>
            <div class="ec-sec">선택한 자막</div>
            <div class="fx" id="fxSel">
              <div class="fx-h"><span class="tri">▼</span><span class="fxi">fx</span><span class="fxn" id="fxSelName">(선택 없음)</span></div>
              <div class="fx-b">
                <div class="prm"><span class="stop"></span><span class="pn">시작</span><div class="pv"><input class="hot wide" type="number" id="selStart" step="0.001" disabled><span class="unit">초</span></div></div>
                <div class="prm"><span class="stop"></span><span class="pn">끝</span><div class="pv"><input class="hot wide" type="number" id="selEnd" step="0.001" disabled><span class="unit">초</span></div></div>
                <div class="prm"><span class="stop" style="visibility:hidden"></span><span class="pn">지속 시간</span><div class="pv"><span class="dimtxt" id="selDur">-</span></div></div>
                <div class="prm"><span class="stop" style="visibility:hidden"></span><span class="pn">텍스트</span><div class="pv"><input class="txt" type="text" id="selText" disabled placeholder="타임라인에서 자막을 선택하세요"></div></div>
                <div class="prm" id="selEnRow" style="display:none"><span class="stop" style="visibility:hidden"></span><span class="pn">영어</span><div class="pv"><input class="txt" type="text" id="selTextEn"></div></div>
              </div>
            </div>
          </div>
        </div>
        <div class="ec-right">
          <div class="kf-ruler" id="kfRuler"></div>
          <div class="kf-body" id="kfBody"></div>
        </div>
      </div>
    </div>
    <div class="pbody" data-body="proj">
      <div class="pj-tools">
        <div class="search"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg><input id="pjSearch" placeholder="검색"></div>
      </div>
      <div class="pj-cols"><span>이름</span><span>미디어 시작</span><span>미디어 지속 시간</span></div>
      <div class="pj-list" id="pjList"></div>
      <div class="pj-foot">
        <button class="fi" id="pjSort" title="이름/시작 시간순 정렬 전환"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h10M4 12h7M4 17h4M17 6v12M14 15l3 3 3-3"/></svg></button>
        <span class="fill"></span>
        <input class="zoom" id="pjZoom" type="range" min="0" max="100" value="50" title="타임라인 확대/축소">
        <button class="fi" id="pjNew" title="새 항목(자막 추가)"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 3H6v18h12V7z"/><path d="M12 11v6M9 14h6"/></svg></button>
        <button class="fi" id="pjDel" title="지우기(선택한 자막)"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13"/></svg></button>
      </div>
    </div>
    <div class="pbody" data-body="media">
      <div class="ec-none">이 프로젝트의 미디어: 원본 영상 1개 (source.mp4). 추가 미디어 가져오기는 홈(후보 목록)에서 합니다.</div>
    </div>
    <div class="pbody" data-body="fx">
      <div class="pj-tools"><div class="search"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg><input placeholder="검색" id="fxSearch"></div></div>
      <div class="fx-tree" id="fxTree">
        <div class="fx-fold"><div class="fh"><span class="tri">▼</span><span class="fico">▰</span>AI 자막 도구</div>
          <div class="fl">
            <div class="fx-item" data-fx="sync"><span class="fxi">fx</span><span class="nm">싱크 맞추기</span><button class="apply">적용</button></div>
            <div class="fx-item" data-fx="correct"><span class="fxi">fx</span><span class="nm">AI 교정</span><button class="apply">적용</button></div>
            <div class="fx-item" data-fx="translate"><span class="fxi">fx</span><span class="nm">영어 번역</span><button class="apply">적용</button></div>
            <div class="fx-item" data-fx="lyrics" id="fxLyrics" style="display:none"><span class="fxi">fx</span><span class="nm">가사 가져오기</span><button class="apply">적용</button></div>
          </div>
        </div>
        <div class="fx-fold closed"><div class="fh"><span class="tri">▼</span><span class="fico">▰</span>사전 설정</div><div class="fl"><div class="ec-none">없음</div></div></div>
        <div class="fx-fold closed"><div class="fh"><span class="tri">▼</span><span class="fico">▰</span>오디오 효과</div><div class="fl"><div class="ec-none">렌더 시 자동 처리</div></div></div>
        <div class="fx-fold closed"><div class="fh"><span class="tri">▼</span><span class="fico">▰</span>비디오 효과</div><div class="fl"><div class="ec-none">렌더 시 자동 처리</div></div></div>
      </div>
    </div>
  </div>

  <div class="split-v" id="splitV"></div>

  <!-- ② 프로그램 모니터 -->
  <div class="panel" id="pProg" style="grid-column:3;grid-row:1">
    <div class="tabs">
      <div class="tab on"><span class="tname">프로그램: <span id="progName">클립 {{ idx + 1 }}</span></span><span class="mm">≡</span></div>
      <div class="fill"></div><div class="tabmenu">»</div>
    </div>
    <div class="pbody on">
      <div class="stage" id="stage">
        <div class="frame showsafe" id="frame">
          <div class="vbox" id="vbox">
            <video id="v" src="/media/{{ video_id }}/source.mp4" playsinline preload="auto"></video>
          </div>
          <div class="safe right"></div><div class="safe bottom"></div>
          <div class="ttl-ov" id="ttlOv"></div>
          <div class="cap-ov" id="capOv" style="display:none"><span class="ko"></span><span class="en"></span></div>
        </div>
        <button class="stageFullBtn" id="stageFullBtn" title="영상만 전체화면으로 보기">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>
        </button>
      </div>
      <div class="mon-bar">
        <span class="tc" id="tCurT">00:00:00.000</span>
        <select class="dd" id="ddFit"><option value="fit">맞춤</option><option value="0.5">50%</option><option value="0.75">75%</option><option value="1">100%</option></select>
        <select class="dd" id="ddRes"><option>전체</option><option>1/2</option><option>1/4</option></select>
        <span class="tc dur" id="tTotal">00:00:00.000</span>
      </div>
      <div class="transport">
        <button class="tb" id="tbIn" title="시작 표시(I) — 클립 시작을 재생 헤드로"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M14 5H9v14h5"/></svg></button>
        <button class="tb" id="tbOut" title="종료 표시(O) — 클립 끝을 재생 헤드로"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M10 5h5v14h-5"/></svg></button>
        <button class="tb" id="tbMarker" title="마커 추가(M)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3l7 7-7 7-7-7z"/></svg></button>
        <span class="tsep"></span>
        <button class="tb" id="tbGoIn" title="시작 지점으로 이동(Home)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M5 5h2v14H5zM19 5v14L8 12z"/></svg></button>
        <button class="tb" id="tbStepB" title="한 프레임 뒤로(←)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5h2v14H8zM19 5v14l-8-7z"/></svg></button>
        <button class="tb play" id="tbPlay" title="재생/정지(Space)"><svg id="icPlay" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4l13 8-13 8z"/></svg><svg id="icPause" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style="display:none"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg></button>
        <button class="tb" id="tbStepF" title="한 프레임 앞으로(→)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M14 5h2v14h-2zM5 5v14l8-7z"/></svg></button>
        <button class="tb" id="tbGoOut" title="종료 지점으로 이동(End)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M17 5h2v14h-2zM5 5v14l11-7z"/></svg></button>
        <span class="tsep"></span>
        <button class="tb" id="tbLift" title="들어내기 — 선택한 자막 삭제(빈자리 유지)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 20h16M12 15V4M8 8l4-4 4 4"/></svg></button>
        <button class="tb" id="tbExtract" title="추출 — 선택한 자막 잔물결 삭제(뒤 자막 당김)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 20h16M12 4v11M8 11l4 4 4-4"/></svg></button>
        <button class="tb" id="tbFrame" title="프레임 내보내기(현재 화면 저장)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3"/></svg></button>
        <button class="tb" id="tbCompare" title="안전 여백 표시 전환"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="8" height="14"/><rect x="13" y="5" width="8" height="14"/></svg></button>
        <button class="tb" id="tbPlus" title="단추 편집기"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg></button>
      </div>
    </div>
  </div>

  <div class="split-h" id="splitH"></div>

  <!-- ③ 도구 + 타임라인 + 오디오 미터 (하단 전체 너비) -->
  <div class="bottom-right" style="grid-column:1 / 4;grid-row:3">
    <div class="tools" id="toolsPanel">
      <button class="tool on" data-tool="select" title="선택 도구 (V)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M5 3l14 9-6 1 3 6-2 1-3-6-4 4z"/></svg></button>
      <button class="tool" data-tool="trackfwd" title="앞으로 트랙 선택 도구 (A)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M4 4l9 8-9 8zM13 4l9 8-9 8z"/></svg><span class="sub"></span></button>
      <button class="tool" data-tool="ripple" title="잔물결 편집 도구 (B)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v18M4 12h5M15 12h5M7 9l-3 3 3 3M17 9l3 3-3 3"/></svg><span class="sub"></span></button>
      <button class="tool" data-tool="roll" title="롤링 편집 도구 (N)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v18M3 12h18M6 9l-3 3 3 3M18 9l3 3-3 3"/></svg></button>
      <button class="tool" data-tool="razor" title="자르기 도구 (C)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2.4"/><circle cx="6" cy="18" r="2.4"/><line x1="20" y1="4" x2="8.5" y2="14.5"/><line x1="8.5" y1="9.5" x2="20" y2="20"/></svg></button>
      <button class="tool" data-tool="slip" title="밀어넣기 도구 (Y)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="8" width="18" height="8"/><path d="M7 4l-3 4 3 4M17 4l3 4-3 4"/></svg><span class="sub"></span></button>
      <button class="tool" data-tool="pen" title="펜 도구 (P)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M14 3l7 7-9 9-6 2 2-6zM5 21h14v-1H5z"/></svg><span class="sub"></span></button>
      <button class="tool" data-tool="hand" title="손 도구 (H)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M8 13V5a1.5 1.5 0 013 0v6M11 11V4a1.5 1.5 0 013 0v7M14 11V6a1.5 1.5 0 013 0v8M17 12a1.5 1.5 0 013 0v3a7 7 0 01-7 7h-1a7 7 0 01-6-3.5L4 14a1.5 1.5 0 012.5-1.5L8 14"/></svg><span class="sub"></span></button>
      <button class="tool" data-tool="type" title="문자 도구 (T)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M5 4h14v4h-2V6h-4v13h2v2H9v-2h2V6H7v2H5z"/></svg><span class="sub"></span></button>
    </div>

    <div class="panel tl-panel" id="tlPanel">
      <div class="tabs">
        <div class="tab on"><span class="tname">타임라인: <span id="tlName">클립 {{ idx + 1 }}</span></span><span class="mm">≡</span></div>
        <div class="fill"></div><div class="tabmenu">»</div>
      </div>
      <div class="pbody on">
        <div class="tl-head">
          <span class="tc" id="tCur">00:00:00.000</span>
          <button class="hb on" id="hbSnap" title="타임라인에서 스냅 (S)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M6 4v8a6 6 0 0012 0V4M6 4h3M15 4h3"/></svg></button>
          <button class="hb on" id="hbLink" title="연결된 선택 — 영어 자막이 한글을 따라감"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M10 13a5 5 0 007 0l2-2a5 5 0 00-7-7l-1 1M14 11a5 5 0 00-7 0l-2 2a5 5 0 007 7l1-1"/></svg></button>
          <button class="hb" id="hbMarker" title="마커 추가 (M)"><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3l7 7-7 7-7-7z"/></svg></button>
          <button class="hb" id="hbSettings" title="영상 트랙 썸네일 표시/숨기기"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14.7 6.3a4 4 0 00-5 5L4 17v3h3l5.7-5.7a4 4 0 005-5l-2.4 2.4-2.6-.6-.6-2.6z"/></svg></button>
          <button class="hb" id="hbCC" title="캡션 트랙 정보"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M9 12H6.5M17.5 12H15M6.5 12a2.5 2.5 0 002.5 2.5M15 12a2.5 2.5 0 002.5 2.5"/></svg></button>
          <button class="hb" id="hbAddCap" title="재생 헤드에 자막 추가 (Ctrl+Shift+N)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M12 9v6M9 12h6"/></svg></button>
          <span class="tsep"></span>
          <button class="hb" id="zoomOut" title="축소 (-)">−</button>
          <button class="hb" id="zoomFit" title="시퀀스에 맞게 (\)"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg></button>
          <button class="hb" id="zoomIn" title="확대 (=)">+</button>
          <span class="status" id="status"></span>
        </div>
        <div class="tl-body">
          <div class="tl-heads">
            <div class="ruler-pad"></div>
            <div class="tl-heads-scroll" id="headsScroll">
              <div class="trk-h h-c" style="height:34px" id="koHead">
                <span class="patch">C1</span>
                <button class="th lock" id="koLock" title="트랙 잠금 전환"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="11" width="14" height="9" rx="1.5"/><path d="M8 11V7a4 4 0 017.8-1.3"/></svg></button>
                <button class="th on" id="syncLock" title="동기화 잠금 — 잔물결 편집 시 마커도 함께 밀림"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h16v6H4zM4 14h16v6H4z"/></svg></button>
                <button class="th on" id="koEye" title="트랙 출력 전환"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg></button>
                <span class="tn">C1 자막(한글)</span>
              </div>
              <div class="trk-h h-c" style="height:34px;display:none" id="enHead">
                <span class="patch">C2</span>
                <button class="th lock on" title="영어 트랙은 항상 한글 시간을 따라갑니다(잠금)"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="11" width="14" height="9" rx="1.5"/><path d="M8 11V7a4 4 0 018 0v4"/></svg></button>
                <button class="th on" id="enEye" title="트랙 출력 전환(미리보기 영어 표시)"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg></button>
                <span class="tn">C2 자막(영어)</span>
              </div>
              <div class="trk-h" style="height:56px" id="vHead">
                <span class="patch">V1</span>
                <button class="th lock on" title="원본 영상은 잠겨 있습니다"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="11" width="14" height="9" rx="1.5"/><path d="M8 11V7a4 4 0 018 0v4"/></svg></button>
                <button class="th on" title="트랙 출력 전환"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg></button>
                <span class="tn">V1 원본 영상</span>
              </div>
              <div class="trk-h" style="height:52px">
                <span class="patch">A1</span>
                <button class="th lock on" title="원본 오디오는 잠겨 있습니다"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="11" width="14" height="9" rx="1.5"/><path d="M8 11V7a4 4 0 018 0v4"/></svg></button>
                <button class="th mute" id="aMute" title="트랙 음소거">M</button>
                <button class="th solo" id="aSolo" title="솔로 트랙 — 재생 시 이 트랙만 들림">S</button>
                <span class="tn">A1 오디오</span>
              </div>
              <div class="trk-h" style="height:52px" id="bgmHead">
                <span class="patch">A2</span>
                <button class="th mute" id="bgmMute" title="배경 음악 음소거" style="display:none">M</button>
                <button class="th" id="bgmAdd" title="배경 음악 추가"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg></button>
                <button class="th" id="bgmRemove" title="배경 음악 제거" style="display:none"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
                <input type="range" id="bgmVol" min="0" max="100" value="25" style="width:44px;display:none" title="배경 음악 볼륨">
                <span class="tn" id="bgmName">A2 배경 음악(없음)</span>
              </div>
              <div class="trk-h addtrk" id="addTrkRow" title="텍스트 트랙 추가 — 자막과 따로 노는 자유 텍스트를 올립니다">
                <button class="addbtn" id="btnAddTrack">+</button>
                <span class="tn">트랙 추가</span>
              </div>
            </div>
          </div>
          <div class="tl-tracks" id="tlTracks">
            <div class="ruler" id="ruler"></div>
            <div class="tl-tracks-scroll" id="tracksScroll">
              <div class="trk ko h-c" id="koTrack"></div>
              <div class="trk en h-c" id="enTrack" style="display:none"></div>
              <div class="trk v h-v" id="vTrack"><div class="thumbs" id="thumbs"></div></div>
              <div class="trk a h-a" id="aTrack"><div class="aclip" id="aClip"><canvas id="wave"></canvas><span class="cn">A1 · 원본 오디오</span></div></div>
              <div class="trk a h-a" id="bgmTrack"><div class="aclip" id="bgmClip" style="display:none"><span class="cn" id="bgmClipCn">A2 · 배경 음악</span></div></div>
              <div class="trk" id="addTrkSpacer" style="height:26px;background:#1a1a1a"></div>
            </div>
            <div class="ph-head" id="phHead" style="display:none"></div>
            <div class="ph-line" id="phLine" style="display:none"></div>
          </div>
        </div>
        <div class="tl-scroll"><div class="pad"></div><div class="navi" id="navi"><div class="thumb" id="naviThumb"></div></div></div>
      </div>
    </div>

    <div class="meters">
      <div class="tabs"><div class="tab on"><span class="tname">오디오 미터</span></div></div>
      <div class="mb">
        <div class="scale"><span>0</span><span>-6</span><span>-12</span><span>-18</span><span>-24</span><span>-30</span><span>-36</span><span>-42</span><span>-48</span><span>-54</span></div>
        <div class="bar"><div class="lv" id="mL"></div><div class="pk" id="pL"></div></div>
        <div class="bar"><div class="lv" id="mR"></div><div class="pk" id="pR"></div></div>
      </div>
    </div>
  </div>
</div>

<div class="ctxmenu" id="ctxMenu"></div>
<input type="file" id="bgmFile" accept=".mp3,.wav,.m4a,.aac,.ogg,.flac,audio/*" style="display:none">
<audio id="bgmAudio" preload="auto" loop style="display:none"></audio>

<div class="modal-bg" id="kbModal"><div class="modal">
  <h2>키보드 단축키</h2>
  <table>
    <tr><td>V / A / B / N</td><td>선택 · 앞으로 트랙 선택 · 잔물결 편집 · 롤링 편집</td></tr>
    <tr><td>C / Y / P / H / T</td><td>자르기 · 밀어넣기 · 펜 · 손 · 문자</td></tr>
    <tr><td>Space, K / L</td><td>재생/정지</td></tr>
    <tr><td>← / →, Shift+← / →</td><td>1프레임 · 5프레임 이동</td></tr>
    <tr><td>Home / End</td><td>클립 시작 / 끝으로</td></tr>
    <tr><td>I / O</td><td>클립 시작 / 끝을 재생 헤드 위치로</td></tr>
    <tr><td>M</td><td>마커 추가</td></tr>
    <tr><td>Ctrl+K</td><td>재생 헤드에서 선택한 자막 분할</td></tr>
    <tr><td>Enter</td><td>선택한 자막 텍스트 수정</td></tr>
    <tr><td>Delete / Shift+Delete</td><td>지우기 / 잔물결 삭제</td></tr>
    <tr><td>Ctrl+C / Ctrl+V</td><td>선택한 자막 복사 / 재생 헤드에 붙여넣기</td></tr>
    <tr><td>Ctrl+Z / Ctrl+Shift+Z</td><td>실행 취소 / 다시 실행</td></tr>
    <tr><td>= / - / \</td><td>확대 / 축소 / 시퀀스에 맞게</td></tr>
    <tr><td>S</td><td>스냅 전환</td></tr>
    <tr><td>Ctrl+S / Ctrl+M</td><td>저장 / 내보내기(만들기)</td></tr>
  </table>
  <button class="close" id="kbClose">닫기</button>
</div></div>

<script>
const VIDEO_ID = {{ video_id | tojson }};
const IDX = {{ idx }};
const $ = (id) => document.getElementById(id);
const v = $('v');
const FPS = 30;
let C = null, L = null;
let caps = [];      // [{start,end,text}] 절대초 (한국어)
let ens = [];       // 영어(인덱스로 한국어와 짝, 시간은 항상 한국어를 따라감)
let selIdx = -1;
let selSet = new Set();   // 다중 선택(앞으로 트랙 선택 도구)
let capClipboard = null;  // Ctrl+C로 복사한 자막들: [{off, dur, text}] (off=첫 자막 기준 상대초)
// 자막 트랙 개수(C1 기본 1개). 붙여넣기가 기존 자막과 겹치면 자동으로 늘어난다(track 필드는
// 화면 배치용일 뿐 서버엔 저장 안 됨 — 내보내기는 항상 start/end/text 평평한 목록 하나다).
let nKoTracks = 1;
let dirty = false;
// 자동 저장 상태. markDirty()가 초기화 도중 불릴 수도 있어(TDZ 방지) 여기서 미리 선언한다.
let autoSaveTimer = null, saving = false;
let view = { a: 0, b: 60 }, homeView = { a: 0, b: 60 };
let dur = 60;
let tool = 'select';
let koLocked = false, enVisible = true, koVisible = true, snapOn = true, syncLock = true;
let markers = [];
let clipRangeDirty = false;
let undoStack = [], redoStack = [];
let peaksFull = null, peaksHome = null;

// 타임코드: 프리미어 형식(시:분:초)에 밀리초 3자리 — 사용자 요청(밀리초 정밀 편집) 유지
const pad2 = (n) => (n < 10 ? '0' : '') + n;
function tc(t) {
  t = Math.max(0, t);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return pad2(h) + ':' + pad2(m) + ':' + (s < 10 ? '0' : '') + s.toFixed(3);
}
function tcShort(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return pad2(m) + ':' + (s < 10 ? '0' : '') + (Number.isInteger(+s.toFixed(3)) ? s.toFixed(0) : s.toFixed(1));
}

// ─── 초기 로드 ───
fetch('/video/' + VIDEO_ID + '/clip/' + IDX + '/preview_info').then(r => r.json()).then(info => {
  C = info.clip; L = info.layout;
  dur = info.source_duration || (C.end + 10);
  caps = (info.caption_lines || []).map(c => ({ start: c.start, end: c.end, text: c.text, track: 0 }));
  ens = (C.caption_overrides_en || []).map(c => ({ start: c.start, end: c.end, text: c.text }));
  nKoTracks = 1;
  initTexts();
  initVSegs();
  const title = (C.title || ('클립 ' + (IDX + 1)));
  ['projName', 'projName2', 'progName', 'tlName'].forEach(id => $(id).textContent = title);
  $('ecMaster').textContent = '마스터 * ' + title; $('ecSeq').textContent = title + ' * 자막';
  if (C.clip_type === 'praise') $('fxLyrics').style.display = '';
  const basePad = Math.max((C.end - C.start) * 0.08, 3);
  view = C.show_full_source_once ? { a: 0, b: dur } : { a: Math.max(0, C.start - basePad), b: Math.min(dur, C.end + basePad) };
  homeView = { a: view.a, b: view.b };
  $('tTotal').textContent = tc(C.end - C.start);
  const defSize = Math.round((L.caption_font_size || 72));
  bindNum('pSize', C.caption_size || defSize);
  bindNum('pX', C.caption_offset_x || 0);
  bindNum('pY', C.caption_offset_y || 0);
  bindNum('pSizeEn', C.caption_size_en || 0);
  bindNum('pSpacing', C.caption_spacing || 0);
  applyPreviewFonts(info);
  initCaptionStyleUi(info);
  v.addEventListener('loadedmetadata', () => { v.currentTime = C.start; fitVideo(); });
  if (v.readyState >= 1) { v.currentTime = C.start; fitVideo(); }
  fitVideo(); renderProject(); renderAll(); syncSelPanel(); loadPeaks(); renderBgm();
});

// ─── 패널 포커스(클릭한 패널에 파란 테두리) ───
document.querySelectorAll('.panel, .tools').forEach(p => p.addEventListener('pointerdown', () => {
  document.querySelectorAll('.panel.focus, .tools.focus').forEach(x => x.classList.remove('focus'));
  p.classList.add('focus');
}));
// 탭 전환
document.querySelectorAll('.tabs').forEach(bar => bar.addEventListener('click', (e) => {
  const t = e.target.closest('.tab'); if (!t || !t.dataset.tab) return;
  const panel = bar.parentElement;
  bar.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x === t));
  panel.querySelectorAll(':scope > .pbody').forEach(b => b.classList.toggle('on', b.dataset.body === t.dataset.tab));
  if (t.dataset.tab === 'src') openSource();
}));

// ─── 스플리터(패널 크기 조절) ───
const ws = $('ws');
function splitter(el, axis) {
  el.addEventListener('pointerdown', (e) => {
    e.preventDefault(); el.classList.add('on');
    const r = ws.getBoundingClientRect();
    const move = (ev) => {
      if (axis === 'v') { const p = Math.max(18, Math.min(70, (ev.clientX - r.left) / r.width * 100)); ws.style.setProperty('--colL', p + '%'); }
      else { const p = Math.max(22, Math.min(80, (ev.clientY - r.top) / r.height * 100)); ws.style.setProperty('--rowT', p + '%'); }
      fitVideo(); renderAll();
    };
    const up = () => { el.classList.remove('on'); document.removeEventListener('pointermove', move); };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up, { once: true });
  });
}
splitter($('splitV'), 'v'); splitter($('splitH'), 'h');

// ─── 프로그램 모니터: '완성될 세로 프레임' 그대로 배치 ───
// 프레임(=출력 해상도) 안에 영상 박스를 layout.video_box 위치에 넣고, 제목·자막을 렌더와
// 같은 기준선(title/caption_base_margin_v)에 그린다. 확인 팝업과 완전히 같은 공식이라
// 여기서 맞춘 위치가 그대로 추출물에 나온다.
let fitMode = 'fit';
let SC = 1;  // 화면 px / 렌더 px
function fitVideo() {
  const st = $('stage'), fr = $('frame'), box = $('vbox');
  if (!L) return;
  const RW = (L.resolution || [1080, 1920])[0], RH = (L.resolution || [1080, 1920])[1];
  const W = st.clientWidth - 16, H = st.clientHeight - 16;
  let s = Math.min(W / RW, H / RH);
  if (fitMode !== 'fit') s = Math.min(s, parseFloat(fitMode) * 0.5);  // 100%는 화면 대비 1/2 크기 기준
  const w = Math.max(60, RW * s), h = Math.max(60, RH * s);
  fr.style.width = w + 'px'; fr.style.height = h + 'px';
  fr.style.left = ((st.clientWidth - w) / 2) + 'px'; fr.style.top = ((st.clientHeight - h) / 2) + 'px';
  SC = w / RW;
  const vb = L.video_box || { x: 0, y: 0, w: RW, h: RH, r: 0 };
  box.style.left = (vb.x * SC) + 'px'; box.style.top = (vb.y * SC) + 'px';
  box.style.width = (vb.w * SC) + 'px'; box.style.height = (vb.h * SC) + 'px';
  box.style.borderRadius = ((vb.r || 0) * SC) + 'px';
  updateTitleOv(); updateOverlay();
}
// 실제 렌더에 쓰이는 글꼴을 미리보기에도 심는다(@font-face). 글꼴이 다르면 글자 폭이
// 달라져서 '몇 글자에서 줄이 넘어가는지'가 어긋난다.
function applyPreviewFonts(info) {
  let css = '';
  for (const f of (info.fonts || [])) {
    css += "@font-face{font-family:'" + f.family + "';src:url('/font/" + f.file + "');font-display:swap}\n";
  }
  if (css) { const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st); }
  if (info.caption_font) $('capOv').style.fontFamily = "'" + info.caption_font.family + "', sans-serif";
  if (info.title_font) $('ttlOv').style.fontFamily = "'" + info.title_font.family + "', sans-serif";
}
// 자막 글꼴/정렬/자간 — 캡컷처럼 스튜디오에서 바로 고른다(2026-09-08 요청). 렌더(build_ass)가
// 이미 caption_font/caption_align/caption_spacing을 지원해서(예전 편집기·팝업에서만 노출됐음),
// 여기선 값만 채우고 저장(payload)에 실어 보내면 된다.
function hexToRgba(hex, opacity) {
  const h = (hex || '#000000').replace('#', '');
  const r = parseInt(h.slice(0, 2), 16) || 0, g = parseInt(h.slice(2, 4), 16) || 0, b = parseInt(h.slice(4, 6), 16) || 0;
  return 'rgba(' + r + ',' + g + ',' + b + ',' + (opacity == null ? 1 : opacity) + ')';
}
let capFont = '', capAlign = 'center';
let capTextColor = '', capBox = false, capBoxColor = '#000000', capBoxOpacity = 0.55, capBoxRadius = 40;
let capBoxW = 28, capBoxH = 28, capBoxOX = 0, capBoxOY = 0;
let capBold = true, capItalic = false, capUnderline = false;
let capOutlineOn = true, capOutlineColor = '#000000';
let capGlow = false, capGlowColor = '#00ffff';
// 캡컷처럼 클릭 한 번으로 색 조합을 통째로 적용하는 프리셋(2026-09-08: "다양한 스타일이
// 있고 우리가 설정할 수 있게", "색상 팔레트 무제한으로" — 프리셋은 지름길일 뿐, 글자색·
// 박스색 스와치(input type=color)가 실제 무제한 선택 경로다). 각 항목이 텍스트색·박스
// 유무·박스색·불투명도·둥글기를 한 번에 정한다(박스 여백은 기본값 28%/28% 유지).
const CAP_PRESETS = [
  { name: '기본', sw: '#ffffff', text: '', box: false },
  { name: '화이트박스', sw: '#ffffff', text: '#111111', box: true, boxColor: '#ffffff', boxOpacity: 0.92, boxRadius: 30 },
  { name: '블랙박스', sw: '#000000', text: '#ffffff', box: true, boxColor: '#000000', boxOpacity: 0.62, boxRadius: 30 },
  { name: '옐로우', sw: '#ffd400', text: '#111111', box: true, boxColor: '#ffd400', boxOpacity: 0.95, boxRadius: 50 },
  { name: '오렌지', sw: '#ff8a00', text: '#111111', box: true, boxColor: '#ff8a00', boxOpacity: 0.95, boxRadius: 50 },
  { name: '레드', sw: '#ff3b30', text: '#ffffff', box: true, boxColor: '#ff3b30', boxOpacity: 0.9, boxRadius: 35 },
  { name: '핑크팝', sw: '#ff2d78', text: '#ffffff', box: true, boxColor: '#ff2d78', boxOpacity: 0.9, boxRadius: 50 },
  { name: '퍼플', sw: '#a259ff', text: '#ffffff', box: true, boxColor: '#a259ff', boxOpacity: 0.9, boxRadius: 50 },
  { name: '블루', sw: '#0a84ff', text: '#ffffff', box: true, boxColor: '#0a84ff', boxOpacity: 0.9, boxRadius: 35 },
  { name: '스카이', sw: '#5ac8fa', text: '#0a2b3d', box: true, boxColor: '#5ac8fa', boxOpacity: 0.92, boxRadius: 50 },
  { name: '민트', sw: '#00d9a3', text: '#0a2b22', box: true, boxColor: '#00d9a3', boxOpacity: 0.9, boxRadius: 20 },
  { name: '그린', sw: '#34c759', text: '#0a2b16', box: true, boxColor: '#34c759', boxOpacity: 0.9, boxRadius: 35 },
  { name: '라임', sw: '#c6f000', text: '#1a1f00', box: true, boxColor: '#c6f000', boxOpacity: 0.95, boxRadius: 50 },
  { name: '베이지', sw: '#e8ddc7', text: '#3a2f1f', box: true, boxColor: '#e8ddc7', boxOpacity: 0.95, boxRadius: 25 },
  { name: '노란글자', sw: '#ffd400', text: '#ffd400', box: false },
  { name: '민트글자', sw: '#00d9a3', text: '#00d9a3', box: false },
];
function applyCapPreset(p) {
  pushUndo();
  capTextColor = p.text || '';
  capBox = !!p.box;
  if (p.box) {
    capBoxColor = p.boxColor; capBoxOpacity = p.boxOpacity; capBoxRadius = p.boxRadius;
    capBoxW = 28; capBoxH = 28; capBoxOX = 0; capBoxOY = 0;
  }
  syncCapStyleInputs();
  markDirty(); updateOverlay();
}
function syncCapStyleInputs() {
  $('pTextColor').value = capTextColor || '#ffffff';
  $('pBoxOn').checked = capBox;
  $('pBoxColor').value = capBoxColor;
  $('pBoxOpacity').value = Math.round(capBoxOpacity * 100);
  $('pBoxRadius').value = capBoxRadius;
  $('pBoxW').value = capBoxW; $('pBoxH').value = capBoxH;
  $('pBoxOX').value = capBoxOX; $('pBoxOY').value = capBoxOY;
  ['pBoxRow1', 'pBoxRow2', 'pBoxRow3', 'pBoxRow4', 'pBoxRow5', 'pBoxRow6'].forEach((id) => {
    $(id).style.display = capBox ? '' : 'none';
  });
  $('pBold').classList.toggle('on', capBold);
  $('pItalic').classList.toggle('on', capItalic);
  $('pUnderline').classList.toggle('on', capUnderline);
  $('pOutlineOn').checked = capOutlineOn;
  $('pOutlineColor').value = capOutlineColor || '#000000';
  $('pOutlineRow1').style.display = capOutlineOn ? '' : 'none';
  $('pOutlineRow2').style.display = capOutlineOn ? '' : 'none';
  $('pGlowOn').checked = capGlow;
  $('pGlowColor').value = capGlowColor || '#00ffff';
  $('pGlowRow1').style.display = capGlow ? '' : 'none';
}
function initCaptionStyleUi(info) {
  capFont = C.caption_font || (info.caption_font && info.caption_font.family) || '';
  capAlign = C.caption_align || 'center';
  capTextColor = C.caption_text_color || '';
  capBox = !!C.caption_box;
  capBoxColor = C.caption_box_color || '#000000';
  capBoxOpacity = (C.caption_box_opacity != null) ? C.caption_box_opacity : 0.55;
  capBoxRadius = (C.caption_box_radius != null) ? C.caption_box_radius : 40;
  capBoxW = (C.caption_box_width_pct != null) ? C.caption_box_width_pct : 28;
  capBoxH = (C.caption_box_height_pct != null) ? C.caption_box_height_pct : 28;
  capBoxOX = C.caption_box_offset_x || 0;
  capBoxOY = C.caption_box_offset_y || 0;
  const sel = $('pFont'); sel.innerHTML = '';
  (info.fonts || []).forEach((f) => {
    const opt = document.createElement('option');
    opt.value = f.family; opt.textContent = f.name || f.family;
    opt.style.fontFamily = "'" + f.family + "', sans-serif";
    if (f.family === capFont) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => {
    pushUndo();
    capFont = sel.value;
    $('capOv').style.fontFamily = "'" + capFont + "', sans-serif";
    markDirty(); updateOverlay();
  });
  document.querySelectorAll('.align-btn').forEach((b) => {
    b.classList.toggle('on', b.dataset.align === capAlign);
    b.addEventListener('click', () => {
      pushUndo();
      capAlign = b.dataset.align;
      document.querySelectorAll('.align-btn').forEach((x) => x.classList.toggle('on', x === b));
      markDirty(); updateOverlay();
    });
  });
  const presetsEl = $('capPresets');
  presetsEl.innerHTML = '';
  CAP_PRESETS.forEach((p) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'cap-preset'; b.title = p.name;
    b.style.background = p.sw;
    b.addEventListener('click', () => {
      applyCapPreset(p);
      document.querySelectorAll('.cap-preset').forEach((x) => x.classList.toggle('on', x === b));
    });
    presetsEl.appendChild(b);
  });
  // 색 피커는 클릭(피커 열기) 시점에 한 번 찍는다 — 피커 안에서 드래그하며 나는 input
  // 이벤트마다 찍으면 Ctrl+Z가 거의 안 움직인 것처럼 보인다.
  $('pTextColor').addEventListener('click', () => pushUndo());
  $('pTextColor').addEventListener('input', () => { capTextColor = $('pTextColor').value; markDirty(); updateOverlay(); });
  $('pTextColorReset').addEventListener('click', () => { pushUndo(); capTextColor = ''; syncCapStyleInputs(); markDirty(); updateOverlay(); });
  $('pBoxOn').addEventListener('change', () => { pushUndo(); capBox = $('pBoxOn').checked; syncCapStyleInputs(); markDirty(); updateOverlay(); });
  $('pBoxColor').addEventListener('click', () => pushUndo());
  $('pBoxColor').addEventListener('input', () => { capBoxColor = $('pBoxColor').value; markDirty(); updateOverlay(); });
  $('pBoxOpacity').addEventListener('input', () => { pushUndoBurst(); capBoxOpacity = (parseFloat($('pBoxOpacity').value) || 0) / 100; markDirty(); updateOverlay(); });
  // 이 넷(둥글기·높이·너비·오프셋)은 pSpacing/pX/pY와 같은 패턴 — 값은 입력칸(DOM)이
  // 곧 원본이고, updateOverlay/payload가 그때그때 읽는다. capBoxW 등 JS 변수는 초기
  // 로드·프리셋 적용 때 입력칸에 값을 밀어넣는 용도로만 쓴다(바인딩 순서상 별도 변수를
  // 여기서 갱신하면 bindNum의 input 리스너가 갱신 전 값으로 먼저 그려 한 틱 밀리는
  // 버그가 났었다 — BGM 오프셋에서 같은 함정을 겪은 뒤 얻은 규칙).
  bindNum('pBoxRadius', capBoxRadius);
  bindNum('pBoxW', capBoxW);
  bindNum('pBoxH', capBoxH);
  bindNum('pBoxOX', capBoxOX);
  bindNum('pBoxOY', capBoxOY);
  ['pBoxRadius', 'pBoxW', 'pBoxH', 'pBoxOX', 'pBoxOY'].forEach((id) => $(id).addEventListener('input', pushUndoBurst));

  // 패턴(B/U/I)·불투명도·획·글로우 — 캡컷 스샷 그대로(2026-09-08: "전부 구현하라").
  capBold = (C.caption_bold != null) ? !!C.caption_bold : true;
  capItalic = !!C.caption_italic;
  capUnderline = !!C.caption_underline;
  capOutlineOn = (C.caption_outline_enabled != null) ? !!C.caption_outline_enabled : true;
  capOutlineColor = C.caption_outline_color || '#000000';
  capGlow = !!C.caption_glow;
  capGlowColor = C.caption_glow_color || '#00ffff';
  const textOpacityInit = (C.caption_text_opacity != null) ? Math.round(C.caption_text_opacity * 100) : 100;
  const outlineWidthInit = (C.caption_outline_width != null && C.caption_outline_width >= 0) ? C.caption_outline_width : 3;
  bindNum('pTextOpacity', textOpacityInit);
  bindNum('pOutlineWidth', outlineWidthInit);
  $('pTextOpacity').addEventListener('input', pushUndoBurst);
  $('pOutlineWidth').addEventListener('input', pushUndoBurst);
  function toggleBtn(id, getVal, onToggle) {
    const b = $(id);
    b.classList.toggle('on', getVal());
    b.addEventListener('click', () => { pushUndo(); onToggle(); b.classList.toggle('on'); markDirty(); updateOverlay(); });
  }
  toggleBtn('pBold', () => capBold, () => { capBold = !capBold; });
  toggleBtn('pItalic', () => capItalic, () => { capItalic = !capItalic; });
  toggleBtn('pUnderline', () => capUnderline, () => { capUnderline = !capUnderline; });
  $('pOutlineOn').checked = capOutlineOn;
  $('pOutlineOn').addEventListener('change', () => { pushUndo(); capOutlineOn = $('pOutlineOn').checked; syncCapStyleInputs(); markDirty(); updateOverlay(); });
  $('pOutlineColor').addEventListener('click', () => pushUndo());
  $('pOutlineColor').addEventListener('input', () => { capOutlineColor = $('pOutlineColor').value; markDirty(); updateOverlay(); });
  $('pOutlineColorReset').addEventListener('click', () => { pushUndo(); capOutlineColor = ''; syncCapStyleInputs(); markDirty(); updateOverlay(); });
  $('pGlowOn').checked = capGlow;
  $('pGlowOn').addEventListener('change', () => { pushUndo(); capGlow = $('pGlowOn').checked; syncCapStyleInputs(); markDirty(); updateOverlay(); });
  $('pGlowColor').addEventListener('click', () => pushUndo());
  $('pGlowColor').addEventListener('input', () => { capGlowColor = $('pGlowColor').value; markDirty(); updateOverlay(); });
  syncCapStyleInputs();
}
// 제목(카드형에서만 존재). 스튜디오에서 편집하지는 않고, 자막 위치를 잡을 때 기준이
// 되도록 실제 크기·위치 그대로 보여준다.
function updateTitleOv() {
  const el = $('ttlOv'); if (!L) return;
  if (L.no_title || !C || !C.title) { el.style.display = 'none'; return; }
  el.style.display = 'block';
  el.textContent = '';
  String(C.title).split('\n').forEach((part, i) => {
    if (i > 0) el.appendChild(document.createElement('br'));
    el.appendChild(document.createTextNode(part));
  });
  const KT = L.title_ass_coeff || 1;
  let fs = (C.title_size || L.title_size || 130) * KT * SC;
  el.style.fontSize = fs + 'px';
  // 렌더는 제목을 항상 한 줄로 폭에 맞춘다(_fit_title_font_size) — 미리보기도 같이 줄인다.
  const cap = (L.resolution[0] - 80) * SC;
  let guard = 0;
  while (el.scrollWidth > cap && fs > 6 && guard++ < 300) { fs -= 0.5; el.style.fontSize = fs + 'px'; }
  el.style.left = ((L.resolution[0] / 2 + (C.title_offset_x || 0)) * SC) + 'px';
  el.style.top = (((L.title_base_margin_v || 0) + (C.title_offset_y || 0)) * SC) + 'px';
}
$('ddFit').addEventListener('change', () => { fitMode = $('ddFit').value; fitVideo(); });
window.addEventListener('resize', () => { fitVideo(); renderAll(); });

// ─── 속성 바인딩(숫자 + 둥근 −/+ 버튼 + 좌우 드래그) ───
// 슬라이더(막대)는 뺐다 — 값이 눈금에 갇히고 미세 조정이 어려워서(사용자 요청 2026-09-07).
function bindNum(numId, initial) {
  const n = $(numId);
  n.value = Math.round(initial);
  n.addEventListener('input', () => { markDirty(); updateOverlay(); });
  hotDrag(n);
}
function nudge(numId, delta) {
  const n = $(numId);
  let val = (parseFloat(n.value) || 0) + delta;
  if (n.min !== '') val = Math.max(parseFloat(n.min), val);
  if (n.max !== '') val = Math.min(parseFloat(n.max), val);
  n.value = Math.round(val);
  markDirty(); updateOverlay();
}
document.querySelectorAll('.stp').forEach((b) => b.addEventListener('click', () => {
  const [id, d] = b.dataset.nudge.split(':');
  nudge(id, parseFloat(d));
}));
// 프리미어 핫텍스트: 파란 숫자를 좌우로 끌면 값이 바뀐다
function hotDrag(n) {
  let x0 = 0, v0 = 0, moved = false;
  n.addEventListener('pointerdown', (e) => {
    if (document.activeElement === n || n.disabled) return;
    x0 = e.clientX; v0 = parseFloat(n.value) || 0; moved = false;
    const step = parseFloat(n.step) || 1;
    const move = (ev) => {
      const dx = ev.clientX - x0;
      if (Math.abs(dx) > 2) moved = true;
      if (!moved) return;
      ev.preventDefault();
      let val = v0 + dx * step * (step < 0.01 ? 5 : 1);
      if (n.min !== '') val = Math.max(parseFloat(n.min), val);
      if (n.max !== '') val = Math.min(parseFloat(n.max), val);
      n.value = step < 1 ? val.toFixed(3) : Math.round(val);
      n.dispatchEvent(new Event('input')); n.dispatchEvent(new Event('change'));
    };
    const up = () => { document.removeEventListener('pointermove', move); if (!moved) n.focus(); };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up, { once: true });
  });
}
$('pSafe').addEventListener('change', () => setSafe($('pSafe').checked));
function setSafe(on) { $('pSafe').checked = on; $('frame').classList.toggle('showsafe', on); $('miSafe').classList.toggle('chk', on); $('tbCompare').classList.toggle('on', on); }
setSafe(true);
document.querySelectorAll('.fx-h').forEach(h => h.addEventListener('click', (e) => { if (e.target.classList.contains('rst')) return; h.parentElement.classList.toggle('closed'); }));
document.querySelector('[data-reset="style"]').addEventListener('click', () => {
  $('pSize').value = Math.round((L && L.caption_font_size) || 72);
  $('pX').value = 0; $('pY').value = 0; $('pSpacing').value = 0;
  capAlign = 'center';
  document.querySelectorAll('.align-btn').forEach((x) => x.classList.toggle('on', x.dataset.align === 'center'));
  document.querySelectorAll('.cap-preset').forEach((x) => x.classList.remove('on'));
  applyCapPreset(CAP_PRESETS[0]);
  markDirty(); updateOverlay();
});
document.querySelector('[data-reset="styleEn"]').addEventListener('click', () => {
  $('pSizeEn').value = 0;  // 0 = 한글의 45%로 자동
  markDirty(); updateOverlay();
});

// ─── 미리보기 자막 오버레이 ───
function activeCapIdx(t) { for (let i = 0; i < caps.length; i++) if (t >= caps[i].start && t < caps[i].end) return i; return -1; }
// 자막이 한 줄로 들어갈 수 있는 폭(렌더 px). captions.py caption_usable_width_px와 같은 공식.
function capUsablePx() {
  const RW = (L.resolution || [1080, 1920])[0];
  const base = L.no_title ? Math.max(40, Math.round(RW * 0.15)) : 40;
  return Math.max(80, RW - 2 * base - 24);
}
function updateOverlay() {
  const ov = $('capOv'); if (!L) return;
  renderTextOverlays();
  const i = activeCapIdx(v.currentTime);
  if (i < 0 || !koVisible) { ov.style.display = 'none'; return; }
  ov.style.display = 'block';
  ov.classList.toggle('onvideo', !!L.no_title);
  ov.style.textAlign = capAlign || 'center';
  ov.style.letterSpacing = ((parseFloat($('pSpacing').value) || 0) * SC) + 'px';
  const koEl = ov.querySelector('.ko');
  const enEl = ov.querySelector('.en');
  const kc = L.caption_ass_coeff || 1;
  const want = parseFloat($('pSize').value) || (L.caption_font_size || 72);
  const usable = capUsablePx() * SC;
  // 렌더(captions.py)와 동일 규칙: 클립 전체가 하나의 크기 + 사용자가 고른 크기가 우선.
  // 한 줄에 안 들어가면 2줄로 내리고, 2줄로도 넘칠 때만 줄인다.
  const koPx = uniformFontPx(caps, want * kc * SC, usable);
  setWrapped(koEl, caps[i].text, koPx, usable);
  ov.style.fontSize = koPx + 'px';
  koEl.style.color = capTextColor || '';
  // 패턴(B/U/I)·획·글로우·불투명도 — CSS로 근사한다(font-weight/style/decoration은 정확히
  // 같은 속성이라 1:1, 획은 -webkit-text-stroke, 글로우는 text-shadow로 흉내낸다).
  koEl.style.fontWeight = capBold ? '800' : '400';
  koEl.style.fontStyle = capItalic ? 'italic' : 'normal';
  koEl.style.textDecoration = capUnderline ? 'underline' : 'none';
  ov.style.opacity = (parseFloat($('pTextOpacity').value) || 100) / 100;
  koEl.style.webkitTextStroke = capOutlineOn
    ? ((parseFloat($('pOutlineWidth').value) || 0) + 'px ' + (capOutlineColor || '#000000')) : '0px transparent';
  koEl.style.textShadow = capGlow
    ? ('0 0 6px ' + (capGlowColor || '#0ff') + ', 0 0 14px ' + (capGlowColor || '#0ff')) : '';
  // 박스(캡컷 스타일): inline 요소에 배경+box-decoration-break:clone을 주면 줄바꿈된 자막도
  // 줄마다 따로 박스가 둘러진다 — 실제 렌더(_emit_caption_box_events, 줄마다 별도 도형)와 같은 느낌.
  // 둥글기·여백은 캡컷처럼 %(렌더와 같은 공식 — 정확한 px 변환은 captions.py 쪽 주석 참고,
  // 여기선 줄 높이 기준 근사치로 충분하다).
  if (capBox) {
    const boxHPct = parseFloat($('pBoxH').value) || 0, boxWPct = parseFloat($('pBoxW').value) || 0;
    const padY = koPx * (boxHPct / 100), padX = koPx * (boxWPct / 100);
    const boxH = koPx * 1.02 + padY * 2;
    const radiusPct = Math.max(0, Math.min(100, parseFloat($('pBoxRadius').value) || 0));
    koEl.style.background = hexToRgba(capBoxColor, capBoxOpacity);
    koEl.style.borderRadius = (boxH / 2 * (radiusPct / 100)) + 'px';
    koEl.style.padding = padY + 'px ' + padX + 'px';
    koEl.style.boxDecorationBreak = 'clone'; koEl.style.webkitBoxDecorationBreak = 'clone';
    koEl.style.transform = 'translate(' + ((parseFloat($('pBoxOX').value) || 0) * SC) + 'px,' + ((parseFloat($('pBoxOY').value) || 0) * SC) + 'px)';
  } else {
    koEl.style.background = ''; koEl.style.borderRadius = ''; koEl.style.padding = ''; koEl.style.transform = '';
  }
  const enWant = parseFloat($('pSizeEn').value) || Math.max(16, Math.round((koPx / (kc * SC)) * 0.60));
  const en = (enVisible && ens[i]) ? ens[i].text : '';
  enEl.style.display = en ? 'block' : 'none';
  if (en) {
    const enPx = uniformFontPx(ens, enWant * kc * SC, usable);
    enEl.style.fontSize = enPx + 'px';
    setWrapped(enEl, en, enPx, usable);
  }
  ov.style.left = (((L.resolution[0] / 2) + parseFloat($('pX').value)) * SC) + 'px';
  const offY = parseFloat($('pY').value) || 0;
  if (L.caption_anchor === 'bottom') {
    // 영상 위 오버레이(찬양): 렌더가 '아래 기준'이라 글씨를 키우면 위로 자란다 — 미리보기도 동일.
    ov.style.top = '';
    ov.style.bottom = (((L.caption_bottom_px || 0) - offY) * SC) + 'px';
  } else {
    ov.style.bottom = '';
    ov.style.top = (((L.caption_base_margin_v || 0) + offY) * SC) + 'px';
  }
}
// ── 한 줄/두 줄 판정(렌더 _make_wrap_planner와 같은 규칙) ────────────────────
let _measEl = null;
function measPx(text, fontPx) {
  if (!_measEl) {
    _measEl = document.createElement('span');
    _measEl.style.cssText = 'position:absolute;visibility:hidden;white-space:pre;left:-99999px;top:0;font-weight:800;line-height:1.25;pointer-events:none';
    document.body.appendChild(_measEl);
  }
  _measEl.style.fontFamily = getComputedStyle($('capOv')).fontFamily;
  _measEl.style.fontSize = fontPx + 'px';
  _measEl.textContent = text || '';
  return _measEl.getBoundingClientRect().width;
}
// 렌더 _make_wrap_planner와 같은 규칙: 한 줄 → 안 되면 2줄 → 3줄, 그래도 넘치면 그때 축소.
// {scale: 이 줄이 요구하는 축소 비율(1이면 그대로), cuts: 줄바꿈할 단어 인덱스들}
const MAX_CAP_LINES = 3;
function bestSplit(toks, fontPx, k) {
  const n = toks.length;
  if (k <= 1) return { cuts: [], widest: measPx(toks.join(' '), fontPx) };
  let best = null;
  const walk = (start, chosen) => {
    if (chosen.length === k - 1) {
      const bounds = [0, ...chosen, n];
      let widest = 0;
      for (let i = 0; i < k; i++) widest = Math.max(widest, measPx(toks.slice(bounds[i], bounds[i + 1]).join(' '), fontPx));
      if (!best || widest < best.widest) best = { cuts: chosen.slice(), widest: widest };
      return;
    }
    for (let i = start; i < n; i++) { chosen.push(i); walk(i + 1, chosen); chosen.pop(); }
  };
  walk(1, []);
  return best || { cuts: [], widest: measPx(toks.join(' '), fontPx) };
}
function planLine(text, fontPx, usable) {
  const toks = String(text || '').split(/\s+/).filter(Boolean);
  if (!toks.length) return { scale: 1, cuts: [] };
  let last = { cuts: [], widest: 0 };
  for (let k = 1; k <= Math.min(MAX_CAP_LINES, toks.length); k++) {
    const r = bestSplit(toks, fontPx, k);
    if (!r.widest || r.widest <= usable) return { scale: 1, cuts: r.cuts };
    last = r;
  }
  return { scale: usable / last.widest, cuts: last.cuts };
}
// 클립 안의 모든 줄에 똑같이 쓸 크기(줄마다 다르면 재생 중 글자가 커졌다 작아졌다 한다).
let _ufKey = '', _ufVal = 0;
function uniformFontPx(list, fontPx, usable) {
  const key = list.map(c => c.text).join('\u0001') + '|' + fontPx.toFixed(2) + '|' + Math.round(usable);
  if (key === _ufKey) return _ufVal;
  let sc = 1;
  for (const c of list) sc = Math.min(sc, planLine(c.text, fontPx, usable).scale);
  _ufKey = key; _ufVal = Math.max(8, fontPx * sc);
  return _ufVal;
}
function setWrapped(el, text, fontPx, usable) {
  const cuts = planLine(text, fontPx, usable).cuts;
  el.textContent = '';
  if (!cuts.length) { el.textContent = text; return; }
  const toks = String(text).split(/\s+/).filter(Boolean);
  const bounds = [0, ...cuts, toks.length];
  for (let i = 0; i < bounds.length - 1; i++) {
    if (i > 0) el.appendChild(document.createElement('br'));
    el.appendChild(document.createTextNode(toks.slice(bounds[i], bounds[i + 1]).join(' ')));
  }
}
// 미리보기에서 자막을 직접 끌어 옮긴다(막대 대신 — 사용자 요청 2026-09-07).
(function () {
  const ov = $('capOv');
  ov.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    ov.classList.add('dragging'); ov.setPointerCapture(e.pointerId);
    const sx = e.clientX, sy = e.clientY;
    const ox = parseFloat($('pX').value) || 0, oy = parseFloat($('pY').value) || 0;
    const move = (ev) => {
      $('pX').value = Math.round(ox + (ev.clientX - sx) / SC);
      $('pY').value = Math.round(oy + (ev.clientY - sy) / SC);
      markDirty(); updateOverlay();
    };
    const up = () => { ov.classList.remove('dragging'); document.removeEventListener('pointermove', move); };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up, { once: true });
  });
})();

// ─── 재생 제어 ───
function playPause() {
  ensureAudio();
  if (v.paused) {
    const doPlay = () => { if (v.currentTime >= C.end - 0.05 || v.currentTime < C.start) v.currentTime = C.start; v.play(); };
    if (v.readyState >= 1) doPlay(); else v.addEventListener('loadedmetadata', doPlay, { once: true });
  }
  else v.pause();
}
function seek(t) {
  const target = Math.min(Math.max(t, 0), dur - 0.05);
  if (v.readyState >= 1) v.currentTime = target;
  else v.addEventListener('loadedmetadata', () => { v.currentTime = target; }, { once: true });
  updateOverlay(); renderPlayhead(); renderKf();
}
$('tbPlay').addEventListener('click', playPause);
v.addEventListener('click', playPause);
// 시간 표시 갱신: timeupdate 이벤트만 쓰면 브라우저가 초당 4~10회 정도만 쏴서
// 밀리초 자릿수가 뚝뚝 끊겨 보인다(사용자 신고: "분절적으로 움직인다"). 재생 중에는
// requestAnimationFrame으로 화면 주사율만큼(보통 60fps) 매끄럽게 갱신하고, 정지 중
// 탐색(seek)·드래그 등은 기존처럼 timeupdate/직접 호출로 커버한다.
function frameTick() {
  if (v.currentTime >= C.end) v.pause();
  const t = tc(Math.max(0, v.currentTime));
  $('tCur').textContent = t; $('tCurT').textContent = tc(Math.max(0, v.currentTime - C.start));
  updateOverlay(); renderPlayhead(); renderKf();
}
let rafId = null;
function rafLoop() { frameTick(); if (!v.paused && !v.ended) rafId = requestAnimationFrame(rafLoop); }
v.addEventListener('play', () => {
  $('icPlay').style.display = 'none'; $('icPause').style.display = '';
  if (rafId == null) rafId = requestAnimationFrame(rafLoop);
});
v.addEventListener('pause', () => {
  $('icPlay').style.display = ''; $('icPause').style.display = 'none';
  if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
  frameTick();  // 정지 직후 마지막 프레임 값으로 한 번 더 맞춘다
});
v.addEventListener('timeupdate', () => { if (v.paused) frameTick(); });
$('tbStepB').addEventListener('click', () => seek(v.currentTime - 1 / FPS));
$('tbStepF').addEventListener('click', () => seek(v.currentTime + 1 / FPS));
$('tbGoIn').addEventListener('click', () => seek(C.start));
$('tbGoOut').addEventListener('click', () => seek(C.end - 0.001));
$('tbIn').addEventListener('click', markIn);
$('tbOut').addEventListener('click', markOut);
$('tbMarker').addEventListener('click', addMarker);
$('hbMarker').addEventListener('click', addMarker);
$('tbLift').addEventListener('click', () => deleteSel(false));
$('tbExtract').addEventListener('click', () => deleteSel(true));
$('tbCompare').addEventListener('click', () => setSafe(!$('pSafe').checked));
$('tbPlus').addEventListener('click', () => setStatus('단추 편집기는 지원하지 않아요'));
// 현재 화면 내보내기: 예전엔 <video> 픽셀만 그대로 캔버스에 떠서 제목·자막이 하나도
// 안 찍혔다(실사고 2026-09-08 — "자막까지 포함되게"). 이제 렌더 해상도(L.resolution)
// 캔버스에 영상 박스(L.video_box) 위치로 영상을 그린 뒤, 화면에 떠 있는 제목·자막·자유
// 텍스트 오버레이를 (이미 정확히 배치·줄바꿈된) 화면 좌표에서 SC로 나눠 렌더 좌표로
// 되돌려 같은 자리에 겹쳐 그린다 — 새 자막 wrap 로직을 다시 구현할 필요가 없다.
function roundRectPath(ctx, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
function drawOverlayToCanvas(ctx, el) {
  if (!el || !el.isConnected) return;
  const cs = getComputedStyle(el);
  if (cs.display === 'none' || parseFloat(cs.opacity || '1') <= 0) return;
  const frameRect = $('frame').getBoundingClientRect();
  const elRect = el.getBoundingClientRect();
  if (!elRect.width && !elRect.height) return;
  const fontPx = parseFloat(cs.fontSize) / SC;
  const lineH = fontPx * 1.25;
  const cx = (elRect.left + elRect.width / 2 - frameRect.left) / SC;
  const topY = (elRect.top - frameRect.top) / SC;
  const lines = (el.innerText || el.textContent || '').split('\n').filter((_, i, a) => a.length > 1 || a[0]);
  if (!lines.length) return;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.font = cs.fontWeight + ' ' + fontPx.toFixed(1) + 'px ' + cs.fontFamily;
  const stroke = el.classList.contains('onvideo');
  lines.forEach((ln, i) => {
    const y = topY + i * lineH;
    if (stroke) { ctx.lineWidth = Math.max(1, fontPx * 0.09); ctx.strokeStyle = '#000'; ctx.strokeText(ln, cx, y); }
    ctx.fillStyle = stroke ? '#fff' : cs.color;
    ctx.fillText(ln, cx, y);
  });
}
$('tbFrame').addEventListener('click', () => {
  try {
    const RW = (L.resolution || [1080, 1920])[0], RH = (L.resolution || [1080, 1920])[1];
    const cv = document.createElement('canvas'); cv.width = RW; cv.height = RH;
    const ctx = cv.getContext('2d');
    const vb = L.video_box || { x: 0, y: 0, w: RW, h: RH, r: 0 };
    ctx.fillStyle = (vb.w < RW - 1 || vb.h < RH - 1) ? '#fff' : '#000';
    ctx.fillRect(0, 0, RW, RH);
    ctx.save();
    roundRectPath(ctx, vb.x, vb.y, vb.w, vb.h, vb.r || 0);
    ctx.clip();
    const vw = v.videoWidth || RW, vh = v.videoHeight || RH;
    const scale = Math.max(vb.w / vw, vb.h / vh);
    const dw = vw * scale, dh = vh * scale;
    ctx.drawImage(v, vb.x + (vb.w - dw) / 2, vb.y + (vb.h - dh) / 2, dw, dh);
    ctx.restore();
    drawOverlayToCanvas(ctx, $('ttlOv'));
    drawOverlayToCanvas(ctx, $('capOv'));
    document.querySelectorAll('.tx-ov').forEach((el) => drawOverlayToCanvas(ctx, el));
    const a = document.createElement('a'); a.href = cv.toDataURL('image/png');
    a.download = 'frame_' + tc(v.currentTime).replace(/[:.]/g, '-') + '.png'; a.click();
    setStatus('프레임 저장(자막 포함)');
  } catch (e) { setStatus('프레임 저장 실패'); }
});
// 영상만 전체화면(우측 하단 버튼) — 상단 메뉴의 '전체 화면'(hFull, 앱 전체)과는 별개로
// stage 하나만 브라우저 전체화면으로 띄운다. fitVideo가 stage 크기 변화에 맞춰 다시
// 계산하므로 fullscreenchange에서 한 번 더 불러 확대 배율을 갱신한다.
$('stageFullBtn').addEventListener('click', () => {
  const stage = $('stage');
  if (document.fullscreenElement === stage) document.exitFullscreen();
  else stage.requestFullscreen().catch(() => {});
});
document.addEventListener('fullscreenchange', () => { if (document.fullscreenElement === $('stage') || !document.fullscreenElement) fitVideo(); });
function markIn(at) {
  const t = (typeof at === 'number') ? at : v.currentTime; if (C.end - t < 1) { setStatus('클립은 최소 1초'); return; }
  pushUndo(); C.start = t; clipRangeDirty = true; markDirty(); renderAll(); setStatus('클립 시작 = ' + tc(t));
}
function markOut(at) {
  const t = (typeof at === 'number') ? at : v.currentTime; if (t - C.start < 1) { setStatus('클립은 최소 1초'); return; }
  pushUndo(); C.end = t; clipRangeDirty = true; markDirty(); renderAll(); setStatus('클립 끝 = ' + tc(t));
}
function addMarker() { markers.push(v.currentTime); renderRuler(); setStatus('마커 추가'); }

// ─── 오디오 미터(WebAudio 분석기) ───
let actx = null, analyserL = null, analyserR = null;
function ensureAudio() {
  if (actx) { if (actx.state === 'suspended') actx.resume(); return; }
  try {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    const src = actx.createMediaElementSource(v);
    const split = actx.createChannelSplitter(2);
    analyserL = actx.createAnalyser(); analyserR = actx.createAnalyser();
    analyserL.fftSize = 1024; analyserR.fftSize = 1024;
    src.connect(split); split.connect(analyserL, 0); split.connect(analyserR, 1);
    src.connect(actx.destination);
    requestAnimationFrame(meterLoop);
  } catch (e) { actx = null; }
}
const bufL = new Float32Array(1024), bufR = new Float32Array(1024);
let pkL = 0, pkR = 0;
function db2pct(db) { return Math.max(0, Math.min(100, (db + 54) / 54 * 100)); }
function meterLoop() {
  if (!analyserL) return;
  analyserL.getFloatTimeDomainData(bufL); analyserR.getFloatTimeDomainData(bufR);
  let sl = 0, sr = 0; for (let i = 0; i < 1024; i++) { sl += bufL[i] * bufL[i]; sr += bufR[i] * bufR[i]; }
  const dl = 20 * Math.log10(Math.sqrt(sl / 1024) + 1e-6), dr = 20 * Math.log10(Math.sqrt(sr / 1024) + 1e-6);
  const l = v.paused ? 0 : db2pct(dl), r = v.paused ? 0 : db2pct(dr);
  pkL = Math.max(l, pkL - 0.6); pkR = Math.max(r, pkR - 0.6);
  const h = $('mL').parentElement.clientHeight;
  $('mL').style.setProperty('--mh', h + 'px'); $('mR').style.setProperty('--mh', h + 'px');
  $('mL').style.height = l + '%'; $('mR').style.height = r + '%';
  $('pL').style.bottom = pkL + '%'; $('pR').style.bottom = pkR + '%';
  requestAnimationFrame(meterLoop);
}
$('aMute').addEventListener('click', () => { v.muted = !v.muted; $('aMute').classList.toggle('on', v.muted); });
$('aSolo').addEventListener('click', () => {
  const on = !$('aSolo').classList.contains('on');
  $('aSolo').classList.toggle('on', on);
  const bgmA = $('bgmAudio');
  if (on) { bgmA.dataset.soloMuted = bgmA.muted ? '0' : '1'; bgmA.muted = true; }
  else if (bgmA.dataset.soloMuted === '1') { bgmA.muted = false; delete bgmA.dataset.soloMuted; }
});
$('syncLock').addEventListener('click', () => { syncLock = !syncLock; $('syncLock').classList.toggle('on', syncLock); setStatus(syncLock ? '동기화 잠금 켬 — 잔물결 편집 시 마커도 밀림' : '동기화 잠금 끔 — 마커는 제자리 유지'); });

// ─── 도구 ───
const TOOL_KEYS = { v: 'select', a: 'trackfwd', b: 'ripple', n: 'roll', c: 'razor', y: 'slip', p: 'pen', h: 'hand', t: 'type' };
function setTool(t) {
  tool = t;
  document.querySelectorAll('.tool').forEach(b => b.classList.toggle('on', b.dataset.tool === t));
  const tp = $('tlPanel');
  tp.className = tp.className.replace(/\btool-\S+/g, '').trim() + ' tool-' + t;
  if (t === 'slip' || t === 'pen') setStatus('자막 트랙에는 ' + (t === 'slip' ? '밀어넣기' : '펜') + ' 도구가 적용되지 않아요');
}
document.querySelectorAll('.tool').forEach(b => b.addEventListener('click', () => setTool(b.dataset.tool)));

// ─── 트랙 헤더 토글 ───
$('koLock').addEventListener('click', () => {
  koLocked = !koLocked; $('koLock').classList.toggle('on', koLocked);
  for (let i = 0; i < nKoTracks; i++) { const l = koLaneEl(i); if (l) l.classList.toggle('locked', koLocked); }
  syncSelPanel();
});
$('koEye').addEventListener('click', () => { koVisible = !koVisible; $('koEye').classList.toggle('on', koVisible); updateOverlay(); });
$('enEye').addEventListener('click', () => setEnVisible(!enVisible));
function setEnVisible(on) { enVisible = on; $('enEye').classList.toggle('on', on); $('miEn').classList.toggle('chk', on); updateOverlay(); }
$('miEn').classList.add('chk');
$('hbSnap').addEventListener('click', () => setSnap(!snapOn));
function setSnap(on) { snapOn = on; $('hbSnap').classList.toggle('on', on); $('miSnap').classList.toggle('chk', on); }
setSnap(true);
$('hbLink').addEventListener('click', () => setStatus('영어 자막은 항상 한글 자막과 연결돼 있어요'));
$('hbSettings').addEventListener('click', () => { const th = $('thumbs'); th.style.display = th.style.display === 'none' ? '' : 'none'; });
$('hbCC').addEventListener('click', () => setStatus('캡션 트랙: C1 한글' + (ens.length ? ' · C2 영어' : '')));

// ─── 타임라인 렌더 ───
const tlTracks = $('tlTracks');
const tw = () => tlTracks.clientWidth;
const t2x = (t) => (t - view.a) / (view.b - view.a) * tw();
const x2t = (px) => view.a + px / tw() * (view.b - view.a);
function renderRuler() {
  const r = $('ruler'); r.innerHTML = '';
  const span = view.b - view.a;
  const step = span > 600 ? 120 : span > 240 ? 60 : span > 90 ? 30 : span > 40 ? 10 : span > 12 ? 5 : span > 4 ? 1 : span > 1.5 ? 0.5 : span > 0.6 ? 0.2 : 0.1;
  const io = document.createElement('div'); io.className = 'inout';
  io.style.left = t2x(C.start) + 'px'; io.style.width = Math.max(0, t2x(C.end) - t2x(C.start)) + 'px'; r.appendChild(io);
  for (let t = Math.ceil(view.a / step) * step; t <= view.b; t += step) {
    const el = document.createElement('div'); el.className = 'tk'; el.style.left = t2x(t) + 'px'; el.textContent = tcShort(t); r.appendChild(el);
    for (let k = 1; k < 5; k++) { const mt = t + step * k / 5; if (mt > view.b) break; const m = document.createElement('div'); m.className = 'tk minor'; m.style.left = t2x(mt) + 'px'; r.appendChild(m); }
  }
  markers.forEach(m => { if (m < view.a || m > view.b) return; const el = document.createElement('div'); el.className = 'mk'; el.style.left = t2x(m) + 'px'; r.appendChild(el); });
}
function renderThumbs() {
  const box = $('thumbs'); box.innerHTML = '';
  const N = Math.min(24, Math.max(8, Math.round(tw() / 80)));
  for (let k = 0; k < N; k++) {
    const t = view.a + (view.b - view.a) * (k + 0.5) / N;
    const img = document.createElement('img'); img.src = '/media/' + VIDEO_ID + '/thumb/' + Math.round(t) + '.jpg'; img.draggable = false; box.appendChild(img);
  }
}
function renderPlayhead() {
  const t = v.currentTime, on = t >= view.a && t <= view.b;
  $('phLine').style.display = on ? 'block' : 'none'; $('phHead').style.display = on ? 'block' : 'none';
  if (on) { $('phLine').style.left = t2x(t) + 'px'; $('phHead').style.left = t2x(t) + 'px'; }
}
function mkBlock(track, item, i, isEn) {
  const x0 = t2x(item.start), x1 = t2x(item.end);
  if (x1 < 0 || x0 > tw()) return;
  const b = document.createElement('div');
  b.className = 'blk' + (!isEn && (i === selIdx || selSet.has(i)) ? ' sel' : '');
  if (!isEn) b.dataset.idx = i;
  b.style.left = Math.max(-2, x0) + 'px';
  b.style.width = Math.max(10, Math.min(tw() + 2, x1) - Math.max(-2, x0)) + 'px';
  const s = document.createElement('span'); s.className = 'txt'; s.textContent = item.text; b.appendChild(s);
  if (!isEn) {
    const hl = document.createElement('div'); hl.className = 'h l';
    const hr = document.createElement('div'); hr.className = 'h r';
    b.appendChild(hl); b.appendChild(hr);
    b.addEventListener('pointerdown', (e) => {
      if (e.button === 2) { if (!selSet.has(i)) select(i); return; }
      if (tool === 'hand') return;
      if (koLocked) { select(i); return; }
      if (tool === 'razor') { splitBlockAt(i, e); return; }
      if (tool === 'type') { select(i); openEdit(i, b); return; }
      if (tool === 'slip' || tool === 'pen') { select(i); return; }
      const edge = e.target === hl ? 'l' : e.target === hr ? 'r' : 'm';
      if (tool === 'trackfwd') { selSet = new Set(); for (let k = i; k < caps.length; k++) selSet.add(k); startDrag(e, i, 'm'); return; }
      if (tool === 'ripple' && edge !== 'm') { startDrag(e, i, 'rip-' + edge); return; }
      if (tool === 'roll' && edge !== 'm') { startDrag(e, i, 'roll-' + edge); return; }
      startDrag(e, i, edge);
    });
    b.addEventListener('dblclick', (e) => { e.stopPropagation(); openEdit(i, b); });
  }
  track.appendChild(b);
}
// 자막이 없는 빈 구간마다 점선 '+' 칸을 둔다 — 누르면 그 구간을 채우는 자막이 생긴다.
function renderAddSlots(track) {
  const gaps = [];
  const ordered = caps.map((c, i) => ({ ...c, i })).sort((a, b) => a.start - b.start);
  let cur = C.start;
  for (const c of ordered) {
    if (c.start - cur > 0.25) gaps.push([cur, c.start]);
    cur = Math.max(cur, c.end);
  }
  if (C.end - cur > 0.25) gaps.push([cur, C.end]);
  for (const [a, b] of gaps) {
    const x0 = t2x(a), x1 = t2x(b);
    if (x1 < 0 || x0 > tw() || x1 - x0 < 16) continue;
    const el = document.createElement('div');
    el.className = 'blk-add';
    el.title = '이 빈 구간에 자막 추가';
    el.textContent = '+';
    el.style.left = Math.max(0, x0) + 'px';
    el.style.width = (Math.min(tw(), x1) - Math.max(0, x0)) + 'px';
    el.addEventListener('pointerdown', (e) => e.stopPropagation());
    el.addEventListener('click', () => addCapAt(a, b));
    track.appendChild(el);
  }
}
function addCapAt(a, b) {
  pushUndo();
  const end = Math.min(b, a + Math.max(0.6, Math.min(b - a, 5)));
  caps.push({ start: a, end: end, text: '새 자막' });
  caps.sort((x, y) => x.start - y.start);
  if (ens.length) { ens = []; setStatus('자막 추가 — 영어 트랙은 다시 번역해 주세요'); }
  selIdx = caps.findIndex((c) => c.start === a); selSet = new Set();
  markDirty(); renderTracks(); updateOverlay(); syncSelPanel();
  const blk = $('koTrack').querySelector('.blk.sel');
  if (blk) openEdit(selIdx, blk);
}
// 자막을 끌거나 늘렸을 때 옆 자막과 겹치지 않도록 옆 자막을 같이 밀어낸다(길이는 유지).
// 겹치면 두 자막이 한 화면에 같이 떠서 화면이 엉킨다 — 밀어내는 편이 항상 옳다.
function pushNeighbors(i) {
  // 겹침 자막(다른 트랙)이 섞여 있으면 배열 인덱스가 곧 '시간순 이웃'이 아니므로, 같은
  // 트랙끼리만 밀어낸다 — 안 그러면 트랙이 다른, 실제로 안 겹치는 자막까지 옆으로 밀렸다.
  const trk = caps[i].track || 0;
  const idxs = caps.map((c, k) => k).filter((k) => (caps[k].track || 0) === trk)
    .sort((a, b) => caps[a].start - caps[b].start);
  const pos = idxs.indexOf(i);
  for (let p = pos + 1; p < idxs.length; p++) {
    const k = idxs[p], kp = idxs[p - 1];
    const need = caps[kp].end - caps[k].start;
    if (need <= 0) break;
    const len = caps[k].end - caps[k].start;
    caps[k].start = Math.min(dur - 0.3, caps[k].start + need);
    caps[k].end = Math.min(dur, caps[k].start + len);
  }
  for (let p = pos - 1; p >= 0; p--) {
    const k = idxs[p], kn = idxs[p + 1];
    const need = caps[k].end - caps[kn].start;
    if (need <= 0) break;
    const len = caps[k].end - caps[k].start;
    caps[k].end = Math.max(0.3, caps[k].end - need);
    caps[k].start = Math.max(0, caps[k].end - len);
  }
}
// 겹침 자막용 추가 트랙(C1+2, C1+3, ...). 기본 트랙(koHead/koTrack, index 0)은 항상 있고,
// nKoTracks가 2 이상이면 그만큼 레인을 헤더 열(C2/en 앞)과 트랙 몸통(en 트랙 앞)에 만든다.
function koHeadEl(i) { return i === 0 ? $('koHead') : $('koHead' + i); }
function koLaneEl(i) { return i === 0 ? $('koTrack') : $('koTrack' + i); }
function ensureKoLanes() {
  const heads = $('headsScroll'), lanes = $('tracksScroll');
  for (let i = 1; i < nKoTracks; i++) {
    if (!koHeadEl(i)) {
      const h = document.createElement('div');
      h.className = 'trk-h h-c'; h.id = 'koHead' + i; h.style.height = '34px';
      h.innerHTML = '<span class="patch">C1+' + (i + 1) + '</span>' +
        '<button class="addbtn" data-kodel="' + i + '" title="이 자막 트랙 삭제">×</button>' +
        '<span class="tn">겹침 자막</span>';
      heads.insertBefore(h, $('enHead'));
    }
    if (!koLaneEl(i)) {
      const l = document.createElement('div');
      l.className = 'trk ko h-c'; l.id = 'koTrack' + i; l.dataset.ko = i;
      lanes.insertBefore(l, $('enTrack'));
    }
  }
  for (let i = nKoTracks; koHeadEl(i); i++) { koHeadEl(i).remove(); const l = koLaneEl(i); if (l) l.remove(); }
}
function deleteKoTrack(i) {
  if (i <= 0) return;
  pushUndo();
  caps = caps.filter((c) => (c.track || 0) !== i).map((c) => ((c.track || 0) > i ? { ...c, track: (c.track || 0) - 1 } : c));
  nKoTracks = Math.max(1, nKoTracks - 1);
  markDirty(); renderTracks(); updateOverlay(); syncSelPanel();
  setStatus('자막 트랙 삭제');
}
function renderTracks() {
  ensureKoLanes();
  const en = $('enTrack');
  for (let i = 0; i < nKoTracks; i++) { const l = koLaneEl(i); if (l) l.querySelectorAll('.blk, .blk-add').forEach(el => el.remove()); }
  en.querySelectorAll('.blk').forEach(el => el.remove());
  caps.forEach((c, i) => { const lane = koLaneEl(c.track || 0); if (lane) mkBlock(lane, c, i, false); });
  renderAddSlots(koLaneEl(0));
  if (ens.length) {
    en.style.display = 'block'; $('enHead').style.display = 'flex';
    ens.forEach((c, i) => { if (caps[i]) mkBlock(en, { start: caps[i].start, end: caps[i].end, text: c.text }, i, true); });
  } else { en.style.display = 'none'; $('enHead').style.display = 'none'; }
  // V1/A1 클립 막대(클립 구간) — 영상을 실제로 여러 조각(vsegs)으로 자를 수 있다.
  const vt = $('vTrack'); vt.querySelectorAll('.vclip').forEach(el => el.remove());
  vsegs.forEach((sg, i) => {
    const x0 = t2x(sg.s), x1 = t2x(sg.e);
    if (x1 < 0 || x0 > tw()) return;
    const vc = document.createElement('div'); vc.className = 'vclip' + (i === selVSeg ? ' sel' : '');
    vc.dataset.vseg = i;
    vc.style.left = Math.max(-2, x0) + 'px'; vc.style.width = Math.max(6, Math.min(tw() + 2, x1) - Math.max(-2, x0)) + 'px';
    vc.style.background = 'transparent'; vc.style.borderColor = i === selVSeg ? '#fff' : '#7f93cf'; vc.style.zIndex = 2;
    const cn = document.createElement('span'); cn.className = 'cn';
    cn.textContent = 'V1 · ' + (C.title || '') + (vsegs.length > 1 ? (' [' + (i + 1) + '/' + vsegs.length + ']') : '');
    vc.appendChild(cn);
    const hl = document.createElement('div'); hl.className = 'h l';
    const hr = document.createElement('div'); hr.className = 'h r';
    vc.appendChild(hl); vc.appendChild(hr);
    vc.addEventListener('pointerdown', (e) => {
      if (e.button === 2) { selVSeg = i; renderTracks(); return; }
      if (tool === 'hand') return;
      if (tool === 'razor') { const rect = tlTracks.getBoundingClientRect(); splitVSegAt(x2t(e.clientX - rect.left)); return; }
      const edge = e.target === hl ? 'l' : e.target === hr ? 'r' : 'm';
      if (edge === 'm') { const rect = tlTracks.getBoundingClientRect(); seek(x2t(e.clientX - rect.left)); }
      startVSegDrag(e, i, edge);
    });
    vt.appendChild(vc);
  });
  const ac = $('aClip'); ac.style.left = '0'; ac.style.width = tw() + 'px';
  const bc = $('bgmClip'); bc.style.left = t2x(C.start) + 'px'; bc.style.width = Math.max(4, t2x(C.end) - t2x(C.start)) + 'px';
  drawWave();
  renderNavi();
  renderTextTracks();
}
// ─── V1 실제 영상 자르기(조각 분할/삭제) ────────────────────────────────────
// 확인 팝업의 '분할' 기능(segs/keep_ranges)과 완전히 같은 저장 구조를 스튜디오
// 타임라인에도 그대로 노출한다 — "영상을 클릭하고 분할했으면 영상이 실제로 잘려야지"
// (2026-09-08). 조각은 절대(원본) 초 좌표를 그대로 쓴다(팝업과 동일 — 지운 자리를
// 화면에서 당겨 붙이지 않고 그냥 빈 구간으로 둔다. 실제 이어붙이기는 렌더가 한다).
// 구간이 바뀌면 저장된 자막 시간이 더는 유효하지 않으므로(팝업과 같은 이유) 저장 시
// 자동으로 비워 렌더가 새 길이로 다시 전사하게 한다(찬양 클립은 재전사가 없어 예외).
let vsegs = [], selVSeg = -1;
function initVSegs() {
  vsegs = (C.keep_ranges && C.keep_ranges.length)
    ? C.keep_ranges.map((r) => ({ s: r[0], e: r[1] }))
    : [{ s: C.start, e: C.end }];
  selVSeg = -1;
}
function vsegsChanged() {
  if (!vsegs.length) return false;
  return vsegs.length > 1 || Math.abs(vsegs[0].s - C.start) > 0.05 || Math.abs(vsegs[vsegs.length - 1].e - C.end) > 0.05;
}
function splitVSegAt(t) {
  for (let i = 0; i < vsegs.length; i++) {
    if (t > vsegs[i].s + 0.3 && t < vsegs[i].e - 0.3) {
      pushUndo();
      vsegs.splice(i + 1, 0, { s: t, e: vsegs[i].e });
      vsegs[i].e = t; selVSeg = i + 1;
      markDirty(); clipRangeDirty = true; renderTracks(); setStatus('영상 분할');
      return;
    }
  }
  setStatus('여기는 이미 조각 경계예요');
}
function deleteVSeg(i) {
  if (vsegs.length <= 1) { setStatus('최소 한 조각은 남아 있어야 해요'); return; }
  pushUndo();
  vsegs.splice(i, 1);
  selVSeg = -1;
  markDirty(); clipRangeDirty = true; renderTracks(); setStatus('영상 조각 삭제 — 저장하면 실제로 잘려요');
}
let vsegDrag = null;
function startVSegDrag(e, i, mode) {
  if (e.button !== undefined && e.button !== 0) return;
  e.preventDefault();
  selVSeg = i; renderTracks();
  vsegDrag = { i, mode, x0: e.clientX, moved: false, snap: vsegs.map((s) => ({ s: s.s, e: s.e })) };
  document.addEventListener('pointermove', onVSegDrag);
  document.addEventListener('pointerup', endVSegDrag, { once: true });
}
function onVSegDrag(e) {
  if (!vsegDrag) return;
  if (!vsegDrag.moved) { if (Math.abs(e.clientX - vsegDrag.x0) < 2) return; vsegDrag.moved = true; pushUndo(); }
  const dt = (e.clientX - vsegDrag.x0) / tw() * (view.b - view.a);
  const i = vsegDrag.i, S = vsegDrag.snap[i], sg = vsegs[i];
  const prevEnd = i > 0 ? vsegDrag.snap[i - 1].e + 0.1 : 0;
  const nextStart = i < vsegs.length - 1 ? vsegDrag.snap[i + 1].s - 0.1 : dur;
  if (vsegDrag.mode === 'l') sg.s = Math.min(Math.max(prevEnd, S.s + dt), sg.e - 0.5);
  else if (vsegDrag.mode === 'r') sg.e = Math.max(Math.min(nextStart, S.e + dt), sg.s + 0.5);
  else if (vsegDrag.mode === 'm') {
    // 가운데를 끌면 조각 전체(길이는 그대로)를 슬라이드 — 자막처럼 드래그로 옮겨진다.
    // 이웃 조각을 넘어갈 수는 없다(순서가 바뀌면 렌더가 시간순으로 다시 정렬해버려
    // 자막·자유 텍스트와 어긋난다).
    const len = S.e - S.s;
    let ns = Math.max(prevEnd, Math.min(nextStart - len, S.s + dt));
    sg.s = ns; sg.e = ns + len;
  }
  markDirty(); clipRangeDirty = true; renderTracks();
}
function endVSegDrag() { vsegDrag = null; document.removeEventListener('pointermove', onVSegDrag); }
// ─── 자유 텍스트 트랙(캡컷식) ──────────────────────────────────────────────
// 자막(caps)은 한 트랙에 시간순으로 늘어서고 서로 겹칠 수 없다. 그와 별개로, 캡컷처럼
// "아무 데나 놓고 아무 때나 띄우는 텍스트"가 필요하다는 요청(2026-09-08). 그래서 요소마다
// 자기 시간·화면 위치·크기를 갖는 자유 텍스트를 별도 트랙(T1, T2 …)에 둔다. 트랙은 좌측
// '+ 트랙 추가'로 늘리고, 블록을 아래로 끌면 새 트랙이 자동으로 생긴다.
let texts = [], nTxTracks = 0, selTx = -1;
const TX_LANE_H = 30;
function txDefSize() { return Math.round((L && L.caption_font_size) || 72); }
function initTexts() {
  texts = ((C && C.free_texts) || []).map(t => ({
    start: +t.start, end: +t.end, text: String(t.text || ''),
    x: +t.x || 0, y: +t.y || 0, size: +t.size || 0, track: Math.max(0, t.track | 0),
  }));
  nTxTracks = texts.reduce((m, t) => Math.max(m, t.track + 1), 0);
}
// 트랙 개수(nTxTracks)에 맞춰 헤더 줄과 레인을 만든다/지운다.
function ensureTxLanes() {
  const heads = $('headsScroll'), lanes = $('tracksScroll');
  for (let i = 0; i < nTxTracks; i++) {
    if (!$('txHead' + i)) {
      const h = document.createElement('div');
      h.className = 'trk-h'; h.id = 'txHead' + i; h.style.height = TX_LANE_H + 'px';
      h.innerHTML = '<span class="patch">T' + (i + 1) + '</span>' +
        '<button class="addbtn" data-txadd="' + i + '" title="재생 헤드에 텍스트 추가">+</button>' +
        '<button class="addbtn" data-txdel="' + i + '" title="이 텍스트 트랙 삭제">×</button>' +
        '<span class="tn">T' + (i + 1) + ' 텍스트</span>';
      heads.insertBefore(h, $('vHead'));
    }
    if (!$('txTrack' + i)) {
      const l = document.createElement('div');
      l.className = 'trk tx'; l.id = 'txTrack' + i; l.dataset.tx = i;
      lanes.insertBefore(l, $('vTrack'));
    }
  }
  for (let i = nTxTracks; $('txHead' + i); i++) { $('txHead' + i).remove(); const t = $('txTrack' + i); if (t) t.remove(); }
}
function renderTextTracks() {
  ensureTxLanes();
  for (let i = 0; i < nTxTracks; i++) {
    const lane = $('txTrack' + i); if (!lane) continue;
    lane.querySelectorAll('.txblk').forEach(el => el.remove());
  }
  texts.forEach((t, i) => {
    const lane = $('txTrack' + t.track); if (!lane) return;
    const x0 = t2x(t.start), x1 = t2x(t.end);
    if (x1 < 0 || x0 > tw()) return;
    const b = document.createElement('div');
    b.className = 'txblk' + (i === selTx ? ' sel' : '');
    b.dataset.tx = i;
    b.style.left = Math.max(-2, x0) + 'px';
    b.style.width = Math.max(10, Math.min(tw() + 2, x1) - Math.max(-2, x0)) + 'px';
    const sp = document.createElement('span'); sp.className = 'txt'; sp.textContent = t.text || '(빈 텍스트)';
    const hl = document.createElement('div'); hl.className = 'h l';
    const hr = document.createElement('div'); hr.className = 'h r';
    b.appendChild(sp); b.appendChild(hl); b.appendChild(hr);
    b.addEventListener('pointerdown', (e) => {
      if (e.button === 2) { selectTx(i); return; }
      if (tool === 'hand') return;
      if (tool === 'razor') { const rect = tlTracks.getBoundingClientRect(); splitTxAt(i, x2t(e.clientX - rect.left)); return; }
      const edge = e.target === hl ? 'l' : e.target === hr ? 'r' : 'm';
      startTxDrag(e, i, edge);
    });
    b.addEventListener('dblclick', (e) => { e.stopPropagation(); selectTx(i); seek(Math.max(t.start + 0.01, Math.min(v.currentTime, t.end - 0.01))); focusTxOverlay(i); });
    lane.appendChild(b);
  });
}
function selectTx(i) { selTx = i; selIdx = -1; selSet = new Set(); renderTracks(); syncSelPanel(); updateOverlay(); }
function addTextAt(track, t) {
  const a = Math.max(C.start, Math.min(t, C.end - 1));
  pushUndo();
  texts.push({ start: a, end: Math.min(a + 3, C.end), text: '새 텍스트', x: 0, y: Math.round((L.resolution[1] || 1920) * 0.45), size: txDefSize(), track: track });
  selTx = texts.length - 1;
  markDirty(); renderTracks(); updateOverlay(); syncSelPanel();
  focusTxOverlay(selTx);
}
function addTextTrack() {
  nTxTracks += 1;
  addTextAt(nTxTracks - 1, v.currentTime);
  setStatus('텍스트 트랙 T' + nTxTracks + ' 추가');
}
function deleteTextTrack(i) {
  pushUndo();
  texts = texts.filter(t => t.track !== i).map(t => (t.track > i ? { ...t, track: t.track - 1 } : t));
  nTxTracks = Math.max(0, nTxTracks - 1); selTx = -1;
  markDirty(); renderTracks(); updateOverlay();
}
function splitTxAt(i, t) {
  const c = texts[i]; if (!c || t <= c.start + 0.05 || t >= c.end - 0.05) return;
  pushUndo();
  texts.splice(i + 1, 0, { ...c, start: t });
  c.end = t; selTx = i;
  markDirty(); renderTracks(); updateOverlay(); setStatus('텍스트 분할');
}
let txDrag = null;
function startTxDrag(e, i, mode) {
  if (e.button !== undefined && e.button !== 0) return;
  e.preventDefault();
  selectTx(i);
  txDrag = { i, mode, x0: e.clientX, y0: e.clientY, snap: texts.map(t => ({ start: t.start, end: t.end, track: t.track })), moved: false };
  document.addEventListener('pointermove', onTxDrag);
  document.addEventListener('pointerup', endTxDrag, { once: true });
}
function onTxDrag(e) {
  if (!txDrag) return;
  if (!txDrag.moved) {
    if (Math.abs(e.clientX - txDrag.x0) < 2 && Math.abs(e.clientY - txDrag.y0) < 2) return;
    txDrag.moved = true;
  }
  const dt = (e.clientX - txDrag.x0) / tw() * (view.b - view.a);
  const i = txDrag.i, S = txDrag.snap[i], t = texts[i];
  if (txDrag.mode === 'm') {
    let ns = snapT(S.start + dt, -1);
    const len = S.end - S.start;
    ns = Math.max(0, Math.min(dur - len, ns));
    t.start = ns; t.end = ns + len;
    // 세로로 끌면 트랙 이동 — 마지막 트랙보다 아래로 내리면 새 트랙이 생긴다.
    const steps = Math.round((e.clientY - txDrag.y0) / TX_LANE_H);
    let nt = Math.max(0, S.track + steps);
    if (nt >= nTxTracks) { nt = nTxTracks; nTxTracks = nt + 1; }
    t.track = nt;
  } else if (txDrag.mode === 'l') {
    t.start = Math.max(0, Math.min(snapT(S.start + dt, -1), t.end - 0.3));
  } else {
    t.end = Math.min(dur, Math.max(snapT(S.end + dt, -1), t.start + 0.3));
  }
  markDirty(); renderTracks(); updateOverlay();
}
function endTxDrag() { txDrag = null; document.removeEventListener('pointermove', onTxDrag); }

// ── 미리보기 위의 자유 텍스트(끌어서 위치, 더블클릭으로 수정) ──
function renderTextOverlays() {
  const frame = $('frame');
  const now = v.currentTime;
  const live = new Set();
  texts.forEach((t, i) => {
    if (now < t.start || now > t.end) return;
    live.add(i);
    let el = frame.querySelector('.tx-ov[data-tx="' + i + '"]');
    if (!el) {
      el = document.createElement('div');
      el.className = 'tx-ov'; el.dataset.tx = String(i);
      bindTxOverlay(el);
      frame.appendChild(el);
    }
    if (document.activeElement !== el) el.textContent = t.text;
    el.classList.toggle('sel', i === selTx);
    const kc = (L.caption_ass_coeff || 1);
    el.style.fontSize = ((t.size || txDefSize()) * kc * SC) + 'px';
    el.style.left = (((L.resolution[0] / 2) + (t.x || 0)) * SC) + 'px';
    el.style.top = ((t.y || 0) * SC) + 'px';
  });
  frame.querySelectorAll('.tx-ov').forEach(el => { if (!live.has(+el.dataset.tx)) el.remove(); });
}
function bindTxOverlay(el) {
  el.addEventListener('pointerdown', (e) => {
    const i = +el.dataset.tx; const t = texts[i]; if (!t) return;
    if (el.classList.contains('editing')) return;      // 편집 중엔 커서 이동이 우선
    e.preventDefault(); selectTx(i);
    el.classList.add('dragging'); el.setPointerCapture(e.pointerId);
    const sx = e.clientX, sy = e.clientY, ox = t.x || 0, oy = t.y || 0;
    const move = (ev) => {
      t.x = Math.round(ox + (ev.clientX - sx) / SC);
      t.y = Math.round(oy + (ev.clientY - sy) / SC);
      markDirty(); renderTextOverlays();
    };
    const up = () => { el.classList.remove('dragging'); document.removeEventListener('pointermove', move); };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up, { once: true });
  });
  el.addEventListener('dblclick', (e) => { e.stopPropagation(); startTxEdit(el); });
  el.addEventListener('blur', () => {
    const i = +el.dataset.tx; if (!texts[i]) return;
    el.classList.remove('editing'); el.contentEditable = 'false';
    const val = el.textContent.trim();
    if (val !== texts[i].text) { texts[i].text = val; markDirty(); renderTracks(); }
  });
  el.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Escape' || (e.key === 'Enter' && !e.shiftKey)) { e.preventDefault(); el.blur(); }
  });
}
function startTxEdit(el) {
  el.classList.add('editing'); el.contentEditable = 'true'; el.focus();
  const r = document.createRange(); r.selectNodeContents(el);
  const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
}
function focusTxOverlay(i) {
  renderTextOverlays();
  const el = $('frame').querySelector('.tx-ov[data-tx="' + i + '"]');
  if (el) startTxEdit(el);
}
$('btnAddTrack').addEventListener('click', addTextTrack);
$('addTrkRow').addEventListener('click', (e) => { if (e.target.id !== 'btnAddTrack') addTextTrack(); });
$('headsScroll').addEventListener('click', (e) => {
  const add = e.target.closest('[data-txadd]'); if (add) { addTextAt(+add.dataset.txadd, v.currentTime); return; }
  const del = e.target.closest('[data-txdel]'); if (del) { deleteTextTrack(+del.dataset.txdel); return; }
  const kodel = e.target.closest('[data-kodel]'); if (kodel) deleteKoTrack(+kodel.dataset.kodel);
});

function renderAll() { if (!C) return; renderRuler(); renderThumbs(); renderTracks(); renderPlayhead(); renderKf(); }
$('tracksScroll').addEventListener('scroll', () => { $('headsScroll').scrollTop = $('tracksScroll').scrollTop; });

// 클릭=탐색 / 휠=확대·축소 / 손 도구=이동
$('ruler').addEventListener('pointerdown', (e) => {
  const rect = $('ruler').getBoundingClientRect();
  const go = (ev) => seek(x2t(ev.clientX - rect.left));
  go(e);
  const up = () => document.removeEventListener('pointermove', go);
  document.addEventListener('pointermove', go); document.addEventListener('pointerup', up, { once: true });
});
$('tracksScroll').addEventListener('pointerdown', (e) => {
  if (e.target.closest('.blk, .txblk')) return;
  if (tool === 'hand') { panDrag(e); return; }
  if (e.target.closest('.trk.v, .trk.a')) { const rect = tlTracks.getBoundingClientRect(); seek(x2t(e.clientX - rect.left)); }
  if (e.target.closest('.trk.ko, .trk.en, .trk.tx')) { selIdx = -1; selSet = new Set(); selTx = -1; renderTracks(); syncSelPanel(); updateOverlay(); }
});
function panDrag(e) {
  const x0 = e.clientX, a0 = view.a, span = view.b - view.a;
  const move = (ev) => { let a = a0 - (ev.clientX - x0) / tw() * span; a = Math.max(0, Math.min(dur - span, a)); view = { a, b: a + span }; renderAll(); };
  document.addEventListener('pointermove', move);
  document.addEventListener('pointerup', () => document.removeEventListener('pointermove', move), { once: true });
}
function zoomBy(factor, pivot) {
  let a = pivot - (pivot - view.a) * factor, b = pivot + (view.b - pivot) * factor;
  if (b - a < 0.3) { const c = (a + b) / 2; a = c - 0.15; b = c + 0.15; }
  view = { a: Math.max(0, a), b: Math.min(dur, b) }; renderAll();
}
tlTracks.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = tlTracks.getBoundingClientRect();
  if (e.altKey || !e.shiftKey && Math.abs(e.deltaY) >= Math.abs(e.deltaX)) zoomBy(e.deltaY > 0 ? 1.25 : 0.8, x2t(e.clientX - rect.left));
  else { const span = view.b - view.a; let a = view.a + (e.deltaX || e.deltaY) / tw() * span; a = Math.max(0, Math.min(dur - span, a)); view = { a, b: a + span }; renderAll(); }
}, { passive: false });
$('zoomIn').addEventListener('click', () => zoomBy(0.75, (view.a + view.b) / 2));
$('zoomOut').addEventListener('click', () => zoomBy(1.34, (view.a + view.b) / 2));
$('zoomFit').addEventListener('click', zoomFit);
function zoomFit() { view = { a: homeView.a, b: homeView.b }; renderAll(); }

// 하단 탐색 막대(줌 스크롤바)
function renderNavi() {
  const n = $('navi'), th = $('naviThumb'); const W = n.clientWidth;
  th.style.left = (view.a / dur * W) + 'px'; th.style.width = Math.max(14, (view.b - view.a) / dur * W) + 'px';
}
$('naviThumb').addEventListener('pointerdown', (e) => {
  e.preventDefault();
  const n = $('navi'), W = n.clientWidth, r = $('naviThumb').getBoundingClientRect();
  const edge = e.clientX - r.left < 7 ? 'l' : r.right - e.clientX < 7 ? 'r' : 'm';
  const x0 = e.clientX, a0 = view.a, b0 = view.b;
  const move = (ev) => {
    const dt = (ev.clientX - x0) / W * dur;
    if (edge === 'm') { const span = b0 - a0; let a = Math.max(0, Math.min(dur - span, a0 + dt)); view = { a, b: a + span }; }
    else if (edge === 'l') view = { a: Math.max(0, Math.min(b0 - 0.3, a0 + dt)), b: b0 };
    else view = { a: a0, b: Math.min(dur, Math.max(a0 + 0.3, b0 + dt)) };
    renderAll();
  };
  document.addEventListener('pointermove', move);
  document.addEventListener('pointerup', () => document.removeEventListener('pointermove', move), { once: true });
});

// ─── 오디오 파형(서버가 ffmpeg로 뽑은 피크) ───
async function loadPeaks() {
  try {
    const hn = Math.min(6000, Math.max(200, Math.round((homeView.b - homeView.a) * 40)));
    const [h, f] = await Promise.all([
      fetch('/media/' + VIDEO_ID + '/peaks.json?a=' + homeView.a.toFixed(2) + '&b=' + homeView.b.toFixed(2) + '&n=' + hn).then(r => r.json()),
      fetch('/media/' + VIDEO_ID + '/peaks.json?a=0&b=' + dur.toFixed(2) + '&n=4000').then(r => r.json()),
    ]);
    peaksHome = h; peaksFull = f; drawWave();
  } catch (e) {}
}
function drawWave() {
  const cv = $('wave'), W = tw(), H = $('aTrack').clientHeight - 6;
  if (W <= 0 || H <= 0) return;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, W, H);
  const src = (peaksHome && view.a >= peaksHome.a - 0.01 && view.b <= peaksHome.b + 0.01) ? peaksHome : peaksFull;
  if (!src || !src.peaks || !src.peaks.length) return;
  const n = src.peaks.length, per = (src.b - src.a) / n;
  ctx.fillStyle = '#7fd9c8';
  for (let x = 0; x < W; x++) {
    const t0 = x2t(x), t1 = x2t(x + 1);
    let i0 = Math.floor((t0 - src.a) / per), i1 = Math.max(i0 + 1, Math.ceil((t1 - src.a) / per));
    if (i1 <= 0 || i0 >= n) continue;
    let pk = 0; for (let i = Math.max(0, i0); i < Math.min(n, i1); i++) pk = Math.max(pk, src.peaks[i]);
    const hh = Math.max(1, pk * H * 0.95);
    ctx.fillRect(x, (H - hh) / 2, 1, hh);
  }
}

// ─── 선택/드래그/편집 ───
function select(i) { selIdx = i; selSet = new Set(); selTx = -1; renderTracks(); syncSelPanel(); updateOverlay(); }
let drag = null;
function startDrag(e, i, mode) {
  if (e.button !== undefined && e.button !== 0) return;
  e.preventDefault();
  if (!selSet.has(i)) { selIdx = i; if (tool !== 'trackfwd') selSet = new Set(); }
  else selIdx = i;
  renderTracks(); syncSelPanel();
  drag = { i, mode, x0: e.clientX, snap: caps.map(c => ({ start: c.start, end: c.end })), markersSnap: markers.slice(), moved: false };
  document.addEventListener('pointermove', onDrag);
  document.addEventListener('pointerup', endDrag, { once: true });
}
function snapT(t, i) {
  if (!snapOn) return t;
  const eps = (view.b - view.a) / tw() * 7;
  const cands = [v.currentTime, C.start, C.end].concat(markers);
  if (i > 0) cands.push(caps[i - 1].end);
  if (i < caps.length - 1) cands.push(caps[i + 1].start);
  for (const c of cands) if (Math.abs(t - c) < eps) return c;
  return t;
}
function onDrag(e) {
  if (!drag) return;
  if (!drag.moved) { if (Math.abs(e.clientX - drag.x0) < 2) return; drag.moved = true; pushUndo(drag.snap); }
  const dt = (e.clientX - drag.x0) / tw() * (view.b - view.a);
  const S = drag.snap, i = drag.i, c = caps[i];
  if (drag.mode === 'm') {
    const idxs = selSet.size ? [...selSet] : [i];
    let ns = snapT(S[i].start + dt, i);
    const minS = Math.min(...idxs.map(k => S[k].start)), maxE = Math.max(...idxs.map(k => S[k].end));
    let d = ns - S[i].start; d = Math.max(-minS, Math.min(dur - maxE, d));
    idxs.forEach(k => { caps[k].start = S[k].start + d; caps[k].end = S[k].end + d; });
  } else if (drag.mode === 'l') {
    c.start = Math.max(0, Math.min(snapT(S[i].start + dt, i), c.end - 0.3));
  } else if (drag.mode === 'r') {
    c.end = Math.min(dur, Math.max(snapT(S[i].end + dt, i), c.start + 0.3));
  } else if (drag.mode === 'rip-r') {
    const ne = Math.min(dur, Math.max(S[i].end + dt, S[i].start + 0.3)); const d = ne - S[i].end;
    c.end = ne; for (let k = i + 1; k < caps.length; k++) { caps[k].start = Math.min(dur, S[k].start + d); caps[k].end = Math.min(dur, S[k].end + d); }
    if (syncLock) markers = drag.markersSnap.map(m => m >= S[i].end ? Math.min(dur, Math.max(0, m + d)) : m);
  } else if (drag.mode === 'rip-l') {
    const nlen = Math.max(0.3, (S[i].end - S[i].start) - dt); const d = nlen - (S[i].end - S[i].start);
    c.end = Math.min(dur, S[i].end + d); for (let k = i + 1; k < caps.length; k++) { caps[k].start = Math.min(dur, S[k].start + d); caps[k].end = Math.min(dur, S[k].end + d); }
    if (syncLock) markers = drag.markersSnap.map(m => m >= S[i].end ? Math.min(dur, Math.max(0, m + d)) : m);
  } else if (drag.mode === 'roll-r' && caps[i + 1]) {
    const t = Math.max(S[i].start + 0.3, Math.min(S[i + 1].end - 0.3, S[i].end + dt)); c.end = t; caps[i + 1].start = t;
  } else if (drag.mode === 'roll-l' && caps[i - 1]) {
    const t = Math.max(S[i - 1].start + 0.3, Math.min(S[i].end - 0.3, S[i].start + dt)); c.start = t; caps[i - 1].end = t;
  } else if (drag.mode.startsWith('roll')) { return; }
  if (drag.mode === 'm' || drag.mode === 'l' || drag.mode === 'r') pushNeighbors(i);
  markDirty(); renderTracks(); updateOverlay(); syncSelPanel(); renderRuler();
}
function endDrag() { drag = null; document.removeEventListener('pointermove', onDrag); }

function splitBlockAt(i, e) {
  const rect = tlTracks.getBoundingClientRect(); splitAtTime(i, x2t(e.clientX - rect.left));
}
function splitAtTime(i, t) {
  const c = caps[i]; if (!c || t <= c.start + 0.05 || t >= c.end - 0.05) return;
  pushUndo();
  const second = { start: t, end: c.end, text: c.text, track: c.track }; c.end = t; caps.splice(i + 1, 0, second);
  if (ens[i]) ens.splice(i + 1, 0, { start: second.start, end: second.end, text: ens[i].text });
  selIdx = i; selSet = new Set(); markDirty(); renderTracks(); updateOverlay(); syncSelPanel(); setStatus('자막 분할');
}
function deleteSel(ripple) {
  if (selTx >= 0) { pushUndo(); texts.splice(selTx, 1); selTx = -1; markDirty(); renderTracks(); updateOverlay(); return; }
  if (selIdx < 0 && !selSet.size) return;
  pushUndo();
  const idxs = (selSet.size ? [...selSet] : [selIdx]).sort((a, b) => b - a);
  const gap = ripple && idxs.length === 1 ? (caps[idxs[0]].end - caps[idxs[0]].start) : 0;
  const cutAt = gap ? caps[idxs[0]].end : 0;
  const cutTrack = gap ? (caps[idxs[0]].track || 0) : 0;
  idxs.forEach(k => { caps.splice(k, 1); if (ens.length > k) ens.splice(k, 1); });
  if (gap) {
    // 같은 트랙(줄)만 당겨 붙인다 — 다른 트랙(겹침 자막)까지 같이 당기면 안 겹쳤던 자막이 겹치게 된다.
    caps.forEach((c) => { if ((c.track || 0) === cutTrack && c.start >= cutAt - 1e-6) { c.start -= gap; c.end -= gap; } });
    if (syncLock) { markers = markers.map(m => m >= cutAt ? Math.max(0, m - gap) : m); renderRuler(); }
  }
  selIdx = -1; selSet = new Set(); markDirty(); renderTracks(); updateOverlay(); syncSelPanel();
}
function syncSelPanel() {
  const has = selIdx >= 0 && !!caps[selIdx];
  ['selStart', 'selEnd'].forEach(id => $(id).disabled = !has || koLocked);
  $('selText').disabled = !has;
  $('fxSelName').textContent = has ? ('자막 ' + (selIdx + 1) + (selSet.size > 1 ? ' 외 ' + (selSet.size - 1) : '')) : '(선택 없음)';
  if (has) {
    if (document.activeElement !== $('selStart')) $('selStart').value = caps[selIdx].start.toFixed(3);
    if (document.activeElement !== $('selEnd')) $('selEnd').value = caps[selIdx].end.toFixed(3);
    if (document.activeElement !== $('selText')) $('selText').value = caps[selIdx].text;
    $('selDur').textContent = (caps[selIdx].end - caps[selIdx].start).toFixed(3) + '초';
    $('selEnRow').style.display = ens[selIdx] ? '' : 'none';
    if (ens[selIdx] && document.activeElement !== $('selTextEn')) $('selTextEn').value = ens[selIdx].text;
  } else { $('selStart').value = ''; $('selEnd').value = ''; $('selText').value = ''; $('selDur').textContent = '-'; $('selEnRow').style.display = 'none'; }
  renderKf();
}
function commitSelTimes() {
  if (selIdx < 0 || !caps[selIdx] || koLocked) return;
  let s = parseFloat($('selStart').value), en = parseFloat($('selEnd').value);
  if (Number.isNaN(s) || Number.isNaN(en)) { syncSelPanel(); return; }
  pushUndo(); s = Math.max(0, s); en = Math.min(dur, en); if (en <= s + 0.05) en = s + 0.05;
  caps[selIdx].start = s; caps[selIdx].end = en; markDirty(); renderTracks(); updateOverlay(); syncSelPanel();
}
$('selStart').addEventListener('change', commitSelTimes); $('selEnd').addEventListener('change', commitSelTimes);
hotDrag($('selStart')); hotDrag($('selEnd'));
$('selText').addEventListener('input', () => { if (selIdx >= 0 && caps[selIdx]) { caps[selIdx].text = $('selText').value; markDirty(); renderTracks(); updateOverlay(); } });
$('selTextEn').addEventListener('input', () => { if (selIdx >= 0 && ens[selIdx]) { ens[selIdx].text = $('selTextEn').value; markDirty(); renderTracks(); updateOverlay(); } });

function openEdit(i, blkEl) {
  closeEdit();
  const wrap = document.createElement('div'); wrap.className = 'blk-edit'; wrap.id = 'blkEdit';
  const inp = document.createElement('input'); inp.type = 'text'; inp.value = caps[i].text;
  const ok = document.createElement('button'); ok.textContent = '확인';
  wrap.appendChild(inp); wrap.appendChild(ok);
  wrap.style.left = Math.max(0, Math.min(parseFloat(blkEl.style.left), tw() - 420)) + 'px';
  wrap.style.top = ((blkEl.parentElement || $('koTrack')).offsetTop + 2) + 'px';
  $('tracksScroll').appendChild(wrap); inp.focus(); inp.select();
  const commit = () => { pushUndo(); caps[i].text = inp.value.trim() || caps[i].text; markDirty(); closeEdit(); renderTracks(); updateOverlay(); syncSelPanel(); };
  ok.addEventListener('click', commit);
  inp.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') commit(); if (ev.key === 'Escape') closeEdit(); ev.stopPropagation(); });
}
function closeEdit() { const el = $('blkEdit'); if (el) el.remove(); }
function addCap() {
  pushUndo();
  const t = Math.max(C.start, Math.min(v.currentTime, C.end - 2));
  caps.push({ start: t, end: Math.min(t + 3, C.end), text: '새 자막' });
  caps.sort((a, b) => a.start - b.start);
  if (ens.length) { ens = []; setStatus('자막 추가 — 영어 트랙은 다시 번역해 주세요'); }
  selIdx = caps.findIndex((c) => c.start === t); selSet = new Set(); markDirty(); renderTracks(); syncSelPanel();
}
$('pjNew').addEventListener('click', addCap); $('pjDel').addEventListener('click', () => deleteSel(false));
$('hbAddCap').addEventListener('click', addCap);

// ─── 자막 복사/붙여넣기(Ctrl+C / Ctrl+V) ───
function copySel() {
  const idxs = (selSet.size ? [...selSet] : (selIdx >= 0 ? [selIdx] : [])).filter((i) => caps[i]).sort((a, b) => a - b);
  if (!idxs.length) return;
  const base = caps[idxs[0]].start;
  capClipboard = idxs.map((i) => ({ off: caps[i].start - base, dur: caps[i].end - caps[i].start, text: caps[i].text }));
  setStatus(idxs.length > 1 ? idxs.length + '개 자막 복사됨' : '자막 복사됨');
}
function rangesOverlap(a0, a1, b0, b1) { return a0 < b1 - 1e-6 && b0 < a1 - 1e-6; }
function pasteSel() {
  if (!capClipboard || !capClipboard.length) return;
  pushUndo();
  const t0 = v.currentTime;
  const added = capClipboard.map((c) => {
    let s = Math.max(0, Math.min(dur, t0 + c.off));
    let en = Math.max(s + 0.05, Math.min(dur, s + c.dur));
    return { start: s, end: en, text: c.text };
  });
  // 겹치는 자리에 붙여넣으면(예전엔 그냥 같은 줄에 쌓여 화면이 뒤섞였다) 캡컷/프리미어처럼
  // 자동으로 자막 트랙을 하나 더 만든다(사용자 요청 2026-09-08). 기존 트랙 중 하나라도
  // 완전히 안 겹치면 그 트랙을 쓰고, 전부 겹치면 새 트랙을 만든다.
  let track = 0;
  for (;; track++) {
    const onTrack = caps.filter((c) => (c.track || 0) === track);
    const conflict = added.some((a) => onTrack.some((c) => rangesOverlap(a.start, a.end, c.start, c.end)));
    if (!conflict) break;
  }
  const addedTrack = track >= nKoTracks;
  if (addedTrack) nKoTracks = track + 1;
  added.forEach((a) => { a.track = track; caps.push(a); });
  caps.sort((a, b) => a.start - b.start);
  let hadEn = false;
  if (ens.length) { hadEn = true; ens = []; }
  selSet = new Set(caps.map((c, i) => i).filter((i) => added.includes(caps[i])));
  selIdx = selSet.size ? Math.min(...selSet) : -1;
  markDirty(); renderTracks(); updateOverlay(); syncSelPanel();
  setStatus((added.length > 1 ? added.length + '개 자막 붙여넣기됨' : '자막 붙여넣기됨')
    + (hadEn ? ' — 영어 트랙은 다시 번역해 주세요' : '')
    + (addedTrack ? ' (겹쳐서 자막 트랙 C1+' + (track + 1) + ' 자동 추가)' : ''));
}

// ─── 효과 컨트롤 우측 키프레임 미니 타임라인 ───
function renderKf() {
  const r = $('kfRuler'), b = $('kfBody'); if (!C) return;
  const W = r.clientWidth; if (W <= 0) return;
  const a = C.start, z = C.end, span = z - a;
  r.innerHTML = ''; b.innerHTML = '';
  const minPx = 64; let step = 1;
  for (const cand of [1, 2, 5, 10, 15, 30, 60, 120, 300]) { step = cand; if (cand / span * W >= minPx) break; }
  for (let t = Math.ceil(a / step) * step; t <= z; t += step) { const el = document.createElement('div'); el.className = 'tk'; el.style.left = ((t - a) / span * W) + 'px'; el.textContent = tcShort(t); r.appendChild(el); }
  const has = selIdx >= 0 && caps[selIdx];
  const lbl = document.createElement('div'); lbl.className = 'lbl'; lbl.style.top = '38px';
  lbl.textContent = has ? '선택한 자막 구간' : '(자막을 선택하면 구간이 표시됩니다)'; b.appendChild(lbl);
  if (has) { const s = document.createElement('div'); s.className = 'span'; s.style.left = ((caps[selIdx].start - a) / span * W) + 'px'; s.style.width = Math.max(2, (caps[selIdx].end - caps[selIdx].start) / span * W) + 'px'; s.style.top = '58px'; b.appendChild(s); }
  const t = v.currentTime; if (t >= a && t <= z) { const p1 = document.createElement('div'); p1.className = 'ph'; p1.style.left = ((t - a) / span * W) + 'px'; r.appendChild(p1); const p2 = p1.cloneNode(); b.appendChild(p2); }
}

// ─── 프로젝트 패널 ───
let pjSel = null;
let pjSortMode = 'time';
function renderProject() {
  const list = $('pjList'); list.innerHTML = '';
  const q = ($('pjSearch').value || '').trim();
  const rows = [
    { ico: 'seq', nm: (C.title || '클립') , meta: [tc(C.start), tc(C.end - C.start)], act: () => seek(C.start) },
    { ico: 'vid', nm: 'source.mp4 (원본 영상)', meta: ['00:00:00.000', tc(dur)], act: () => { view = { a: 0, b: dur }; renderAll(); } },
    { ico: 'cc', nm: '자막(한글) · ' + caps.length + '개', meta: [caps.length ? tc(caps[0].start) : '-', ''], act: zoomFit },
  ];
  if (ens.length) rows.push({ ico: 'cc', nm: '자막(영어) · ' + ens.length + '개', meta: [caps.length ? tc(caps[0].start) : '-', ''], act: zoomFit });
  if (C && C.bgm) rows.push({ ico: 'cc', nm: '배경 음악 · ' + C.bgm.filename, meta: ['', ''], act: () => {} });
  if (pjSortMode === 'name') rows.sort((a, b) => a.nm.localeCompare(b.nm));
  const ICO = { seq: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="14"/><path d="M3 10h18M8 5v14"/></svg>',
    vid: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="6" width="13" height="12" rx="1"/><path d="M16 10l5-3v10l-5-3z"/></svg>',
    cc: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M9 12H6.5M17.5 12H15"/></svg>' };
  rows.forEach((r) => {
    if (q && !r.nm.includes(q)) return;
    const el = document.createElement('div'); el.className = 'pj-row' + (r.nm === pjSel ? ' sel' : '');
    el.innerHTML = '<span class="ico">' + ICO[r.ico] + '</span><span class="nm"></span><span class="meta"></span><span class="meta"></span>';
    el.querySelector('.nm').textContent = r.nm; const ms = el.querySelectorAll('.meta'); ms[0].textContent = r.meta[0]; ms[1].textContent = r.meta[1];
    el.addEventListener('click', () => { pjSel = r.nm; renderProject(); });
    el.addEventListener('dblclick', r.act);
    list.appendChild(el);
  });
}
$('pjSearch').addEventListener('input', renderProject);
$('pjSort').addEventListener('click', () => {
  pjSortMode = pjSortMode === 'time' ? 'name' : 'time';
  $('pjSort').classList.toggle('on', pjSortMode === 'name');
  renderProject();
});
$('pjZoom').addEventListener('input', () => {
  const t = parseFloat($('pjZoom').value) / 100;
  const minSpan = 1, maxSpan = Math.max(2, dur);
  const span = maxSpan * Math.pow(minSpan / maxSpan, t);
  const center = (view.a + view.b) / 2;
  let a = Math.max(0, center - span / 2);
  const b = Math.min(dur, a + span);
  a = Math.max(0, b - span);
  view = { a, b }; renderAll();
});
document.querySelectorAll('.fx-fold > .fh').forEach(h => h.addEventListener('click', () => h.parentElement.classList.toggle('closed')));
$('fxSearch').addEventListener('input', () => { const q = $('fxSearch').value.trim(); document.querySelectorAll('.fx-item').forEach(it => { if (it.id === 'fxLyrics' && C && C.clip_type !== 'praise') return; it.style.display = (!q || it.querySelector('.nm').textContent.includes(q)) ? '' : 'none'; }); });
document.querySelectorAll('.fx-item').forEach(it => {
  it.querySelector('.apply').addEventListener('click', (e) => { e.stopPropagation(); applyFx(it.dataset.fx); });
  it.addEventListener('dblclick', () => applyFx(it.dataset.fx));
});

// ─── AI 도구(효과 적용) ───
function collect() { return caps.map(c => ({ start: c.start, end: c.end, text: c.text })); }
async function aiCall(kind, url, body, apply) {
  const it = document.querySelector('.fx-item[data-fx="' + kind + '"]'); if (it && it.classList.contains('busy')) return;
  if (it) it.classList.add('busy'); setStatus('처리 중…');
  let r = null;
  try { r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); } catch (e) {}
  const j = r && r.ok ? await r.json().catch(() => null) : null;
  if (it) it.classList.remove('busy');
  if (!j || j.error || !j.lines) { alert('실패' + (j && j.error ? ': ' + j.error : '')); setStatus(''); return false; }
  pushUndo(); apply(j); markDirty(); renderTracks(); updateOverlay(); syncSelPanel(); renderProject();
  // AI 도구(번역/교정/싱크/가사)는 결과를 바로 저장한다 — "한 번 만들었으면 계속
  // 남아있어야지"(사용자 신고 2026-09-06): 저장은 수동(Ctrl+S)뿐이라, 만들자마자
  // 저장을 안 누르고 나가면(뒤로가기 등) 방금 만든 결과가 통째로 사라져 보였다.
  await save();
  return true;   // 호출부가 성공 여부로 후속 작업(싱크 → 영어 번역)을 이어갈 수 있게.
}
function applyFx(kind) {
  const base = '/video/' + VIDEO_ID + '/clip/' + IDX + '/';
  if (kind === 'sync') {
    // 싱크가 끝나면 영어 자막도 이어서 만든다(사용자 요청 2026-09-07). 번역 입력은 방금
    // 싱크된 줄(collect())이라 영어 트랙의 시간이 한국어와 정확히 같아진다.
    aiCall(kind, base + 'sync_captions', { captions: collect() }, (j) => { caps = j.lines.map(c => ({ start: c.start, end: c.end, text: c.text })); setStatus('싱크 맞춤: ' + (j.source || '')); })
      .then((ok) => {
        if (!ok) return;
        setStatus('싱크 완료 — 영어 자막 만드는 중…');
        return aiCall('translate', base + 'translate_captions', { captions: collect(), model: '' }, (j) => { ens = j.lines.map(c => ({ start: c.start, end: c.end, text: c.text })); setStatus('싱크 + 영어 ' + j.lines.length + '줄'); });
      });
  }
  else if (kind === 'correct') aiCall(kind, base + 'correct_captions', { captions: collect(), model: '' }, (j) => { j.lines.forEach((ln, i) => { if (caps[i]) caps[i].text = ln.text; }); setStatus('교정 ' + (j.changed || 0) + '줄'); });
  else if (kind === 'translate') aiCall(kind, base + 'translate_captions', { captions: collect(), model: '' }, (j) => { ens = j.lines.map(c => ({ start: c.start, end: c.end, text: c.text })); setStatus('영어 ' + j.lines.length + '줄'); });
  else if (kind === 'lyrics') {
    const t = prompt('곡 제목 또는 벅스 트랙 URL (music.bugs.co.kr에서 가사를 가져옵니다)', (C && C.title) || ''); if (t == null || !t.trim()) return;
    aiCall(kind, base + 'fetch_lyrics', { title: t.trim() }, (j) => { caps = j.lines.map(c => ({ start: c.start, end: c.end, text: c.text })); ens = []; setStatus('가사 ' + j.lines.length + '줄'); });
  }
}

// ─── 실행 취소 ───
// 실행 취소가 자막 시간·구간뿐 아니라 글꼴·정렬·색·박스·자유 텍스트까지 전부 되돌리게
// 한다(2026-09-08: "컨트롤Z 누르면 이전으로 돌아가게" — 스타일 바꾼 뒤 Ctrl+Z가 안 먹던
// 문제. 색·박스 컨트롤들이 markDirty만 부르고 pushUndo를 안 불러서 되돌릴 지점 자체가
// 없었다). 슬라이더성 값(크기·위치 등)은 입력칸에서 직접 읽어 스냅샷에 담는다.
function snapshot() {
  return {
    caps: caps.map(c => ({ ...c })), ens: ens.map(c => ({ ...c })), cs: C.start, ce: C.end,
    vsegs: vsegs.map(s => ({ ...s })), texts: texts.map(t => ({ ...t })), nTxTracks, nKoTracks,
    capFont, capAlign, capTextColor, capBox, capBoxColor, capBoxOpacity,
    capBold, capItalic, capUnderline, capOutlineOn, capOutlineColor, capGlow, capGlowColor,
    pSize: $('pSize').value, pX: $('pX').value, pY: $('pY').value, pSizeEn: $('pSizeEn').value,
    pSpacing: $('pSpacing').value, pBoxRadius: $('pBoxRadius').value,
    pBoxW: $('pBoxW').value, pBoxH: $('pBoxH').value, pBoxOX: $('pBoxOX').value, pBoxOY: $('pBoxOY').value,
    pTextOpacity: $('pTextOpacity').value, pOutlineWidth: $('pOutlineWidth').value,
  };
}
function pushUndo(prevTimes) {
  const s = snapshot();
  if (prevTimes) s.caps = s.caps.map((c, i) => ({ ...c, start: prevTimes[i].start, end: prevTimes[i].end }));
  undoStack.push(s); if (undoStack.length > 100) undoStack.shift(); redoStack = [];
}
function restore(s) {
  caps = s.caps.map(c => ({ ...c })); ens = s.ens.map(c => ({ ...c })); C.start = s.cs; C.end = s.ce;
  nKoTracks = s.nKoTracks || caps.reduce((m, c) => Math.max(m, (c.track || 0) + 1), 1);
  vsegs = (s.vsegs || [{ s: s.cs, e: s.ce }]).map(x => ({ ...x }));
  if (s.texts) { texts = s.texts.map(t => ({ ...t })); nTxTracks = s.nTxTracks || 0; }
  if (s.capFont !== undefined) {
    capFont = s.capFont; capAlign = s.capAlign; capTextColor = s.capTextColor;
    capBox = s.capBox; capBoxColor = s.capBoxColor; capBoxOpacity = s.capBoxOpacity;
    capBoxRadius = s.pBoxRadius; capBoxW = s.pBoxW; capBoxH = s.pBoxH; capBoxOX = s.pBoxOX; capBoxOY = s.pBoxOY;
    capBold = s.capBold; capItalic = s.capItalic; capUnderline = s.capUnderline;
    capOutlineOn = s.capOutlineOn; capOutlineColor = s.capOutlineColor;
    capGlow = s.capGlow; capGlowColor = s.capGlowColor;
    $('pFont').value = capFont;
    document.querySelectorAll('.align-btn').forEach((x) => x.classList.toggle('on', x.dataset.align === capAlign));
    $('pSize').value = s.pSize; $('pX').value = s.pX; $('pY').value = s.pY;
    $('pSizeEn').value = s.pSizeEn; $('pSpacing').value = s.pSpacing;
    $('pTextOpacity').value = s.pTextOpacity; $('pOutlineWidth').value = s.pOutlineWidth;
    syncCapStyleInputs();
  }
  selTx = -1; selVSeg = -1; selIdx = -1; selSet = new Set();
  markDirty(); renderAll(); updateOverlay(); syncSelPanel(); renderProject();
}
// 색 피커·드래그형 스테퍼처럼 한 번의 '조작'이 input 이벤트를 여러 번(계속) 쏘는 컨트롤은
// 매번 pushUndo하면 Ctrl+Z가 아주 조금씩만 되돌아간다. 조작이 시작될 때 한 번만 찍고,
// 500ms 동안 잠잠하면 다음 조작을 새 체크포인트로 잡는다.
let undoBurstTimer = null;
function pushUndoBurst() {
  if (!undoBurstTimer) pushUndo();
  clearTimeout(undoBurstTimer);
  undoBurstTimer = setTimeout(() => { undoBurstTimer = null; }, 500);
}
function undo() { if (!undoStack.length) return; redoStack.push(snapshot()); restore(undoStack.pop()); setStatus('실행 취소'); }
function redo() { if (!redoStack.length) return; undoStack.push(snapshot()); restore(redoStack.pop()); setStatus('다시 실행'); }

// ─── 저장/내보내기 ───
function markDirty() { dirty = true; $('projEdited').textContent = ' - 편집됨'; scheduleAutoSave(); }
function payload() {
  const p = {
    captions: collect(),
    caption_size: parseInt($('pSize').value, 10),
    caption_size_en: parseInt($('pSizeEn').value, 10) || 0,
    caption_offset_x: parseFloat($('pX').value),
    caption_offset_y: parseFloat($('pY').value),
    caption_font: capFont,
    caption_align: capAlign,
    caption_spacing: parseFloat($('pSpacing').value) || 0,
    caption_text_color: capTextColor,
    caption_box: capBox,
    caption_box_color: capBoxColor,
    caption_box_opacity: capBoxOpacity,
    caption_box_radius: parseFloat($('pBoxRadius').value) || 0,
    caption_box_width_pct: parseFloat($('pBoxW').value) || 0,
    caption_box_height_pct: parseFloat($('pBoxH').value) || 0,
    caption_box_offset_x: parseFloat($('pBoxOX').value) || 0,
    caption_box_offset_y: parseFloat($('pBoxOY').value) || 0,
    caption_bold: capBold, caption_italic: capItalic, caption_underline: capUnderline,
    caption_text_opacity: (parseFloat($('pTextOpacity').value) || 0) / 100,
    caption_outline_enabled: capOutlineOn, caption_outline_color: capOutlineColor,
    caption_outline_width: parseFloat($('pOutlineWidth').value) || 0,
    caption_glow: capGlow, caption_glow_color: capGlowColor,
  };
  if (ens.length === caps.length && ens.length) p.caption_overrides_en = ens.map((c, i) => ({ start: caps[i].start, end: caps[i].end, text: c.text }));
  p.free_texts = texts.map(t => ({ start: t.start, end: t.end, text: t.text, x: t.x, y: t.y, size: t.size, track: t.track }));
  // 영상을 실제로 잘랐으면(vsegsChanged) 구간·keep_ranges를 같이 보내고, 저장된 자막
  // 시간은 새 길이에 안 맞으므로 비워서 렌더가 다시 전사하게 한다(팝업의 분할과 동일한
  // 규칙 — 찬양은 재전사가 없어 예외로 자막을 지키고, keep_ranges만 반영한다).
  const vChanged = vsegsChanged();
  if (clipRangeDirty || vChanged) {
    p.clip_start = vsegs[0].s; p.clip_end = vsegs[vsegs.length - 1].e;
  }
  if (vChanged) {
    p.keep_ranges = vsegs.map((s) => [s.s, s.e]);
    if (C.clip_type !== 'praise') p.captions = [];
  }
  if (C.bgm) { p.bgm_volume = C.bgm.volume; p.bgm_muted = C.bgm.muted; p.bgm_offset = C.bgm.offset || 0; }
  return p;
}
async function save() {
  const vChanged = vsegsChanged();
  const r = await fetch('/video/' + VIDEO_ID + '/clip/' + IDX + '/position', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload()) });
  if (!r.ok) { setStatus('저장 실패'); return false; }
  dirty = false; clipRangeDirty = false; $('projEdited').textContent = '';
  if (vChanged) {
    // 영상 길이가 바뀌었으니(진짜 컷) 새 구간 기준으로 페이지를 다시 불러온다 — 자막은
    // 서버가 비웠고, 다음 로드에서 새 길이에 맞는 자막 초안이 자동으로 다시 뜬다.
    setStatus('영상이 잘렸어요 — 새 길이로 다시 불러오는 중…');
    location.reload();
    return true;
  }
  setStatus('저장됨 ✓'); return true;
}
async function render() {
  if (!(await save())) return;
  const r = await fetch('/video/' + VIDEO_ID + '/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ indices: [IDX], outro: true }) });
  setStatus(r.ok ? '내보내기 시작 — 후보 목록에서 진행률 확인' : '내보내기 요청 실패');
}
$('hExport').addEventListener('click', render); $('modeExport').addEventListener('click', render);
// 우측 상단 '저장하기'. 예전엔 저장 수단이 Ctrl+S(와 메뉴)뿐이라, 크기·가로·세로를 고쳐도
// 저장을 안 한 채 내보내면 그 값이 영상에 안 들어갔다(실신고 2026-09-07).
const saveBtnEl = $('hSave');
saveBtnEl.addEventListener('click', async () => {
  saveBtnEl.disabled = true; saveBtnEl.textContent = '저장 중…';
  const ok = await save();
  saveBtnEl.textContent = ok ? '저장됨 ✓' : '저장 실패';
  saveBtnEl.classList.toggle('saved', ok);
  saveBtnEl.disabled = false;
  setTimeout(() => { saveBtnEl.textContent = '저장하기'; saveBtnEl.classList.remove('saved'); refreshSaveBtn(); }, 2500);
});
// 저장 안 된 편집이 있으면 버튼을 노란색으로 — 자동 저장이 끝나면 다시 흰색이 된다.
function refreshSaveBtn() { saveBtnEl.classList.toggle('dirty', !!dirty); }
setInterval(refreshSaveBtn, 600);
$('modeImport').addEventListener('click', () => saveThenGo('/video/' + VIDEO_ID));
// 홈(후보 목록) 링크도 저장을 마친 뒤에 이동한다.
document.querySelector('.hdr .home').addEventListener('click', (e) => {
  e.preventDefault(); saveThenGo(e.currentTarget.getAttribute('href'));
});
$('hFull').addEventListener('click', () => { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen(); });
$('hWorkspace').addEventListener('click', resetWs);
function resetWs() { ws.style.removeProperty('--colL'); ws.style.removeProperty('--rowT'); fitVideo(); renderAll(); }
function setStatus(m) { $('status').textContent = m; setTimeout(() => { if ($('status').textContent === m) $('status').textContent = ''; }, 5000); }
// ─── 자동 저장 ───────────────────────────────────────────────────────────
// 사용자 요청(2026-09-07): "저장 버튼 따로 안 눌러도, 홈으로 가더라도 자동 저장되게."
// (1) 편집이 멈추면 1.2초 뒤 자동 저장 — 슬라이더를 끌 때마다 저장하지 않도록 디바운스.
// (2) 홈/가져오기로 나갈 땐 저장이 끝난 뒤에 이동 — 이동이 저장을 앞질러 값이 날아가지 않게.
// (3) 탭을 닫거나 뒤로가기처럼 기다릴 수 없는 경우엔 sendBeacon으로 마지막 상태를 밀어넣는다.
function scheduleAutoSave() {
  if (autoSaveTimer) clearTimeout(autoSaveTimer);
  autoSaveTimer = setTimeout(async () => {
    autoSaveTimer = null;
    if (!dirty || saving) return;
    saving = true;
    try { await save(); } finally { saving = false; refreshSaveBtn(); }
  }, 1200);
}
async function saveThenGo(href) {
  if (dirty && !saving) { saving = true; try { await save(); } catch (e) {} saving = false; }
  location.href = href;
}
function beaconSave() {
  if (!dirty) return;
  try {
    navigator.sendBeacon('/video/' + VIDEO_ID + '/clip/' + IDX + '/position',
      new Blob([JSON.stringify(payload())], { type: 'application/json' }));
    dirty = false;
  } catch (e) {}
}
window.addEventListener('pagehide', beaconSave);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') beaconSave(); });
document.addEventListener('keydown', (e) => { if (e.key === 'F11') { e.preventDefault(); $('hFull').click(); } });

// ─── 배경 음악(A2) — 미리듣기는 별도 <audio>로 근사 동기화, 정밀 반복/페이드는 렌더가 처리 ───
function renderBgm() {
  const has = !!C.bgm;
  $('bgmMute').style.display = has ? '' : 'none';
  $('bgmRemove').style.display = has ? '' : 'none';
  $('bgmVol').style.display = has ? '' : 'none';
  $('bgmClip').style.display = has ? '' : 'none';
  $('bgmName').textContent = has ? ('A2 · ' + C.bgm.filename) : 'A2 배경 음악(없음)';
  const a = $('bgmAudio');
  if (has) {
    $('bgmClipCn').textContent = 'A2 · ' + C.bgm.filename;
    $('bgmMute').classList.toggle('on', !!C.bgm.muted);
    $('bgmVol').value = Math.round((C.bgm.volume != null ? C.bgm.volume : 0.25) * 100);
    const src = '/media/' + VIDEO_ID + '/bgm/' + encodeURIComponent(C.bgm.filename);
    if (!a.src || !a.src.endsWith(src)) a.src = src;
    a.volume = C.bgm.volume != null ? C.bgm.volume : 0.25;
    a.muted = !!C.bgm.muted;
  } else {
    a.pause(); a.removeAttribute('src'); a.load();
  }
  renderProject();
}
$('bgmAdd').addEventListener('click', () => $('bgmFile').click());
$('bgmFile').addEventListener('change', async () => {
  const f = $('bgmFile').files[0]; $('bgmFile').value = '';
  if (!f) return;
  setStatus('배경 음악 업로드 중…');
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/video/' + VIDEO_ID + '/clip/' + IDX + '/bgm', { method: 'POST', body: fd });
  if (r.ok) { const j = await r.json(); C.bgm = j.bgm; renderBgm(); setStatus('배경 음악 추가됨 — 내보내기(렌더) 결과물에 반영됩니다'); }
  else setStatus('배경 음악 업로드 실패');
});
$('bgmRemove').addEventListener('click', async () => {
  if (!C.bgm) return;
  await fetch('/video/' + VIDEO_ID + '/clip/' + IDX + '/bgm', { method: 'DELETE' });
  C.bgm = null; renderBgm(); setStatus('배경 음악 제거됨');
});
$('bgmMute').addEventListener('click', () => { if (!C.bgm) return; C.bgm.muted = !C.bgm.muted; renderBgm(); markDirty(); });
$('bgmVol').addEventListener('input', () => {
  if (!C.bgm) return;
  C.bgm.volume = parseInt($('bgmVol').value, 10) / 100; $('bgmAudio').volume = C.bgm.volume; markDirty();
});
v.addEventListener('play', () => {
  const a = $('bgmAudio'); if (!C.bgm || !a.src) return;
  try { if (a.duration) a.currentTime = Math.max(0, ((C.bgm.offset || 0) + v.currentTime - C.start) % a.duration); } catch (e) { /* 메타데이터 전이면 무시 */ }
  a.play().catch(() => {});
});
v.addEventListener('pause', () => $('bgmAudio').pause());
v.addEventListener('seeked', () => {
  const a = $('bgmAudio'); if (!C.bgm || !a.duration) return;
  a.currentTime = Math.max(0, ((C.bgm.offset || 0) + v.currentTime - C.start) % a.duration);
});
// BGM 막대를 좌우로 끌면 '이 클립에서 재생되는 시간대'가 아니라 '배경음악 파일 안에서
// 어디부터 틀지'(offset)가 바뀐다 — 막대 자체는 항상 클립 구간(C.start~C.end)에 고정.
// 음원이 무한 반복이라 오프셋에 상한이 없고, 드래그 중엔 미리듣기 오디오도 같이 옮겨 준다.
(function () {
  const bc = $('bgmClip');
  bc.addEventListener('pointerdown', (e) => {
    if (!C.bgm || tool === 'hand') return;
    e.preventDefault();
    const x0 = e.clientX, off0 = C.bgm.offset || 0;
    const move = (ev) => {
      const dt = (ev.clientX - x0) / tw() * (view.b - view.a);
      C.bgm.offset = Math.max(0, off0 + dt);
      markDirty();
      const a = $('bgmAudio');
      if (a.duration) { try { a.currentTime = (C.bgm.offset + v.currentTime - C.start) % a.duration; } catch (err) {} }
      setStatus('배경 음악 시작 지점 = ' + tc(C.bgm.offset));
    };
    const up = () => document.removeEventListener('pointermove', move);
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up, { once: true });
  });
})();

// ─── 소스 모니터(원본 영상, 탭을 열 때 로드) ───
let sv = null;
function openSource() {
  if (sv) return;
  const st = $('srcStage'); st.innerHTML = '';
  sv = document.createElement('video'); sv.src = '/media/' + VIDEO_ID + '/source.mp4'; sv.preload = 'metadata'; sv.playsInline = true; st.appendChild(sv);
  sv.addEventListener('loadedmetadata', () => { $('srcDur').textContent = tc(sv.duration); sv.currentTime = C ? C.start : 0; });
  sv.addEventListener('timeupdate', () => { $('srcTc').textContent = tc(sv.currentTime); });
  sv.addEventListener('click', () => sv.paused ? sv.play() : sv.pause());
}
$('srcPlay').addEventListener('click', () => { if (sv) sv.paused ? sv.play() : sv.pause(); });
$('srcBack').addEventListener('click', () => { if (sv) sv.currentTime -= 1 / FPS; });
$('srcFwd').addEventListener('click', () => { if (sv) sv.currentTime += 1 / FPS; });

// ─── 메뉴 막대 ───
const menubar = $('menubar');
let menuOpen = false;
menubar.querySelectorAll('.mi').forEach(mi => {
  mi.addEventListener('click', (e) => {
    if (e.target.closest('.menu')) return;
    const was = mi.classList.contains('open');
    menubar.querySelectorAll('.mi.open').forEach(x => x.classList.remove('open'));
    if (!was) mi.classList.add('open'); menuOpen = !was;
  });
  mi.addEventListener('pointerenter', () => { if (menuOpen) { menubar.querySelectorAll('.mi.open').forEach(x => x.classList.remove('open')); mi.classList.add('open'); } });
});
document.addEventListener('pointerdown', (e) => { if (!e.target.closest('.menubar')) { menubar.querySelectorAll('.mi.open').forEach(x => x.classList.remove('open')); menuOpen = false; } });
menubar.querySelectorAll('.menu .it').forEach(it => it.addEventListener('click', () => {
  menubar.querySelectorAll('.mi.open').forEach(x => x.classList.remove('open')); menuOpen = false;
  menuAct(it.dataset.act);
}));
function menuAct(a) {
  const A = {
    save, render, back: () => location.href = '/video/' + VIDEO_ID,
    undo, redo, 'delete': () => deleteSel(false), 'ripple-delete': () => deleteSel(true),
    copy: copySel, paste: pasteSel,
    'select-all': () => { selSet = new Set(caps.map((_, i) => i)); selIdx = caps.length ? 0 : -1; renderTracks(); syncSelPanel(); },
    deselect: () => { selSet = new Set(); selIdx = -1; selTx = -1; renderTracks(); syncSelPanel(); updateOverlay(); },
    'edit-text': () => { const b = $('koTrack').querySelector('.blk.sel'); if (b && selIdx >= 0) openEdit(selIdx, b); },
    split: () => { const i = selIdx >= 0 ? selIdx : activeCapIdx(v.currentTime); if (i >= 0) splitAtTime(i, v.currentTime); },
    add: addCap, 'mark-in': () => markIn(), 'mark-out': () => markOut(),
    'ctx-mark-in': () => markIn(ctxTime), 'ctx-mark-out': () => markOut(ctxTime),
    'tx-edit-text': () => focusTxOverlay(selTx),
    'tx-split': () => { if (selTx >= 0) splitTxAt(selTx, v.currentTime); },
    'v-split': () => splitVSegAt(ctxTime),
    'v-delete': () => { if (selVSeg >= 0) deleteVSeg(selVSeg); },
    'zoom-in': () => zoomBy(0.75, (view.a + view.b) / 2), 'zoom-out': () => zoomBy(1.34, (view.a + view.b) / 2), 'zoom-fit': zoomFit,
    snap: () => setSnap(!snapOn), marker: addMarker, 'marker-clear': () => { markers = []; renderRuler(); },
    'fx-correct': () => applyFx('correct'), 'fx-translate': () => applyFx('translate'), 'fx-sync': () => applyFx('sync'),
    safe: () => setSafe(!$('pSafe').checked), 'en-vis': () => setEnVisible(!enVisible), fit: () => { fitMode = 'fit'; $('ddFit').value = 'fit'; fitVideo(); },
    'ws-reset': resetWs, fullscreen: () => $('hFull').click(), shortcuts: () => $('kbModal').classList.add('on'),
  };
  if (A[a]) A[a]();
}

// ─── 타임라인 우클릭 컨텍스트 메뉴(프리미어처럼 클립/빈 영역에서 다르게 뜬다) ───
function ctxItem(label, act, opts) {
  opts = opts || {};
  const cls = ['it']; if (opts.dis) cls.push('dis'); if (opts.chk) cls.push('chk');
  return '<div class="' + cls.join(' ') + '" data-act="' + act + '">' + label + (opts.k ? '<span class="k">' + opts.k + '</span>' : '') + '</div>';
}
function ctxSep() { return '<div class="sep"></div>'; }
function showCtxMenu(x, y, html) {
  const m = $('ctxMenu');
  m.innerHTML = html; m.classList.add('on'); m.style.left = '0px'; m.style.top = '0px';
  const r = m.getBoundingClientRect();
  m.style.left = Math.max(2, Math.min(x, window.innerWidth - r.width - 4)) + 'px';
  m.style.top = Math.max(2, Math.min(y, window.innerHeight - r.height - 4)) + 'px';
}
function hideCtxMenu() { $('ctxMenu').classList.remove('on'); }
$('ctxMenu').addEventListener('click', (e) => {
  const it = e.target.closest('.it'); hideCtxMenu();
  if (it && !it.classList.contains('dis')) menuAct(it.dataset.act);
});
document.addEventListener('pointerdown', (e) => { if (!e.target.closest('#ctxMenu')) hideCtxMenu(); });
window.addEventListener('blur', hideCtxMenu);
window.addEventListener('resize', hideCtxMenu);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideCtxMenu(); }, true);

function timelineCtx(e) {
  e.preventDefault();
  const vclip = e.target.closest('.vclip');
  if (vclip && vclip.dataset.vseg !== undefined) {
    const i = parseInt(vclip.dataset.vseg, 10);
    selVSeg = i; renderTracks();
    const rect = tlTracks.getBoundingClientRect();
    const t = x2t(e.clientX - rect.left);
    const canSplit = t > vsegs[i].s + 0.3 && t < vsegs[i].e - 0.3;
    showCtxMenu(e.clientX, e.clientY,
      ctxItem('여기서 영상 분할', 'v-split', { k: 'Ctrl+K', dis: !canSplit }) +
      ctxSep() +
      ctxItem('이 조각 삭제(잘라내기)', 'v-delete', { dis: vsegs.length <= 1 }) +
      ctxSep() +
      ctxItem('모두 선택 해제', 'deselect', { k: 'Ctrl+Shift+A' })
    );
    ctxTime = t;
    return;
  }
  const txblk = e.target.closest('.txblk');
  if (txblk && txblk.dataset.tx !== undefined) {
    const i = parseInt(txblk.dataset.tx, 10);
    selectTx(i);
    showCtxMenu(e.clientX, e.clientY,
      ctxItem('텍스트 수정…', 'tx-edit-text', { k: '더블클릭' }) +
      ctxItem('재생 헤드에서 분할', 'tx-split', { k: 'Ctrl+K' }) +
      ctxSep() +
      ctxItem('지우기', 'delete', { k: 'Delete' }) +
      ctxSep() +
      ctxItem('모두 선택 해제', 'deselect', { k: 'Ctrl+Shift+A' })
    );
    return;
  }
  const blk = e.target.closest('.blk');
  if (blk && blk.dataset.idx !== undefined) {
    const i = parseInt(blk.dataset.idx, 10);
    if (!selSet.has(i)) select(i);
    showCtxMenu(e.clientX, e.clientY,
      ctxItem('텍스트 수정…', 'edit-text', { k: 'Enter' }) +
      ctxItem('재생 헤드에서 분할', 'split', { k: 'Ctrl+K' }) +
      ctxSep() +
      ctxItem('복사', 'copy', { k: 'Ctrl+C' }) +
      ctxItem('붙여넣기', 'paste', { k: 'Ctrl+V', dis: !capClipboard || !capClipboard.length }) +
      ctxSep() +
      ctxItem('지우기', 'delete', { k: 'Delete' }) +
      ctxItem('잔물결 삭제', 'ripple-delete', { k: 'Shift+Delete' }) +
      ctxSep() +
      ctxItem('마커 추가', 'marker', { k: 'M' }) +
      ctxSep() +
      ctxItem('모두 선택 해제', 'deselect', { k: 'Ctrl+Shift+A' })
    );
  } else {
    // 클립이 아닌 곳(영상/오디오 트랙·빈 자막 구간·눈금자)에서도 '분할'이 보여야 한다.
    // 예전엔 자막 막대 위에서만 떠서 "우클릭해도 분할이 없다"는 말이 나왔다(2026-09-08).
    // 재생 헤드 아래에 자막이 있으면 그걸 자르고, 없으면 비활성으로 보여만 준다.
    const rect = tlTracks.getBoundingClientRect();
    const t = x2t(e.clientX - rect.left);
    const overV = !!e.target.closest('.trk.v');
    const overA = !!e.target.closest('.trk.a');
    const canSplit = (selIdx >= 0 && caps[selIdx] && v.currentTime > caps[selIdx].start + 0.05 && v.currentTime < caps[selIdx].end - 0.05)
      || activeCapIdx(v.currentTime) >= 0;
    showCtxMenu(e.clientX, e.clientY,
      ctxItem('재생 헤드에서 분할', 'split', { k: 'Ctrl+K', dis: !canSplit }) +
      ctxItem('재생 헤드에 자막 추가', 'add', { k: 'Ctrl+Shift+N' }) +
      ctxSep() +
      ((overV || overA) ? (
        ctxItem('여기를 클립 시작(In)으로', 'ctx-mark-in', { k: 'I' }) +
        ctxItem('여기를 클립 끝(Out)으로', 'ctx-mark-out', { k: 'O' }) +
        ctxSep()) : '') +
      ctxItem('마커 추가', 'marker', { k: 'M' }) +
      ctxSep() +
      ctxItem('타임라인에서 스냅', 'snap', { k: 'S', chk: snapOn }) +
      ctxItem('시퀀스에 맞게 확대/축소', 'zoom-fit', { k: '\\' }) +
      ctxSep() +
      ctxItem('모두 선택 해제', 'deselect', { k: 'Ctrl+Shift+A', dis: selIdx < 0 && selSet.size === 0 })
    );
    ctxTime = t;
  }
}
let ctxTime = 0;   // 마지막 우클릭 지점의 시각(트랙 메뉴의 In/Out이 쓴다)
$('tracksScroll').addEventListener('contextmenu', timelineCtx);
$('ruler').addEventListener('contextmenu', timelineCtx);

$('kbClose').addEventListener('click', () => $('kbModal').classList.remove('on'));
$('kbModal').addEventListener('click', (e) => { if (e.target === $('kbModal')) $('kbModal').classList.remove('on'); });

// ─── 키보드 ───
document.addEventListener('keydown', (e) => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) { if (e.key === 'Escape') e.target.blur(); return; }
  const k = e.key.toLowerCase();
  if (e.ctrlKey || e.metaKey) {
    if (k === 's') { e.preventDefault(); save(); }
    else if (k === 'm') { e.preventDefault(); render(); }
    else if (k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
    else if (k === 'z' && e.shiftKey) { e.preventDefault(); redo(); }
    else if (k === 'k') { e.preventDefault(); menuAct('split'); }
    else if (k === 'a' && !e.shiftKey) { e.preventDefault(); menuAct('select-all'); }
    else if (k === 'a' && e.shiftKey) { e.preventDefault(); menuAct('deselect'); }
    else if (k === 'n' && e.shiftKey) { e.preventDefault(); addCap(); }
    else if (k === 'c') { e.preventDefault(); copySel(); }
    else if (k === 'v') { e.preventDefault(); pasteSel(); }
    else if (e.altKey && k === 'k') { e.preventDefault(); menuAct('shortcuts'); }
    return;
  }
  if (e.code === 'Space' || k === 'k' || k === 'l') { e.preventDefault(); if (k === 'k') v.pause(); else if (k === 'l') { if (v.paused) playPause(); } else playPause(); return; }
  if (k === 'j') { seek(v.currentTime - 1); return; }
  if (TOOL_KEYS[k]) { setTool(TOOL_KEYS[k]); return; }
  if (k === 'i') markIn(); else if (k === 'o') markOut(); else if (k === 'm') addMarker(); else if (k === 's') setSnap(!snapOn);
  else if (e.key === 'ArrowLeft') seek(v.currentTime - (e.shiftKey ? 5 : 1) / FPS);
  else if (e.key === 'ArrowRight') seek(v.currentTime + (e.shiftKey ? 5 : 1) / FPS);
  else if (e.key === 'Home') seek(C.start); else if (e.key === 'End') seek(C.end - 0.001);
  else if (e.key === '=' || e.key === '+') zoomBy(0.75, v.currentTime); else if (e.key === '-') zoomBy(1.34, v.currentTime); else if (e.key === '\\') zoomFit();
  else if (e.key === 'Enter') menuAct('edit-text');
  else if (e.key === 'Delete' || e.key === 'Backspace') deleteSel(e.shiftKey);
  else if (e.key === 'Escape') { closeEdit(); menuAct('deselect'); }
  else return;
  e.preventDefault();
});
</script>
</body>
</html>
"""


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
PREVIEW_MODAL_JS = r"""
(function () {
  'use strict';
  const VIDEO_ID = location.pathname.split('/')[2];

  // 만들기 흐름: 선택한 클립들을 순서대로 팝업 확인 → 모두 확인되면 onAllConfirmed() 실행.
  window.__previewFlow = function (indices, onAllConfirmed) {
    window.__pvHorizontal = new Set();  // '가로 원본'으로 만들 클립 인덱스(팝업에서 선택)
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
  .pv-backdrop { position: fixed; inset: 0; background: rgba(15,23,42,.38);
    -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px); z-index: 1000;
    display: flex; align-items: center; justify-content: center; padding: 16px;
    animation: pvFade .18s ease; }
  @keyframes pvFade { from { opacity: 0; } to { opacity: 1; } }
  .pv-card { background: #fff; border: 1px solid var(--glass-border, rgba(255,255,255,.55));
    border-radius: 26px; box-shadow: 0 24px 80px rgba(15,23,42,.3);
    width: min(600px, 94vw); max-height: 94vh; overflow-y: auto; padding: 18px 18px 16px;
    animation: pvUp .22s cubic-bezier(.22,.61,.36,1); }
  @keyframes pvUp { from { opacity: 0; transform: translateY(14px) scale(.98); } to { opacity: 1; transform: none; } }
  .pv-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
  .pv-head b { font-size: 16px; letter-spacing: -0.01em; }
  .pv-x { cursor: pointer; border: none; background: none; font-size: 22px; color: #8b95a1; line-height: 1; padding: 2px 6px; }
  .pv-head-r { display: flex; align-items: center; gap: 6px; }
  /* 스튜디오로 가는 버튼. 예전엔 팝업 맨 아래 작은 링크라 잘 안 보였다(사용자 요청
     2026-09-07: "구간 재분석 우측으로 올려줘"). 우측 상단 액션 줄에 pill로 둔다. */
  .pv-studio-btn { display: inline-flex; align-items: center; text-decoration: none; white-space: nowrap;
    padding: 6px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 700; letter-spacing: -0.01em;
    background: #fff; color: #1d1d1f; box-shadow: 0 1px 2px rgba(0,0,0,.10), 0 4px 12px rgba(0,0,0,.08);
    transition: transform .12s ease, box-shadow .12s ease; }
  .pv-studio-btn:hover { transform: translateY(-1px); box-shadow: 0 2px 5px rgba(0,0,0,.12), 0 8px 20px rgba(0,0,0,.12); }
  .pv-studio-btn:active { transform: translateY(0); }
  .pv-fullbtn { cursor: pointer; border: none; white-space: nowrap;
    padding: 6px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 700; letter-spacing: -0.01em;
    background: #fff; color: #1d1d1f; box-shadow: 0 1px 2px rgba(0,0,0,.10), 0 4px 12px rgba(0,0,0,.08);
    transition: transform .12s ease, box-shadow .12s ease; }
  .pv-fullbtn:hover { transform: translateY(-1px); box-shadow: 0 2px 5px rgba(0,0,0,.12), 0 8px 20px rgba(0,0,0,.12); }
  .pv-fullbtn:active { transform: translateY(0); }
  /* '전체화면'은 실제 OS/브라우저 풀스크린(F11류)이 아니라 팝업 카드 자체를 화면 가득 키우는
     CSS 확대다(사용자 요청: 진짜 풀스크린 API는 쓰지 말 것). 미리보기 확대는 JS가 캔버스에 scale. */
  .pv-backdrop.pv-maxed { padding: 0; background: rgba(15,23,42,.92); }
  .pv-backdrop.pv-maxed .pv-card { width: min(1280px, 96vw); max-height: 100vh; border-radius: 0; }
  .pv-reanalyze { cursor: pointer; border: none; background: rgba(120,120,128,.12); color: #0a84ff;
    font-size: 12.5px; font-weight: 700; font-family: inherit; border-radius: 9px; padding: 6px 10px; white-space: nowrap; }
  .pv-reanalyze:hover { background: rgba(10,132,255,.14); }
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
  /* 가로 정중앙 스냅 가이드선(파워포인트/피그마 스타일) — 드래그로 중앙에 가까워지면 표시. */
  .pv-guide-v { position: absolute; top: 0; bottom: 0; left: 50%; width: 0; border-left: 1.5px dashed #ff3b8d;
    transform: translateX(-50%); pointer-events: none; z-index: 5; display: none; }
  .pv-guide-v.show { display: block; }
  .pv-resize { position: absolute; right: -8px; bottom: -8px; width: 15px; height: 15px;
    border-radius: 4px; background: #3182f6; border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,.35);
    cursor: nwse-resize; display: none; touch-action: none; }
  .pv-title:hover .pv-resize, .pv-title.dragging .pv-resize { display: block; }
  /* 실제 렌더는 제목을 항상 한 줄로 맞춘다(_fit_title_font_size). 팝업이 normal로 줄바꿈하면
     (1) 2줄로 보여 실제와 배열이 다르고 (2) 줄바꿈 때문에 scrollWidth가 한계를 안 넘어
     fitToWidth 축소가 아예 작동하지 않아 크기까지 다르게 보였다 — 반드시 nowrap. */
  .pv-title { white-space: nowrap; }
  /* 제목 텍스트: 여러 줄(사용자 줄바꿈)에서도 폭 측정(scrollWidth)이 정확하려면 inline-block. */
  .pv-txt { display: inline-block; text-align: center; outline: none; }
  .pv-title.editing { border-color: #3182f6; background: rgba(49,130,246,.10); cursor: text; }
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
  /* 진행바 우클릭 메뉴(분할) — 별도 '분할' 버튼 대신 진행바에서 바로 우클릭해 분할 */
  .pv-ctxmenu { display: none; position: fixed; min-width: 120px; background: #fff; border: 1px solid #e5e7eb;
    border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,.18); padding: 4px; z-index: 500; }
  .pv-ctxmenu.on { display: block; }
  .pv-ctxmenu .it { padding: 8px 12px; font-size: 13px; font-weight: 600; color: #191f28; border-radius: 6px;
    cursor: default; white-space: nowrap; }
  .pv-ctxmenu .it:hover { background: #eef4ff; color: #3182f6; }
  .pv-ctxmenu .it.dis { color: #b0b4ba; pointer-events: none; }
  .pv-times { display: flex; justify-content: space-between; font-size: 12px; color: #6b7684;
    font-variant-numeric: tabular-nums; margin-top: 8px; }
  .pv-times b { color: #191f28; }
  .pv-tools { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
  .pv-speedlbl { margin-left: auto; font-size: 12.5px; font-weight: 700; color: #6b7684; }
  .pv-speed { font-size: 12.5px; font-family: inherit; border: 1.5px solid #f0f1f3; border-radius: 8px;
    padding: 5px 7px; background: #fafbfc; color: #191f28; cursor: pointer; }
  .pv-speed:hover { border-color: #3182f6; }
  .pv-tool { padding: 7px 11px; font-size: 12.5px; font-weight: 700; font-family: inherit;
    border: 1.5px solid #f0f1f3; background: #fafbfc; color: #191f28; border-radius: 9px; cursor: pointer; }
  .pv-tool:hover { border-color: #3182f6; color: #3182f6; }
  .pv-capedit-btn { cursor: pointer; border: 1.5px solid #f0f1f3; background: #fafbfc; color: #191f28;
    font-size: 12.5px; font-weight: 700; font-family: inherit; border-radius: 9px; padding: 6px 10px; white-space: nowrap; }
  .pv-capedit-btn:hover { border-color: #3182f6; color: #3182f6; }
  .pv-capsec { margin-top: 12px; }
  .pv-capsec.hidden { display: none; }
  .pv-capfont-row { display: flex; align-items: center; gap: 8px; font-size: 12.5px;
    color: #6b7684; font-weight: 600; margin-bottom: 8px; }
  .pv-capfont { flex: 1; min-width: 0; padding: 7px 9px; font-size: 13px; font-family: inherit;
    border: 1.5px solid #f0f1f3; border-radius: 8px; background: #fafbfc; color: #191f28; }
  .pv-capcopyall { flex: 0 0 auto; margin-left: auto; padding: 6px 9px; font-size: 13px;
    border: 1.5px solid #f0f1f3; background: #fafbfc; border-radius: 8px; cursor: pointer; }
  .pv-capcopyall:hover { border-color: #3182f6; background: #f0f6ff; }
  .pv-retrans, .pv-syncbtn, .pv-shiftm, .pv-shiftp,
  .pv-correctbtn, .pv-translatebtn, .pv-karaoke-toggle, .pv-lyricsbtn { flex: 0 0 auto; padding: 6px 10px;
    font-size: 12px; font-weight: 700; font-family: inherit; border: 1.5px solid #f0f1f3;
    background: #fafbfc; color: #3182f6; border-radius: 8px; cursor: pointer; white-space: nowrap; }
  .pv-retrans:hover, .pv-syncbtn:hover, .pv-shiftm:hover, .pv-shiftp:hover,
  .pv-correctbtn:hover, .pv-translatebtn:hover, .pv-karaoke-toggle:hover, .pv-lyricsbtn:hover {
    border-color: #3182f6; background: #f0f6ff; }
  .pv-retrans:disabled, .pv-syncbtn:disabled,
  .pv-correctbtn:disabled, .pv-translatebtn:disabled, .pv-lyricsbtn:disabled { opacity: .6; cursor: default; }
  .pv-shiftm, .pv-shiftp { color: #191f28; }
  .pv-karaoke-toggle.on { background: #3182f6; color: #fff; border-color: #3182f6; }
  .pv-karaoke-toggle.on:hover { background: #2b74d9; }
  .pv-capfont-row { flex-wrap: wrap; }
  .pv-savebtn { flex: 0 0 auto; padding: 13px 18px; border: 1.5px solid #3182f6; border-radius: 12px;
    background: #fff; color: #3182f6; font-weight: 700; font-size: 14px; font-family: inherit; cursor: pointer; }
  .pv-savebtn:hover { background: #f0f6ff; }
  .pv-capsel-toggle, .pv-capsel-merge, .pv-capsel-del { flex: 0 0 auto; padding: 6px 10px;
    font-size: 12px; font-weight: 700; font-family: inherit; border: 1.5px solid #f0f1f3;
    background: #fafbfc; color: #191f28; border-radius: 8px; cursor: pointer; white-space: nowrap; }
  .pv-capsel-toggle:hover { border-color: #3182f6; color: #3182f6; }
  .pv-capsel-toggle.on { border-color: #3182f6; background: #eef4ff; color: #1b64da; }
  .pv-capsel-merge { color: #3182f6; }
  .pv-capsel-merge:hover { border-color: #3182f6; background: #f0f6ff; }
  .pv-capsel-del { color: #e02424; }
  .pv-capsel-del:hover { border-color: #e02424; background: #ffe2e2; }
  .pv-cap-sel { display: none; flex: 0 0 auto; width: 16px; height: 16px; accent-color: #3182f6; }
  .pv-caprows.selmode .pv-cap-sel { display: block; }
  .pv-caprows { display: flex; flex-direction: column; gap: 6px; max-height: 220px; overflow-y: auto; padding: 2px; }
  .pv-caprow { display: flex; gap: 6px; align-items: center; }
  /* input[type=number]/[type=text]의 전역 width:100%(BASE_STYLE)보다 상위 명시도가
     필요해 .pv-caprow input.pv-cap-*로 부모+태그+클래스 조합을 쓴다(단일 클래스는 짐). */
  .pv-caprow input.pv-cap-start, .pv-caprow input.pv-cap-end {
    width: 50px; flex: 0 0 auto; padding: 6px 5px; font-size: 12px;
    border: 1.5px solid #f0f1f3; border-radius: 7px; font-family: inherit; margin-top: 0; }
  .pv-caprow input.pv-cap-text { flex: 1; min-width: 0; padding: 6px 8px; font-size: 13px;
    border: 1.5px solid #f0f1f3; border-radius: 7px; font-family: inherit; margin-top: 0; }
  .pv-cap-del { flex: 0 0 auto; width: 24px; height: 24px; border-radius: 50%; border: none;
    background: #f0f1f3; color: #6b7684; font-size: 13px; cursor: pointer; }
  .pv-cap-del:hover { background: #ffe2e2; color: #e02424; }
  .pv-capadd { margin-top: 8px; padding: 7px 11px; font-size: 12.5px; font-weight: 700; font-family: inherit;
    border: 1.5px dashed #d3d8de; background: none; color: #6b7684; border-radius: 9px; cursor: pointer; width: 100%; }
  .pv-capadd:hover { border-color: #3182f6; color: #3182f6; }
  .pv-cands { display: flex; flex-direction: column; gap: 7px; margin-top: 12px; }
  .pv-cand { text-align: left; padding: 10px 12px; font-size: 13.5px; font-weight: 600; font-family: inherit;
    color: #191f28; background: #fafbfc; border: 1.5px solid #f0f1f3; border-radius: 11px;
    cursor: pointer; line-height: 1.35; }
  .pv-cand:hover { border-color: #3182f6; background: #f0f6ff; }
  .pv-cand.sel { border-color: #3182f6; background: #eef4ff; color: #1b64da; }
  .pv-foot { display: flex; gap: 9px; margin-top: 14px; align-items: center; }
  .pv-cancel { flex: 0 0 auto; padding: 13px 16px; border: none; border-radius: 12px; background: #f0f1f3;
    color: #191f28; font-weight: 600; font-size: 14px; font-family: inherit; cursor: pointer; }
  .pv-wide { flex: 0 0 auto; padding: 13px 18px; border: 1.5px solid #d1d6db; border-radius: 12px;
    background: #fff; color: #191f28; font-size: 15px; font-weight: 700; font-family: inherit; cursor: pointer; }
  .pv-wide:hover { border-color: #3182f6; background: #f0f6ff; }
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
    // 실제 렌더 글꼴을 팝업에서도 그대로 보여준다. 자막 글꼴 드롭다운에서 바로 미리보기가
    // 되도록 현재 선택된 폰트뿐 아니라 등록된 폰트 전체의 @font-face를 넣는다.
    let faceCss = '';
    for (const f of (info.fonts || [])) {
      faceCss += "@font-face{font-family:'" + f.family + "';src:url('/font/" + f.file + "');font-display:swap}\n";
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
      '      <button type="button" class="pv-capedit-btn">자막 수정</button>' +
      '      <button class="pv-reanalyze" title="주제는 그대로 두고 이 장면의 시작·끝만 다시 잡아 새 후보로 추가합니다(원본 유지)">구간 재분석</button>' +
      '      <a class="pv-studio-btn" href="/video/' + VIDEO_ID + '/clip/' + idx + '/studio" title="타임라인·트랙이 있는 프리미어식 편집 화면으로 이동합니다">스튜디오</a>' +
      '      <button type="button" class="pv-fullbtn" title="팝업을 화면 가득 띄우고 미리보기를 크게 봅니다">전체화면</button>' +
      '      <button class="pv-x" title="취소">&times;</button>' +
      '    </div></div>' +
      '  <div class="pv-canvas-wrap"><div class="pv-canvas" style="width:' + W + 'px;height:' + H + 'px">' +
      '    <div class="pv-vbox"><video playsinline preload="metadata"></video></div>' +
      '    <div class="pv-guide-v"></div>' +
      '    <div class="pv-drag pv-title"><span class="pv-txt"></span><i class="pv-resize" title="드래그해서 제목 크기 조절"></i></div>' +
      '    <div class="pv-drag pv-caption"></div>' +
      '    <button class="pv-play">▶</button>' +
      '  </div></div>' +
      '  <div class="pv-trimwrap">' +
      '    <div class="pv-trim">' +
      '      <div class="pv-strip"></div>' +
      '      <div class="pv-segs"></div>' +
      '      <div class="pv-play-head" style="display:none"></div>' +
      '      <div class="pv-ctxmenu"><div class="it" data-act="split">분할</div></div>' +
      '    </div>' +
      '    <div class="pv-times"><span>시작 <b class="pv-t0"></b></span><span class="pv-dur"></span><span>끝 <b class="pv-t1"></b></span></div>' +
      '    <div class="pv-tools">' +
      '      <button type="button" class="pv-tool pv-zoomout">− 축소</button>' +
      '      <button type="button" class="pv-tool pv-zoomin">+ 확대</button>' +
      '      <label class="pv-speedlbl" title="영상·소리 배속(음정 유지). 자막도 함께 배속돼 싱크가 유지됩니다.">배속 ' +
      '        <select class="pv-speed"><option value="1">1.0×</option><option value="1.1">1.1×</option>' +
      '        <option value="1.2">1.2×</option><option value="1.3">1.3×</option><option value="1.5">1.5×</option>' +
      '        <option value="1.75">1.75×</option><option value="2">2.0×</option></select></label>' +
      '    </div>' +
      '  </div>' +
      '  <div class="pv-capsec hidden">' +
      '    <div class="pv-capfont-row"><span>자막 글꼴</span> <select class="pv-capfont"></select>' +
      '      <button type="button" class="pv-retrans" title="이 구간만 정밀 음성인식(large-v3)을 새로 돌려 자막 초안을 다시 뽑습니다 (1~3분, 토큰 비용 없음)">꼼꼼 재분석</button>' +
      '      <button type="button" class="pv-syncbtn" title="자막 내용·분할은 그대로 두고 각 줄의 시작·끝 시간만 실제 발화에 다시 맞춥니다. 맞춘 뒤 영어 자막까지 자동으로 만들어 둡니다. 한 번 더 누르면 최고 정밀 모델로 처음부터 재분석합니다">싱크 맞추기</button>' +
      '      <button type="button" class="pv-correctbtn" title="AI가 문맥·성경지식으로 자막 오타를 고치고 핵심 단어를 형광 강조합니다 (줄 수·시간 유지, 토큰 사용)">AI 자막 교정</button>' +
      '      <button type="button" class="pv-lyricsbtn" hidden title="곡 제목으로 정식 가사를 인터넷에서 검색해 가져오고, 이 영상의 노래 속도에 맞춰 자막을 채웁니다 (토큰 사용)">가사 자동 가져오기</button>' +
      '      <button type="button" class="pv-translatebtn" title="자막을 영어로 번역해 영어 트랙으로 저장합니다. 한국어는 그대로 유지되고, 만들 때 \'영어 자막\' 옵션으로 전환됩니다 (토큰 사용)">영어 자막 만들기</button>' +
      '      <button type="button" class="pv-karaoke-toggle" hidden title="켜면 말에 따라 단어가 파란색으로 강조됩니다. 끄면(기본) 흰 자막이 그대로 떠 있습니다.">파란 강조: 끔</button>' +
      '      <button type="button" class="pv-shiftm" title="자막 전체를 0.1초 앞으로(빠르게)">◀ 0.1s</button>' +
      '      <button type="button" class="pv-shiftp" title="자막 전체를 0.1초 뒤로(늦게)">0.1s ▶</button>' +
      '      <button type="button" class="pv-capcopyall" title="자막 내용 전체 복사">복사</button>' +
      '      <button type="button" class="pv-capsel-toggle">선택하기</button>' +
      '      <button type="button" class="pv-capsel-merge" hidden>병합</button>' +
      '      <button type="button" class="pv-capsel-del" hidden>삭제</button>' +
      '    </div>' +
      '    <div class="pv-caprows"></div>' +
      '    <button type="button" class="pv-capadd">+ 자막 줄 추가</button>' +
      '  </div>' +
      '  <div class="pv-cands"></div>' +
      '  <div class="pv-foot"><button class="pv-cancel">취소</button><button class="pv-savebtn">저장</button>' +
      '<button class="pv-wide" hidden title="쇼츠 세로 레이아웃 없이 원본 가로 비율 그대로, 자막·제목 없이 이 구간만 잘라냅니다 (곡별 개별 업로드용)">가로 원본</button>' +
      '<button class="pv-ok">이 설정으로 만들기</button></div>' +
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
    if (L.no_title) {
      // 업로드 찬양: 실제 렌더가 원본 풀프레임+제목 없음이므로 미리보기도 똑같이 —
      // 흰 카드 배경 대신 영상이 캔버스 전체를 채우고, 제목 드래그 요소를 숨긴다.
      $('.pv-canvas').style.background = '#000';
      video.style.objectFit = 'cover';  // 렌더의 crop(여백 없음)과 동일한 보기
    }

    // ── NLE식 타임라인: 휠 확대 · 분할 · 조각 삭제 ──
    const dur = info.source_duration;
    const trimEl = $('.pv-trim'), strip = $('.pv-strip'), segsLayer = $('.pv-segs'), headEl = $('.pv-play-head');
    const playBtn = $('.pv-play');
    // 남길 구간(절대초). 이전에 분할·저장했으면 복원, 아니면 클립 전체 한 조각.
    let segs = (C.keep_ranges && C.keep_ranges.length)
      ? C.keep_ranges.map((r) => ({ s: r[0], e: r[1] }))
      : [{ s: C.start, e: C.end }];
    // 처음엔 클립 주변을 적당히 확대해서 보여준다: 16초짜리가 21분 전체 위에선 손톱만 해서
    // 좌우 핸들을 못 잡아 '늘리기/줄이기'가 안 됐다. 이제 조각이 화면의 30~40%를 차지하게 띄운다.
    // (Ctrl+휠 또는 − 축소로 설교 전체까지 볼 수 있다.)
    // 단, 방금 '복제'한 클립이면 처음부터 원본 영상 전체를 보여준다(사용자 요청: 복제본은
    // 같은 장면을 다듬기보다 다른 구간을 새로 고르는 용도로도 쓰이므로).
    const _cs = segs[0].s, _ce = segs[segs.length - 1].e;
    const _pad = Math.max((_ce - _cs) * 1.3, 12);
    let view = C.show_full_source_once
      ? { a: 0, b: dur }
      : { a: Math.max(0, _cs - _pad), b: Math.min(dur, _ce + _pad) };
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
    function renderHeadAt(t) {  // 지정 시각으로 파란 재생선을 즉시 그린다(currentTime 비동기 문제 회피)
      if (t >= view.a && t <= view.b) { headEl.style.display = 'block'; headEl.style.left = t2x(t) + 'px'; }
      else headEl.style.display = 'none';
    }
    function renderHead() { renderHeadAt(video.currentTime); }
    function positionSeg(el, sg) {   // DOM 재생성 없이 한 조각의 좌우 위치만 갱신(드래그 중 사용)
      const W = trimEl.clientWidth;
      const x0 = Math.max(0, t2x(sg.s)), x1 = Math.min(W, t2x(sg.e));
      el.style.left = x0 + 'px'; el.style.width = Math.max(6, x1 - x0) + 'px';
    }
    function updateTimes() {
      const total = segs.reduce((a, s) => a + (s.e - s.s), 0);
      $('.pv-t0').textContent = fmt(segs[0].s);
      $('.pv-t1').textContent = fmt(segs[segs.length - 1].e);
      $('.pv-dur').textContent = '남는 길이 ' + total.toFixed(1) + '초' + (segs.length > 1 ? (' · ' + segs.length + '조각') : '');
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
      updateTimes();
    }
    function redraw() { renderStrip(); renderSegs(); renderHead(); }
    // 확대 중엔 조각·재생선(계산만, 즉시)만 갱신하고 썸네일(네트워크)은 살짝 지연 로딩한다 →
    // 휠 확대가 끊김 없이 즉각 반응하게. (매 틱마다 12장을 다시 불러오던 게 '느린' 주범이었다.)
    let stripTimer = null;
    function redrawLight() {
      renderSegs(); renderHead();
      if (stripTimer) clearTimeout(stripTimer);
      stripTimer = setTimeout(renderStrip, 110);
    }

    function bindSeg(el, i) {
      const Wpx = () => trimEl.clientWidth;
      const prevEnd = () => (segs[i - 1] ? segs[i - 1].e + 0.1 : 0);
      const nextStart = () => (segs[i + 1] ? segs[i + 1].s - 0.1 : dur);
      // 크기 조절은 '핸들'에서만. 조각 본체엔 드래그/포인터캡처를 안 걸어야 본체 위 클릭·더블클릭이
      // 타임라인으로 전달된다(포인터 캡처가 브라우저 더블클릭 인식을 막던 게 '노란색 위 더블클릭
      // 안 됨/오류'의 원인). 핸들은 stopPropagation으로 스크럽과 충돌하지 않게 한다.
      function bindHandle(handleEl, isL) {
        if (!handleEl) return;
        handleEl.addEventListener('pointerdown', (e) => {
          e.preventDefault(); e.stopPropagation();
          activeSeg = i;
          handleEl.setPointerCapture(e.pointerId);
          const startX = e.clientX, o = { s: segs[i].s, e: segs[i].e };
          const move = (ev) => {
            const dt = (ev.clientX - startX) / Wpx() * (view.b - view.a);
            if (isL) segs[i].s = Math.min(Math.max(prevEnd(), o.s + dt), segs[i].e - 0.5);
            else segs[i].e = Math.max(Math.min(nextStart(), o.e + dt), segs[i].s + 0.5);
            video.pause(); playBtn.classList.remove('hidden');
            const t = isL ? segs[i].s : segs[i].e;
            video.currentTime = t;
            positionSeg(el, segs[i]); updateTimes(); renderHeadAt(t);
          };
          const up = () => {
            handleEl.removeEventListener('pointermove', move); handleEl.removeEventListener('pointerup', up);
            try { handleEl.releasePointerCapture(e.pointerId); } catch (err) {}
            renderSegs();
          };
          handleEl.addEventListener('pointermove', move); handleEl.addEventListener('pointerup', up);
        });
      }
      bindHandle(el.querySelector('.hL'), true);
      bindHandle(el.querySelector('.hR'), false);
      el.querySelector('.del').addEventListener('click', (ev) => {
        ev.stopPropagation();
        if (segs.length <= 1) return;   // 최소 한 조각은 남긴다
        segs.splice(i, 1);
        activeSeg = Math.max(0, Math.min(activeSeg, segs.length - 1));
        redraw();
      });
    }

    // Ctrl(맥은 ⌘)+휠로만 확대/축소한다(일반 휠은 페이지 스크롤 유지). 커서 위치를 중심으로,
    // deltaY 크기에 비례한 지수 배율이라 빠르고 부드럽다.
    trimEl.addEventListener('wheel', (e) => {
      if (!e.ctrlKey && !e.metaKey) return;  // 일반 휠은 통과(페이지 스크롤)
      e.preventDefault();
      const pivot = x2t(e.clientX - trimEl.getBoundingClientRect().left);
      // 위로(음수)=확대(f<1), 아래로=축소. 한 틱이 확 반응하도록 계수 상향, 튐 방지로 범위 제한.
      const f = Math.min(2.2, Math.max(0.45, Math.exp(e.deltaY * 0.006)));
      view.a = pivot - (pivot - view.a) * f;
      view.b = pivot + (view.b - pivot) * f;
      clampView(); redrawLight();
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

    // 분할: 지정한 지점이 든 조각을 그 지점에서 둘로 나눔
    function splitAt(t) {
      for (let i = 0; i < segs.length; i++) {
        if (t > segs[i].s + 0.3 && t < segs[i].e - 0.3) {
          segs.splice(i + 1, 0, { s: t, e: segs[i].e });
          segs[i].e = t; activeSeg = i + 1; redraw(); return;
        }
      }
    }

    const clickT = (e) => Math.max(0, Math.min(dur, x2t(e.clientX - trimEl.getBoundingClientRect().left)));
    const onHandle = (e) => e.target.closest('.h') || e.target.closest('.del');  // 핸들/삭제는 제외

    // 진행바 우클릭 → '분할' 메뉴(별도 버튼 없이 우클릭한 그 지점에서 바로 분할).
    const pvCtxMenu = $('.pv-ctxmenu');
    let pvCtxSplitT = 0;
    trimEl.addEventListener('contextmenu', (e) => {
      if (onHandle(e)) return;
      e.preventDefault();
      pvCtxSplitT = clickT(e);
      pvCtxMenu.style.left = '0px'; pvCtxMenu.style.top = '0px'; pvCtxMenu.classList.add('on');
      const r = pvCtxMenu.getBoundingClientRect();
      pvCtxMenu.style.left = Math.max(2, Math.min(e.clientX, window.innerWidth - r.width - 4)) + 'px';
      pvCtxMenu.style.top = Math.max(2, Math.min(e.clientY, window.innerHeight - r.height - 4)) + 'px';
    });
    pvCtxMenu.addEventListener('click', (e) => {
      const it = e.target.closest('.it'); pvCtxMenu.classList.remove('on');
      if (it && it.dataset.act === 'split') splitAt(pvCtxSplitT);
    });
    document.addEventListener('pointerdown', (e) => { if (!e.target.closest('.pv-ctxmenu')) pvCtxMenu.classList.remove('on'); });
    // 단일 클릭(노란 조각 위 포함): 파란 재생선만 그 지점으로 이동(스크럽). 경계는 안 건드림.
    trimEl.addEventListener('click', (e) => {
      if (onHandle(e)) return;
      const t = clickT(e);
      video.currentTime = t; video.pause(); playBtn.classList.remove('hidden'); renderHeadAt(t);
    });
    // 더블 클릭(어느 지점이든, 노란 조각 위 포함): 그 지점이 클립 '시작'이 되고 파란 바도 이동.
    trimEl.addEventListener('dblclick', (e) => {
      if (onHandle(e)) return;
      e.preventDefault();
      const t = clickT(e);
      let i = segs.findIndex((sg) => t < sg.e - 0.5);  // 이 지점이 시작이 될 조각
      if (i === -1) i = segs.length - 1;
      const prev = segs[i - 1];
      segs[i].s = Math.max(prev ? prev.e + 0.1 : 0, Math.min(t, segs[i].e - 0.5));  // 시작을 클릭 지점으로
      activeSeg = i;
      video.currentTime = segs[i].s; video.pause(); playBtn.classList.remove('hidden');
      renderSegs(); renderHeadAt(segs[i].s);
    });

    video.addEventListener('loadedmetadata', () => { video.currentTime = segs[0].s; renderHeadAt(segs[0].s); });
    video.addEventListener('seeked', renderHead);  // 스크럽/시크 후 파란 바 위치 동기화
    redraw();
    renderHeadAt(segs[0].s);  // 영상 로드 전에도 파란 바를 클립 시작에 즉시 표시

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
      title: { x: C.title_offset_x, y: C.title_offset_y, size: C.title_size || 0 },
      caption: { x: C.caption_offset_x, y: C.caption_offset_y },
    };
    const titleEl = $('.pv-title'), capEl = $('.pv-caption');
    const titleTxt = titleEl.querySelector('.pv-txt');
    let chosenTitle = C.title;   // 후보 선택·직접 수정으로 바뀔 수 있음(payload에 저장)
    let editing = false;         // 제목 인라인 편집 중이면 드래그를 막는다
    // 제목 텍스트를 '\n'→<br>로 그린다. textContent만 쓰면 nowrap에서 줄바꿈이 공백으로 뭉개진다.
    function setTitleText(str) {
      titleTxt.textContent = '';
      String(str == null ? '' : str).split('\n').forEach((p, i) => {
        if (i > 0) titleTxt.appendChild(document.createElement('br'));
        titleTxt.appendChild(document.createTextNode(p));
      });
    }
    setTitleText(C.title);
    titleEl.title = '더블클릭하면 제목을 직접 고칠 수 있어요 · Enter로 줄바꿈';
    if (L.no_title) {
      titleEl.style.display = 'none';  // 업로드 찬양: 렌더에 제목이 없다
      // 실제 렌더 자막은 검은 외곽선의 흰 글씨(영상 위 오버레이) — 미리보기도 맞춘다.
      capEl.style.color = '#fff';
      capEl.style.textShadow = '0 0 6px rgba(0,0,0,.9)';
    }
    capEl.textContent = info.caption_preview;
    // libass는 ASS Fontsize를 셀 높이로 해석해 같은 숫자라도 브라우저보다 작게 그린다
    // (Gmarket Sans ≈ 0.87배, 서버가 폰트별 계수를 계산해 내려줌). 미리보기 px에 이
    // 계수를 곱해야 팝업에서 본 크기 = 실제 영상 크기가 된다.
    const KT = L.title_ass_coeff || 1, KC = L.caption_ass_coeff || 1;
    function applyTitleSize() {
      titleEl.style.fontSize = Math.round((state.title.size || L.title_size) * KT * SC) + 'px';
    }
    applyTitleSize();
    capEl.style.fontSize = Math.round((C.caption_size || L.caption_font_size) * KC * SC) + 'px';
    // 실제 렌더는 제목이 항상 1줄 — 팝업도 무조건 1줄로. 스타일시트 순서/캐시에 좌우되지
    // 않게 인라인으로 박는다(인라인이 어떤 시트 규칙보다 우선).
    titleEl.style.whiteSpace = 'nowrap';
    titleEl.style.maxWidth = 'none';
    // maxWidth를 주면 nowrap과 충돌해 글자가 박스 밖으로 잘려 보인다. 폭 제한은
    // fitToWidth(폰트 축소)가 담당하므로 박스 자체는 제한하지 않는다.
    if (info.title_font) titleEl.style.fontFamily = "'" + info.title_font.family + "', sans-serif";
    if (info.caption_font) capEl.style.fontFamily = "'" + info.caption_font.family + "', sans-serif";
    const USABLE_W = (L.resolution[0] - 80) * SC;
    function fitToWidth(el, measureEl) {
      measureEl = measureEl || el;
      let fs = parseFloat(getComputedStyle(el).fontSize), guard = 0;
      while (measureEl.scrollWidth > USABLE_W && fs > 5 && guard < 300) { fs -= 0.5; el.style.fontSize = fs + 'px'; guard++; }
    }
    // 전체화면에서 미리보기 캔버스를 CSS scale로 키운다. 화면 px -> 렌더 px 변환은
    // SC 하나였는데, 확대하면 SC*ZOOM이 된다 — 드래그/크기조절이 확대 배율만큼 어긋나지
    // 않도록 아래 계산에 전부 ZOOM을 곱한다.
    let ZOOM = 1;
    function paintBox(el, key) {
      el.style.left = (bases[key].left + state[key].x * SC) + 'px';
      if (key === 'caption' && L.caption_anchor === 'bottom') {
        // 렌더가 '아래 기준'인 오버레이 자막(찬양): 아래에서 띄운 거리로 그린다.
        el.style.top = '';
        el.style.bottom = (((L.caption_bottom_px || 0) - state.caption.y) * SC) + 'px';
      } else {
        el.style.bottom = '';
        el.style.top = (bases[key].top + state[key].y * SC) + 'px';
      }
    }
    // 가로 정중앙 스냅: x(가운데 기준 오프셋)가 거의 0이면 0으로 딱 붙이고 가이드선을 보여준다.
    const guideV = $('.pv-guide-v');
    const SNAP_PX = 6;  // 팝업 화면 px 기준 스냅 반경
    function makeDraggable(el, key, onDbl) {
      let lastDown = 0;
      el.addEventListener('pointerdown', (e) => {
        if (editing) return;   // 편집 중엔 커서/선택이 우선 — 드래그 금지
        if (onDbl && e.timeStamp - lastDown < 320) {  // 빠른 두 번 클릭 = 더블클릭
          lastDown = 0; e.preventDefault(); onDbl(); return;
        }
        lastDown = e.timeStamp;
        e.preventDefault();
        el.classList.add('dragging');
        el.setPointerCapture(e.pointerId);
        const sx = e.clientX, sy = e.clientY, ox = state[key].x, oy = state[key].y;
        const move = (ev) => {
          let nx = ox + (ev.clientX - sx) / (SC * ZOOM);
          const ny = oy + (ev.clientY - sy) / (SC * ZOOM);
          if (Math.abs(nx * SC * ZOOM) < SNAP_PX) { nx = 0; guideV.classList.add('show'); }
          else { guideV.classList.remove('show'); }
          state[key].x = nx; state[key].y = ny;
          paintBox(el, key);
        };
        const up = () => {
          el.classList.remove('dragging');
          guideV.classList.remove('show');
          el.removeEventListener('pointermove', move);
          el.removeEventListener('pointerup', up);
        };
        el.addEventListener('pointermove', move);
        el.addEventListener('pointerup', up);
      });
    }

    // ── 제목 직접 수정: 더블클릭 → 인라인 편집, Enter=줄바꿈, 포커스 잃으면 확정 ──
    function startEdit() {
      if (editing) return;
      editing = true;
      titleEl.classList.add('editing');
      titleTxt.setAttribute('contenteditable', 'true');
      titleTxt.style.whiteSpace = 'pre-wrap';   // 편집 중엔 긴 줄·줄바꿈이 다 보이게
      titleTxt.focus();
      const r = document.createRange(); r.selectNodeContents(titleTxt);
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
    }
    // contenteditable 내용을 텍스트로 읽는다: <br>/블록 노드를 줄바꿈으로 바꿔 반환한다
    // (innerText는 white-space 렌더에 따라 줄바꿈이 공백으로 뭉개져 신뢰할 수 없었다).
    function readEditedText() {
      let out = '';
      titleTxt.childNodes.forEach((n) => {
        if (n.nodeName === 'BR') out += '\n';
        else if (n.nodeType === 1) { if (out && !out.endsWith('\n')) out += '\n'; out += (n.textContent || ''); }
        else out += (n.textContent || '');
      });
      return out;
    }
    function commitEdit() {
      if (!editing) return;
      editing = false;
      titleTxt.removeAttribute('contenteditable');
      titleTxt.style.whiteSpace = '';
      titleEl.classList.remove('editing');
      let t = readEditedText().split('\n').map(function (s) { return s.replace(/\s+$/, ''); }).join('\n').replace(/\n+$/, '');
      if (!t.trim()) t = chosenTitle;   // 빈 제목 방지 — 되돌린다
      chosenTitle = t;
      setTitleText(t);                  // DOM 정규화(편집 흔적 제거)
      applyTitleSize(); fitToWidth(titleEl, titleTxt);
      candsBox.querySelectorAll('.pv-cand').forEach((x) => x.classList.remove('sel'));
    }
    titleTxt.addEventListener('keydown', (e) => {
      if (!editing) return;
      if (e.key === 'Enter') { e.preventDefault(); document.execCommand('insertLineBreak'); }
      else if (e.key === 'Escape') { e.preventDefault(); setTitleText(chosenTitle); titleTxt.blur(); }
    });
    titleTxt.addEventListener('blur', commitEdit);

    fitToWidth(titleEl, titleTxt); fitToWidth(capEl);
    paintBox(titleEl, 'title'); paintBox(capEl, 'caption');
    makeDraggable(titleEl, 'title', startEdit); makeDraggable(capEl, 'caption');
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => {
      applyTitleSize();
      fitToWidth(titleEl, titleTxt);
    });

    // ── 제목 크기 조절 핸들(파워포인트처럼 모서리를 드래그) ──
    const resizeHandle = titleEl.querySelector('.pv-resize');
    resizeHandle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();  // 부모(titleEl)의 이동 드래그가 같이 반응하지 않게
      titleEl.classList.add('dragging');
      resizeHandle.setPointerCapture(e.pointerId);
      const sx = e.clientX;
      const startSize = state.title.size || L.title_size;
      const move = (ev) => {
        const dx = ev.clientX - sx;
        state.title.size = Math.max(30, Math.min(280, Math.round(startSize + dx / (SC * ZOOM))));
        titleEl.style.fontSize = Math.round(state.title.size * KT * SC) + 'px';
      };
      const up = () => {
        titleEl.classList.remove('dragging');
        resizeHandle.removeEventListener('pointermove', move);
        resizeHandle.removeEventListener('pointerup', up);
        fitToWidth(titleEl, titleTxt);  // 한 줄 안에 들어가게 강제(브라우저 실측 폭 기준)
        // 화면에 보이는 크기 그대로 저장해야 실제 렌더도 똑같이 나온다. 드래그한 원값을
        // 그대로 저장하면 한 줄에 안 맞아 화면상 줄어든 걸 무시한 채 큰 값이 저장되고,
        // 그 값이 렌더의 상한(max_title_size)이 되어 미리보기보다 커 보이는 원인이 된다.
        // (표시 px → ASS 크기 역변환에도 KT를 반영해야 한다.)
        const shownPx = parseFloat(getComputedStyle(titleEl).fontSize);
        state.title.size = Math.max(20, Math.round(shownPx / (KT * SC)));
      };
      resizeHandle.addEventListener('pointermove', move);
      resizeHandle.addEventListener('pointerup', up);
    });

    // ── 제목 후보 5개 ──
    const candsBox = $('.pv-cands');
    (C.title_candidates || []).forEach((t, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'pv-cand' + (t === chosenTitle ? ' sel' : '');
      b.textContent = t;
      b.addEventListener('click', () => {
        chosenTitle = t;
        candsBox.querySelectorAll('.pv-cand').forEach((x) => x.classList.remove('sel'));
        b.classList.add('sel');
        setTitleText(t);
        applyTitleSize();
        fitToWidth(titleEl, titleTxt);
      });
      candsBox.appendChild(b);
    });

    // ── 자막 수정(접었다 폈다, 시작·끝·내용 편집) ──
    const capSec = $('.pv-capsec'), capRowsBox = $('.pv-caprows');
    // 찬양 클립은 자막(가사)이 핵심이라 자막 영역을 처음부터 펼쳐 둔다 — 가사 가져오기·
    // 싱크 같은 도구가 '자막 수정'을 눌러야만 보여서 못 찾는 문제(실신고) 방지.
    if (C.clip_type === 'praise') capSec.classList.remove('hidden');

    // ── 배속(1.0~2.0): 미리보기도 즉시 그 배속으로 재생, 저장 시 렌더에 반영 ──
    const speedSel = $('.pv-speed');
    speedSel.value = String(C.playback_speed || 1);
    if (![...speedSel.options].some(o => o.value === speedSel.value)) speedSel.value = '1';
    video.playbackRate = parseFloat(speedSel.value) || 1;
    speedSel.addEventListener('change', () => {
      video.playbackRate = parseFloat(speedSel.value) || 1;
    });
    // ── 자막 글꼴 선택 ──
    const capFontSel = $('.pv-capfont');
    let chosenCaptionFont = (info.caption_font && info.caption_font.family) || '';
    (info.fonts || []).forEach((f) => {
      const o = document.createElement('option');
      o.value = f.family; o.textContent = f.name;
      if (f.family === chosenCaptionFont) o.selected = true;
      capFontSel.appendChild(o);
    });
    capFontSel.addEventListener('change', () => {
      chosenCaptionFont = capFontSel.value;
      capEl.style.fontFamily = "'" + chosenCaptionFont + "', sans-serif";
      capRowsBox.querySelectorAll('.pv-cap-text').forEach((el) => {
        el.style.fontFamily = "'" + chosenCaptionFont + "', sans-serif";
      });
    });
    function addCapRow(startRel, endRel, text) {
      const row = document.createElement('div');
      row.className = 'pv-caprow';
      row.innerHTML =
        '<input type="checkbox" class="pv-cap-sel" title="선택">' +
        '<input type="number" class="pv-cap-start" step="0.1">' +
        '<input type="number" class="pv-cap-end" step="0.1">' +
        '<input type="text" class="pv-cap-text">' +
        '<button type="button" class="pv-cap-del" title="삭제">&times;</button>';
      row.querySelector('.pv-cap-start').value = startRel.toFixed(1);
      row.querySelector('.pv-cap-end').value = endRel.toFixed(1);
      const textEl = row.querySelector('.pv-cap-text');
      textEl.value = text || '';
      if (chosenCaptionFont) textEl.style.fontFamily = "'" + chosenCaptionFont + "', sans-serif";
      capRowsBox.appendChild(row);
    }

    // ── 자막 줄 선택 모드: 체크박스로 여러 줄 골라 병합(최대 4개)·삭제 ──
    const selToggle = $('.pv-capsel-toggle'), selMerge = $('.pv-capsel-merge'), selDel = $('.pv-capsel-del');
    let selMode = false;
    selToggle.addEventListener('click', () => {
      selMode = !selMode;
      capRowsBox.classList.toggle('selmode', selMode);
      selToggle.classList.toggle('on', selMode);
      selToggle.textContent = selMode ? '선택 취소' : '선택하기';
      selMerge.hidden = selDel.hidden = !selMode;
      if (!selMode) capRowsBox.querySelectorAll('.pv-cap-sel').forEach((c) => { c.checked = false; });
    });
    function selectedRows() {
      return [...capRowsBox.querySelectorAll('.pv-caprow')]
        .filter((r) => r.querySelector('.pv-cap-sel').checked);
    }
    selDel.addEventListener('click', () => {
      const rows = selectedRows();
      if (!rows.length) { alert('삭제할 자막 줄을 먼저 체크하세요'); return; }
      rows.forEach((r) => r.remove());
      capsDirty = true;
    });
    selMerge.addEventListener('click', () => {
      const rows = selectedRows();
      if (rows.length < 2) { alert('병합할 자막 줄을 2개 이상 체크하세요'); return; }
      if (rows.length > 4) { alert('병합은 최대 4개까지만 가능합니다'); return; }
      // 시간순으로 합친다: 시작=가장 이른 시작, 끝=가장 늦은 끝, 내용은 시간순 이어붙임.
      rows.sort((a, b) =>
        (parseFloat(a.querySelector('.pv-cap-start').value) || 0)
        - (parseFloat(b.querySelector('.pv-cap-start').value) || 0));
      const s = Math.min(...rows.map((r) => parseFloat(r.querySelector('.pv-cap-start').value) || 0));
      const e = Math.max(...rows.map((r) => parseFloat(r.querySelector('.pv-cap-end').value) || 0));
      const text = rows.map((r) => r.querySelector('.pv-cap-text').value.trim()).filter(Boolean).join(' ');
      const first = rows[0];
      first.querySelector('.pv-cap-start').value = s.toFixed(1);
      first.querySelector('.pv-cap-end').value = e.toFixed(1);
      first.querySelector('.pv-cap-text').value = text;
      first.querySelector('.pv-cap-sel').checked = false;
      rows.slice(1).forEach((r) => r.remove());
      capsDirty = true;
    });
    (info.caption_lines || []).forEach((c) => addCapRow(c.start - C.start, c.end - C.start, c.text));
    // 실제로 자막을 고쳤을 때만 저장한다. 안 고쳤는데 매번 초안을 '사용자 확정본'으로
    // 저장하면, (아직 정밀 재전사 전인 클립은) 부정확한 자동자막 초안이 그대로 굳어서
    // 렌더의 정밀 재전사(더 정확한 자막)가 영영 건너뛰어진다.
    let capsDirty = false;
    // 업로드 찬양 클립의 카라오케(단어별 색 변경, 파란 강조) 자막 토글.
    // 기본은 꺼짐(정적 흰 자막이 쭉 떠 있음) — 사용자 요청 2026-09-05: "말 따라 파란색으로
    // 가지 말고 그냥 흰 자막이 계속 떠 있게". 켜고 싶으면 아래 '파란 강조' 버튼으로만 켠다
    // (예전엔 '싱크 맞추기'가 자동으로 켰는데, 그게 원치 않는 파란색의 원인이었다).
    let capKaraoke = !!C.caption_karaoke;
    // AI 자막 교정이 뽑은 형광 강조어. 저장 시 caption_highlights로 넘어가 렌더가 강조한다.
    let capHighlights = Array.isArray(C.caption_highlights) ? C.caption_highlights.slice() : [];
    // 영어 자막 트랙(번역). 저장 시 caption_overrides_en으로 넘어간다. 한국어는 그대로 유지.
    let capOverridesEn = Array.isArray(C.caption_overrides_en) ? C.caption_overrides_en.slice() : [];
    const karaokeBtn = $('.pv-karaoke-toggle');
    // 파란 강조는 업로드 찬양 렌더에서만 의미가 있다 — 그 경우에만 버튼을 보여준다.
    const isUploadPraise = (C.clip_type === 'praise') && VIDEO_ID.indexOf('upload_') === 0;
    function renderKaraokeBtn() {{
      karaokeBtn.textContent = capKaraoke ? '파란 강조: 켬' : '파란 강조: 끔';
      karaokeBtn.classList.toggle('on', capKaraoke);
    }}
    if (isUploadPraise) {{
      karaokeBtn.hidden = false;
      renderKaraokeBtn();
      karaokeBtn.addEventListener('click', () => {{
        capKaraoke = !capKaraoke;
        capsDirty = true;  // 저장 시 caption_karaoke가 반영되도록
        renderKaraokeBtn();
      }});
    }}
    capRowsBox.addEventListener('input', () => { capsDirty = true; });
    $('.pv-capedit-btn').addEventListener('click', () => capSec.classList.toggle('hidden'));
    $('.pv-capadd').addEventListener('click', () => {
      const rows = capRowsBox.querySelectorAll('.pv-caprow');
      const s = rows.length ? (parseFloat(rows[rows.length - 1].querySelector('.pv-cap-end').value) || 0) : 0;
      addCapRow(s, s + 2, '');
      capsDirty = true;
      capSec.classList.remove('hidden');
    });
    capRowsBox.addEventListener('click', (e) => {
      if (e.target.classList.contains('pv-cap-del')) { e.target.closest('.pv-caprow').remove(); capsDirty = true; }
    });
    function collectCaptions() {
      return [...capRowsBox.querySelectorAll('.pv-caprow')].map((r) => {
        const s = parseFloat(r.querySelector('.pv-cap-start').value);
        const en = parseFloat(r.querySelector('.pv-cap-end').value);
        return {
          start: C.start + (isNaN(s) ? 0 : s), end: C.start + (isNaN(en) ? 0 : en),
          text: r.querySelector('.pv-cap-text').value,
        };
      });
    }

    // ── 구간 재분석: 주제는 그대로, 이 장면 시작·끝만 다시 잡아 새 후보로 추가(원본 유지) ──
    // 실제 작업은 서버 스레드에서 도니 팝업을 닫아도 계속 진행된다. 화면 아래 고정
    // 위젯(mini-prog, __startRenderWatch 옆의 pollReanalyze)이 팝업이 닫힌 뒤에도 진행률·완료를
    // 이어서 보여준다 — 이 팝업 안 코드는 "열려 있는 동안"의 버튼 표시만 담당한다.
    const reBtn = $('.pv-reanalyze');
    reBtn.addEventListener('click', async () => {
      reBtn.disabled = true; reBtn.textContent = '재분석 중…';
      let started = null;
      try { started = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/reanalyze', { method: 'POST' }); }
      catch (e) { started = null; }
      if (!started || !started.ok) {
        const d = started ? await started.json().catch(() => ({})) : {};
        alert('재분석 시작 실패: ' + (d.error || '네트워크 오류'));
        reBtn.disabled = false; reBtn.textContent = '구간 재분석'; return;
      }
      const poll = () => {
        if (!back.isConnected) return;  // 팝업이 닫혔으면 여기선 멈추고 화면 아래 위젯에 맡긴다
        fetch('/video/' + VIDEO_ID + '/reanalyze_status').then((r) => r.json()).then((j) => {
          if (!back.isConnected) return;
          if (j.running) { reBtn.textContent = '재분석 중… ' + Math.round((j.pct || 0) * 100) + '%'; setTimeout(poll, 1000); return; }
          if (j.error) { reBtn.disabled = false; reBtn.textContent = '구간 재분석'; alert('재분석 실패: ' + j.error); return; }
          reBtn.textContent = '✓ 완료 (새로고침하면 후보에 추가됨)';
        }).catch(() => setTimeout(poll, 1500));
      };
      poll();
    });

    // ── 자막 꼼꼼 재분석: 이 구간만 정밀 음성인식을 새로 돌려 자막 초안을 다시 뽑는다 ──
    const retransBtn = $('.pv-retrans');
    retransBtn.addEventListener('click', async () => {
      retransBtn.disabled = true;
      const t0 = Date.now();
      retransBtn.textContent = '정밀 인식 중… (1~3분)';
      let started = null;
      try { started = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/retranscribe', { method: 'POST' }); }
      catch (e) { started = null; }
      if (!started || !started.ok) {
        alert('재분석 시작 실패');
        retransBtn.disabled = false; retransBtn.textContent = '꼼꼼 재분석'; return;
      }
      const poll = () => {
        if (!back.isConnected) return;  // 팝업 닫혔으면 중단(서버 작업은 계속 돌지만 결과 반영처가 없음)
        fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/retranscribe_status')
          .then((r) => r.json()).then((j) => {
            if (!back.isConnected) return;
            if (j.running) {
              retransBtn.textContent = '정밀 인식 중… ' + Math.round((Date.now() - t0) / 1000) + '초';
              setTimeout(poll, 2000); return;
            }
            if (j.error || !j.lines) {
              alert('자막 재분석 실패: ' + (j.error || '결과 없음'));
              retransBtn.disabled = false; retransBtn.textContent = '꼼꼼 재분석'; return;
            }
            // 편집 목록을 새 정밀 자막으로 교체하고, 확정 시 저장되도록 dirty 표시.
            capRowsBox.innerHTML = '';
            j.lines.forEach((c) => addCapRow(c.start - C.start, c.end - C.start, c.text));
            capsDirty = true;
            capSec.classList.remove('hidden');
            retransBtn.disabled = false; retransBtn.textContent = '✓ 새 자막 적용됨 (저장 필요)';
          }).catch(() => setTimeout(poll, 2500));
      };
      poll();
    });

    // ── 싱크 맞추기: 자막 내용·분할은 그대로, 각 줄의 시작·끝만 실제 발화 시각에 재정렬 ──
    // 두 번째 이상 누르면 fresh=true — 캐시 재사용이 아니라 최고 정밀 모델(large-v3)로
    // 처음부터 다시 전사해 다시 맞춘다("계속 눌러도 결과가 똑같다" 신고 대응).
    const syncBtn = $('.pv-syncbtn');
    let syncPresses = 0;
    syncBtn.addEventListener('click', async () => {
      const caps = collectCaptions();
      if (!caps.length) { alert('맞출 자막이 없습니다'); return; }
      const fresh = syncPresses > 0;
      syncPresses += 1;
      syncBtn.disabled = true;
      syncBtn.textContent = fresh ? '정밀 재분석 중… (1~3분)' : '맞추는 중…';
      let r = null;
      try {
        r = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/sync_captions', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ captions: caps, fresh: fresh }),
        });
      } catch (e) { r = null; }
      const j = r && r.ok ? await r.json().catch(() => null) : null;
      syncBtn.disabled = false;
      if (!j || !j.lines) {
        syncBtn.textContent = '싱크 맞추기';
        alert('싱크 맞추기 실패' + (j && j.error ? ': ' + j.error : '')); return;
      }
      // 행 순서는 그대로 두고 시간만 갱신(텍스트·분할 불변).
      const rows = [...capRowsBox.querySelectorAll('.pv-caprow')];
      j.lines.forEach((ln, i) => {
        if (!rows[i]) return;
        rows[i].querySelector('.pv-cap-start').value = (ln.start - C.start).toFixed(1);
        rows[i].querySelector('.pv-cap-end').value = (ln.end - C.start).toFixed(1);
      });
      capsDirty = true;
      // 싱크 맞추기는 '줄의 시작·끝 시간'만 다시 맞춘다 — 파란 강조(카라오케)는 켜지 않는다.
      // (사용자 요청 2026-09-05: 흰 자막이 그대로 떠 있어야 하고, 파란색은 '파란 강조' 버튼으로만.)
      capSec.classList.remove('hidden');
      // 영어 자막도 여기서 같이 만든다(사용자 요청 2026-09-07: "영어 자막 만들기를 따로
      // 안 눌러도 되게"). 싱크 결과는 위에서 이미 화면에 반영했고 번역은 그 다음에 돌린다 —
      // 싱크 자체가 번역(10~30초)을 기다리느라 느려 보이지 않게 하려는 순서다.
      // 번역 입력은 서버가 돌려준 j.lines(분할까지 반영된 최종 줄)를 쓴다.
      syncBtn.textContent = '✓ ' + j.matched + '/' + j.total + '줄 맞춤 — 영어 자막 만드는 중…';
      let enMsg = '';
      try {
        const rt = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/translate_captions', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ captions: j.lines, model: '' }),
        });
        const jt = rt && rt.ok ? await rt.json().catch(() => null) : null;
        if (jt && jt.lines) { capOverridesEn = jt.lines; enMsg = ' + 영어 ' + jt.lines.length + '줄'; }
        else enMsg = ' (영어 번역 실패 — 아래 버튼으로 다시)';
      } catch (e) { enMsg = ' (영어 번역 실패 — 아래 버튼으로 다시)'; }
      syncBtn.textContent = '✓ ' + j.matched + '/' + j.total + '줄 맞춤' + enMsg + ' (저장 필요)';
      setTimeout(() => { syncBtn.textContent = '싱크 맞추기'; }, 6000);
    });

    // ── AI 자막 교정: 문맥·성경지식으로 오타 교정 + 핵심어 형광 강조(줄 수·시간 유지) ──
    const correctBtn = $('.pv-correctbtn');
    correctBtn.addEventListener('click', async () => {
      const caps = collectCaptions();
      if (!caps.length) { alert('교정할 자막이 없습니다'); return; }
      correctBtn.disabled = true; correctBtn.textContent = '교정 중… (10~30초)';
      let r = null;
      try {
        r = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/correct_captions', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ captions: caps, model: '' }),
        });
      } catch (e) { r = null; }
      const j = r && r.ok ? await r.json().catch(() => null) : null;
      correctBtn.disabled = false;
      if (!j || !j.lines) {
        correctBtn.textContent = 'AI 자막 교정';
        alert('AI 자막 교정 실패' + (j && j.error ? ': ' + j.error : '')); return;
      }
      // 텍스트만 교체(시간·행 순서 유지). 강조어 저장.
      const rows = [...capRowsBox.querySelectorAll('.pv-caprow')];
      j.lines.forEach((ln, i) => { if (rows[i]) rows[i].querySelector('.pv-cap-text').value = ln.text; });
      capHighlights = Array.isArray(j.highlights) ? j.highlights : [];
      capsDirty = true;
      capSec.classList.remove('hidden');
      correctBtn.textContent = '✓ ' + (j.changed || 0) + '줄 교정 · 강조 ' + capHighlights.length + '개 (저장 필요)';
      setTimeout(() => { correctBtn.textContent = 'AI 자막 교정'; }, 5000);
    });

    // ── 가사 자동 가져오기(찬양): 곡 제목으로 인터넷 검색 → 정식 가사 + 노래 속도 싱크 ──
    const lyricsBtn = $('.pv-lyricsbtn');
    if (C.clip_type === 'praise') {
      lyricsBtn.hidden = false;
      lyricsBtn.addEventListener('click', async () => {
        const t = prompt('곡 제목 (이 제목으로 인터넷에서 정식 가사를 검색합니다)', chosenTitle || C.title || '');
        if (t == null || !t.trim()) return;
        lyricsBtn.disabled = true; lyricsBtn.textContent = '가사 검색·싱크 중… (1~3분)';
        let r = null;
        try {
          r = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/fetch_lyrics', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: t.trim() }),
          });
        } catch (e) { r = null; }
        const j = r && r.ok ? await r.json().catch(() => null) : null;
        lyricsBtn.disabled = false;
        if (!j || !j.lines) {
          lyricsBtn.textContent = '가사 자동 가져오기';
          const d = r && !r.ok ? await r.json().catch(() => ({})) : {};
          alert('가사 가져오기 실패' + (d.error ? ': ' + d.error : (j && j.error ? ': ' + j.error : ''))); return;
        }
        capRowsBox.innerHTML = '';
        j.lines.forEach((c2) => addCapRow(c2.start - C.start, c2.end - C.start, c2.text));
        capsDirty = true;
        capSec.classList.remove('hidden');
        lyricsBtn.textContent = '✓ ' + j.lines.length + '줄 (저장 필요)';
        setTimeout(() => { lyricsBtn.textContent = '가사 자동 가져오기'; }, 5000);
      });
    }

    // ── 영어 자막 만들기: 현재 자막을 영어로 번역해 영어 트랙에 저장(한국어 유지) ──
    const translateBtn = $('.pv-translatebtn');
    translateBtn.addEventListener('click', async () => {
      const caps = collectCaptions();
      if (!caps.length) { alert('번역할 자막이 없습니다'); return; }
      translateBtn.disabled = true; translateBtn.textContent = '번역 중… (10~30초)';
      let r = null;
      try {
        r = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/translate_captions', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ captions: caps, model: '' }),
        });
      } catch (e) { r = null; }
      const j = r && r.ok ? await r.json().catch(() => null) : null;
      translateBtn.disabled = false;
      if (!j || !j.lines) {
        translateBtn.textContent = '영어 자막 만들기';
        alert('영어 번역 실패' + (j && j.error ? ': ' + j.error : '')); return;
      }
      capOverridesEn = j.lines;   // 시간은 현재 자막과 동일, 텍스트만 영어
      capsDirty = true;           // 저장 시 caption_overrides_en 반영
      translateBtn.textContent = '✓ 영어 ' + j.lines.length + '줄 (저장 후 \'영어 자막\'으로 만들기)';
      setTimeout(() => { translateBtn.textContent = '영어 자막 만들기'; }, 6000);
    });

    // ── 전체 밀기: 모든 자막 줄의 시작·끝을 한 번에 ±0.1초 이동(귀로 미세 조정용) ──
    function shiftAll(delta) {
      capRowsBox.querySelectorAll('.pv-caprow').forEach((r) => {
        const s = r.querySelector('.pv-cap-start'), e = r.querySelector('.pv-cap-end');
        s.value = ((parseFloat(s.value) || 0) + delta).toFixed(1);
        e.value = ((parseFloat(e.value) || 0) + delta).toFixed(1);
      });
      capsDirty = true;
      capSec.classList.remove('hidden');
    }
    $('.pv-shiftm').addEventListener('click', () => shiftAll(-0.1));
    $('.pv-shiftp').addEventListener('click', () => shiftAll(0.1));

    // 자막 내용 전체 복사(시간 없이 텍스트만, 줄바꿈으로).
    const capCopyAll = $('.pv-capcopyall');
    capCopyAll.addEventListener('click', () => {
      const txt = [...capRowsBox.querySelectorAll('.pv-cap-text')]
        .map((el) => el.value.trim()).filter(Boolean).join('\n');
      if (!txt) { alert('복사할 자막이 없습니다'); return; }
      const done = () => { capCopyAll.textContent = '복사됨'; setTimeout(() => { capCopyAll.textContent = '복사'; }, 1500); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(done, () => {
          const ta = document.createElement('textarea'); ta.value = txt; document.body.appendChild(ta);
          ta.select(); document.execCommand('copy'); ta.remove(); done();
        });
      } else {
        const ta = document.createElement('textarea'); ta.value = txt; document.body.appendChild(ta);
        ta.select(); document.execCommand('copy'); ta.remove(); done();
      }
    });

    // ── 전체화면(사용자 요청 2026-09-07, 2026-09-08: 진짜 풀스크린 API 아님) ──────
    // 팝업 전체를 화면 가득 띄우고, 캔버스 밖 UI(헤더·자막 목록·버튼)가 쓰는 높이를 뺀
    // 남는 공간만큼 미리보기 캔버스를 확대한다. 확대는 CSS scale이라 자막/제목 오버레이가
    // 캔버스와 함께 정확히 같은 비율로 커진다(좌표계는 ZOOM으로 보정).
    // document.requestFullscreen()은 쓰지 않는다 — 브라우저 실제 풀스크린으로 들어가면
    // 탭/주소창이 사라지고 Esc로만 빠져나올 수 있어 사용자가 원하는 "팝업만 크게"와 다르다.
    // 대신 .pv-maxed 클래스로 카드를 뷰포트 전체에 고정 배치하는 순수 CSS 확대만 쓴다.
    const fullBtn = $('.pv-fullbtn');
    const cardEl = $('.pv-card'), wrapEl = $('.pv-canvas-wrap'), canvasEl = $('.pv-canvas');
    let maxed = false;
    function applyZoom() {
      if (!document.body.contains(back)) return;
      wrapEl.style.height = ''; canvasEl.style.transform = '';   // 원래 크기로 되돌려 실측
      ZOOM = 1;
      if (maxed) {
        const other = Math.max(0, cardEl.scrollHeight - wrapEl.offsetHeight);
        ZOOM = Math.max(1, Math.min(
          (window.innerHeight - other - 24) / H, (window.innerWidth - 48) / W, 3));
      }
      if (ZOOM > 1.001) {
        canvasEl.style.transformOrigin = 'top center';
        canvasEl.style.transform = 'scale(' + ZOOM + ')';
        wrapEl.style.height = Math.round(H * ZOOM) + 'px';
      }
      fullBtn.textContent = maxed ? '전체화면 해제' : '전체화면';
    }
    fullBtn.addEventListener('click', () => {
      maxed = !maxed;
      back.classList.toggle('pv-maxed', maxed);
      applyZoom();
    });
    window.addEventListener('resize', applyZoom);

    // ── 닫기/저장/확정 ──
    function close() {
      video.pause();
      window.removeEventListener('resize', applyZoom);
      back.remove();
    }
    $('.pv-x').addEventListener('click', close);
    $('.pv-cancel').addEventListener('click', close);
    function buildPayload() {
      const payload = {
        title: chosenTitle,
        title_offset_x: state.title.x, title_offset_y: state.title.y,
        title_size: state.title.size,
        caption_offset_x: state.caption.x, caption_offset_y: state.caption.y,
        caption_font: chosenCaptionFont,
      };
      // 자막은 사용자가 실제로 고쳤을 때만 확정본으로 저장(위 capsDirty 주석 참고).
      if (capsDirty) payload.captions = collectCaptions();
      payload.caption_karaoke = capKaraoke;
      payload.caption_highlights = capHighlights;
      payload.caption_overrides_en = capOverridesEn;
      payload.playback_speed = parseFloat(speedSel.value) || 1;
      const outS = segs[0].s, outE = segs[segs.length - 1].e;
      const changed = segs.length > 1 || Math.abs(outS - C.start) > 0.05 || Math.abs(outE - C.end) > 0.05;
      if (changed) {
        payload.clip_start = outS; payload.clip_end = outE;
        payload.keep_ranges = segs.map((sg) => [sg.s, sg.e]);
        // 구간을 바꾸면 저장된 자막 타임스탬프가 무효 → 설교는 비워서 렌더 때 재전사를
        // 유도한다. 단 업로드 찬양은 재전사 폴백이 없어(전사본 없음) 비우는 순간 자막이
        // 영구 소실되고 영상에 아무 자막도 안 구워진다(실사고 2026-09-05: "저장해둔 자막이
        // 사라져") — 찬양은 자막을 유지한다(시각은 절대초라 트림과 무관하게 유효).
        if (!isUploadPraise) payload.captions = [];
      }
      return { payload, outS, outE };
    }
    async function saveNow() {
      const { payload, outS, outE } = buildPayload();
      const r = await fetch('/video/' + VIDEO_ID + '/clip/' + idx + '/position', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      if (!r.ok) return false;
      // 저장 성공 → 현재 상태를 새 기준으로: 다음 저장에서 '구간 바뀜' 오판 방지, dirty 해제.
      C.start = outS; C.end = outE;
      capsDirty = false;
      const card = document.getElementById('cand-' + idx);
      if (card) { const h = card.querySelector('h3.title'); if (h) h.textContent = chosenTitle; }
      return true;
    }
    const saveBtn = $('.pv-savebtn');
    saveBtn.addEventListener('click', async () => {
      saveBtn.disabled = true;
      const ok = await saveNow();
      saveBtn.disabled = false;
      saveBtn.textContent = ok ? '저장됨 ✓' : '저장 실패';
      setTimeout(() => { saveBtn.textContent = '저장'; }, 2000);
      if (!ok) alert('저장에 실패했어요');
    });
    $('.pv-ok').addEventListener('click', async () => {
      const ok = await saveNow();
      if (!ok) { alert('저장에 실패했어요'); return; }
      // 이 클립을 이전에 '가로 원본'으로 찍어뒀다가 마음을 바꿔 일반 만들기를 눌렀을 수
      // 있으므로, 일반 확정은 가로 목록에서 확실히 뺀다.
      if (window.__pvHorizontal) window.__pvHorizontal.delete(idx);
      close();
      onConfirm();
    });
    // '가로 원본': 찬양 클립 전용 — 쇼츠 세로 레이아웃 없이 원본 가로 비율 그대로,
    // 자막·제목 없이 이 구간만 잘라낸다(곡별 개별 업로드용, 사용자 요청 2026-09-06).
    const wideBtn = $('.pv-wide');
    if (C.clip_type === 'praise') wideBtn.hidden = false;
    wideBtn.addEventListener('click', async () => {
      const ok = await saveNow();
      if (!ok) { alert('저장에 실패했어요'); return; }
      (window.__pvHorizontal = window.__pvHorizontal || new Set()).add(idx);
      close();
      onConfirm();
    });
  }
})();
"""


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
        # 하이라이트 후보 목록 '맨 아래'에 새 후보로 추가한다(원본은 그대로 유지).
        with CLIPS_LOCK:
            clips = load_clips_json(clips_path)  # 그 사이 바뀌었을 수 있어 다시 읽는다
            clips.append(new_clip)
            new_idx = len(clips) - 1
            save_clips_json(clips, clips_path)
        _reanalyze_jobs[video_id] = {"running": False, "error": None, "new_idx": new_idx, "pct": 1.0}
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
        # 가창 단어가 거의 안 잡혔다 → 매핑 불가, 원래 시각 유지.
        lines = [
            {"start": float(l.get("start", 0)), "end": float(l.get("end", 0)),
             "text": str(l.get("text", "")).strip()}
            for l in in_lines
        ]
        return jsonify({"lines": lines, "matched": 0, "total": len(lines),
                        "source": "전사 단어 부족 — 원래 시각 유지"})
    src = "최고 정밀(large-v3) 재분석" if fresh else "가창 단어 시각 기준(노래 싱크)"
    mapped = _split_long_caption_lines(video_dir.name, clip, cfg, mapped)
    return jsonify({"lines": mapped, "matched": len(mapped), "total": len(mapped), "source": src})


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
    if segs is None:
        segs = tdata["segments"]
        used = "참조 전사(캐시 없음 — '꼼꼼 재분석' 후 다시 누르면 더 정확)"
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
    wi = 0  # 순차 정렬: 다음 줄은 이전 줄 매칭 지점 이후에서 찾는다
    n_words = len(words)
    matched_n = 0
    for ln in in_lines:
        text = str(ln.get("text", "")).strip()
        target = norm(text)
        cur = {"start": float(ln.get("start", 0)), "end": float(ln.get("end", 0)), "text": text}
        if not target or wi >= n_words:
            out.append(cur)
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
        out.append(cur)
    # 줄끼리 겹치지 않게(다음 줄 시작 - 0.02까지만) 정리해 화면에 두 줄이 겹쳐 뜨는 것 방지.
    for k in range(len(out) - 1):
        if out[k]["end"] > out[k + 1]["start"]:
            out[k]["end"] = round(max(out[k]["start"] + 0.2, out[k + 1]["start"] - 0.02), 2)
    out = _split_long_caption_lines(video_id, clip, cfg, out)
    return jsonify({"lines": out, "matched": matched_n, "total": len(out), "source": used})


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
