
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
  .pv-truth-btn { padding: 8px 12px; border: 1.5px solid #d1d6db; border-radius: 999px; background: #fff; color: #191f28;
    font-size: 12.5px; font-weight: 700; font-family: inherit; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,.06); }
  .pv-truth-btn:hover { border-color: #3182f6; background: #f0f6ff; }
  .pv-truth-ov { position: absolute; inset: 0; z-index: 30; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,.45); }
  .pv-truth-card { background: #fff; border-radius: 18px; padding: 12px; box-shadow: 0 24px 64px rgba(0,0,0,.35); }
  .pv-truth-head { display: flex; align-items: center; gap: 8px; margin: 0 2px 8px; font-size: 13px; color: #4e5968; }
  .pv-truth-head b { color: #191f28; font-size: 14px; }
  .pv-truth-head .pv-x { margin-left: auto; }
  .pv-truth-card img { display: block; border-radius: 12px; }
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
      '      <button type="button" class="pv-truth-btn" title="지금 화면 시점을 실제 렌더와 똑같은 자막 엔진(ffmpeg+ASS)으로 그려서 보여줍니다. 미리보기와 다르면 이쪽이 진짜 결과입니다.">실제 결과</button>' +
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
      '      <button type="button" class="pv-syncbtn" title="한국어 자막의 시작·끝 시간만 실제 발화에 다시 맞추고, 줄 사이 빈 칸을 메웁니다(내용·분할은 유지). 영어 자막은 오른쪽 [영어 자막 만들기] 버튼으로 따로 만듭니다. 한 번 더 누르면 최고 정밀 모델로 처음부터 재분석합니다">싱크 맞추기</button>' +
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
    // ── 실제 결과: 이 시점을 진짜 렌더 엔진으로 한 장 그려 겹쳐 보여준다(미리보기 검증용) ──
    const truthBtn = $('.pv-truth-btn');
    truthBtn.addEventListener('click', async () => {
      const t = video.currentTime;
      truthBtn.disabled = true; truthBtn.textContent = '그리는 중…';
      const url = '/media/' + VIDEO_ID + '/truth/' + idx + '.jpg?t=' + t.toFixed(2) + '&_=' + Date.now();
      let r = null;
      try { r = await fetch(url); } catch (e) { r = null; }
      truthBtn.disabled = false; truthBtn.textContent = '실제 결과';
      if (!r || !r.ok) {
        const d = r ? await r.json().catch(() => ({})) : {};
        alert('실제 결과를 못 그렸어요: ' + (d.error || '네트워크 오류')); return;
      }
      const blob = await r.blob();
      const ov = document.createElement('div');
      ov.className = 'pv-truth-ov';
      ov.innerHTML = '<div class="pv-truth-card"><div class="pv-truth-head"><b>실제 결과</b> <span>' + t.toFixed(1) + '초 · 렌더 엔진이 그린 그대로</span><button class="pv-x">&times;</button></div><img></div>';
      ov.querySelector('img').src = URL.createObjectURL(blob);
      ov.querySelector('img').style.height = H + 'px';
      const close = () => { ov.remove(); };
      ov.querySelector('.pv-x').addEventListener('click', close);
      ov.addEventListener('click', (e) => { if (e.target === ov) close(); });
      back.appendChild(ov);
    });
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
      // 서버가 빈 줄을 지우거나 긴 줄을 쪼개면 줄 수가 달라진다 — 그럴 땐 목록을 통째로
      // 다시 그린다(예전엔 순서대로 시간만 덮어써서, 줄 수가 달라지면 아래 줄들의 시간이
      // 한 칸씩 밀려 엉뚱한 자막에 박혔다).
      const rows = [...capRowsBox.querySelectorAll('.pv-caprow')];
      if (j.lines.length !== rows.length) {
        capRowsBox.innerHTML = '';
        j.lines.forEach((ln) => addCapRow(ln.start - C.start, ln.end - C.start, ln.text));
      } else {
        j.lines.forEach((ln, i) => {
          rows[i].querySelector('.pv-cap-start').value = (ln.start - C.start).toFixed(1);
          rows[i].querySelector('.pv-cap-end').value = (ln.end - C.start).toFixed(1);
          rows[i].querySelector('.pv-cap-text').value = ln.text;
        });
      }
      capsDirty = true;
      // 싱크 맞추기는 '줄의 시작·끝 시간'만 다시 맞춘다 — 파란 강조(카라오케)는 켜지 않는다.
      // (사용자 요청 2026-09-05: 흰 자막이 그대로 떠 있어야 하고, 파란색은 '파란 강조' 버튼으로만.)
      capSec.classList.remove('hidden');
      // 영어 자막은 여기서 만들지 않는다(2026-09-21 요청: "싱크 맞추기 눌렀다고 영어 자막을
      // 넣지는 말고, 한글·영어를 따로 고를 수 있게"). 영어가 필요하면 바로 옆 '영어 자막
      // 만들기' 버튼으로 따로 만든다 — 싱크도 그만큼 빨리 끝난다.
      const tidy = [];
      if (j.dropped) tidy.push('빈 줄 ' + j.dropped + '개 삭제');
      if (j.filled) tidy.push('빈 칸 ' + j.filled + '개 메움');
      syncBtn.textContent = '✓ ' + j.matched + '/' + j.total + '줄 맞춤'
        + (tidy.length ? ' · ' + tidy.join(', ') : '') + ' (저장 필요)';
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
      if (card) { const h = card.querySelector('h3.title'); if (h) h.textContent = (h.dataset.prefix || '') + chosenTitle; }
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
