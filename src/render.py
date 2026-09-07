"""ffmpeg로 클립 컷 -> 9:16 변환(+Ken Burns) -> 무음 제거 -> 자막 번인까지 한 번에 처리"""
from __future__ import annotations

import json
import subprocess
import traceback
from pathlib import Path

from src.captions import build_ass_for_clip
from src.highlights import Clip
from src.transcribe import Segment


def _probe_fps(video_path: Path) -> float:
    """소스 영상의 실제 프레임레이트를 가져온다. zoompan에 하드코딩된 fps를 쓰면
    (예: 25) 실제 소스가 60fps일 때 영상 길이가 fps 비율만큼(60/25=2.4배) 늘어나버리는
    버그가 생기므로, 반드시 소스에 맞는 fps를 zoompan에 넘겨야 한다."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
         str(video_path)],
        capture_output=True, text=True,
    )
    raw = proc.stdout.strip()
    try:
        if "/" in raw:
            num, den = raw.split("/")
            return float(num) / float(den)
        return float(raw)
    except (ValueError, ZeroDivisionError):
        return 30.0  # 안전한 기본값


def _probe_resolution(video_path: Path) -> tuple[int, int]:
    """소스 영상의 실제 가로x세로 픽셀 크기를 가져온다 (원본 비율을 유지한 채 둥근 박스를
    만들려면 이 비율을 알아야 크롭/확대 없이 정확히 맞출 수 있다)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        w_str, h_str = proc.stdout.strip().split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError):
        return 1920, 1080  # 안전한 기본값 (16:9)


def _probe_display_resolution(video_path: Path) -> tuple[int, int]:
    """실제로 화면에 보이는 가로x세로를 반환한다(회전 메타데이터 반영).

    스마트폰 세로 촬영 영상은 흔히 센서 그대로(가로 픽셀, 예: 1280x720)로 저장하고
    90/270도 회전 태그로 재생기가 세로로 돌려 보여준다(특히 iOS). ffmpeg CLI는 기본
    -autorotate가 켜져 있어 디코딩 단계에서 프레임을 실제로 돌리는데, 우리 필터 체인의
    목표 해상도를 '회전 반영 전' width/height로 계산하면 이미 세로로 돌아간 프레임을
    가로 캔버스에 욱여넣어 영상이 찌그러진다. 회전이 90/270이면 width/height를 바꿔 반환."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json",
         str(video_path)],
        capture_output=True, text=True,
    )
    w, h, rot = 1920, 1080, 0
    try:
        stream = (json.loads(proc.stdout).get("streams") or [{}])[0]
        w, h = int(stream["width"]), int(stream["height"])
        tag = (stream.get("tags") or {}).get("rotate")
        if tag is not None:
            rot = int(tag) % 360
        else:
            for sd in stream.get("side_data_list") or []:
                if "rotation" in sd:
                    rot = int(round(float(sd["rotation"]))) % 360
                    break
    except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        pass
    if rot in (90, 270):
        w, h = h, w
    return w, h


def _detect_silences(
    video_path: Path,
    start: float,
    end: float,
    threshold_db: float,
    min_duration_sec: float,
) -> list[tuple[float, float]]:
    """clip 구간(start~end, 원본 기준) 안에서 무음 구간을 찾아 (클립 내부 상대시작, 상대끝) 리스트로 반환."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner",
        "-ss", str(start), "-to", str(end), "-i", str(video_path),
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration_sec}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stderr = proc.stderr

    silences: list[tuple[float, float]] = []
    silence_start = None
    for line in stderr.splitlines():
        if "silence_start" in line:
            try:
                silence_start = float(line.split("silence_start:")[1].strip().split()[0])
            except (IndexError, ValueError):
                silence_start = None
        elif "silence_end" in line and silence_start is not None:
            try:
                part = line.split("silence_end:")[1].strip().split("|")[0].strip()
                silences.append((silence_start, float(part)))
            except (IndexError, ValueError):
                pass
            silence_start = None
    return silences


def _build_keep_segments(
    clip_duration: float, silences: list[tuple[float, float]], min_gap_sec: float = 0.15
) -> list[tuple[float, float]]:
    """무음 구간을 제외한, 실제로 남길 구간들(클립 내부 상대시간)을 계산.
    무음을 완전히 0으로 자르지 않고 min_gap_sec만큼만 남겨 자연스러운 호흡을 유지."""
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in silences:
        keep_until = s + min_gap_sec / 2 if e - s > min_gap_sec else e
        if keep_until > cursor:
            keep.append((cursor, keep_until))
        cursor = max(cursor, e - min_gap_sec / 2)
    if cursor < clip_duration:
        keep.append((cursor, clip_duration))
    return keep if keep else [(0.0, clip_duration)]


def _intersect_intervals(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """두 '남길 구간' 목록의 교집합(둘 다 남기라고 한 구간만 남긴다). 짧은 파편은 버린다."""
    out: list[tuple[float, float]] = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e - s > 0.05:
                out.append((s, e))
    out.sort()
    return out


def _combine_keep(
    duration: float,
    clip_start: float,
    user_keep_abs: list | None,
    silence_keep_rel: list[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    """사용자 분할·삭제(keep_ranges, 절대초)와 무음 제거(상대초)를 합쳐 최종 '남길 구간'(상대초)을 만든다.
    전체(0~duration) 한 조각이면 None을 반환해 select 필터 없는 빠른 경로를 탄다."""
    if user_keep_abs:
        user_rel = [
            (max(0.0, float(s) - clip_start), min(duration, float(e) - clip_start))
            for s, e in user_keep_abs
        ]
        user_rel = sorted((s, e) for s, e in user_rel if e - s > 0.05)
        if not user_rel:
            user_rel = [(0.0, duration)]
    else:
        user_rel = [(0.0, duration)]

    combined = user_rel if silence_keep_rel is None else _intersect_intervals(user_rel, silence_keep_rel)
    if not combined:
        return None
    if len(combined) == 1 and combined[0][0] <= 0.01 and combined[0][1] >= duration - 0.01:
        return None  # 통째로 남김 = 자를 것 없음(빠른 경로)
    return combined


def _vertical_transform(background_mode: str, resolution: tuple[int, int], pad_color: str = "white") -> str:
    w, h = resolution
    if background_mode == "crop":
        return f"scale={w}:-2,crop={w}:{h}"
    if background_mode == "pad":
        # 단색 여백 + 중앙 원본 (설교자 화면이 안 잘리고, 원본 자체 자막이 블러로 비치지 않아 산만하지 않음)
        return (
            f"scale={w}:-2:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={pad_color}"
        )
    # blur 배경 + 중앙 원본 (설교자 화면이 안 잘리도록)
    return (
        f"split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},gblur=sigma=20[bgblur];"
        f"[fg]scale={w}:-2:force_original_aspect_ratio=decrease[fgscaled];"
        f"[bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2"
    )


def _rounded_mask_expr(w: int, h: int, r: int) -> str:
    """geq 필터용 마스크 표현식: 네 모서리 RxR 영역에서 라운딩 원 밖이면 검정(0), 나머지는 흰색(255)."""
    corners = [
        f"lte(X,{r})*lte(Y,{r})*gt(pow({r}-X,2)+pow({r}-Y,2),pow({r},2))",
        f"lte({w}-X,{r})*lte(Y,{r})*gt(pow({r}-({w}-X),2)+pow({r}-Y,2),pow({r},2))",
        f"lte(X,{r})*lte({h}-Y,{r})*gt(pow({r}-X,2)+pow({r}-({h}-Y),2),pow({r},2))",
        f"lte({w}-X,{r})*lte({h}-Y,{r})*gt(pow({r}-({w}-X),2)+pow({r}-({h}-Y),2),pow({r},2))",
    ]
    expr = "255"
    for c in reversed(corners):
        expr = f"if({c},0,{expr})"
    return expr


_MASK_CACHE_DIR = Path("assets/cache")


def _get_or_create_rounded_mask(vbw: int, vbh: int, r: int) -> Path:
    """둥근 모서리 마스크 이미지를 캐시해서 재사용한다.

    geq는 픽셀마다 수식을 계산해야 해서 영상 전체 프레임에 매번 적용하면 극도로 느리다
    (30초 클립 하나에 몇 분씩 걸릴 정도). 그래서 이 마스크를 '정지 이미지 한 장'에만 한 번
    geq로 구워두고, 실제 영상에는 훨씬 가벼운 alphamerge로 그 마스크를 입힌다.
    """
    _MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = _MASK_CACHE_DIR / f"rounded_mask_{vbw}x{vbh}_r{r}.png"
    if mask_path.exists():
        return mask_path

    mask_expr = _rounded_mask_expr(vbw, vbh, r)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=white:s={vbw}x{vbh}",
        "-frames:v", "1",
        "-vf", f"geq=lum='{mask_expr}'",
        # -pix_fmt gray 없으면 PNG가 RGB로 저장되며 YUV<->RGB 변환 과정에서 흰색(255)이
        # 178로 떨어진다. 그러면 alphamerge 시 영상이 70%만 불투명해져 흰 배경이 비쳐
        # 화면 전체가 뿌옇게 보인다("불투명 박스 올린 것 같은" 현상). gray로 저장해야 255 유지.
        "-pix_fmt", "gray",
        str(mask_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"둥근 모서리 마스크 생성 실패:\n{proc.stderr[-2000:]}")
    return mask_path


def _probe_audio_params(video_path: Path) -> tuple[int, int]:
    """(sample_rate, channels)를 가져온다. 아웃트로 concat(-c copy)은 오디오 샘플레이트가
    본 클립과 정확히 같아야 한다 — 다르면 아웃트로 오디오 프레임이 본 클립 타임베이스로
    잘못 해석되어 오디오 트랙만 몇 초씩 길어지고, 플레이어가 남은 오디오 동안 마지막
    프레임(로고)을 정지 표시해 '로고가 3초가 아니라 10초씩 뜨는' 실측 사고가 났다
    (본 클립 44100Hz vs 아웃트로 48000Hz, 2026-09-03)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        sr, ch = proc.stdout.split()
        return int(sr), int(ch)
    except (ValueError, IndexError):
        return 44100, 2  # 유튜브 오디오의 흔한 기본값


def _get_or_create_outro_segment(
    image_path: Path, duration_sec: float, resolution: tuple[int, int], fps: float, encoder: str,
    sample_rate: int, channels: int,
) -> Path:
    """정지 이미지 + 무음으로 된 아웃트로 영상을 캐시해서 재사용한다.

    본 클립과 concat demuxer(-c copy, 재인코딩 없음)로 이어 붙이려면 코덱/해상도/fps/오디오
    샘플레이트·채널까지 정확히 같아야 한다. fps·샘플레이트는 소스 영상마다 달라서 전역
    상수로 캐시할 수 없다 — 파라미터 조합별로 별도 캐시 파일을 둔다."""
    _MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    w, h = resolution
    codec = "libx264" if encoder != "h264_qsv" else "h264_qsv"
    fps_key = f"{fps:.3f}".replace(".", "_")
    seg_path = _MASK_CACHE_DIR / (
        f"outro_{image_path.stem}_{w}x{h}_{fps_key}fps_{sample_rate}hz{channels}ch_{codec}.mp4"
    )
    if seg_path.exists() and seg_path.stat().st_mtime >= image_path.stat().st_mtime:
        return seg_path

    video_args = (
        ["-c:v", "h264_qsv", "-global_quality", "23", "-preset", "veryfast"]
        if codec == "h264_qsv"
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    )
    ch_layout = "mono" if channels == 1 else "stereo"
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-i", str(image_path),
        "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl={ch_layout}",
        "-t", str(duration_sec),
        # 이미지 비율을 유지한 채 해상도에 맞추고 남는 부분은 흰 패딩(레터박스).
        # 예전 scale={w}:{h} 강제 스케일은 세로 로고(1080x1920)를 16:9 업로드 찬양에
        # 붙일 때 찌그러뜨렸다(그래서 찬양 아웃트로를 끄는 임시조치가 있었음 — 이제
        # 비율이 달라도 안전하므로 켤 수 있다). 로고 배경이 흰색이라 패딩도 흰색.
        "-vf", (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=white,fps={fps}"
        ),
        "-pix_fmt", "yuv420p",
        *video_args,
        "-c:a", "aac", "-b:a", "192k", "-ar", str(sample_rate), "-ac", str(channels),
        "-shortest",
        str(seg_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and codec == "h264_qsv":
        fallback = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        cmd = cmd[: cmd.index("-c:v")] + fallback + cmd[cmd.index("-c:a") :]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"아웃트로 세그먼트 생성 실패:\n{proc.stderr[-2000:]}")
    return seg_path


def _append_outro(output_path: Path, outro_path: Path) -> None:
    """concat demuxer로 본 클립 끝에 아웃트로를 이어 붙인다(-c copy: 재인코딩 없이 빠름).

    실패해도 본 렌더는 이미 output_path에 완성돼 있으므로, 아웃트로만 못 붙이고
    원본 클립 그대로 두는 쪽이 렌더 전체를 실패시키는 것보다 안전하다(호출자가 로그만 남김)."""
    list_path = output_path.with_suffix(".concat.txt")
    tmp_path = output_path.with_suffix(".withoutro.mp4")
    list_path.write_text(
        f"file '{output_path.resolve().as_posix()}'\nfile '{outro_path.resolve().as_posix()}'\n",
        encoding="utf-8",
    )
    try:
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy", str(tmp_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp_path.exists():
            raise RuntimeError(f"아웃트로 이어붙이기 실패:\n{proc.stderr[-2000:]}")
        tmp_path.replace(output_path)
    finally:
        list_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)  # replace 성공 시 이미 없음; 실패 시 잔여물 정리


def _apply_speed(output_path: Path, speed: float, encoder: str, fps: float) -> None:
    """완성본(자막 포함)에 배속을 후처리로 적용한다(영상 setpts + 오디오 atempo, 음정 유지).

    자막이 이미 프레임에 구워진 뒤라 영상과 함께 자연히 배속된다 — 자막 시각을 따로
    계산할 필요가 없어 싱크가 어긋날 수 없다. fps 필터로 CFR을 유지해 이후의 아웃트로
    concat(-c copy, 파라미터 일치 필수)도 안전하다."""
    speed = min(2.0, max(1.0, float(speed)))
    if abs(speed - 1.0) < 0.01:
        return
    tmp = output_path.with_suffix(".spd.mp4")
    video_args = (
        ["-c:v", "h264_qsv", "-global_quality", "23", "-preset", "veryfast"]
        if encoder == "h264_qsv"
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    )
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(output_path),
        "-vf", f"setpts=PTS/{speed:.4f},fps={fps}",
        "-af", f"atempo={speed:.4f}",
        *video_args,
        "-c:a", "aac", "-b:a", "192k",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 and encoder == "h264_qsv":
            cmd = cmd[: cmd.index("-c:v")] + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] + cmd[cmd.index("-c:a") :]
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp.exists():
            raise RuntimeError(f"배속 적용 실패:\n{proc.stderr[-2000:]}")
        tmp.replace(output_path)
    finally:
        tmp.unlink(missing_ok=True)


def _add_sfx(output_path: Path, times: list[float], sfx_cfg: dict) -> None:
    """효과음(whoosh)을 지정 시각에 합성해 본 클립 오디오 아래로 믹싱한다(외부 파일 없음).

    게시물의 '효과음은 받는 게 아니라 만든다'처럼 ffmpeg로 직접 합성한다: 핑크노이즈를
    밴드로 걸러 짧게 페이드인/아웃한 '스위시(whoosh)'를 각 시각에 adelay로 배치하고
    amix(normalize=0)로 원 오디오와 섞는다. 비디오는 재인코딩 없이 복사(-c:v copy).

    실패해도 본 렌더는 이미 완성돼 있으므로 호출자가 로그만 남기고 넘어간다."""
    times = sorted({round(max(0.0, t), 2) for t in times})
    if not times:
        return
    vol = float(sfx_cfg.get("volume", 0.35) or 0.35)
    dur = float(sfx_cfg.get("whoosh_dur_sec", 0.45) or 0.45)
    parts = ["[0:a]aformat=sample_rates=48000:channel_layouts=stereo[base]"]
    labels = ["[base]"]
    for i, t in enumerate(times):
        ms = int(round(t * 1000))
        # 핑크노이즈 → 밴드패스(700~6000) → 페이드인/아웃 → 볼륨 → 지연 배치.
        parts.append(
            f"anoisesrc=d={dur:.2f}:c=pink:a=0.9,highpass=f=700,lowpass=f=6000,"
            f"afade=t=in:d=0.08,afade=t=out:st={max(0.0, dur-0.25):.2f}:d=0.25,"
            f"volume={vol:.2f},aformat=sample_rates=48000:channel_layouts=stereo,"
            f"adelay={ms}|{ms}[s{i}]"
        )
        labels.append(f"[s{i}]")
    n = len(labels)
    parts.append(f"{''.join(labels)}amix=inputs={n}:normalize=0:dropout_transition=0[aout]")
    filter_complex = ";".join(parts)
    tmp_path = output_path.with_suffix(".sfx.mp4")
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(output_path),
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp_path.exists():
            raise RuntimeError(f"효과음 믹싱 실패:\n{proc.stderr[-2000:]}")
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def _mix_bgm(output_path: Path, bgm_path: Path, volume: float, offset: float = 0.0) -> None:
    """스튜디오에서 업로드한 배경 음악을 결과물 오디오 아래로 믹싱한다(외부 파일 있음).

    _add_sfx와 같은 ffmpeg amix 패턴: 클립 길이(이미 완성된 output_path의 실제 길이 —
    무음 제거/배속을 다 반영한 최종 길이)에 맞춰 반복재생 후 자르고, 시작/끝을 짧게
    페이드해 튀지 않게 한다. 비디오는 재인코딩하지 않는다(-c:v copy).
    실패해도 본 렌더는 이미 완성돼 있으므로 호출자가 로그만 남기고 넘어간다.

    offset: 스튜디오에서 BGM 막대를 끌어 정한, 배경음악 파일 안에서 재생을 시작할 지점(초).
    입력을 -stream_loop -1로 무한 반복시키므로 그 반복 스트림 위에서 atrim의 시작점만
    옮기면 된다(음원 길이로 나눠 감쌀 필요 없음 — 반복이 무한이라 임의의 양수 오프셋이
    항상 유효하다). 기본 0이면 기존과 완전히 동일한 동작."""
    dur = _probe_duration(output_path)
    if dur <= 0:
        return
    vol = max(0.0, min(1.0, volume))
    off = max(0.0, offset)
    fade_d = min(1.0, dur / 4)
    filter_complex = (
        f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo[base];"
        f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,"
        f"atrim={off:.3f}:{off + dur:.3f},asetpts=PTS-STARTPTS,volume={vol:.3f},"
        f"afade=t=in:d={fade_d:.2f},afade=t=out:st={max(0.0, dur - fade_d):.2f}:d={fade_d:.2f}[bgm];"
        f"[base][bgm]amix=inputs=2:normalize=0:dropout_transition=0[aout]"
    )
    tmp_path = output_path.with_suffix(".bgm.mp4")
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(output_path),
        "-stream_loop", "-1", "-i", str(bgm_path),
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{dur:.3f}",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp_path.exists():
            raise RuntimeError(f"배경 음악 믹싱 실패:\n{proc.stderr[-2000:]}")
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# 소스 하단 크롭 기본값. 렌더(_build_card_filter_complex)와 편집 미리보기(web_app)가
# 서로 다른 기본값을 쓰면 config에 이 키가 없을 때 미리보기와 결과물이 어긋난다 — 반드시
# 이 상수 하나만 참조할 것.
SOURCE_CROP_BOTTOM_PCT_DEFAULT = 0.08


def card_source_video_filter(card: dict, vbw: int, vbh: int, pan_x_expr: str = "") -> str:
    """소스 프레임을 카드 영상 박스 크기로 만드는 crop+scale 필터 체인.

    실제 렌더(_build_card_filter_complex)와 편집 미리보기(web_app.clip_preview_frame)가
    이 함수 하나를 공유한다 — 각자 문자열을 복제하면 fill_mode/크롭 기본값이 조금만
    어긋나도 '편집 화면에서 본 프레임 ≠ 결과물'이 된다.

    pan_x_expr가 주어지면(cover 모드) 중앙 고정 크롭 대신 그 x 표현식으로 크롭을 움직여
    화자 얼굴을 따라간다(facetrack). 미리보기는 pan_x_expr 없이 호출해 중앙 크롭을 보인다."""
    parts = []
    pct = card.get("source_crop_bottom_pct", SOURCE_CROP_BOTTOM_PCT_DEFAULT)
    if pct > 0:
        parts.append(f"crop=iw:ih*{1 - pct}:0:0")
    if card.get("fill_mode", "cover") == "cover":
        # 박스를 꽉 채우도록 확대 후 박스 크기로 잘라낸다. 기본은 중앙(좌우 빈 배경만 트리밍),
        # facetrack이 있으면 x를 시간에 따라 움직여 화자를 따라간다.
        if pan_x_expr:
            parts.append(
                f"scale={vbw}:{vbh}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={vbw}:{vbh}:x='{pan_x_expr}':y=0"
            )
        else:
            parts.append(
                f"scale={vbw}:{vbh}:force_original_aspect_ratio=increase:flags=lanczos,crop={vbw}:{vbh}"
            )
    else:
        # fit: 원본 가로세로 비율 그대로(옆을 안 자름). vbh가 이미 그 비율로 계산됨.
        parts.append(f"scale={vbw}:{vbh}:flags=lanczos")
    return ",".join(parts)


def _compute_card_video_box_height(card: dict, source_resolution: tuple[int, int]) -> int:
    """영상 박스 높이를 결정한다.

    - fill_mode == "cover"(기본): 세로가 확 큰 소스(16:9)를 카드에 작게 letterbox하면
      화면 절반이 빈 흰 여백이 되어 "유튜브 쇼츠 같지 않고 하단이 텅 빈" 문제가 생긴다.
      그래서 config의 video_box_aspect(가로:세로 비, 예 [4,3])로 박스를 크게 잡고, 소스는
      좌우의 빈 배경만 살짝 잘라(cover) 박스를 꽉 채운다. 설교자는 화면 중앙에 있으므로
      좌우를 조금 잘라도 잘리지 않는다.
    - fill_mode == "fit": 원본 가로세로 비율을 그대로 유지(옛 동작, 옆을 안 자름).
    """
    vbw = card["video_box_width"]
    fill_mode = card.get("fill_mode", "cover")
    if fill_mode == "cover":
        aspect = card.get("video_box_aspect", [4, 3])
        return round(vbw * aspect[1] / aspect[0])
    source_crop_bottom_pct = card.get("source_crop_bottom_pct", SOURCE_CROP_BOTTOM_PCT_DEFAULT)
    src_w, src_h = source_resolution
    effective_src_h = src_h * (1 - source_crop_bottom_pct)
    return round(vbw * effective_src_h / src_w)


def _compute_card_video_box_y(card: dict, resolution: tuple[int, int], vbh: int) -> int:
    """영상 박스를 제목 영역과 캡션 영역을 제외한 나머지 중간 공간에 수직 중앙 정렬한다.
    (원본 비율을 유지하면 박스 높이가 짧아지는데, 고정된 y좌표에 그대로 두면 제목 바로
    아래 붙어서 아래쪽에 여백만 커 보인다 — 사용자 요청으로 중앙 정렬로 변경.)"""
    _, h = resolution
    title_h = card.get("title_area_height", 200)
    caption_h = card.get("caption_area_height", 380)
    available = h - title_h - caption_h
    return title_h + max(0, (available - vbh) // 2)


def _build_card_filter_complex(
    render_cfg: dict,
    captions_cfg: dict,
    resolution: tuple[int, int],
    clip_duration: float,
    select_expr: str | None,
    ass_path_ff: str,
    font_dir: str,
    source_fps: float,
    vbh: int,
    vby: int,
    speedup: tuple[float, float] | None = None,
    pan_x_expr: str = "",
) -> tuple[str, str]:
    """카드형 레이아웃(흰 배경 + 둥근 모서리 영상 박스): filter_complex 문자열과
    최종 비디오 출력 라벨을 반환한다.
    blur/crop/pad와 달리 흰 배경을 합성해야 해서 단순 -vf 체인이 아니라 filter_complex가 필요하다.

    color 소스의 fps를 소스 영상과 반드시 맞춰야 한다 (fps=25 기본값과 실제 소스 fps가
    다르면 overlay가 두 입력의 프레임레이트를 맞추는 과정에서 길이가 틀어질 수 있음 —
    이미 zoompan에서 같은 이유로 길이가 늘어나는 버그를 겪었으므로 여기서도 명시적으로 맞춘다).

    둥근 모서리는 매 프레임 geq를 돌리지 않고, 미리 구워둔 마스크 이미지를 alphamerge로
    입힌다 (마스크는 입력 1번, `-loop 1 -i mask.png`로 별도 추가되어야 함 — render_clip 참고).

    vbh/vby(영상 박스 높이/y좌표)는 호출자가 미리 계산해서 넘긴다 (build_ass_for_clip의
    캡션 위치 계산과 같은 값을 써야 하므로 한 곳에서만 계산한다).
    """
    w, h = resolution
    card = render_cfg["card_layout"]
    vbw = card["video_box_width"]
    vbx = (w - vbw) // 2
    video_parts = []
    if select_expr:
        video_parts.append(f"select='{select_expr}'")
        video_parts.append("setpts=N/FRAME_RATE/TB")
    elif speedup:
        # 훅 배속: 초반 f초를 k배로 감는 시간 워프(초 단위 T). setpts는 select와 동시엔 안
        # 온다(warp는 keep_segments 없을 때만 활성). captions.warp_time과 동일한 식이어야 함.
        f, k = speedup
        video_parts.append(f"setpts='(if(lt(T,{f}),T/{k},{f}/{k}+(T-{f})))/TB'")
    video_parts.append(card_source_video_filter(card, vbw, vbh, pan_x_expr))
    video_chain = ",".join(video_parts)

    filter_complex = (
        f"[0:v]{video_chain}[rgbsrc];"
        f"[1:v]format=gray[maskgray];"
        f"[rgbsrc][maskgray]alphamerge[rounded];"
        f"color=white:s={w}x{h}:d={clip_duration}:r={source_fps}[cardbg];"
        f"[cardbg][rounded]overlay={vbx}:{vby}[composited]"
    )
    if captions_cfg.get("enabled", True):
        filter_complex += f";[composited]subtitles='{ass_path_ff}':fontsdir='{font_dir}'[vout]"
        return filter_complex, "[vout]"
    return filter_complex, "[composited]"


def _ken_burns_filter(resolution: tuple[int, int], zoom_per_sec: float, fps: float) -> str:
    """이미 9:16으로 합성된 최종 프레임에 미세한 줌인 효과를 사후 적용한다.

    주의: crop 필터는 w/h를 프레임마다(eval=frame) 다시 계산하는 옵션 자체가 없어
    시간 가변 줌에 쓸 수 없다 (x/y만 프레임마다 재계산 가능). 그래서 zoompan을 쓰되,
    d=1(입력 프레임 하나당 출력 프레임도 하나)로 못박고 zoompan 내부에 유지되는
    자기참조 변수 'zoom'을 프레임마다 조금씩 늘리는 방식으로, 프레임 수/길이를
    전혀 바꾸지 않으면서 부드러운 줌인을 만든다. (d에 총 프레임 수를 넣으면
    입력 프레임 하나당 그만큼 프레임을 복제해버려 영상이 크게 길어지는 버그가 남.)"""
    w, h = resolution
    increment = zoom_per_sec / fps
    zoom_expr = f"zoom+{increment}"
    return (
        f"zoompan=z='{zoom_expr}':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}"
    )


def _build_video_filter(
    render_cfg: dict,
    captions_cfg: dict,
    resolution: tuple[int, int],
    select_expr: str | None,
    ass_path_ff: str,
    font_dir: str,
    source_fps: float,
) -> str:
    parts = []
    if select_expr:
        parts.append(f"select='{select_expr}'")
        parts.append("setpts=N/FRAME_RATE/TB")
    parts.append(_vertical_transform(
        render_cfg.get("background_mode", "blur"), resolution, render_cfg.get("pad_color", "white")
    ))
    ken_burns = render_cfg.get("ken_burns", {})
    if ken_burns.get("enabled", False):
        # 9:16으로 합성된 최종 프레임에 대해 줌인 (blur/crop 어느 모드든 동일하게 적용 가능)
        parts.append(_ken_burns_filter(resolution, ken_burns.get("zoom_per_sec", 0.008), source_fps))
    if captions_cfg.get("enabled", True):
        parts.append(f"subtitles='{ass_path_ff}':fontsdir='{font_dir}'")
    return ",".join(parts)


def render_clip(
    video_path: Path,
    segments: list[Segment],
    clip: Clip,
    output_path: Path,
    render_cfg: dict,
    captions_cfg: dict,
) -> None:
    resolution = tuple(render_cfg.get("resolution", [1080, 1920]))
    duration = clip.end - clip.start

    # 편집기에서 클립별로 화면모드(풀 화면=fit / 화면 확대=cover)를 고른 경우 config 기본값을 덮어쓴다.
    clip_fill = getattr(clip, "fill_mode", "") or ""
    if clip_fill:
        _card = {**render_cfg.get("card_layout", {}), "fill_mode": clip_fill}
        render_cfg = {**render_cfg, "card_layout": _card}
    # 얼굴 추적 리프레이밍은 cover 크롭에서만 의미가 있다(fit은 옆을 안 자름) → 켜지면 cover 강제.
    if render_cfg.get("facetrack"):
        _card = {**render_cfg.get("card_layout", {}), "fill_mode": "cover"}
        render_cfg = {**render_cfg, "card_layout": _card}

    silences: list[tuple[float, float]] = []
    if render_cfg.get("remove_silence", True):
        silences = _detect_silences(
            video_path,
            clip.start,
            clip.end,
            render_cfg.get("silence_threshold_db", -35),
            render_cfg.get("silence_min_duration_sec", 0.6),
        )
    # 자막 생성보다 먼저 계산해야 한다: 무음 제거로 영상 타임라인이 압축되는데,
    # 자막 타임스탬프도 똑같이 압축해서 리매핑하지 않으면 뒤로 갈수록 자막이 밀린다.
    silence_keep = _build_keep_segments(duration, silences) if silences else None
    # 확인 팝업에서 분할·삭제한 구간(keep_ranges)이 있으면 무음 제거와 합쳐 최종 남길 구간을 만든다.
    keep_segments = _combine_keep(
        duration, clip.start, getattr(clip, "keep_ranges", None) or None, silence_keep
    )

    is_card = render_cfg.get("background_mode", "blur") == "card"
    card_layout = render_cfg.get("card_layout") if is_card else None
    if is_card:
        # 영상 박스 높이는 원본 비율을 그대로 유지해서 계산 (옆을 잘라 확대하지 않음).
        # 자막 위치 계산(build_ass_for_clip)과 실제 필터(_build_card_filter_complex)가
        # 반드시 같은 값을 써야 캡션이 영상 박스 바로 아래에 정확히 붙는다.
        source_resolution = _probe_resolution(video_path)
        vbh = _compute_card_video_box_height(card_layout, source_resolution)
        vby = _compute_card_video_box_y(card_layout, resolution, vbh)
        card_layout = {**card_layout, "video_box_height": vbh, "video_box_y": vby}

    # 자막 싱크 교정용 무음 지도: Whisper가 쉼(pause)을 다음 단어 발화 시간에 흡수해
    # 자막이 실제 말보다 1~2초 먼저 뜨는 문제(실측)를, 실제 오디오의 무음 구간으로
    # 단어 start를 교정해 잡는다. remove_silence(-35dB/1.2s)와 별개로, 짧은 쉼까지
    # 잡도록 더 민감한 값(-32dB/0.3s)을 쓴다. ffmpeg 한 번이라 클립당 2~3초면 끝난다.
    voice_silences: list[tuple[float, float]] | None = None
    if captions_cfg.get("enabled", True):
        # 편집 자막(caption_overrides) 경로에도 적용한다: 카라오케 단어 시각은 어느 경로든
        # 전사 단어를 쓰므로, 여기서 빼면 편집 클립만 "자막이 말보다 빠른" 문제가 남는다
        # (실제 신고된 잔존 싱크 문제의 원인 중 하나).
        voice_silences = _detect_silences(video_path, clip.start, clip.end, -32.0, 0.3)

    # 훅 배속: 초반 first_sec초를 factor배로 빠르게(card + 분할/무음제거 안 한 클립에서만).
    hs_cfg = render_cfg.get("hook_speedup") or {}
    _hs_first = float(hs_cfg.get("first_sec", 0) or 0)
    _hs_factor = float(hs_cfg.get("factor", 1.0) or 1.0)
    warp_active = bool(
        is_card and hs_cfg.get("enabled") and _hs_factor > 1.0 and _hs_first > 0
        and not keep_segments and duration > _hs_first + 0.5
    )
    hook_speedup = (_hs_first, _hs_factor) if warp_active else None
    # 배속 후 실제 본문 길이(warp_time(duration))
    warped_dur = (_hs_first / _hs_factor + (duration - _hs_first)) if warp_active else duration

    ass_path = output_path.with_suffix(".ass")
    ass_content = build_ass_for_clip(
        segments=segments,
        clip_start=clip.start,
        clip_end=clip.end,
        config_captions=captions_cfg,
        config_hook=render_cfg.get("hook", {"enabled": False}),
        resolution=resolution,
        hook_text=clip.title,
        card_layout=card_layout,
        keep_segments=keep_segments,
        title_offset_x=clip.title_offset_x,
        title_offset_y=clip.title_offset_y,
        caption_offset_x=clip.caption_offset_x,
        caption_offset_y=clip.caption_offset_y,
        caption_overrides=getattr(clip, "caption_overrides", None) or None,
        font_style={
            "title_font": getattr(clip, "title_font", "") or "",
            "title_size": getattr(clip, "title_size", 0) or 0,
            "title_align": getattr(clip, "title_align", "") or "",
            "title_spacing": getattr(clip, "title_spacing", 0.0) or 0.0,
            "caption_font": getattr(clip, "caption_font", "") or "",
            "caption_size": getattr(clip, "caption_size", 0) or 0,
            "caption_size_en": getattr(clip, "caption_size_en", 0) or 0,
            "caption_align": getattr(clip, "caption_align", "") or "",
            "caption_spacing": getattr(clip, "caption_spacing", 0.0) or 0.0,
            "caption_text_color": getattr(clip, "caption_text_color", "") or "",
            "caption_box": bool(getattr(clip, "caption_box", False)),
            "caption_box_color": getattr(clip, "caption_box_color", "") or "#000000",
            "caption_box_opacity": getattr(clip, "caption_box_opacity", 0.55),
            "caption_box_radius": getattr(clip, "caption_box_radius", 40.0),
            "caption_box_width_pct": getattr(clip, "caption_box_width_pct", 28.0),
            "caption_box_height_pct": getattr(clip, "caption_box_height_pct", 28.0),
            "caption_box_offset_x": getattr(clip, "caption_box_offset_x", 0.0),
            "caption_box_offset_y": getattr(clip, "caption_box_offset_y", 0.0),
            "caption_bold": bool(getattr(clip, "caption_bold", True)),
            "caption_italic": bool(getattr(clip, "caption_italic", False)),
            "caption_underline": bool(getattr(clip, "caption_underline", False)),
            "caption_text_opacity": getattr(clip, "caption_text_opacity", 1.0),
            "caption_outline_enabled": bool(getattr(clip, "caption_outline_enabled", True)),
            "caption_outline_color": getattr(clip, "caption_outline_color", "") or "",
            "caption_outline_width": getattr(clip, "caption_outline_width", -1.0),
            "caption_glow": bool(getattr(clip, "caption_glow", False)),
            "caption_glow_color": getattr(clip, "caption_glow_color", "") or "",
        },
        voice_silences=voice_silences,
        hook_speedup=hook_speedup,
        # 자막 형광 강조 대상: 클립의 명시적 강조어(caption_highlights)를 우선하고,
        # 없으면 성경 고유명사(keywords)로 폴백한다. captions.highlight_keywords_enabled로 on/off.
        highlight_keywords=(
            (getattr(clip, "caption_highlights", None) or getattr(clip, "keywords", None) or [])
        ),
        # 이중언어(한글 아래 영어): render_selected(caption_lang="bilingual")일 때만 켜진다.
        # 한국어(caption_overrides)는 그대로 두고 영어 번역 트랙을 같은 줄에 작게 이어붙인다.
        bilingual_overrides=(
            getattr(clip, "caption_overrides_en", None) if render_cfg.get("caption_bilingual") else None
        ),
        free_texts=getattr(clip, "free_texts", None) or None,
    )
    ass_path.write_text(ass_content, encoding="utf-8")

    # 편집기에서 고른 글꼴이 어느 폴더에 있든 libass가 찾도록 모든 폰트를 모은 통합 폴더를 fontsdir로 넘긴다.
    from src.fonts import default_font_dir_for_ass

    font_dir = default_font_dir_for_ass()
    ass_path_ff = str(ass_path).replace("\\", "/").replace(":", "\\:")

    select_expr = None
    audio_filter = None
    if keep_segments:
        select_expr = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in keep_segments)
        audio_filter = f"aselect='{select_expr}',asetpts=N/SR/TB"

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", str(clip.start), "-to", str(clip.end), "-i", str(video_path),
    ]

    # 클립 끝 여운: 마지막 프레임을 정지(tpad clone)시키고 오디오엔 무음(apad)을 덧붙여
    # 문장이 끝나자마자 뚝 끊기는 느낌을 없앤다. 영상·오디오 둘 다 같은 길이만큼 늘려야
    # (-shortest / 기본 종료 조건에서) 여운이 잘리지 않는다.
    end_pad = float(render_cfg.get("end_pad_sec", 0.0) or 0.0)
    vpad_vf = f"tpad=stop_mode=clone:stop_duration={end_pad}" if end_pad > 0 else None
    apad_af = f"apad=pad_dur={end_pad}" if end_pad > 0 else None

    # 끝 소리 페이드아웃: 본 영상이 끝나고 로고(아웃트로)로 넘어갈 때 소리가 뚝 끊기지 않게
    # 마지막 구간을 부드럽게 줄인다(사용자 요청, 2026-09-03). 무음 여운(end_pad) 시작 직전에
    # 걸어서, 말끝 → 페이드 → 무음 여운 → 로고로 자연스럽게 이어진다.
    fade_d = float(render_cfg.get("end_audio_fade_sec", 0.6) or 0.0)
    content_dur = (
        sum(e - s for s, e in keep_segments) if keep_segments else warped_dur
    )
    afade_af = (
        f"afade=t=out:st={max(0.0, content_dur - fade_d):.3f}:d={fade_d:.3f}"
        if fade_d > 0 else None
    )

    def _join_af(base: str | None) -> list[str]:
        chain = ",".join(p for p in (base, afade_af, apad_af) if p)
        return ["-af", chain] if chain else []

    source_fps = _probe_fps(video_path)
    # 얼굴 추적 팬 표현식(cover 크롭용): 켜져 있고 cover일 때만 계산한다. 실패/저검출이면
    # 빈 문자열 → 중앙 크롭 폴백(기존 동작). select(무음 제거)로 시간축이 바뀌면 팬 시각이
    # 어긋나므로 keep_segments가 있으면 적용하지 않는다(안전).
    pan_x_expr = ""
    if is_card and render_cfg.get("facetrack") and card_layout.get("fill_mode", "cover") == "cover" \
            and not keep_segments and not warp_active:
        try:
            from src.facetrack import compute_face_centers, build_crop_x_expr

            src_w, src_h = source_resolution
            bottom_pct = card_layout.get("source_crop_bottom_pct", SOURCE_CROP_BOTTOM_PCT_DEFAULT)
            eff_h = src_h * (1 - bottom_pct)
            vbw_ = card_layout["video_box_width"]
            vbh_ = card_layout["video_box_height"]
            factor = max(vbw_ / src_w, vbh_ / max(1.0, eff_h))
            scaled_w = src_w * factor
            max_x = scaled_w - vbw_
            if max_x > 2:
                centers = compute_face_centers(video_path, clip.start, clip.end)
                pan_x_expr = build_crop_x_expr(centers, scaled_w, vbw_, max_x)
        except Exception:  # noqa: BLE001 - 얼굴 추적 실패는 치명적이지 않음(중앙 크롭 폴백)
            traceback.print_exc()
            pan_x_expr = ""
    if is_card:
        mask_path = _get_or_create_rounded_mask(
            card_layout["video_box_width"], card_layout["video_box_height"], card_layout["corner_radius"]
        )
        cmd += ["-loop", "1", "-i", str(mask_path)]  # 입력 1번: 둥근 모서리 마스크 이미지
        # color 배경 길이도 배속된 본문 길이에 맞춘다(warp 시 duration보다 짧아짐).
        filter_complex, vout_label = _build_card_filter_complex(
            render_cfg, captions_cfg, resolution, warped_dur, select_expr, ass_path_ff, font_dir,
            source_fps, card_layout["video_box_height"], card_layout["video_box_y"],
            speedup=hook_speedup, pan_x_expr=pan_x_expr,
        )
        if vpad_vf:
            filter_complex += f";{vout_label}{vpad_vf}[vpad]"
            vout_label = "[vpad]"
        cmd += ["-filter_complex", filter_complex, "-map", vout_label]
        if warp_active:
            # 오디오도 초반 f초만 k배(atempo, 피치 유지) 후 나머지와 concat → afade/apad 적용.
            f, k = hook_speedup
            a_post = ",".join(p for p in (afade_af, apad_af) if p)
            afx = (
                f"[0:a]atrim=0:{f},asetpts=PTS-STARTPTS,atempo={k}[hsa0];"
                f"[0:a]atrim=start={f},asetpts=PTS-STARTPTS[hsa1];"
                f"[hsa0][hsa1]concat=n=2:v=0:a=1[hsac]"
            )
            filter_complex += ";" + afx
            if a_post:
                filter_complex += f";[hsac]{a_post}[aout]"
                aout_label = "[aout]"
            else:
                aout_label = "[hsac]"
            # filter_complex를 이미 cmd에 넣었으므로 문자열을 갱신해 재지정한다.
            fc_idx = cmd.index("-filter_complex") + 1
            cmd[fc_idx] = filter_complex
            cmd += ["-map", aout_label, "-shortest"]
        else:
            cmd += _join_af(audio_filter)
            # 마스크 입력(-loop 1)이 무한 스트림이라 -shortest 없이는 ffmpeg가 멈출 시점을 몰라
            # 인코딩이 끝나지 않는다. 반드시 필요. (tpad/apad로 영상·오디오 둘 다 +end_pad가 되어
            # -shortest가 여운을 자르지 않는다.)
            cmd += ["-map", "0:a", "-shortest"]
    else:
        video_filter = _build_video_filter(
            render_cfg, captions_cfg, resolution, select_expr, ass_path_ff, font_dir, source_fps
        )
        if vpad_vf:
            video_filter += f",{vpad_vf}"
        cmd += ["-vf", video_filter]
        cmd += _join_af(audio_filter)

    # 인코더: 이 PC엔 Intel Arc iGPU가 있어 h264_qsv 하드웨어 인코딩이 가능하다(실측 인코딩
    # 17~28초 → 수 초). 쇼츠는 플랫폼이 재인코딩하므로 화질 차이는 체감 없음. QSV가 드라이버
    # 문제 등으로 실패하면 자동으로 libx264(소프트웨어)로 다시 인코딩한다.
    encoder = render_cfg.get("video_encoder", "h264_qsv")
    if encoder == "h264_qsv":
        video_args = ["-c:v", "h264_qsv", "-global_quality", "23", "-preset", "veryfast"]
    else:
        # preset slow/crf16은 화질은 최고지만 클립 하나에 1~2분씩 걸려 렌더가 느렸다.
        # crf 20/veryfast면 체감 화질 차이 없이 인코딩이 5~8배 빠르다(속도 우선).
        video_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    tail = ["-c:a", "aac", "-b:a", "192k", str(output_path)]

    proc = subprocess.run(cmd + video_args + tail, capture_output=True, text=True)
    used_encoder = encoder
    if proc.returncode != 0 and encoder == "h264_qsv":
        fallback_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        proc = subprocess.run(cmd + fallback_args + tail, capture_output=True, text=True)
        used_encoder = "libx264"
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 렌더링 실패:\n{proc.stderr[-3000:]}")

    # 배속(선택, 팝업에서 1.0~2.0): 자막까지 구워진 완성본에 후처리로 적용 — 자막도
    # 프레임과 함께 배속되므로 싱크 재계산이 필요 없다. 아웃트로/효과음보다 먼저.
    playback_speed = min(2.0, max(1.0, float(getattr(clip, "playback_speed", 1.0) or 1.0)))
    if abs(playback_speed - 1.0) >= 0.01:
        try:
            _apply_speed(output_path, playback_speed, used_encoder, source_fps)
        except Exception:  # noqa: BLE001 - 배속 실패 시 원속 결과물 유지(로그만)
            traceback.print_exc()
            playback_speed = 1.0

    # 효과음(선택): 본편 완성 후·아웃트로 이전에 whoosh를 합성해 믹싱한다. 도입부(0초)에
    # 하나, 무음 제거로 생긴 컷 경계마다 하나(전환음). 배속 구간이 있으면 컷 시각이 어긋날
    # 수 있어 컷 전환음은 배속이 없을 때만 넣는다. 실패해도 본 렌더는 유지(로그만).
    sfx_cfg = render_cfg.get("sfx") or {}
    if sfx_cfg.get("enabled"):
        sfx_times = [0.0]
        if sfx_cfg.get("whoosh_at_cuts", True) and keep_segments and len(keep_segments) > 1 and not warp_active:
            acc = 0.0
            for s, e in keep_segments[:-1]:
                acc += (e - s)
                sfx_times.append(round(acc / playback_speed, 2))  # 배속 후 시간축 보정
        try:
            _add_sfx(output_path, sfx_times, sfx_cfg)
        except Exception:  # noqa: BLE001 - 효과음 실패는 치명적이지 않음
            traceback.print_exc()

    # 배경 음악(선택, 스튜디오 A2 트랙에서 업로드): 효과음 믹싱 다음·아웃트로 이전.
    bgm = getattr(clip, "bgm", None)
    if bgm and not bgm.get("muted") and bgm.get("filename"):
        bgm_path = video_path.parent / "bgm" / bgm["filename"]
        if bgm_path.exists():
            try:
                _mix_bgm(output_path, bgm_path, float(bgm.get("volume", 0.25) or 0.25), float(bgm.get("offset", 0.0) or 0.0))
            except Exception:  # noqa: BLE001 - 배경 음악 실패는 치명적이지 않음
                traceback.print_exc()

    # 끝에 로고 이미지 아웃트로(사용자 요청, 2026-09-03). 본 클립은 이미 output_path에
    # 완성됐으므로, 아웃트로 붙이기가 실패해도 렌더 전체를 실패시키지 않고 로그만 남긴다.
    outro_cfg = render_cfg.get("outro") or {}
    if outro_cfg.get("enabled"):
        image_path = Path(outro_cfg.get("image_path", ""))
        outro_dur = float(outro_cfg.get("duration_sec", 3) or 0)
        if image_path.exists() and outro_dur > 0:
            try:
                # 방금 렌더된 결과물의 실제 오디오 파라미터에 정확히 맞춘다(-c copy concat 필수 조건).
                sample_rate, channels = _probe_audio_params(output_path)
                outro_seg = _get_or_create_outro_segment(
                    image_path, outro_dur, resolution, source_fps, used_encoder,
                    sample_rate, channels,
                )
                _append_outro(output_path, outro_seg)
            except Exception:  # noqa: BLE001
                traceback.print_exc()


def render_all_clips(
    video_path: Path,
    segments: list[Segment],
    clips: list[Clip],
    output_dir: Path,
    render_cfg: dict,
    captions_cfg: dict,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for i, clip in enumerate(clips, start=1):
        out_path = output_dir / f"short_{i}.mp4"
        render_clip(video_path, segments, clip, out_path, render_cfg, captions_cfg)
        outputs.append(out_path)
    return outputs
