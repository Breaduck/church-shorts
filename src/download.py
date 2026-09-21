"""유튜브 링크 -> 영상 다운로드 (yt-dlp 래퍼)"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yt_dlp


@dataclass
class DownloadResult:
    video_id: str
    title: str
    video_path: Path
    duration_sec: float


def extract_video_id(url: str) -> str:
    # live/: 실시간 스트리밍 다시보기 URL(youtube.com/live/<id>) — 전체 예배 실황이 이 형식
    match = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})", url)
    if not match:
        raise ValueError(f"유튜브 URL에서 video id를 찾을 수 없습니다: {url}")
    return match.group(1)


def _probe_duration_sec(path: Path) -> float:
    import subprocess

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def _has_audio_stream(path: Path) -> bool:
    """파일에 오디오 스트림이 실제로 들어있는지 확인한다. 영상 전용 조각(source.fNNN.mp4)을
    완성본으로 오인해 소스로 쓰면 소리 없는 쇼츠가 나오거나 렌더가 깨지므로 방어에 쓴다."""
    import subprocess

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return bool(proc.stdout.strip())


_FRAGMENT_RE = re.compile(r"\.f\d+\.")  # yt-dlp 포맷별 스트림 조각: source.f298.mp4, source.f140.m4a


def _is_partial(name: str) -> bool:
    """중단·임시·포맷조각 파일인지 판별 (완성 병합본 source.mp4가 아닌 것들)."""
    return (
        ".part" in name or ".ytdl" in name or ".temp" in name
        or name.endswith(".part-Frag") or bool(_FRAGMENT_RE.search(name))
    )


def _cleanup_partials(video_dir: Path) -> None:
    """이전 다운로드가 중단되며 남긴 조각/임시 파일을 지운다. 이게 남아 있으면 재다운로드
    시 병합이 꼬이거나, 완성본으로 오인돼 source.mp4 없이 파이프라인이 진행되는 버그가 난다."""
    for p in video_dir.glob("source.*"):
        if p.name != "source.mp4" and _is_partial(p.name):
            try:
                p.unlink()
            except OSError:
                pass


def _purge_non_final(video_dir: Path) -> None:
    """유효한 source.mp4가 확보된 뒤, 남아있는 다른 source.* (조각/다른 컨테이너)를 정리한다."""
    for p in video_dir.glob("source.*"):
        if p.name != "source.mp4":
            try:
                p.unlink()
            except OSError:
                pass


def find_cached(url: str, output_root: Path) -> Optional[DownloadResult]:
    """이미 받아둔 완성본(source.mp4, 오디오 포함)이 있으면 즉시 반환, 없으면 None."""
    video_id = extract_video_id(url)
    target = output_root / video_id / "source.mp4"
    if target.exists() and target.stat().st_size > 0 and _has_audio_stream(target):
        return DownloadResult(
            video_id=video_id, title=video_id, video_path=target,
            duration_sec=_probe_duration_sec(target),
        )
    return None


def probe_video(url: str, output_root: Path) -> DownloadResult:
    """다운로드 없이 메타데이터(제목/길이)만 몇 초 만에 가져온다.

    하이라이트 '후보 뽑기'는 자막(텍스트)만 있으면 되고 영상 파일은 렌더 때에야 필요하다.
    그래서 분석 크리티컬 패스에서 몇 분짜리 다운로드를 빼고(백그라운드로 돌리고),
    여기서 얻은 길이/제목만으로 분석을 바로 진행한다."""
    video_id = extract_video_id(url)
    with yt_dlp.YoutubeDL({"quiet": True, "noprogress": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return DownloadResult(
        video_id=video_id,
        title=info.get("title", video_id),
        video_path=output_root / video_id / "source.mp4",
        duration_sec=float(info.get("duration") or 0.0),
    )


def download_video(
    url: str, output_root: Path, on_progress: Optional[Callable[[float], None]] = None
) -> DownloadResult:
    """주어진 유튜브 URL의 영상을 output_root/<video_id>/source.mp4 로 저장한다.
    이미 다운로드된 파일이 있으면 재다운로드하지 않고 재사용한다.
    on_progress가 주어지면 0~100 사이의 다운로드 진행률(%)을 실시간으로 콜백한다."""
    video_id = extract_video_id(url)
    video_dir = output_root / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    target = video_dir / "source.mp4"

    # 재사용은 "완성된 병합본 source.mp4 + 오디오 포함"일 때만 한다. 조각 파일
    # (source.f298.mp4, source.f140.m4a.part 등)은 완성본이 아니므로 절대 재사용하지 않는다.
    # (예전엔 glob("source.*")로 조각을 완성본으로 오인 → 오디오 없는 소스로 진행하다
    #  렌더 단계에서 source.mp4 없음 크래시가 났다.)
    if target.exists() and target.stat().st_size > 0 and _has_audio_stream(target):
        if on_progress:
            on_progress(100.0)
        return DownloadResult(
            video_id=video_id,
            title=video_id,
            video_path=target,
            duration_sec=_probe_duration_sec(target),
        )

    # 재다운로드 전에 이전 실패가 남긴 조각/임시 파일을 정리한다(병합 꼬임/오인 방지).
    _cleanup_partials(video_dir)

    # 영상/오디오가 별도 스트림으로 순차 다운로드되어 각각 0~100%를 다시 찍으므로,
    # 진행률 바가 뒤로 튀지 않도록 지금까지 본 최댓값만 콜백한다.
    _max_pct = [0.0]

    def _hook(d: dict) -> None:
        if on_progress is None or d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes")
        if total and downloaded:
            pct = min(100.0, downloaded / total * 100)
            if pct > _max_pct[0]:
                _max_pct[0] = pct
                on_progress(pct)

    ydl_opts = {
        # 최종 영상 박스가 1000px 폭이라 720p 소스면 화질 손실 없이 충분하다(1280→1000 다운스케일).
        # 최고화질(1080p+, 수백 MB)을 통째로 받으면 링크 넣은 직후 다운로드가 몇 분씩 걸려
        # 크리티컬 패스를 잡아먹으므로, 720p로 상한을 둬서 다운로드 용량/시간을 대폭 줄인다.
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        # yt-dlp 기본 정렬은 "코덱 효율" 위주라 저비트레이트 AV1을 고비트레이트 H264보다
        # 우선시할 때가 있다 (예: 515kbps AV1을 1592kbps H264보다 선호) — 실제로는 훨씬
        # 흐릿하게 나오므로, 해상도 다음으로 비트레이트(tbr)를 명시적으로 우선시한다.
        "format_sort": ["res", "tbr"],
        "outtmpl": str(target.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        # 조각 다운로드가 네트워크 문제로 끊겨 병합이 안 되는 걸 줄인다(이번 크래시의 근본 원인).
        "retries": 5,
        "fragment_retries": 10,
        "continuedl": True,
        "quiet": False,
        "noprogress": False,
        "progress_hooks": [_hook] if on_progress else [],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # 병합 결과가 source.mp4가 아니면(단일 webm/mkv 등) mp4로 맞춰 파이프라인 계약을 항상
    # 만족시킨다(렌더는 source.mp4를 하드코딩으로 기대함).
    if not target.exists():
        singles = [p for p in video_dir.glob("source.*") if not _is_partial(p.name)]
        if singles:
            src = singles[0]
            import subprocess

            # 먼저 무손실 remux(-c copy) 시도, 실패하면 재인코딩.
            for cmd in (
                ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-i", str(src), "-c", "copy", str(target)],
                ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-i", str(src), str(target)],
            ):
                subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if target.exists() and target.stat().st_size > 0:
                    break

    # 계약 검증: 오디오 포함 source.mp4가 없으면 반쪽짜리 상태로 다음 단계로 넘기지 않고
    # 조각을 지운 뒤 명확한 오류로 실패시킨다(재시도하면 깨끗이 다시 받는다).
    if not target.exists() or target.stat().st_size == 0 or not _has_audio_stream(target):
        _cleanup_partials(video_dir)
        raise RuntimeError(
            "영상 다운로드가 완결되지 않았습니다(오디오 포함 source.mp4 생성 실패). "
            "네트워크 문제로 중단됐을 수 있으니 다시 시도하세요."
        )

    _purge_non_final(video_dir)  # 검증 통과 후 남은 조각/다른 컨테이너 정리
    return DownloadResult(
        video_id=video_id,
        title=info.get("title", video_id),
        video_path=target,
        duration_sec=float(info.get("duration") or _probe_duration_sec(target)),
    )


def download_video_hd(
    video_dir: Path, on_progress: Optional[Callable[[float], None]] = None
) -> Path | None:
    """가로 원본 렌더용 고화질(1080p 상한) 원본을 source_hd.mp4로 받는다.

    기본 source.mp4는 720p 상한이다 — 세로 쇼츠는 영상 박스가 1000px 폭이라 충분하지만,
    가로 원본 잘라내기(찬양 곡별 업로드)는 화면 전체가 1:1로 보여 720p 저비트레이트가
    그대로 드러난다(실신고 2026-09-06: "화질이 왜이래"). 렌더는 백그라운드라 다운로드
    시간이 크리티컬 패스를 안 잡으므로, 이 경로만 1080p를 따로 받는다.

    실패하면 None(호출자가 기존 source.mp4로 폴백 — 화질은 낮아도 결과는 나온다)."""
    target = video_dir / "source_hd.mp4"
    if target.exists() and target.stat().st_size > 0 and _has_audio_stream(target):
        return target
    url = f"https://www.youtube.com/watch?v={video_dir.name}"
    _max_pct = [0.0]

    def _hook(d: dict) -> None:
        if on_progress is None or d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes")
        if total and downloaded:
            pct = min(100.0, downloaded / total * 100)
            if pct > _max_pct[0]:
                _max_pct[0] = pct
                on_progress(pct)

    try:
        with yt_dlp.YoutubeDL({
            "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "format_sort": ["res", "tbr"],  # 해상도 → 비트레이트 우선(저비트레이트 AV1 회피)
            "outtmpl": str(target.with_suffix("")) + ".%(ext)s",
            "merge_output_format": "mp4",
            "retries": 5,
            "fragment_retries": 10,
            "continuedl": True,
            "quiet": True,
            "noprogress": True,
            "progress_hooks": [_hook] if on_progress else [],
        }) as ydl:
            ydl.extract_info(url, download=True)
    except Exception:  # noqa: BLE001 - 고화질 확보 실패는 치명적이지 않음(720p 폴백)
        import traceback

        traceback.print_exc()
        return None
    if target.exists() and target.stat().st_size > 0 and _has_audio_stream(target):
        return target
    # 병합 결과가 mp4가 아닌 단일 파일(webm 등)로 남았으면 mp4로 remux.
    singles = [p for p in video_dir.glob("source_hd.*") if p.suffix != ".mp4" and not _is_partial(p.name)]
    if singles:
        import subprocess

        subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-i", str(singles[0]), "-c", "copy", str(target)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if target.exists() and target.stat().st_size > 0 and _has_audio_stream(target):
            return target
    return None


if __name__ == "__main__":
    import sys

    url = sys.argv[1]
    result = download_video(url, Path("output"))
    print(result)
