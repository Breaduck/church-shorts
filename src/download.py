"""유튜브 링크 -> 영상 다운로드 (yt-dlp 래퍼)"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yt_dlp


@dataclass
class DownloadResult:
    video_id: str
    title: str
    video_path: Path
    duration_sec: float


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    if not match:
        raise ValueError(f"유튜브 URL에서 video id를 찾을 수 없습니다: {url}")
    return match.group(1)


def _probe_duration_sec(path: Path) -> float:
    import subprocess

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def download_video(url: str, output_root: Path) -> DownloadResult:
    """주어진 유튜브 URL의 영상을 output_root/<video_id>/source.mp4 로 저장한다.
    이미 다운로드된 파일이 있으면 재다운로드하지 않고 재사용한다."""
    video_id = extract_video_id(url)
    video_dir = output_root / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    target = video_dir / "source.mp4"

    existing = list(video_dir.glob("source.*"))
    if existing:
        return DownloadResult(
            video_id=video_id,
            title=video_id,
            video_path=existing[0],
            duration_sec=_probe_duration_sec(existing[0]),
        )

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        # yt-dlp 기본 정렬은 "코덱 효율" 위주라 저비트레이트 AV1을 고비트레이트 H264보다
        # 우선시할 때가 있다 (예: 515kbps AV1을 1592kbps H264보다 선호) — 실제로는 훨씬
        # 흐릿하게 나오므로, 해상도 다음으로 비트레이트(tbr)를 명시적으로 우선시한다.
        "format_sort": ["res", "tbr"],
        "outtmpl": str(target.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # yt-dlp가 병합 후 실제로 만든 파일명을 확인 (확장자가 mp4가 아닐 수도 있으므로)
    resolved = video_dir / f"source.mp4"
    if not resolved.exists():
        candidates = list(video_dir.glob("source.*"))
        if not candidates:
            raise FileNotFoundError(f"다운로드된 파일을 찾을 수 없습니다: {video_dir}")
        resolved = candidates[0]

    return DownloadResult(
        video_id=video_id,
        title=info.get("title", video_id),
        video_path=resolved,
        duration_sec=float(info.get("duration") or 0.0),
    )


if __name__ == "__main__":
    import sys

    url = sys.argv[1]
    result = download_video(url, Path("output"))
    print(result)
