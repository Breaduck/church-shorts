"""YouTube Data API v3를 통한 쇼츠 업로드 (OAuth, 최초 1회 인증 필요)"""
from __future__ import annotations

import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",  # 업로드 후 조회수/좋아요/댓글 통계 조회용
]
TOKEN_PATH = Path("secrets/youtube_token.json")


def get_credentials() -> Credentials:
    client_secrets_path = os.environ.get(
        "YOUTUBE_CLIENT_SECRETS_PATH", "secrets/youtube_client_secret.json"
    )
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(client_secrets_path).exists():
                raise RuntimeError(
                    "YouTube 업로드에 필요한 OAuth 인증 파일이 없습니다: "
                    f"{client_secrets_path}\n"
                    "이 파일은 Google에서 직접 발급받아야 합니다(코드로 만들 수 없음):\n"
                    "1) https://console.cloud.google.com 에서 프로젝트 생성\n"
                    "2) 'API 및 서비스 > 라이브러리'에서 'YouTube Data API v3' 사용 설정\n"
                    "3) 'OAuth 동의 화면' 구성(외부, 테스트 사용자에 본인 Gmail 추가)\n"
                    "4) '사용자 인증 정보 > 사용자 인증 정보 만들기 > OAuth 클라이언트 ID > "
                    "애플리케이션 유형: 데스크톱 앱' 생성\n"
                    "5) 내려받은 JSON을 이 경로에 저장: "
                    f"{Path(client_secrets_path).resolve()}\n"
                    "저장 후 업로드 버튼을 다시 누르면 브라우저에서 1회 로그인 창이 뜹니다."
                )
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return creds


def upload_short(
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    category_id: str = "22",
    privacy_status: str = "unlisted",
) -> str:
    """쇼츠를 업로드하고 video id를 반환한다."""
    creds = get_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)

    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
    return response["id"]


def fetch_video_stats(youtube_video_id: str) -> dict:
    """업로드된 영상의 조회수/좋아요/댓글 수를 가져온다 (YouTube Data API v3, statistics part).
    저장(saves)/공유(shares)/평균 조회율(retention_pct)은 이 API로 얻을 수 없다
    (YouTube Analytics API/Studio 전용 지표라서 Data API v3 statistics에는 없음)."""
    creds = get_credentials()
    youtube = build("youtube", "v3", credentials=creds)
    resp = youtube.videos().list(part="statistics", id=youtube_video_id).execute()
    items = resp.get("items", [])
    if not items:
        raise RuntimeError(f"영상을 찾을 수 없습니다: {youtube_video_id}")
    stats = items[0]["statistics"]
    return {
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats["likeCount"]) if "likeCount" in stats else None,
        "comments": int(stats["commentCount"]) if "commentCount" in stats else None,
    }
