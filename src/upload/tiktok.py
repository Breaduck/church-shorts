"""TikTok Content Posting API 자동 업로드 (Phase C - 아직 미구현/비활성)

전제 조건 (config.yaml의 upload.tiktok.enabled=true 로 켜기 전에 준비 필요):
  - 미심사(unaudited) 앱은 콘텐츠가 무조건 SELF_ONLY(비공개)로만 게시되고,
    게시 대상 계정도 비공개 상태여야 함. 공개 자동 게시를 하려면 TikTok의
    앱 audit 승인이 필요함 (소규모 앱은 승인이 불확실하고 수 주 소요될 수 있음).
  - audit 승인 전에는 draft(MEDIA_UPLOAD) 모드로 테스트만 가능.

API 흐름 (audit 통과 후):
  1. POST /v2/post/publish/video/init/  (게시 정보, 소스: PULL_FROM_URL 또는 FILE_UPLOAD)
  2. 업로드 상태 폴링

그 전까지는 편집기(web_app.py)에서 완성된 파일 + 캡션/해시태그를 다운로드해
수동으로 틱톡에 업로드하는 방식으로 대체한다.
"""
from __future__ import annotations


def upload_video(video_path: str, caption: str) -> str:
    raise NotImplementedError(
        "TikTok 자동 업로드는 앱 audit 승인 후 구현됩니다. "
        "그 전까지는 편집기(web_app.py)에서 파일을 다운로드해 수동으로 업로드하세요."
    )
