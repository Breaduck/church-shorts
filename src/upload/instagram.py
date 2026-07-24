"""Instagram Reels 자동 업로드 (Phase C - 아직 미구현/비활성)

전제 조건 (config.yaml의 upload.instagram.enabled=true 로 켜기 전에 준비 필요):
  1. Instagram 비즈니스 계정 + 연결된 Facebook 페이지
  2. Meta 개발자 앱 생성 + 앱 심사 통과
     - 필요 권한: instagram_business_basic, instagram_business_content_publish
     - 심사에는 전체 플로우를 보여주는 스크린캐스트 제출 필요, 2~4주 소요
  3. 영상 파일이 공개적으로 접근 가능한 URL에 먼저 호스팅되어 있어야 함
     (Graph API는 로컬 파일 업로드가 아니라 URL을 요구함)

API 흐름 (심사 통과 후):
  1. POST /{ig-user-id}/media  (video_url, media_type=REELS, caption)
  2. POST /{ig-user-id}/media_publish (creation_id)

그 전까지는 review_app.py에서 완성된 파일 + 캡션/해시태그를 다운로드해
수동으로 인스타그램에 업로드하는 방식으로 대체한다.
"""
from __future__ import annotations

from pathlib import Path


def upload_reel(video_public_url: str, caption: str) -> str:
    raise NotImplementedError(
        "Instagram 자동 업로드는 Meta 앱 심사 통과 후 구현됩니다. "
        "그 전까지는 review_app.py에서 파일을 다운로드해 수동으로 업로드하세요."
    )
