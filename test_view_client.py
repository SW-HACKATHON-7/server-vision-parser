"""
View Endpoint Test Client - 부분 스크린샷으로 AI 분석 결과 조회
"""

import requests
import json
from pathlib import Path
from typing import List, Dict, Any


class ViewClient:
    """View 엔드포인트 테스트 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def view_screenshots(
        self,
        session_id: str,
        image_paths: List[str]
    ) -> Dict[str, Any]:
        """
        이미 분석된 세션에서 부분 스크린샷으로 결과 조회

        Args:
            session_id: 세션 ID (이미 process가 완료된 세션)
            image_paths: 조회할 스크린샷 경로들

        Returns:
            매칭된 메시지 데이터 (AI 분석 결과 포함)
        """
        print("="*60)
        print(f"📋 View 요청")
        print("="*60)
        print(f"Session ID: {session_id}")
        print(f"스크린샷 개수: {len(image_paths)}개")
        print()

        # 파일 존재 확인
        for path in image_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"파일이 없습니다: {path}")

        # View 요청
        print("🔍 View 요청 중...")
        files = [
            ('files', (Path(path).name, open(path, 'rb'), 'image/jpeg'))
            for path in image_paths
        ]

        try:
            response = requests.post(
                f"{self.base_url}/sessions/{session_id}/view",
                files=files
            )
            response.raise_for_status()

            result = response.json()

            print(f"\n✅ View 완료!")
            print(f"  - OCR 추출: {result['total_ocr_extracted']}개")
            print(f"  - 매칭 성공: {result['total_matched']}개")
            print(f"  - 매칭 여부: {'✓' if result['matched'] else '✗'}")

            return result

        finally:
            # 파일 핸들 닫기
            for _, (_, file_obj, _) in files:
                file_obj.close()

    def print_messages(self, result: Dict[str, Any]):
        """메시지 출력"""
        messages = result.get('messages', [])

        if not messages:
            print("\n매칭된 메시지가 없습니다.")
            return

        print(f"\n📝 매칭된 메시지 ({len(messages)}개):")
        print("-"*80)

        for idx, msg in enumerate(messages, 1):
            speaker_icon = "🟢" if msg['speaker'] == 'user' else "🔵"
            speaker_name = "나" if msg['speaker'] == 'user' else "상대방"

            text = msg['text']
            if len(text) > 50:
                text = text[:50] + "..."

            # AI 분석 결과
            score_str = ""
            if msg.get('score'):
                score_str = f" [점수: {msg['score']:.1f}/10]"

            emotional_tone_str = ""
            if msg.get('emotional_tone'):
                emotional_tone_str = f" [감정: {msg['emotional_tone']}]"

            print(f"{idx:3d}. {speaker_icon} {speaker_name:5s}: {text}{score_str}{emotional_tone_str}")

            # AI 코멘트
            if msg.get('ai_message'):
                print(f"      💬 AI: {msg['ai_message'][:80]}...")

            # 대안 제안
            if msg.get('suggested_alternative'):
                print(f"      💡 제안: {msg['suggested_alternative'][:80]}...")

        print("-"*80)

        # 통계
        user_messages = [m for m in messages if m['speaker'] == 'user']

        print(f"\n📊 통계:")
        print(f"   전체 매칭된 메시지: {len(messages)}개")
        print(f"   내 메시지: {len(user_messages)}개")

        if user_messages:
            scores = [m['score'] for m in user_messages if m.get('score')]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"   평균 점수: {avg_score:.2f}/10")

    def save_to_file(self, result: Dict[str, Any], output_path: str = None):
        """결과를 JSON 파일로 저장"""
        if output_path is None:
            session_id = result.get('session_id', 'unknown')
            output_path = f"view_result_{session_id}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n💾 결과 저장: {output_path}")
        return output_path


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("   📱 View 엔드포인트 테스트")
    print("="*60 + "\n")

    # 설정
    SESSION_ID = input("세션 ID 입력 (이미 process 완료된 세션): ").strip()
    if not SESSION_ID:
        print("❌ 세션 ID를 입력해주세요.")
        return

    # 조회할 스크린샷 경로
    print("\n조회할 스크린샷 경로들을 입력하세요 (쉼표로 구분):")
    print("예: target1.png, target2.png")
    paths_input = input("> ").strip()

    if not paths_input:
        print("❌ 스크린샷 경로를 입력해주세요.")
        return

    image_paths = [p.strip() for p in paths_input.split(',')]

    # 파일 존재 확인
    for path in image_paths:
        if not Path(path).exists():
            print(f"❌ 파일이 없습니다: {path}")
            return

    try:
        # 클라이언트 생성
        client = ViewClient()

        # View 요청
        result = client.view_screenshots(
            session_id=SESSION_ID,
            image_paths=image_paths
        )

        # 메시지 출력
        client.print_messages(result)

        # 파일 저장
        client.save_to_file(result)

        print(f"\n✨ 완료!")

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP 에러: {e}")
        if e.response is not None:
            print(f"   상세: {e.response.text}")
    except requests.exceptions.ConnectionError:
        print("\n❌ API 서버에 연결할 수 없습니다.")
        print("   서버 실행: python main.py")
    except Exception as e:
        print(f"\n❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
