"""
Simple Single Image OCR Client
"""

import requests
import json
from pathlib import Path
from typing import Dict, Any
import time


class SimpleOCRClient:
    """단일 이미지 OCR 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None

    def process_single_image(
        self, 
        image_path: str,
        relationship: str = "FRIEND",
        relationship_info: str = "친한 친구"
    ) -> Dict[str, Any]:
        """
        단일 이미지 처리 (세션 생성 -> 업로드 -> 처리 -> 메시지 조회)
        
        Args:
            image_path: 이미지 파일 경로
            relationship: 관계 타입 (FRIEND, FAMILY, LOVER 등)
            relationship_info: 관계 상세 정보
            
        Returns:
            처리된 메시지 데이터
        """
        print("="*60)
        print(f"📸 이미지 OCR 처리 시작")
        print("="*60)
        print(f"파일: {image_path}")
        print(f"관계: {relationship} ({relationship_info})")
        print()

        # 파일 존재 확인
        if not Path(image_path).exists():
            raise FileNotFoundError(f"파일이 없습니다: {image_path}")

        start_time = time.time()

        # 1. 세션 생성
        print("1️⃣  세션 생성 중...")
        response = requests.post(f"{self.base_url}/sessions")
        response.raise_for_status()
        self.session_id = response.json()['session_id']
        print(f"   ✓ Session ID: {self.session_id}")

        # 2. 이미지 업로드
        print("\n2️⃣  이미지 업로드 중...")
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(
                f"{self.base_url}/sessions/{self.session_id}/upload",
                files=files
            )
            response.raise_for_status()
        
        upload_data = response.json()
        print(f"   ✓ Screenshot ID: {upload_data['screenshot_id']}")

        # 3. OCR 처리
        print("\n3️⃣  OCR 처리 중... (시간이 걸릴 수 있습니다)")
        response = requests.post(
            f"{self.base_url}/sessions/{self.session_id}/process",
            params={
                "relationship": relationship,
                "relationship_info": relationship_info
            }
        )
        response.raise_for_status()
        
        process_data = response.json()
        print(f"   ✓ 총 {process_data['total_messages']}개 메시지 추출")
        print(f"   ✓ 외부 API 호출: {process_data['external_api_called']}")

        # 4. 메시지 조회
        print("\n4️⃣  메시지 조회 중...")
        response = requests.get(
            f"{self.base_url}/sessions/{self.session_id}/messages"
        )
        response.raise_for_status()
        
        messages_data = response.json()

        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ 처리 완료! ({elapsed_time:.1f}초 소요)")
        print(f"{'='*60}")

        return messages_data

    def print_messages(self, messages_data: Dict[str, Any]):
        """메시지 출력"""
        messages = messages_data.get('messages', [])
        
        if not messages:
            print("\n메시지가 없습니다.")
            return

        print(f"\n📝 추출된 메시지 ({len(messages)}개):")
        print("-"*60)

        for idx, msg in enumerate(messages, 1):
            speaker_icon = "🟢" if msg['speaker'] == 'user' else "🔵"
            speaker_name = "나" if msg['speaker'] == 'user' else "상대방"
            
            text = msg['text']
            # 긴 텍스트는 줄바꿈
            if len(text) > 50:
                text = text[:50] + "..."
            
            score_info = f" (점수: {msg['score']:.1f})" if msg.get('score') else ""
            
            print(f"{idx:3d}. {speaker_icon} {speaker_name:5s}: {text}{score_info}")

        # 통계
        user_messages = [m for m in messages if m['speaker'] == 'user']
        other_messages = [m for m in messages if m['speaker'] == 'other']
        
        print(f"\n📊 통계:")
        print(f"   전체 메시지: {len(messages)}개")
        print(f"   내 메시지: {len(user_messages)}개")
        print(f"   상대방 메시지: {len(other_messages)}개")
        
        # user 메시지 평균 점수
        if user_messages:
            scores = [m['score'] for m in user_messages if m.get('score')]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"   내 메시지 평균 점수: {avg_score:.2f}/10")

    def save_to_file(self, messages_data: Dict[str, Any], output_path: str = None):
        """결과를 JSON 파일로 저장"""
        if output_path is None:
            output_path = f"result_{self.session_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장: {output_path}")
        return output_path


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("   📱 카카오톡 스크린샷 OCR 처리")
    print("="*60 + "\n")

    # 이미지 파일 확인
    image_file = "target3.png"
    
    if not Path(image_file).exists():
        print(f"❌ 이미지 파일이 없습니다: {image_file}")
        print(f"\n사용법:")
        print(f"  1. 카카오톡 스크린샷을 '{image_file}' 이름으로 저장")
        print(f"  2. 이 스크립트 실행")
        return

    try:
        # 클라이언트 생성
        client = SimpleOCRClient()

        # 이미지 처리
        messages_data = client.process_single_image(
            image_path=image_file,
            relationship="FRIEND",
            relationship_info="친한 친구"
        )

        # 메시지 출력
        client.print_messages(messages_data)

        # 파일 저장
        output_file = client.save_to_file(messages_data)

        print(f"\n✨ 완료! Session ID: {client.session_id}")

    except requests.exceptions.ConnectionError:
        print("\n❌ API 서버에 연결할 수 없습니다.")
        print("   서버 실행: python main.py")
    except Exception as e:
        print(f"\n❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()