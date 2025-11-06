"""
Test Client for Session-based Chat OCR API V2
"""

import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import time


class ChatOCRClient:
    """Chat OCR API V2 클라이언트"""

    def __init__(self, base_url: str = "http://3.239.81.172/"):
        """
        Args:
            base_url: API 서버 URL
        """
        self.base_url = base_url
        self.session_id = None

    def create_session(self) -> Dict[str, Any]:
        """새 세션 생성"""
        print("\n" + "="*70)
        print("1. 세션 생성")
        print("="*70)

        response = requests.post(f"{self.base_url}/sessions")
        response.raise_for_status()

        data = response.json()
        self.session_id = data['session_id']

        print(f"✓ 세션 생성 완료")
        print(f"  Session ID: {self.session_id}")
        print(f"  Created at: {data['created_at']}")

        return data

    def upload_screenshots(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """스크린샷 업로드"""
        print("\n" + "="*70)
        print("2. 스크린샷 업로드")
        print("="*70)

        if not self.session_id:
            raise ValueError("세션이 생성되지 않았습니다. create_session()을 먼저 호출하세요.")

        results = []

        for idx, image_path in enumerate(image_paths, 1):
            print(f"\n[{idx}/{len(image_paths)}] 업로드: {image_path}")

            if not Path(image_path).exists():
                print(f"  ⚠ 파일이 없습니다: {image_path}")
                continue

            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                response = requests.post(
                    f"{self.base_url}/sessions/{self.session_id}/upload",
                    files=files
                )
                response.raise_for_status()

            data = response.json()
            results.append(data)

            print(f"  ✓ 업로드 완료")
            print(f"    Screenshot ID: {data['screenshot_id']}")
            print(f"    Upload Order: {data['upload_order']}")

        print(f"\n총 {len(results)}개 스크린샷 업로드 완료")
        return results

    def process_session(self, relationship: str = "FRIEND", relationship_info: str = "친한 친구") -> Dict[str, Any]:
        """세션 처리 (OCR + 병합 + 외부 API)"""
        print("\n" + "="*70)
        print("3. 세션 처리 (OCR + 병합 + 외부 API)")
        print("="*70)

        if not self.session_id:
            raise ValueError("세션이 생성되지 않았습니다.")

        print(f"대화 상대: {relationship} ({relationship_info})")
        print("처리 시작... (시간이 걸릴 수 있습니다)")
        start_time = time.time()

        response = requests.post(
            f"{self.base_url}/sessions/{self.session_id}/process",
            params={
                "relationship": relationship,
                "relationship_info": relationship_info
            }
        )
        response.raise_for_status()

        elapsed_time = time.time() - start_time
        data = response.json()

        print(f"\n✓ 세션 처리 완료 ({elapsed_time:.1f}초 소요)")
        print(f"  Status: {data['status']}")
        print(f"  Total Screenshots: {data['total_screenshots']}")
        print(f"  Total Messages: {data['total_messages']}")
        print(f"  External API Called: {data['external_api_called']}")

        # 병합 정보
        merge_info = data.get('merge_info', {})
        if merge_info.get('merge_history'):
            print(f"\n  병합 히스토리:")
            for history in merge_info['merge_history']:
                if history.get('overlap_found'):
                    print(f"    Step {history['step']}: {history['overlap_length']}개 메시지 겹침 발견")
                else:
                    print(f"    Step {history['step']}: 겹침 없음 (순서대로 이어붙임)")

        return data

    def get_messages(self) -> Dict[str, Any]:
        """메시지 조회"""
        print("\n" + "="*70)
        print("4. 메시지 조회")
        print("="*70)

        if not self.session_id:
            raise ValueError("세션이 생성되지 않았습니다.")

        response = requests.get(f"{self.base_url}/sessions/{self.session_id}/messages")
        response.raise_for_status()

        data = response.json()

        print(f"✓ 메시지 조회 완료")
        print(f"  Total Messages: {data['total_messages']}")
        print(f"  Total Screenshots: {data['total_screenshots']}")

        # 메시지 미리보기 (처음 5개 + 마지막 5개)
        messages = data['messages']

        if messages:
            print(f"\n  메시지 미리보기 (처음 5개):")
            for msg in messages[:5]:
                speaker_icon = "🟢" if msg['speaker'] == 'user' else "🔵"
                score_info = f" [score: {msg['score']:.1f}]" if msg.get('score') else ""
                print(f"    {speaker_icon} {msg['text'][:50]}{'...' if len(msg['text']) > 50 else ''}{score_info}")

            if len(messages) > 10:
                print(f"\n  ... ({len(messages) - 10}개 메시지 생략) ...\n")

            if len(messages) > 5:
                print(f"  메시지 미리보기 (마지막 5개):")
                for msg in messages[-5:]:
                    speaker_icon = "🟢" if msg['speaker'] == 'user' else "🔵"
                    score_info = f" [score: {msg['score']:.1f}]" if msg.get('score') else ""
                    print(f"    {speaker_icon} {msg['text'][:50]}{'...' if len(msg['text']) > 50 else ''}{score_info}")

            # user 메시지 통계
            user_messages = [msg for msg in messages if msg['speaker'] == 'user']
            if user_messages:
                scores = [msg['score'] for msg in user_messages if msg.get('score')]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"\n  User 메시지 평균 점수: {avg_score:.2f}")

        return data

    def search_by_screenshot(self, search_image_path: str) -> Dict[str, Any]:
        """스크린샷으로 검색"""
        print("\n" + "="*70)
        print("5. 스크린샷으로 검색")
        print("="*70)

        if not self.session_id:
            raise ValueError("세션이 생성되지 않았습니다.")

        if not Path(search_image_path).exists():
            raise FileNotFoundError(f"검색 이미지가 없습니다: {search_image_path}")

        print(f"검색 이미지: {search_image_path}")

        with open(search_image_path, 'rb') as f:
            files = {'file': (Path(search_image_path).name, f, 'image/jpeg')}
            response = requests.post(
                f"{self.base_url}/sessions/{self.session_id}/search",
                files=files
            )
            response.raise_for_status()

        data = response.json()

        print(f"✓ 검색 완료")
        print(f"  Matched: {data['matched']}")
        print(f"  Message: {data['message']}")

        if data.get('results'):
            print(f"\n  매칭된 메시지:")
            for msg in data['results'][:10]:  # 최대 10개만 출력
                speaker_icon = "🟢" if msg['speaker'] == 'user' else "🔵"
                print(f"    {speaker_icon} {msg['text'][:60]}{'...' if len(msg['text']) > 60 else ''}")

        return data

    def save_results_to_file(self, messages_data: Dict[str, Any], output_path: str = "session_result.json"):
        """결과를 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 결과 저장: {output_path}")


def main():
    """메인 테스트 함수"""
    print("="*70)
    print("    Chat OCR API V2 - 세션 기반 다중 스크린샷 테스트")
    print("="*70)

    # 이미지 파일 확인
    image_files = ["target.jpg", "target1.jpg"]
    existing_files = [f for f in image_files if Path(f).exists()]

    if not existing_files:
        print(f"\n⚠ 테스트 이미지 파일이 없습니다.")
        print(f"   다음 파일들을 준비해주세요: {', '.join(image_files)}")
        return

    print(f"\n📁 사용 가능한 이미지 파일: {', '.join(existing_files)}")

    try:
        # 클라이언트 생성
        client = ChatOCRClient()

        # 1. 세션 생성
        session_data = client.create_session()

        # 2. 스크린샷 업로드
        upload_results = client.upload_screenshots(existing_files)

        # 3. 세션 처리 (relationship, relationship_info 지정)
        process_result = client.process_session(
            relationship="FRIEND",
            relationship_info="2년 지기"
        )

        # 4. 메시지 조회
        messages_data = client.get_messages()

        # 5. 결과 저장
        output_file = f"session_{client.session_id}_result.json"
        client.save_results_to_file(messages_data, output_file)

        # 6. (선택) 첫 번째 이미지로 검색 테스트
        if existing_files:
            print("\n" + "="*70)
            print("6. 검색 기능 테스트 (첫 번째 이미지로)")
            print("="*70)
            search_result = client.search_by_screenshot(existing_files[0])

        print("\n" + "="*70)
        print("✓ 모든 테스트 완료!")
        print("="*70)
        print(f"\nSession ID: {client.session_id}")
        print(f"결과 파일: {output_file}")

    except requests.exceptions.ConnectionError:
        print("\n❌ API 서버에 연결할 수 없습니다.")
        print("   서버가 실행 중인지 확인하세요: python main.py")

    except Exception as e:
        print(f"\n❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
